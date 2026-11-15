"""Day 3, Module 07: retrieval and ranking metrics.

Two questions, kept apart on purpose:

  ranking   Did retrieval put the relevant chunks near the top? Measured on
            the ranked list the `retrieve` span recorded (its `kept` ids, in
            rank order), not on what reached the model.
  fate      What happened to each relevant chunk afterwards? Reached the
            model, dropped at a named stage, or never retrieved at all.

Relevance is a human judgment, made before any metric runs. The suite writes
it down as expected-evidence phrases beside each question, and a chunk's
grade is read from them over the WHOLE corpus, because recall needs every
relevant chunk, including the ones retrieval never returned:

  2  relevant: the chunk carries all of the expected evidence
  1  partly relevant: it carries most of it (used only as a smaller gain in nDCG)
  0  not relevant

The binary metrics (P@k, R@k, RR, AP) count grade 2 only. nDCG uses both
grades, which is the case it was designed for (Järvelin and Kekäläinen 2002).
The formulas are short and worth seeing once, so they are written out here.
"""

from __future__ import annotations

import math

from tracing.schema import Chunk, Trace

RELEVANT, PARTLY = 2, 1


# ---------------------------------------------------------------------------
# Relevance judgments
# ---------------------------------------------------------------------------
def needles(evidence: str | None) -> list[str]:
    return [p.strip().lower() for p in (evidence or "").split("|") if p.strip()]


def grade(chunk: Chunk, evidence: str | None) -> int:
    """2 if the chunk carries all of the expected evidence, 1 if most, else 0."""
    parts = needles(evidence)
    if not parts:
        return 0
    # Whitespace is normalised: a structural chunk keeps the page's line
    # breaks, and a phrase that wraps onto the next line is still the phrase.
    body = " ".join((chunk.text + " " + chunk.heading_path).lower().split())
    found = sum(1 for p in parts if " ".join(p.split()) in body)
    if found == len(parts):
        return RELEVANT
    if found >= max(1, (len(parts) + 1) // 2):
        return PARTLY
    return 0


def judgments(chunks: list[Chunk], evidence: str | None) -> dict[str, int]:
    """Grades for every chunk in `chunks` that is at least partly relevant."""
    out = {}
    for c in chunks:
        g = grade(c, evidence)
        if g:
            out[c.chunk_id] = g
    return out


def relevant(grades: dict[str, int]) -> set[str]:
    return {cid for cid, g in grades.items() if g >= RELEVANT}


# ---------------------------------------------------------------------------
# The ranked list
# ---------------------------------------------------------------------------
def ranking(trace: Trace) -> list[str]:
    """Chunk ids in the order retrieval ranked them (the `retrieve` span)."""
    s = trace.span("retrieve")
    if s is not None and s.kept:
        return list(s.kept)
    return [c.chunk_id for c in trace.retrieved]


def precision_at_k(ranked: list[str], rel: set[str], k: int) -> float:
    """Share of the top k that is relevant. The denominator is k."""
    return sum(1 for cid in ranked[:k] if cid in rel) / k if k else 0.0


def recall_at_k(ranked: list[str], rel: set[str], k: int) -> float | None:
    """Share of all relevant chunks in the corpus that appear in the top k."""
    if not rel:
        return None
    return sum(1 for cid in ranked[:k] if cid in rel) / len(rel)


def reciprocal_rank(ranked: list[str], rel: set[str]) -> float:
    """1 / rank of the first relevant chunk; 0 when none was retrieved."""
    for i, cid in enumerate(ranked, 1):
        if cid in rel:
            return 1.0 / i
    return 0.0


def average_precision(ranked: list[str], rel: set[str]) -> float | None:
    """Mean of precision@k at the rank of each relevant chunk, over ALL relevant.

    A relevant chunk that was never retrieved contributes 0, so AP falls
    when retrieval misses evidence, not only when it ranks it low.
    """
    if not rel:
        return None
    hits, total = 0, 0.0
    for i, cid in enumerate(ranked, 1):
        if cid in rel:
            hits += 1
            total += hits / i
    return total / len(rel)


def ndcg_at_k(ranked: list[str], grades: dict[str, int], k: int) -> float | None:
    """Discounted cumulative gain at k, divided by the ideal ranking's."""
    if not grades:
        return None
    dcg = sum(grades.get(cid, 0) / math.log2(i + 2) for i, cid in enumerate(ranked[:k]))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else None


def hit_at_k(ranked: list[str], rel: set[str], k: int) -> float | None:
    """1 if any relevant chunk is in the top k, else 0. Its mean is the hit rate.

    Often the most practical first metric: for a question one chunk can
    answer, one relevant chunk in the context is enough.
    """
    if not rel:
        return None
    return 1.0 if any(cid in rel for cid in ranked[:k]) else 0.0


def f1_at_k(p: float | None, r: float | None) -> float | None:
    """Harmonic mean of precision@k and recall@k, at the same k."""
    if p is None or r is None:
        return None
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------------------
# Each stage scored on its own
# ---------------------------------------------------------------------------
def stage_list(trace: Trace, name: str) -> list[str]:
    s = trace.span(name)
    return list(s.kept) if s is not None else []


def stage_scores(trace: Trace, grades: dict[str, int]) -> dict:
    """Recall for the first stage, nDCG before and after rerank, and arrival.

    The first stage should find the evidence (recall); the reranker should
    put it at the top of what it keeps (nDCG on the same cutoff, before and
    after). Context assembly then decides what the model sees.
    """
    rel = relevant(grades)
    first = stage_list(trace, "retrieve") or ranking(trace)
    reranked = stage_list(trace, "rerank")
    cut = len(reranked) or 3
    reached = {c.chunk_id for c in trace.retrieved if not c.dropped_at}
    return {
        "recall_first": recall_at_k(first, rel, len(first)) if first else None,
        "ndcg_before": ndcg_at_k(first[:cut], grades, cut),
        "ndcg_after": ndcg_at_k(reranked, grades, cut) if reranked else None,
        "cut": cut,
        "reached": bool(rel & reached),
    }


# ---------------------------------------------------------------------------
# Ingestion: two versions of one document in the index
# ---------------------------------------------------------------------------
def stale_versions(corpus: list[Chunk], status: dict[str, dict]) -> list[dict]:
    """Superseded documents whose chunks are still in the index.

    Two versions of the same policy in one index is an ingestion bug: the
    old one can be retrieved and quoted as current. This finds it before
    anyone asks a question.
    """
    out: dict[str, dict] = {}
    for c in corpus:
        st = status.get(c.doc_id, {})
        if st.get("status") == "superseded":
            row = out.setdefault(c.doc_id, {"doc_id": c.doc_id, "version": st.get("doc_version"),
                                             "last_modified": st.get("last_modified"),
                                             "superseded_by": st.get("superseded_by"), "chunks": 0})
            row["chunks"] += 1
    return sorted(out.values(), key=lambda r: r["doc_id"])


# ---------------------------------------------------------------------------
# Retrieval success against answer correctness
# ---------------------------------------------------------------------------
def retrieval_vs_answer(hits: dict[str, bool], correct: dict[str, bool]) -> dict[tuple[bool, bool], list[str]]:
    """Which questions fall in each cell: (retrieved, correct) -> ids.

    Retrieved but wrong sends you to generation or context assembly. Missed
    but right means the answer did not come from the evidence at all.
    """
    cells: dict[tuple[bool, bool], list[str]] = {(a, b): [] for a in (True, False) for b in (True, False)}
    for qid, hit in hits.items():
        if qid in correct:
            cells[(hit, correct[qid])].append(qid)
    return cells


# ---------------------------------------------------------------------------
# What happened to the evidence
# ---------------------------------------------------------------------------
def fate(trace: Trace, rel: set[str]) -> dict[str, str]:
    """For each relevant chunk: reached the model, dropped at <stage>, or not retrieved."""
    seen = {c.chunk_id: c for c in trace.considered}
    reached = {c.chunk_id for c in trace.retrieved if not c.dropped_at}
    out = {}
    for cid in sorted(rel):
        if cid in reached:
            out[cid] = "reached the model"
        elif cid in seen and seen[cid].dropped_at:
            out[cid] = f"dropped at {seen[cid].dropped_at}"
        else:
            out[cid] = "not retrieved"
    return out


def context_recall_from_dropped(trace: Trace, relevant_ids: set[str]) -> dict[str, int]:
    """Where relevant evidence was lost, by stage (only stages that dropped some).

    Only computable because every stage logged what it dropped. Evidence
    that retrieval never returned is counted as `not retrieved`.
    """
    lost: dict[str, int] = {}
    for where in fate(trace, relevant_ids).values():
        if where != "reached the model":
            key = where.removeprefix("dropped at ")
            lost[key] = lost.get(key, 0) + 1
    return lost


def retrieval_row(trace: Trace, corpus: list[Chunk], evidence: str | None, k: int = 5) -> dict:
    """Everything Module 07's table prints for one question, all at the same k.

    nDCG here divides by the ideal ranking of every graded chunk in the
    corpus, so a relevant chunk that retrieval missed lowers it. Some texts
    re-sort only the retrieved chunks for the ideal; that version cannot see
    a miss, which is why it must be read beside recall.
    """
    grades = judgments(corpus, evidence)
    rel = relevant(grades)
    ranked = ranking(trace)
    p = precision_at_k(ranked, rel, k) if rel else None
    r = recall_at_k(ranked, rel, k)
    return {
        "relevant": rel,
        "grades": grades,
        "ranks": [i for i, cid in enumerate(ranked, 1) if cid in rel],
        "p": p,
        "r": r,
        "f1": f1_at_k(p, r),
        "hit": hit_at_k(ranked, rel, k),
        "rr": reciprocal_rank(ranked, rel) if rel else None,
        "ap": average_precision(ranked, rel),
        "ndcg": ndcg_at_k(ranked, grades, k),
        "fate": fate(trace, rel),
        "stages": stage_scores(trace, grades),
    }
