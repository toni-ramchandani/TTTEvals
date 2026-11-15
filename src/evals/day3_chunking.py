"""Day 3, Module 07: chunking as a variable you measure.

The same questions, the same relevance judgments (the suite's expected
evidence), and one ingestion setting changed at a time. Each configuration
re-chunks and re-indexes the handbook and is scored with the retrieval
metrics from `day3_retrieval`: Recall@5 for the first stage, nDCG@5 for the
order, and whether the evidence survived to the model under the default
context budget.

Two things only a grid shows:

* chunks too small scatter a fact over two chunks, so no chunk carries all
  the evidence and the question cannot be answered from any single chunk;
* chunks too large carry the fact with a lot of noise, and they use up the
  context budget sooner, so a relevant chunk can be retrieved and still
  dropped.

No answer is generated: this measures retrieval and context assembly only.
Live, each configuration costs one embedding pass over the handbook plus one
embedding per question.
"""

from __future__ import annotations

from dataclasses import replace

from helix.config import DEFAULT, Config
from helix.ingest import build_index
from helix.retrieve import rerank, retrieve
from helix.llm import count_tokens

from .day3_retrieval import (
    hit_at_k,
    judgments,
    mean,
    ndcg_at_k,
    recall_at_k,
    relevant,
)

#: (label, changes from the default). One family of settings at a time.
GRID: list[tuple[str, dict]] = [
    ("fixed 60, no overlap", {"chunk_strategy": "fixed", "chunk_tokens": 60, "chunk_overlap": 0}),
    ("fixed 60, overlap 30", {"chunk_strategy": "fixed", "chunk_tokens": 60, "chunk_overlap": 30}),
    ("fixed 110, no overlap (default)", {}),
    ("fixed 110, overlap 55", {"chunk_strategy": "fixed", "chunk_tokens": 110, "chunk_overlap": 55}),
    ("fixed 220, no overlap", {"chunk_strategy": "fixed", "chunk_tokens": 220, "chunk_overlap": 0}),
    ("fixed 220, overlap 110", {"chunk_strategy": "fixed", "chunk_tokens": 220, "chunk_overlap": 110}),
    ("structural, headings off", {"chunk_strategy": "structural", "prepend_heading": False}),
    ("structural, headings on", {"chunk_strategy": "structural", "prepend_heading": True}),
]


def configs(base: Config = DEFAULT) -> list[tuple[str, Config]]:
    return [(label, replace(base, **changes)) for label, changes in GRID]


def first_stages(index, query: str, cfg: Config, user_role: str) -> dict[str, list[str]]:
    """The pipeline's retrieve, rerank and context stages, without generation."""
    candidates, _ = retrieve(index, query, cfg, user_role=user_role)
    ranked, _ = rerank(candidates, query, cfg)
    reached, used = [], 0
    for c in ranked:
        cost = count_tokens(c.text)
        if used + cost > cfg.context_token_budget:
            continue
        reached.append(c.chunk_id)
        used += cost
    return {"retrieve": [c.chunk_id for c in candidates], "reached": reached}


def score_config(cfg: Config, suite: list[dict]) -> dict:
    """Retrieval metrics for one chunking configuration, over the suite."""
    index = build_index(cfg)
    rows, no_chunk = [], []
    for q in suite:
        ev = q.get("expected_evidence")
        if not ev:
            continue
        grades = judgments(index.chunks, ev)
        rel = relevant(grades)
        if not rel:
            # The evidence exists in the handbook, but no single chunk carries
            # all of it: the fact was split across chunks.
            no_chunk.append(q["id"])
            continue
        st = first_stages(index, q["text"], cfg, q.get("user_role", "employee"))
        rows.append({
            "id": q["id"],
            "recall": recall_at_k(st["retrieve"], rel, 5),
            "hit": hit_at_k(st["retrieve"], rel, 5),
            "ndcg": ndcg_at_k(st["retrieve"], grades, 5),
            "reached": bool(rel & set(st["reached"])),
        })
    asked = sum(1 for q in suite if q.get("expected_evidence"))
    return {
        "chunks": len(index.chunks),
        "recall": mean([r["recall"] for r in rows] + [0.0] * len(no_chunk)),
        "ndcg": mean([r["ndcg"] for r in rows] + [0.0] * len(no_chunk)),
        "hit": mean([r["hit"] for r in rows] + [0.0] * len(no_chunk)),
        "reached": sum(1 for r in rows if r["reached"]),
        "asked": asked,
        "split": no_chunk,
        "rows": rows,
    }
