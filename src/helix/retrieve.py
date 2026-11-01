"""Keyword, embedding and hybrid retrieval.

Start with keyword search: it is fast, interpretable,
and a strong baseline, and you can always see *why* a document was retrieved.
Hybrid search combines the two with reciprocal rank fusion, which is the
default in most production RAG systems because keyword matching catches exact
terms embeddings miss, and embeddings catch meaning keywords miss.

The entitlement filter lives here. With ``Config.enforce_entitlements`` off it
does nothing, which is FP11.
"""

from __future__ import annotations

import numpy as np

from . import bm25
from .config import Config
from .ingest import Index
from .llm import cosine, embed
from tracing.schema import Chunk

#: Which access labels each role may see. A real system reads this from the
#: identity provider; a workshop reads it from a dict.
ENTITLEMENTS: dict[str, set[str]] = {
    "employee": {"all"},
    "manager": {"all"},
    "people-team": {"all", "hr-only"},
}


tokenize = bm25.tokenize


def _bm25_ranking(index: Index, query: str) -> list[int]:
    s = bm25.scores([tokenize(c.text) for c in index.chunks], tokenize(query))
    return list(np.argsort(s)[::-1])


def _dense_ranking(index: Index, query: str, cfg: Config) -> list[int]:
    qv = embed([query], cfg)[0]
    scores = [cosine(qv, v) for v in index.vectors]
    return list(np.argsort(scores)[::-1])


def _rrf(rankings: list[list[int]], k: int) -> dict[int, float]:
    """Reciprocal rank fusion (Cormack et al. 2009)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


def retrieve(
    index: Index, query: str, cfg: Config, user_role: str = "employee"
) -> tuple[list[Chunk], list[Chunk]]:
    """Return (kept, filtered_out).

    ``filtered_out`` is the set of chunks the entitlement filter removed. It
    is returned rather than silently dropped so the trace can record it: a
    filter that fires invisibly is as hard to debug as one that never runs.
    """
    if cfg.retriever == "bm25":
        rankings = [_bm25_ranking(index, query)]
    elif cfg.retriever == "dense":
        rankings = [_dense_ranking(index, query, cfg)]
    else:
        rankings = [_bm25_ranking(index, query), _dense_ranking(index, query, cfg)]

    fused = _rrf(rankings, cfg.rrf_k)
    order = sorted(fused, key=lambda i: fused[i], reverse=True)

    allowed = ENTITLEMENTS.get(user_role, {"all"})
    kept: list[Chunk] = []
    filtered: list[Chunk] = []

    for idx in order:
        chunk = index.chunks[idx]
        scored = Chunk(**{**chunk.__dict__, "score": round(fused[idx], 6)})
        if cfg.enforce_entitlements and scored.access_label not in allowed:
            filtered.append(scored)
            continue
        kept.append(scored)
        if len(kept) >= cfg.top_k:
            break

    return kept, filtered


def rerank(candidates: list[Chunk], query: str, cfg: Config) -> tuple[list[Chunk], list[Chunk]]:
    """Re-score the shortlist against the query and keep the top few.

    Production systems use a cross-encoder here. This uses BM25 scored over
    the candidate set alone, which is enough to show the behaviour that
    matters for evaluation: a rare query term such as an office name
    outweighs a common one such as "leave", and the reranker therefore
    reorders what the first-stage retriever handed it.

    The teaching point is the same whatever the model: reranking *drops*
    candidates, and what it drops has to be logged, or FP2 is invisible.
    """
    if not candidates:
        return [], []
    s = bm25.scores([tokenize(c.text) for c in candidates], tokenize(query))
    order = sorted(range(len(candidates)), key=lambda i: s[i], reverse=True)
    ranked = [candidates[i] for i in order]
    return ranked[: cfg.rerank_cutoff], ranked[cfg.rerank_cutoff :]


def relevance(chunks: list[Chunk], query: str) -> dict[str, float]:
    """BM25 relevance per chunk id. Used by the Day 1 triage detectors to ask
    whether a dropped chunk was more relevant than what survived."""
    if not chunks:
        return {}
    s = bm25.scores([tokenize(c.text) for c in chunks], tokenize(query))
    return {c.chunk_id: float(v) for c, v in zip(chunks, s)}
