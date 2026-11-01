"""Day 3, Module 07: retrieval and ranking metrics.

These are rank-aware metrics over the retrieval span of each trace. They are
meaningless until the room has agreed what counts as a relevant chunk, which
is the ground-truth work done in the Day 1 charter and the Day 2 golden set.
That ordering is the reason this module is empty of judgment: it computes,
and something else decides what "relevant" means.

Implemented here because the formulas are short and worth seeing once.
"""

from __future__ import annotations

import math

from tracing.schema import Trace


def _relevant_ranks(trace: Trace, relevant_ids: set[str]) -> list[int]:
    return [
        i + 1
        for i, c in enumerate(trace.retrieved)
        if c.chunk_id in relevant_ids
    ]


def precision_at_k(trace: Trace, relevant_ids: set[str], k: int) -> float:
    top = trace.retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for c in top if c.chunk_id in relevant_ids) / len(top)


def recall_at_k(trace: Trace, relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = trace.retrieved[:k]
    return sum(1 for c in top if c.chunk_id in relevant_ids) / len(relevant_ids)


def reciprocal_rank(trace: Trace, relevant_ids: set[str]) -> float:
    ranks = _relevant_ranks(trace, relevant_ids)
    return 1.0 / ranks[0] if ranks else 0.0


def mrr(traces: list[Trace], relevant: dict[str, set[str]]) -> float:
    if not traces:
        return 0.0
    return sum(reciprocal_rank(t, relevant.get(t.trace_id, set())) for t in traces) / len(traces)


def ndcg_at_k(trace: Trace, gains: dict[str, float], k: int) -> float:
    """Normalised discounted cumulative gain with graded relevance."""
    dcg = sum(
        gains.get(c.chunk_id, 0.0) / math.log2(i + 2)
        for i, c in enumerate(trace.retrieved[:k])
    )
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def context_recall_from_dropped(trace: Trace, relevant_ids: set[str]) -> dict[str, int]:
    """Where relevant evidence was lost, by stage.

    This is the metric Day 1's trace schema was designed to make possible. It
    is only computable because every stage logged what it dropped.
    """
    lost: dict[str, int] = {}
    for c in trace.considered:
        if c.chunk_id in relevant_ids and c.dropped_at:
            lost[c.dropped_at] = lost.get(c.dropped_at, 0) + 1
    return lost
