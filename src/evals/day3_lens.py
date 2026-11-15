"""Day 3 through the toolkits: every curriculum item, ours beside DeepEval, G-Eval and Ragas.

Day 3's curriculum names nineteen things to evaluate, from Precision@k to task
completion. For each one this module records the workshop's own instrument
and the DeepEval, G-Eval and Ragas metric that measures the same thing, or a
dash where no toolkit has one. `python run.py toolkits --module 07|08|09`
prints the map and runs the toolkit side on one or two questions;
the deck's coverage slide is drawn from the same table.

It also holds the arithmetic the toolkits' contextual metrics use, so the
reference run can show what those metrics give when the judge agrees with
people. Both toolkits compute the same thing (read in their source at the
pinned versions):

    Ragas 0.4.3   ContextPrecisionWithReference._calculate_average_precision
    DeepEval 4.2.8  ContextualPrecisionMetric._calculate_score

    score = sum over relevant positions k of (relevant in top k / k)
            ------------------------------------------------------------
                    number of relevant chunks IN THE LIST

The denominator is the relevant chunks the list contains, not the relevant
chunks that exist. A chunk retrieval never returned cannot lower it; that is
what recall is for. Our MAP divides by every relevant chunk in the corpus.

No toolkit import here: this runs in the core environment and offline.
"""

from __future__ import annotations

from dataclasses import dataclass

from evals.day3_retrieval import ranking, relevant, stage_list
from tracing.schema import Chunk, Trace


@dataclass(frozen=True)
class Item:
    module: str
    item: str          # the curriculum's words
    ours: str          # the workshop's instrument
    deepeval: str      # DeepEval metric, or "-"
    geval: str         # G-Eval criterion, or "-"
    ragas: str         # Ragas metric, or "-"
    where: str         # slide numbers in the Day 3 deck


#: The Day 3 curriculum, item by item. "-" means the toolkit has no metric
#: for it; the reason is in NO_TOOLKIT.
CURRICULUM: tuple[Item, ...] = (
    Item("07", "Precision@k, Recall@k, F1@k", "P@5, R@5, F1@5 on the ranked list", "-", "-", "-", "7.2, 7.3"),
    Item("07", "MRR, MAP, nDCG", "MRR, MAP and nDCG@5, relevance over the whole corpus", "-", "-", "-",
         "7.2, 7.4"),
    Item("07", "Contextual metrics", "relevance written by people in the suite",
         "contextual precision, recall, relevancy", "-", "context precision, context recall", "7.7, 7.10"),
    Item("07", "Ranking and reranking", "one score per stage; nDCG before and after rerank",
         "contextual precision per stage", "-", "context precision per stage", "7.4, 7.10"),
    Item("08", "Correctness", "correctness checker; our judge", "-", "correctness", "factual correctness",
         "8.9, 8.16"),
    Item("08", "Relevance", "direct-answer check", "answer relevancy", "-", "answer relevancy", "8.12, 8.15, 8.16"),
    Item("08", "Completeness", "nugget check", "-", "completeness", "factual correctness, recall mode (note)",
         "8.11, 8.16"),
    Item("08", "Faithfulness", "word checker; citation support; our judge", "faithfulness", "-", "faithfulness",
         "8.3, 8.4, 8.16"),
    Item("08", "Context utilization", "each nugget used or OMITTED; the context swap", "-", "-",
         "context utilization", "8.11, 8.13, 8.16"),
    Item("08", "Citations", "citation support; lineage (version, access)", "citation faithfulness (community)",
         "citation support", "-",
         "8.12, 8.16"),
    Item("08", "Consistency", "the same question N times, facts compared", "-", "-", "-", "8.13"),
    Item("08", "Unsupported vs faithful to stale evidence", "grounded crossed with correct; lineage",
         "faithfulness", "correctness", "faithfulness, factual correctness", "8.9, 8.10, 8.16"),
    Item("09", "Adversarial stress cases", "23 behavioural tests, 20 injection patterns, the planted page",
         "faithfulness on the planted page", "correctness on the planted page", "noise sensitivity",
         "9.4 to 9.6, 9.14"),
    Item("09", "Evaluator red-teaming", "7 attacks on 8 failing answers, every instrument",
         "faithfulness under attack", "correctness under attack", "factual correctness under attack",
         "9.7, 9.8, 9.14"),
    Item("09", "GraphRAG patterns", "six architectures, GR codes, the six-question checklist",
         "-", "correctness on graph answers", "faithfulness on graph facts", "9.9, 9.13, 9.14"),
    Item("09", "Tool routing", "Select and Args per tool call", "tool correctness, argument correctness", "-",
         "tool call F1", "9.10, 9.14"),
    Item("09", "Trajectories", "four stages per call, first failure, heatmap",
         "tool correctness in order (subsequence)", "-", "tool call accuracy (exact order)",
         "9.10, 9.11, 9.14"),
    Item("09", "Retries, fallback, recovery", "planted faults, retries, re-search, budget",
         "task completion under faults", "-", "goal accuracy under faults", "9.12, 9.14"),
    Item("09", "Task completion", "right or wrong against the suite", "task completion", "-", "goal accuracy",
         "9.10, 9.14"),
)

#: Why a toolkit column is empty, or what to know before reading it.
NO_TOOLKIT = {
    "Precision@k, Recall@k, F1@k": "DeepEval has none. Ragas has ID-based context precision and recall "
                                   "only in its deprecated legacy module (not in ragas.metrics.collections); "
                                   "on a top-k list they are set ratios, P@k and R@k. We compute them ourselves",
    "MRR, MAP, nDCG": "neither has MRR or nDCG; their context precision averaged over questions is close "
                      "to MAP, but divides by the relevant chunks in each list",
    "Completeness": "Ragas recall mode counts the answer's claims the reference supports, over those plus "
                    "the reference's claims the answer misses: close to coverage, not equal to it",
    "Citations": "DeepEval's citation faithfulness reads [N] markers and gives one verdict per answer, so "
                 "Helix's chunk-id citations are renumbered first; G-Eval is written for the chunk ids and "
                 "scores each cited sentence. Ragas's quoted-span alignment needs quotes, which Helix never writes",
    "Consistency": "one answer cannot show it: run the question N times (day3 --only consistency)",
    "Trajectories": "DeepEval's order allows extra calls between the expected ones; Ragas's needs the exact "
                    "sequence. DeepEval's step and plan metrics need DeepEval's own tracing, not this trace",
}


def by_module(module: str) -> list[Item]:
    return [i for i in CURRICULUM if i.module == module]


# ---------------------------------------------------------------------------
# Module 07: the toolkits' contextual precision, computed with people's grades
# ---------------------------------------------------------------------------
def toolkit_precision(verdicts: list[int]) -> float:
    """Average precision as both toolkits compute it: over relevant chunks IN the list.

    Returns 0.0 for a list with no relevant chunk, as both do.
    """
    hits, total = 0, 0.0
    for k, v in enumerate(verdicts, 1):
        if v:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


@dataclass
class Stage:
    name: str
    chunks: list[Chunk]
    verdicts: list[int]           # 1 where people call the chunk relevant
    precision: float              # toolkit_precision on people's verdicts
    recall: float | None          # relevant in this list / relevant in the corpus


def stages(trace: Trace, grades: dict[str, int]) -> list[Stage]:
    """The three lists a ranking passes through, each with people's verdicts.

    retrieved   the `retrieve` span's ranking (top 5)
    reranked    what the reranker kept, in its order
    reached     what context assembly gave the model
    """
    rel = relevant(grades)
    pool = {c.chunk_id: c for c in [*trace.considered, *trace.retrieved]}
    lists = [("retrieved", stage_list(trace, "retrieve") or ranking(trace)),
             ("reranked", stage_list(trace, "rerank")),
             ("reached the model", [c.chunk_id for c in trace.retrieved if not c.dropped_at])]
    out = []
    for name, ids in lists:
        chunks = [pool[i] for i in ids if i in pool]
        if not chunks:
            continue
        v = [1 if c.chunk_id in rel else 0 for c in chunks]
        out.append(Stage(name, chunks, v, toolkit_precision(v),
                         sum(v) / len(rel) if rel else None))
    return out


def verdict_string(verdicts: list[int]) -> str:
    """People's verdicts down the list: 'no no YES' for a hit at rank 3."""
    return " ".join("YES" if v else "no" for v in verdicts)
