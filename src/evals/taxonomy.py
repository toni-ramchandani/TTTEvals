"""The failure-point vocabulary used across all four days.

FP1 to FP7 are the seven failure points Barnett et al. catalogued from three
production RAG deployments (CAIN 2024). FP8 to FP11 are the four this workshop
adds for enterprise corpora, where the documents are owned by someone else,
contradict each other, and carry access rules.

This numbering is load-bearing: Day 2 builds its taxonomy on top of it, Day 3
chooses metrics per failure point, and Day 4 gates on them. Renaming anything
here means changing all four days.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FailurePoint:
    code: str
    name: str
    born_at: str
    looks_like: str
    trace_evidence: str


CATALOG: list[FailurePoint] = [
    FailurePoint(
        "FP1", "Missing content", "corpus",
        "A confident answer the documents cannot support",
        "No retrieved chunk contains the fact",
    ),
    FailurePoint(
        "FP2", "Missed top-ranked documents", "retrieval",
        "The answer exists in the corpus but ranked below the cut-off",
        "The supporting chunk appears in rerank.dropped",
    ),
    FailurePoint(
        "FP3", "Not in context", "context.assemble",
        "Retrieved, then dropped in consolidation or truncation",
        "The supporting chunk appears in context.assemble.dropped",
    ),
    FailurePoint(
        "FP4", "Not extracted", "generate",
        "It was in the context and the model did not use it",
        "The supporting chunk is in retrieved but absent from the answer",
    ),
    FailurePoint(
        "FP5", "Wrong format", "generate",
        "Instructions about structure ignored",
        "Answer violates the output format in the prompt template",
    ),
    FailurePoint(
        "FP6", "Incorrect specificity", "generate",
        "Too general or too specific for what was asked",
        "Answer granularity does not match the question",
    ),
    FailurePoint(
        "FP7", "Incomplete", "generate",
        "Part of the available answer omitted",
        "Context holds more of the answer than the response gives",
    ),
    FailurePoint(
        "FP8", "Stale or superseded evidence", "ingestion",
        "Faithful to a document that no longer reflects policy",
        "doc_version / last_modified on the cited chunk is behind the current one",
    ),
    FailurePoint(
        "FP9", "Contradictory evidence", "corpus",
        "Two valid-looking sources disagree and the system silently picks one",
        "Two retrieved chunks from different docs assert different values",
    ),
    FailurePoint(
        "FP10", "Severed context", "chunking",
        "The qualifier and the fact ended up in different chunks",
        "The cited chunk's heading_path names a scope its text does not carry",
    ),
    FailurePoint(
        "FP11", "Entitlement leakage", "retrieval",
        "Content surfaced to a user who should not see it",
        "A retrieved chunk's access_label is outside the user's entitlement",
    ),
]

#: The failure points born before the model is called. Seven of eleven.
BORN_BEFORE_GENERATION = [fp.code for fp in CATALOG if fp.born_at != "generate"]


# ---------------------------------------------------------------------------
# Day 3: failure codes that only exist in the other two architectures.
#
# FP1 to FP11 still apply to agentic RAG and GraphRAG: both retrieve, both
# generate, both can cite a stale page or leak a restricted one. These are the
# failures each architecture adds. They are born in places vanilla RAG does
# not have: the agent's choice of tool and argument, and the graph's build.
# ---------------------------------------------------------------------------
AGENT_CATALOG: list[FailurePoint] = [
    FailurePoint(
        "AG1", "Missed tool call", "agent.plan",
        "An answer that would have been right with one more lookup",
        "A tool the question needed (profile, document status) never appears in the step sequence",
    ),
    FailurePoint(
        "AG2", "Wrong tool argument", "agent.step",
        "The right tool, pointed at the wrong thing",
        "A search_handbook step's office argument differs from the office the question names",
    ),
    FailurePoint(
        "AG3", "Premature finish", "agent.plan",
        "One part of a multi-part question answered, the rest silently dropped",
        "Fewer search steps than the question has parts, then finish",
    ),
    FailurePoint(
        "AG4", "Step budget exhausted", "agent.run",
        "A non-answer on exactly the questions that needed the most work",
        "agent.run.budget_exhausted is true",
    ),
    FailurePoint(
        "AG5", "Observation ignored", "agent.step",
        "The agent was told, and did not listen",
        "check_document_status returned superseded for a document the answer still cites",
    ),
]

GRAPH_CATALOG: list[FailurePoint] = [
    FailurePoint(
        "GR1", "Entity linking miss", "graph.link",
        "The graph knows the answer under a name the question did not use",
        "An office named or implied by the question is absent from graph.link.offices_linked",
    ),
    FailurePoint(
        "GR2", "Extraction error", "graph build",
        "A fact missed, or written down against the wrong entity, once for every future question",
        "The cited fact is company-wide but its source page is office-specific, or no fact holds the answer",
    ),
    FailurePoint(
        "GR3", "Stale fact traversed", "graph build / graph.filter",
        "A superseded page's figure, presented as a graph fact",
        "The cited fact's doc_id is superseded in the corpus",
    ),
    FailurePoint(
        "GR4", "Access label lost", "graph build",
        "A restricted figure served as an ordinary fact",
        "The cited fact's source page is restricted although the fact's own label says all",
    ),
    FailurePoint(
        "GR5", "Coverage gap", "graph schema / graph.traverse",
        "A refusal for a question the documents can answer",
        "The graph has no fact on the topic, or none within reach, and nothing fell back",
    ),
]

BY_CODE: dict[str, FailurePoint] = {
    fp.code: fp for fp in CATALOG + AGENT_CATALOG + GRAPH_CATALOG
}
