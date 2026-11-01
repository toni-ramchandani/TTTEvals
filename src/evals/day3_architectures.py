"""Day 3: evaluating three architectures with one set of instruments.

Vanilla Helix, the agent and the graph all write the same trace schema, so the
same correctness check runs over all three. What differs is *where a failure
can be born*, and therefore which trace fields you read to attribute it:

* vanilla:  the six request-time spans (FP1 to FP11)
* agentic:  the step sequence, its arguments and its observations (AG1 to AG5)
* graph:    the link, traverse and filter spans, and the build (GR1 to GR5)

Two rules this module follows, both worth carrying into your own work:

1. **Score the answer and the path separately.** An agent can take the right
   steps and still answer wrongly (the generator misread a good extract), or
   take the wrong steps and answer rightly by luck. Trajectory metrics and
   answer metrics disagree often enough that reporting only one hides the
   other.
2. **Attribute against the corpus, not against the system's own labels.** A
   graph that lost an access label reports ``access_label="all"`` on a
   restricted fact. A detector that trusts the trace's label cannot see the
   leak. The detectors below check every cited source against the handbook
   itself.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from helix.ingest import HANDBOOK, parse
from helix.retrieve import ENTITLEMENTS
from helix.structure import offices_in, topics_in
from tracing.schema import Trace

OFFICE_DEPENDENT_TOPICS = {"parental_leave", "notice_period"}


# ---------------------------------------------------------------------------
# Ground truth about the corpus (what the corpus owner knows)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def corpus() -> dict[str, dict[str, Any]]:
    """doc_id -> access label, title, and whether the page is superseded."""
    docs = [parse(p) for p in sorted(HANDBOOK.glob("*.md"))]
    by_title: dict[str, list] = {}
    for d in docs:
        by_title.setdefault(d.title.strip().lower(), []).append(d)
    out: dict[str, dict[str, Any]] = {}
    for group in by_title.values():
        group.sort(key=lambda d: d.last_modified)
        for d in group:
            out[d.doc_id] = {
                "access_label": d.access_label,
                "title": d.title,
                "superseded_by": None if d is group[-1] else group[-1].doc_id,
            }
    return out


def _cited(trace: Trace):
    return [c for c in trace.retrieved if c.cite() in trace.answer]


# ---------------------------------------------------------------------------
# Answer correctness
# ---------------------------------------------------------------------------
@dataclass
class Grade:
    qid: str
    system: str
    correct: bool
    reasons: list[str] = field(default_factory=list)


def grade(trace: Trace, q: dict[str, Any]) -> Grade:
    text = trace.answer.lower()
    reasons: list[str] = []
    bad = [s for s in q.get("must_not_include", []) if s.lower() in text]
    if bad:
        reasons.append(f"contains {bad}")
    if q.get("should_refuse"):
        if not trace.refused:
            reasons.append("should have refused")
    else:
        if trace.refused:
            reasons.append("refused an answerable question")
        missing = [s for s in q.get("must_include", []) if s.lower() not in text]
        if missing:
            reasons.append(f"missing {missing}")
        doc = q.get("must_cite_doc")
        if doc and not any(doc in c for c in trace.citations):
            reasons.append(f"does not cite {doc}")
    return Grade(q["id"], trace.system, not reasons, reasons)


# ---------------------------------------------------------------------------
# Trajectory metrics (agentic)
# ---------------------------------------------------------------------------
def steps(trace: Trace) -> list[dict[str, Any]]:
    return [
        {"tool": s.attributes.get("tool"), "args": s.attributes.get("args", {}),
         "observation": s.attributes.get("observation"), "kept": s.kept}
        for s in trace.spans if s.name.startswith("agent.step.")
    ]


def trajectory(trace: Trace, q: dict[str, Any]) -> dict[str, Any]:
    run = trace.span("agent.run")
    seq = [s["tool"] for s in steps(trace)]
    expected = Counter(q.get("expected_tools", []))
    got = Counter(seq)
    hit = sum(min(n, got[t]) for t, n in expected.items())
    checked = {s["args"].get("doc_id") for s in steps(trace) if s["tool"] == "check_document_status"}
    cited_docs = {c.doc_id for c in _cited(trace)}
    return {
        "steps_used": run.attributes.get("steps_used") if run else len(seq),
        "budget_exhausted": bool(run and run.attributes.get("budget_exhausted")),
        "tool_recall": round(hit / sum(expected.values()), 2) if expected else 1.0,
        "searches": got["search_handbook"],
        "version_checked_share": round(len(cited_docs & checked) / len(cited_docs), 2) if cited_docs else None,
    }


# ---------------------------------------------------------------------------
# Graph metrics
# ---------------------------------------------------------------------------
def graph_metrics(trace: Trace) -> dict[str, Any]:
    link = trace.span("graph.link")
    filt = trace.span("graph.filter")
    a = link.attributes if link else {}
    return {
        "offices_linked": a.get("offices_linked", []),
        "offices_named": a.get("offices_in_question_any_alias", []),
        "profile_used": a.get("profile_used", []),
        "facts_in_context": len(trace.retrieved),
        "dropped_superseded": len(filt.attributes.get("dropped_superseded", [])) if filt else 0,
        "blocked_by_entitlement": len(filt.attributes.get("blocked_by_entitlement", [])) if filt else 0,
        "fallback": trace.span("fallback.retrieve") is not None,
        "provenance_complete": all(c.doc_version and c.last_modified for c in trace.retrieved),
    }


# ---------------------------------------------------------------------------
# Attribution: which code explains a failing trace
# ---------------------------------------------------------------------------
def _evidence_in(chunks, q: dict[str, Any]) -> bool:
    ev = (q.get("expected_evidence") or "").lower()
    need = [s.lower() for s in q.get("must_include", [])]
    for c in chunks:
        t = c.text.lower()
        if ev and ev in t:
            return True
        if need and all(s in t for s in need):
            return True
    return False


def attribute(trace: Trace, q: dict[str, Any], graph=None) -> list[str]:
    """Failure codes for a trace that ``grade`` marked wrong. Empty if correct."""
    if grade(trace, q).correct:
        return []
    if trace.system == "helix":
        # Vanilla traces get the Day 1 detectors, so a code means the same
        # thing here as it did in hands-on H1.
        from .day1_triage import triage

        found = [f.code for f in triage(trace, q.get("expected_evidence"))]
        if found:
            return list(dict.fromkeys(found))

    codes: list[str] = []
    truth = corpus()
    allowed = ENTITLEMENTS.get(trace.user_role, {"all"})
    cited = _cited(trace)

    # --- shared: born before generation, visible through lineage ---------------
    true_labels = [truth.get(c.doc_id, {}).get("access_label", c.access_label) for c in trace.retrieved]
    if any(lbl not in allowed for lbl in true_labels):
        leaked_by_label = any(c.access_label not in allowed for c in trace.retrieved)
        codes.append("FP11")
        if trace.system == "graphrag" and not leaked_by_label:
            codes.append("GR4")
    stale = [c for c in cited if truth.get(c.doc_id, {}).get("superseded_by")]
    if stale:
        codes.append("FP8")
        if trace.system == "graphrag":
            codes.append("GR3")

    # --- agentic ---------------------------------------------------------------
    if trace.system == "agentic":
        st = steps(trace)
        seq = [s["tool"] for s in st]
        if trajectory(trace, q)["budget_exhausted"]:
            codes.append("AG4")
        missing = [t for t in dict.fromkeys(q.get("expected_tools", [])) if t not in seq]
        if "check_document_status" in missing and "FP8" not in codes:
            missing.remove("check_document_status")  # skipped, but it did not matter here
        if missing and "AG4" not in codes:
            codes.append("AG1")
        named = offices_in(q["text"])
        if named and any(s["args"].get("office") not in (None, *named)
                         for s in st if s["tool"] == "search_handbook"):
            codes.append("AG2")
        parts = topics_in(q["text"])
        if len(parts) > 1 and seq.count("search_handbook") < len(parts) and "AG4" not in codes:
            codes.append("AG3")
        superseded_seen = {
            s["args"].get("doc_id") for s in st
            if s["tool"] == "check_document_status"
            and isinstance(s["observation"], dict) and s["observation"].get("status") == "superseded"
        }
        if superseded_seen & {c.doc_id for c in cited}:
            codes.append("AG5")

    # --- graph -----------------------------------------------------------------
    if trace.system == "graphrag":
        m = graph_metrics(trace)
        topics = topics_in(q["text"])
        link = trace.span("graph.link").attributes
        missed_named = [o for o in m["offices_named"] if o not in m["offices_linked"]]
        implied_missed = (
            not m["offices_linked"] and not link.get("comparative")
            and OFFICE_DEPENDENT_TOPICS & set(topics)
        )
        if missed_named or implied_missed:
            codes.append("GR1")
        if graph is not None:
            for c in cited:
                f = graph.facts.get(c.chunk_id.removesuffix("+ranked"))
                if f and f.office is None and any(
                    o.office for o in graph.facts.values() if o.doc_id == f.doc_id and o.topic == f.topic
                ):
                    codes.append("GR2")
                    break
        trav = trace.span("graph.traverse").attributes
        if trace.refused and not q.get("should_refuse") and not m["fallback"]:
            codes.append("GR5")
        elif not any(c in codes for c in ("GR1", "GR2", "GR3", "GR4")) and topics and \
                any(trav.get("graph_has_topic", {}).values()) and not _evidence_in(trace.retrieved, q):
            codes.append("GR2")

    # --- generic fall-through: where did the evidence get to? --------------------
    if not codes:
        if q.get("should_refuse") and not trace.refused:
            codes.append("FP1")
        elif _evidence_in(trace.retrieved, q):
            codes.append("FP4")
        elif _evidence_in([c for c in trace.considered if c.dropped_at], q):
            codes.append("FP2")
        else:
            codes.append("FP1" if not trace.refused else "FP2")
    return list(dict.fromkeys(codes))
