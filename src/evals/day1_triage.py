"""Day 1, hands-on H1: localise a failure from its trace.

These are *detectors*, not metrics. They answer "which span created this?"
rather than "how good is this?". Quantification starts on Day 2, after
participants have read traces with their own eyes.

The module makes one distinction on purpose, and it is the lesson of the
exercise:

**Signals** are neutral facts the trace always carries: what each stage kept
and what it dropped. A drop is not a failure. Every healthy request drops
candidates; that is what a cut-off is for.

**Findings** are failure points. Some can be read from the trace alone
(FP8, FP9, FP10, FP11), because the trace records enough provenance to prove
them. Others cannot (FP2, FP3): knowing that a dropped chunk *mattered*
requires knowing what the right answer was. Those detectors therefore take an
``expected_doc`` and stay silent without one.

That gap is the bridge into Day 2. Error analysis exists because a trace can
show you where a failure *could* have been born but not, on its own, whether
it was.
"""

from __future__ import annotations

from dataclasses import dataclass

from tracing.schema import Trace

from .taxonomy import BY_CODE

#: Supersession relationships that nobody recorded in the index. In a real
#: system this comes from the document management system.
SUPERSEDED = {"HB-PAR-2022": "HB-PAR-2024"}

ENTITLED = {
    "employee": {"all"},
    "manager": {"all"},
    "people-team": {"all", "hr-only"},
}

#: Scope qualifiers that must travel with the figures they govern, grouped so
#: that "Netherlands" and "Amsterdam" count as one scope rather than two. A
#: chunk naming both is fine; a chunk naming Amsterdam *and* Dublin is not.
SCOPE_GROUPS = {
    "ireland": "IE", "dublin": "IE",
    "netherlands": "NL", "amsterdam": "NL",
    "germany": "DE", "berlin": "DE",
    "london": "UK",
    "singapore": "SG",
}


@dataclass
class Finding:
    code: str
    span: str
    evidence: str

    def __str__(self) -> str:
        return (
            f"{self.code} {BY_CODE[self.code].name}  (born at {self.span})\n"
            f"        {self.evidence}"
        )


# ---------------------------------------------------------------------------
# Signals: always true, never a verdict
# ---------------------------------------------------------------------------
def signals(trace: Trace) -> list[str]:
    out = []
    for s in trace.spans:
        if s.dropped:
            out.append(f"{s.name} dropped {len(s.dropped)}: {', '.join(s.dropped)}")
    return out


def outcome_is_suspect(trace: Trace) -> str:
    """A cheap, production-available smell test. Not a correctness judgment."""
    if trace.refused:
        return "the system refused to answer"
    if not trace.citations:
        return "the answer carries no citation"
    return ""


# ---------------------------------------------------------------------------
# Findings readable from the trace alone
# ---------------------------------------------------------------------------
def detect_fp11_entitlement(trace: Trace, expected_doc: str | None = None) -> Finding | None:
    allowed = ENTITLED.get(trace.user_role, {"all"})
    for c in trace.retrieved:
        if c.access_label not in allowed:
            return Finding(
                "FP11", "retrieve",
                f"chunk {c.chunk_id} carries access_label='{c.access_label}' but role "
                f"'{trace.user_role}' may only see {sorted(allowed)}. The answer is "
                f"perfectly faithful to a document the user was not entitled to read, "
                f"so no groundedness metric would ever flag it.",
            )
    return None


def detect_fp10_severed(trace: Trace, expected_doc: str | None = None) -> Finding | None:
    """A chunk that spans a scope boundary, so its figures are ambiguous.

    If one chunk contains two different office or country names together with
    numbers, nothing in that chunk says which figures belong to which scope.
    The model is not guessing; it was handed an ambiguous document.
    """
    for c in trace.retrieved:
        body = c.text.lower()
        groups = {g for word, g in SCOPE_GROUPS.items() if word in body}
        if len(groups) >= 2 and any(ch.isdigit() for ch in c.text):
            named = sorted({w for w in SCOPE_GROUPS if w in body})
            return Finding(
                "FP10", "chunking",
                f"chunk {c.chunk_id} carries figures and spans {len(groups)} different "
                f"scopes ({', '.join(named)}). The qualifier and the fact were severed "
                f"by the chunker, so nothing in the chunk says which figures belong to "
                f"which office.",
            )
    return None


def detect_fp9_contradiction(trace: Trace, expected_doc: str | None = None) -> Finding | None:
    docs = {c.doc_id for c in trace.retrieved}
    for old, new in SUPERSEDED.items():
        if old in docs and new in docs:
            return Finding(
                "FP9", "corpus",
                f"both {old} and {new} reached the model for the same question. "
                f"The answer names one of them without reporting the disagreement.",
            )
    return None


def detect_fp8_stale(trace: Trace, expected_doc: str | None = None) -> Finding | None:
    for c in trace.retrieved:
        if c.doc_id in SUPERSEDED and c.cite() in trace.answer:
            return Finding(
                "FP8", "ingestion",
                f"the answer cites {c.doc_id} v{c.doc_version} ({c.last_modified}), which "
                f"{SUPERSEDED[c.doc_id]} superseded. Grounded, and wrong. Only "
                f"doc_version and last_modified make this visible.",
            )
    return None


# ---------------------------------------------------------------------------
# Findings that need to know what the right answer was
# ---------------------------------------------------------------------------
def _evidence_reached(trace: Trace, evidence: str) -> bool:
    return any(evidence.lower() in c.text.lower() for c in trace.retrieved)


def _evidence_dropped_at(trace: Trace, evidence: str, stage: str):
    for c in trace.considered:
        if c.dropped_at == stage and evidence.lower() in c.text.lower():
            return c
    return None


def detect_fp3_truncated(trace: Trace, evidence: str | None = None) -> Finding | None:
    if not evidence or _evidence_reached(trace, evidence):
        return None
    c = _evidence_dropped_at(trace, evidence, "context.assemble")
    if c is None:
        return None
    s = trace.span("context.assemble")
    budget = s.attributes.get("token_budget") if s else "?"
    return Finding(
        "FP3", "context.assemble",
        f"chunk {c.chunk_id} held the supporting evidence, was retrieved, and was then "
        f"dropped at the {budget}-token context budget. Retrieval did its job and "
        f"assembly undid it, a distinction no end-to-end score can make.",
    )


def detect_fp2_demoted(trace: Trace, evidence: str | None = None) -> Finding | None:
    if not evidence or _evidence_reached(trace, evidence):
        return None
    c = _evidence_dropped_at(trace, evidence, "rerank")
    if c is not None:
        return Finding(
            "FP2", "rerank",
            f"chunk {c.chunk_id} held the supporting evidence and was demoted below the "
            f"rerank cut-off of {(trace.span('rerank').attributes.get('cutoff') if trace.span('rerank') else '?')}.",
        )
    if not any(evidence.lower() in c.text.lower() for c in trace.considered):
        return Finding(
            "FP2", "retrieve",
            "the supporting evidence exists in the corpus and no chunk containing it "
            "entered the shortlist at all. The retriever never saw the answer.",
        )
    return None


DETECTORS = [
    detect_fp11_entitlement,
    detect_fp10_severed,
    detect_fp9_contradiction,
    detect_fp8_stale,
    detect_fp2_demoted,
    detect_fp3_truncated,
]


def triage(trace: Trace, evidence: str | None = None) -> list[Finding]:
    """Failure points this trace provides evidence for, most upstream first.

    `evidence` is a distinctive snippet of the passage that *should* have
    answered the question. Without it, FP2 and FP3 cannot be decided and the
    function says so by staying silent on them.

    Record the *first* failure in a trace rather than
    every symptom, because one upstream error usually cascades into several.
    The ordering here follows that advice.
    """
    return [f for det in DETECTORS if (f := det(trace, evidence)) is not None]


def report(traces: list[Trace], expected: dict[str, str] | None = None) -> str:
    """`expected` maps query text to the evidence snippet that should have answered it."""
    expected = expected or {}
    lines: list[str] = []
    clean = 0
    for t in traces:
        findings = triage(t, expected.get(t.query))
        lines.append(f"\n{'─' * 78}\n{t.query}")
        lines.append(f"  answer   {t.answer[:150]}")
        smell = outcome_is_suspect(t)
        lines.append(f"  outcome  {smell or 'answered with a citation'}")
        for s in signals(t):
            lines.append(f"  signal   {s}")
        if not findings:
            clean += 1
            lines.append("  FINDING  none: no failure point has evidence in this trace")
        for f in findings:
            lines.append(f"  FINDING  {f}")
    lines.append(
        f"\n{'═' * 78}\n{len(traces) - clean} of {len(traces)} traces carry at least one "
        f"failure point."
    )
    return "\n".join(lines)
