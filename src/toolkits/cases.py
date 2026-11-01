"""From a workshop trace to the test case every toolkit reads.

DeepEval calls it an `LLMTestCase`, Ragas a sample. Both want the same few
things, and the Day 1 trace already records all of them:

    trace field                 DeepEval LLMTestCase     Ragas sample
    -------------------------   ----------------------   ------------------
    query (+ asker)             input                    user_input
    answer                      actual_output            response
    retrieved (reached model)   retrieval_context        retrieved_contexts
    suite ground_truth          expected_output          reference
    agent.step.N spans          tools_called             tool calls in messages
    suite expected_tools        expected_tools           reference_tool_calls

The trace holds more than either toolkit asks for: versions, access labels,
what each stage dropped. Those are what the code checks read, and they ride
along in `metadata` so a custom DeepEval metric can read them too.

`Case` is plain Python, built in memory from a trace whenever a command
needs it; nothing here writes to disk. Only `to_deepeval` and
`to_ragas_messages` import a toolkit, and only when called.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from tracing.schema import Chunk, Trace


@dataclass
class Case:
    id: str
    system: str
    question: str
    asker_role: str
    asker_office: str
    answer: str
    refused: bool
    contexts: list[str]
    context_ids: list[str]
    context_meta: list[dict[str, str]]
    cited_ids: list[str]
    reference: str = ""
    reference_context_ids: list[str] = field(default_factory=list)
    should_refuse: bool | None = None
    tools_called: list[dict[str, Any]] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    trace_id: str = ""
    config_hash: str = ""

    @property
    def input_text(self) -> str:
        """The question as a judge must read it: with who is asking.

        "How many weeks do I get?" has a different right answer in Dublin and
        in Amsterdam. A judge that sees only the question cannot tell an
        answer for the wrong office from a right one.
        """
        return (f"Asker: {self.asker_role}, {self.asker_office.title()} office\n"
                f"Question: {self.question}")


def _cited(trace: Trace) -> list[Chunk]:
    blob = " ".join(trace.citations) + " " + trace.answer
    return [c for c in trace.retrieved if not c.dropped_at and c.chunk_id in blob]


def _relevant(trace: Trace, evidence: str | None) -> list[str]:
    """Chunks carrying the suite's expected evidence, wherever they ended up.

    The same rule as Day 3's retrieval metrics, applied to every chunk the
    pipeline considered, so a relevant chunk that was dropped still counts as
    one the system should have used.
    """
    needles = [p.strip().lower() for p in (evidence or "").split("|") if p.strip()]
    if not needles:
        return []
    need = max(1, (len(needles) + 1) // 2)
    pool = {c.chunk_id: c for c in [*trace.considered, *trace.retrieved]}
    return [cid for cid, c in pool.items()
            if sum(1 for n in needles if n in (c.text + " " + c.heading_path).lower()) >= need]


def _tools(trace: Trace) -> list[dict[str, Any]]:
    out = []
    for s in trace.spans:
        if re.fullmatch(r"agent\.step\.\d+", s.name) and s.attributes.get("tool"):
            if s.attributes["tool"] == "finish":
                continue
            out.append({"name": s.attributes["tool"], "args": s.attributes.get("args") or {}})
    return out


def case_from_trace(trace: Trace, item: dict[str, Any] | None = None) -> Case:
    item = item or {}
    reached = [c for c in trace.retrieved if not c.dropped_at]
    should_refuse = item.get("should_refuse")
    return Case(
        id=item.get("id") or trace.trace_id,
        system=trace.system,
        question=trace.query,
        asker_role=trace.user_role,
        asker_office=trace.user_office,
        answer=trace.answer,
        refused=trace.refused,
        # Exactly the text the model was given. heading_path stays out: with
        # fixed chunks the model never saw it, and a toolkit must judge the
        # same evidence the model had.
        contexts=[c.text for c in reached],
        context_ids=[c.chunk_id for c in reached],
        context_meta=[{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "doc_version": c.doc_version,
                       "last_modified": c.last_modified, "access_label": c.access_label}
                      for c in reached],
        cited_ids=[c.chunk_id for c in _cited(trace)],
        reference=item.get("ground_truth", "") or "",
        reference_context_ids=_relevant(trace, item.get("expected_evidence")),
        should_refuse=bool(should_refuse) if should_refuse is not None else None,
        tools_called=_tools(trace),
        expected_tools=list(item.get("expected_tools") or []),
        trace_id=trace.trace_id,
        config_hash=trace.config_hash,
    )


def cases_from_traces(traces: list[Trace], suite: list[dict[str, Any]] | None = None) -> list[Case]:
    """Pair traces with suite items by question text, falling back to order."""
    suite = suite or []
    by_text = {q.get("text"): q for q in suite}
    out = []
    for i, t in enumerate(traces):
        item = by_text.get(t.query) or (suite[i] if i < len(suite) else None)
        out.append(case_from_trace(t, item))
    return out


def show(case: Case, trace: Trace, width: int = 78) -> list[str]:
    """The case as printable lines: what each toolkit reads, and what only the trace holds.

    Built in memory from the trace whenever a command needs it. Nothing is
    written to disk.
    """
    import textwrap

    def wrap(text: str) -> list[str]:
        text = " ".join(str(text).split())
        lines = textwrap.wrap(text, width - 6, break_on_hyphens=False) or ["-"]
        if len(lines) > 2:
            lines = lines[:2]
            lines[1] = lines[1][: width - 9] + "..."
        return ["      " + ln for ln in lines]

    rows = [
        ("DeepEval input, Ragas user_input", case.input_text.replace("\n", " | ")),
        ("DeepEval actual_output, Ragas response", case.answer),
        ("DeepEval retrieval_context, Ragas retrieved_contexts",
         f"{len(case.contexts)} chunks: " + ", ".join(case.context_ids)),
        ("DeepEval expected_output, Ragas reference", case.reference or "(none in the suite)"),
    ]
    out: list[str] = []
    for name, value in rows:
        out += [f"  {name}"] + wrap(value)
    versions = ", ".join(f"{m['doc_id']} v{m['doc_version']} {m['last_modified']}" for m in case.context_meta)
    labels = ", ".join(sorted({m["access_label"] for m in case.context_meta}))
    dropped = ", ".join(f"{c.chunk_id} at {c.dropped_at}" for c in trace.considered if c.dropped_at)
    out += ["  Only the trace holds: version, access label, what each stage dropped"]
    out += wrap(f"versions: {versions}") + wrap(f"access labels: {labels or '-'}")
    out += wrap(f"dropped: {dropped or 'nothing'}")
    return out


# ---------------------------------------------------------------------------
# Toolkit shapes. These import the toolkit, so call them only after
# toolkits.env.require(...).
# ---------------------------------------------------------------------------
def to_deepeval(case: Case):
    from deepeval.test_case import LLMTestCase, ToolCall

    return LLMTestCase(
        name=case.id,
        input=case.input_text,
        actual_output=case.answer or "(empty answer)",
        expected_output=case.reference or None,
        retrieval_context=case.contexts or None,
        tools_called=[ToolCall(name=t["name"], input_parameters=t["args"]) for t in case.tools_called] or None,
        expected_tools=[ToolCall(name=n) for n in case.expected_tools] or None,
        metadata={
            "asker_role": case.asker_role,
            "context_meta": case.context_meta,
            "cited_ids": case.cited_ids,
            "refused": case.refused,
            "should_refuse": case.should_refuse,
            "trace_id": case.trace_id,
            "config_hash": case.config_hash,
        },
    )


def to_ragas_messages(case: Case):
    """The agent's run as Ragas messages, tool names only.

    The suite records which tools a question needs, not their exact
    arguments. Ragas compares name and arguments together, so both sides drop
    the arguments; otherwise every wording difference in a search query would
    count as a wrong tool call.
    """
    from ragas.messages import AIMessage, HumanMessage, ToolCall

    messages: list[Any] = [HumanMessage(content=case.input_text)]
    if case.tools_called:
        messages.append(AIMessage(content="", tool_calls=[ToolCall(name=t["name"], args={})
                                                          for t in case.tools_called]))
    messages.append(AIMessage(content=case.answer or ""))
    reference = [ToolCall(name=n, args={}) for n in case.expected_tools]
    return messages, reference
