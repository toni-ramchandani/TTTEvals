"""Agentic RAG: a tool loop over the same index Helix uses.

Vanilla Helix runs one fixed path: retrieve, rerank, assemble, generate. The
agent decides its own path. It can look up who is asking, search more than
once, search with a filter, and check whether a page is still current before
it trusts it. Each of those fixes a Day 1 failure. Each also adds a decision
the agent can get wrong, and the answer text will not tell you which one it
got wrong. Only the trajectory will.

Two policies drive the loop, and they write identical spans:

* **offline** (``--offline``): a deterministic planner. Every choice it makes is one
  of the ``AgentConfig`` switches below, so turning a switch off reproduces one
  agent failure on demand, and the same failure every time.
* **live** (the default): a real model chooses tools through its own tool
  calling. The same switches still apply, because they decide which tools
  the model is offered and which instructions it is given (see
  ``live_tools`` and ``live_system_prompt``). Turning a switch off takes the
  capability away; whether the model then fails is something you observe,
  not something the code forces.

=========================  ====================================================
``use_profile``            look up the asker before answering. Off: an office
                           implied by who is asking is never found. AG1.
``prefer_query_office``    an office named in the question beats the asker's
                           own. Off: the profile overrides it. AG2.
``office_filter``          pass the office to search. Off: other offices'
                           sections compete for the top slots.
``decompose``              one search per part of a multi-part question. Off:
                           the agent answers the first part and stops. AG3.
``check_versions``         check each page's status before using it. Off: a
                           superseded page is used as if current. AG1.
``act_on_observations``    drop what a status check says is superseded. Off:
                           the check runs and its result is ignored. AG5.
``max_steps``              the step budget. Too small for the question: AG4.
=========================  ====================================================

Every tool call is one span named ``agent.step.N``, carrying the tool, its
arguments, the agent's stated reason, and a summary of what came back. Search
steps also fill ``kept`` and ``dropped`` exactly as Helix's retrieve span does.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from helix.config import REPAIRED, Config
from helix.ingest import Index
from helix.llm import _REFUSAL, chat_with_tools, complete, is_refusal
from helix.prompts import PROMPT_VERSION, build_messages
from helix.structure import is_comparative, offices_in, role_in, tenure_in, topics_in
from tracing.schema import Chunk, Trace, _now
from tracing.tracer import span

from .tools import TOOL_SPECS, Toolbox

BUDGET_REFUSAL = (
    "I don't have enough information to answer that within my step limit. Please check with HR."
)


@dataclass(frozen=True)
class AgentConfig:
    use_profile: bool = True
    prefer_query_office: bool = True
    office_filter: bool = True
    decompose: bool = True
    check_versions: bool = True
    act_on_observations: bool = True
    max_steps: int = 8
    base: Config = field(default_factory=lambda: REPAIRED)

    def hash(self) -> str:
        payload = asdict(self)
        payload["base"] = self.base.hash()
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def describe(self) -> str:
        flags = [k for k in ("use_profile", "prefer_query_office", "office_filter", "decompose",
                             "check_versions", "act_on_observations") if getattr(self, k)]
        return f"agent max_steps={self.max_steps} on=[{', '.join(flags)}] [{self.hash()}]"


#: Every switch on, eight steps. The configuration the Day 3 comparison uses.
AGENT_DEFAULT = AgentConfig()

#: One search and done: the agent equivalent of a fixed pipeline.
AGENT_NAIVE = AgentConfig(
    use_profile=False, prefer_query_office=False, office_filter=False, decompose=False,
    check_versions=False, act_on_observations=False, max_steps=4,
)

#: Instruction lines for the live agent, each tied to the switch that grants it.
_LINES = [
    ("use_profile", "Look up who is asking (get_user_profile) before answering anything that "
                    "depends on their office or length of service."),
    (None, "Search the handbook with search_handbook."),
    ("office_filter", "Pass `office` to search_handbook when the question is about one office."),
    ("prefer_query_office", "If the question names an office, use that office, not the asker's."),
    ("check_versions", "Before you rely on a document, call check_document_status with its doc id."),
    ("act_on_observations", "Never use or cite a document whose status is superseded; search again "
                            "for the current one."),
    ("decompose", "If the question has several parts, search for each part separately."),
    (None, "Answer only from extracts the tools returned. If they do not contain the answer, "
           "say exactly: \"" + _REFUSAL + "\""),
    (None, "Finish by calling `finish` with the answer, keeping each bracketed reference exactly "
           "as the tool returned it, for example [HB-EXAMPLE-00 v1 2026-01-01]."),
]


def live_system_prompt(acfg: "AgentConfig") -> str:
    """The live agent's instructions, with each switched-off capability removed."""
    lines = [text for flag, text in _LINES if flag is None or getattr(acfg, flag)]
    return ("### Role\nYou are Helix, an assistant for an internal employee handbook, "
            "working with tools.\n\n### Instructions\n" + "\n".join(f"- {l}" for l in lines) + "\n")


def live_tools(acfg: "AgentConfig") -> list[dict[str, Any]]:
    """The tools the live agent is offered, with each switched-off capability removed."""
    out = []
    for spec in TOOL_SPECS:
        name = spec["function"]["name"]
        if name == "get_user_profile" and not acfg.use_profile:
            continue
        if name == "check_document_status" and not acfg.check_versions:
            continue
        if name == "search_handbook" and not acfg.office_filter:
            spec = json.loads(json.dumps(spec))
            spec["function"]["parameters"]["properties"].pop("office", None)
            spec["function"]["description"] = spec["function"]["description"].split(" Pass `office`")[0]
        out.append(spec)
    return out


#: The live prompt with every switch on, for reading and for the docs.
SYSTEM_PROMPT = live_system_prompt(AGENT_DEFAULT)


class _Budget(Exception):
    pass


class _Recorder:
    """Counts steps and writes one span per tool call."""

    def __init__(self, trace: Trace, max_steps: int, on_step=None) -> None:
        self.trace = trace
        self.max_steps = max_steps
        self.on_step = on_step
        self.n = 0
        self.calls: list[dict[str, Any]] = []

    def step(self, tool: str, args: dict[str, Any], thought: str):
        if self.n >= self.max_steps:
            raise _Budget()
        self.n += 1
        self.calls.append({"tool": tool, "args": args})
        if self.on_step:
            self.on_step(self.n, tool, args)
        return span(self.trace, f"agent.step.{self.n}", tool=tool, args=args, thought=thought)


def _summ(chunks: list[Chunk]) -> list[str]:
    return [f"{c.chunk_id} v{c.doc_version} {c.last_modified}" for c in chunks]


def _rewrite(topic: str, role: str | None, office: str | None, tenure: float | None) -> str:
    """The sub-question the agent writes for one part of a multi-part question."""
    if topic == "parental_leave":
        return f"{role or 'primary'} caregiver parental leave weeks working days {office or ''}".strip()
    if topic == "notice_period":
        years = f"{int(tenure)} years" if tenure is not None else ""
        return f"notice period resign {office or ''} {years}".strip()
    return topic.replace("_", " ")


# ---------------------------------------------------------------------------
# Offline policy (the reference run)
# ---------------------------------------------------------------------------
def _stale(c: Chunk, status: dict[str, dict]) -> bool:
    return status.get(c.doc_id, {}).get("status") == "superseded"


def _check(results: list[Chunk], status: dict[str, dict], tb: Toolbox, rec: _Recorder) -> None:
    for doc_id in dict.fromkeys(c.doc_id for c in results):
        if doc_id in status:
            continue
        with rec.step("check_document_status", {"doc_id": doc_id},
                      f"Check {doc_id} is current before relying on it.") as s:
            status[doc_id] = tb.check_document_status(doc_id)
            s.attributes["observation"] = status[doc_id]


def _run_offline(question: str, tb: Toolbox, acfg: AgentConfig, rec: _Recorder) -> tuple[str, list[Chunk], dict]:
    used: list[Chunk] = []
    plan: dict[str, Any] = {}

    profile = None
    if acfg.use_profile:
        with rec.step("get_user_profile", {}, "Find out who is asking before answering.") as s:
            profile = tb.get_user_profile()
            s.attributes["observation"] = profile

    q_offices = offices_in(question)
    role = role_in(question)
    tenure = tenure_in(question)
    if tenure is None and profile:
        tenure = profile.get("tenure_years")
    comparative = is_comparative(question)
    topics = topics_in(question) or ["general"]

    if comparative:
        with rec.step("list_offices", {}, "The question compares offices, so check each one.") as s:
            offices = tb.list_offices()
            s.attributes["observation"] = offices
        subqs = [(_rewrite("parental_leave", role, o, None), o, "parental_leave") for o in offices]
    else:
        if q_offices and (acfg.prefer_query_office or not profile):
            office = q_offices[0]
        elif profile:
            office = profile.get("office")
        else:
            office = None
        if acfg.decompose and len(topics) > 1:
            subqs = [(_rewrite(t, role, office, tenure), office, t) for t in topics]
        else:
            subqs = [(question, office, topics[0])]
    plan.update(comparative=comparative, topics=topics, offices_in_question=q_offices,
                subquestions=[q for q, _, _ in subqs])

    status: dict[str, dict] = {}
    sub_answers: list[tuple[str, str | None, str, list[Chunk]]] = []
    for subq, office, topic in subqs:
        args = {"query": subq}
        if acfg.office_filter and office:
            args["office"] = office
        with rec.step("search_handbook", args, f"Search for: {topic.replace('_', ' ')}.") as s:
            results, filtered = tb.search_handbook(**args)
            s.kept = [c.chunk_id for c in results]
            s.dropped = [c.chunk_id for c in filtered]
            s.attributes["observation"] = _summ(results)

        if acfg.check_versions:
            _check(results, status, tb, rec)
            if acfg.act_on_observations:
                top_was_stale = bool(results) and _stale(results[0], status)
                results = [c for c in results if not _stale(c, status)]
                if top_was_stale and office and not comparative:
                    # The best extract was invalidated. Search again, narrower.
                    retry = {"query": _rewrite(topic, role, office, tenure), "office": office}
                    with rec.step("search_handbook", retry,
                                  "The top extract was superseded; search again for the current page.") as s:
                        again, filtered = tb.search_handbook(**retry)
                        s.kept = [c.chunk_id for c in again]
                        s.dropped = [c.chunk_id for c in filtered]
                        s.attributes["observation"] = _summ(again)
                    _check(again, status, tb, rec)
                    results = [c for c in again if not _stale(c, status)] or results

        text, _ = complete(build_messages(subq, results), acfg.base)
        sub_answers.append((text, office, topic, results))
        used.extend(results)  # what reached the generator, cited or not

    # Compose.
    if comparative:
        best = None
        for text, office, _, results in sub_answers:
            m = re.search(r"caregiver,\s*(\d+)", text)
            cited = [c for c in results if c.cite() in text]
            # Only count a figure if the extract it came from is about that office.
            if m and office and cited and office in offices_in(cited[0].text):
                weeks = int(m.group(1))
                if best is None or weeks > best[0]:
                    best = (weeks, office, text)
        final = (f"The {best[1].title()} office has the longest: {best[2]}" if best else _REFUSAL)
    else:
        parts = [t for t, _, _, _ in sub_answers if t != _REFUSAL]
        final = " ".join(parts) if parts else _REFUSAL
    plan["sub_answers"] = [t for t, _, _, _ in sub_answers]
    return final, used, plan


# ---------------------------------------------------------------------------
# Live policy
# ---------------------------------------------------------------------------
def _run_live(question: str, tb: Toolbox, acfg: AgentConfig, rec: _Recorder) -> tuple[str, list[Chunk], dict]:
    """A real model picks every step. The harness only runs the tools and counts."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": live_system_prompt(acfg)},
        {"role": "user", "content": question},
    ]
    tools = live_tools(acfg)
    returned: dict[str, Chunk] = {}
    while True:
        if rec.n >= rec.max_steps:
            raise _Budget()
        turn = chat_with_tools(messages, tools, acfg.base)
        if not turn["tool_calls"]:
            # Some models answer in plain text instead of calling finish.
            text = turn["content"] or _REFUSAL
            return text, list(returned.values()), {"ended": "text_reply", "finished_with_tool": False}
        messages.append({
            "role": "assistant",
            "content": turn["content"] or None,
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                           for c in turn["tool_calls"]],
        })
        for call in turn["tool_calls"]:
            name, args = call["name"], call["arguments"]
            thought = turn["content"] or f"The model chose {name}."
            if name == "finish":
                with rec.step("finish", {"answer": str(args.get("answer", ""))[:120]}, thought):
                    pass
                text = str(args.get("answer", "")) or _REFUSAL
                return text, list(returned.values()), {
                    "ended": "finish", "finished_with_tool": True,
                    "cited_refs": [str(r) for r in args.get("citations") or []],
                }
            with rec.step(name, args, thought) as s:
                if name == "search_handbook":
                    office = args.get("office") if acfg.office_filter else None
                    results, filtered = tb.search_handbook(str(args.get("query", "")), office or None)
                    s.kept = [c.chunk_id for c in results]
                    s.dropped = [c.chunk_id for c in filtered]
                    returned.update({c.chunk_id: c for c in results})
                    obs: Any = [{"ref": c.cite(), "doc_id": c.doc_id, "text": c.text} for c in results]
                    s.attributes["observation"] = _summ(results)
                elif name == "get_user_profile" and acfg.use_profile:
                    obs = tb.get_user_profile()
                    s.attributes["observation"] = obs
                elif name == "list_offices":
                    obs = tb.list_offices()
                    s.attributes["observation"] = obs
                elif name == "check_document_status" and acfg.check_versions:
                    obs = tb.check_document_status(str(args.get("doc_id", "")))
                    s.attributes["observation"] = obs
                else:
                    obs = {"error": f"tool {name} is not available"}
                    s.attributes["observation"] = obs
            messages.append({"role": "tool", "tool_call_id": call["id"], "name": name,
                             "content": json.dumps(obs)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def answer(
    question: str,
    index: Index,
    acfg: AgentConfig = AGENT_DEFAULT,
    user_office: str = "amsterdam",
    user_role: str = "employee",
    user_tenure_years: float | None = 3,
    on_step=None,
) -> Trace:
    """Run the agent on one question and return its trace.

    `on_step(n, tool, args)`, if given, is called as each step starts, so a
    script can show the agent working in real time.
    """
    trace = Trace(query=question, config_hash=acfg.hash(), user_office=user_office, user_role=user_role)
    trace.system = "agentic"
    tb = Toolbox(index, acfg.base, {"office": user_office, "role": user_role, "tenure_years": user_tenure_years})
    rec = _Recorder(trace, acfg.max_steps, on_step=on_step)

    budget_hit = False
    plan: dict[str, Any] = {}
    with span(trace, "agent.run", policy="offline" if acfg.base.offline else "live",
              max_steps=acfg.max_steps, model=acfg.base.generation_model,
              prompt_version=PROMPT_VERSION) as run:
        try:
            if acfg.base.offline:
                text, used, plan = _run_offline(question, tb, acfg, rec)
            else:
                text, used, plan = _run_live(question, tb, acfg, rec)
        except _Budget:
            budget_hit = True
            text, used = BUDGET_REFUSAL, []
        if not budget_hit and plan.get("finished_with_tool") is not True:
            try:
                with rec.step("finish", {"answer": text[:120]}, "Answer from the evidence gathered."):
                    pass
            except _Budget:
                budget_hit = True
                text, used = BUDGET_REFUSAL, []
        run.attributes.update(plan=plan, steps_used=rec.n, budget_exhausted=budget_hit,
                              tool_sequence=[c["tool"] for c in rec.calls])

    # retrieved = everything that reached the generator, as in Helix.
    # considered = everything any tool returned, including what was filtered.
    trace.retrieved = list({c.chunk_id: c for c in used}.values())
    used_ids = {c.chunk_id for c in trace.retrieved}
    for c in tb.seen.values():
        if c.chunk_id not in used_ids:
            c.dropped_at = c.dropped_at or "agent.unused"
    trace.considered = list(tb.seen.values())

    with span(trace, "postprocess") as s:
        cited_text = text + " " + " ".join(plan.get("cited_refs", []))
        citations = [c.cite() for c in trace.retrieved
                     if c.cite() in cited_text or c.chunk_id in cited_text]
        refused = is_refusal(text)
        s.attributes.update(citations_emitted=citations, refused=refused)

    trace.answer, trace.citations, trace.refused = text, citations, refused
    trace.ended_at = _now()
    return trace
