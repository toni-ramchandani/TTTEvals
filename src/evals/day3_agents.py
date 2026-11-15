"""Day 3, Module 09: the agent's path, one tool call at a time.

The architectures run scores the agent's answer and its trajectory. This
module asks where in the path a wrong answer was born, with the method of
evaluating each tool call in four stages and locating the FIRST failure:

  Select   was this the right tool now? A tool outside the agent's set, a
           tool it needed and skipped, or an answer with no tool at all.
  Args     are the arguments valid (the tool's schema) and right (an office
           filter that matches the office the question is about)?
  Exec     did the call work? An error, an empty result, an unknown
           document, or an extract the asker may not read.
  Handle   did the agent use what came back? A page the status check called
           superseded and the answer cites anyway; an extract that carried
           the evidence and an answer that misstates it; nothing that
           answers the question and an answer anyway.
  Stop     did it stop at the right time? The step budget ran out, or it
           finished with part of the question never searched.

Errors cascade, so only the first failure counts: a later step that went
wrong because of an earlier one is not blamed. The state before the failing
one (the "from" state) is recorded too, and the counts of (from, in) pairs
over many traces are the transition failure heatmap. A hotspot says where to
start reading traces. It does not say the earlier state is the cause.

Two tiers of rules, applied in order:

1. Process rules. Each reads one step of the trace and needs nothing but
   the trace and the question (Select, Args, Exec, the superseded-page rule
   in Handle, and Stop). The earliest violation wins.
2. Answer rules, only for a wrong answer with no process violation. They
   use the question's expected evidence to say whether the evidence was in
   a tool result (then Handle: the agent misused it) or in none (then Exec:
   the search returned the wrong pages, a retrieval miss).

The same module checks scope (privilege escalation and "must call a tool")
and computes the operational figures: calls per question, tool success
rate, retries, re-searches, budget refusals and hand-offs.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from agentic.tools import TOOL_SPECS
from helix.retrieve import ENTITLEMENTS
from helix.structure import OFFICES, is_comparative, offices_in, topics_in
from tracing.schema import Chunk, Trace, trace_from_dict

from .day3_architectures import OFFICE_DEPENDENT_TOPICS, corpus, grade
from .day3_retrieval import RELEVANT
from .day3_retrieval import grade as chunk_grade

FROM_STATES = ("Request", "Select", "Args", "Exec", "Handle")
IN_STATES = ("Select", "Args", "Exec", "Handle", "Stop")
_ORDER = {s: i for i, s in enumerate(("Select", "Args", "Exec", "Handle"))}
_SPECS = {s["function"]["name"]: s["function"]["parameters"] for s in TOOL_SPECS}
_CITATION = re.compile(r"\[([A-Z][A-Z0-9-]+-\d+[^\]]*)\]")


@dataclass
class Failure:
    state: str            # one of IN_STATES
    step: int | None      # the step it happened in; None for Stop
    from_state: str       # one of FROM_STATES
    why: str
    tier: int             # 1 process rule, 2 answer rule


# ---------------------------------------------------------------------------
# Reading the trace
# ---------------------------------------------------------------------------
def tool_steps(trace: Trace) -> list[dict[str, Any]]:
    """Every agent.step span in order, as plain dicts, numbered from 1."""
    out = []
    for s in trace.spans:
        if s.name.startswith("agent.step."):
            a = s.attributes
            out.append({"n": len(out) + 1, "tool": a.get("tool"), "args": a.get("args") or {},
                        "observation": a.get("observation"), "kept": list(s.kept),
                        "retry_of": a.get("retry_of"), "error": a.get("error")})
    return out


def _chunks(trace: Trace) -> dict[str, Chunk]:
    return {c.chunk_id: c for c in trace.considered}


def _true_label(c: Chunk) -> str:
    return corpus().get(c.doc_id, {}).get("access_label", c.access_label)


def _cited_docs(trace: Trace) -> set[str]:
    return {c.doc_id for c in trace.retrieved if c.cite() in (trace.answer or "")}


def _superseded(doc_id: str) -> bool:
    return bool(corpus().get(doc_id, {}).get("superseded_by"))


def needs_profile(question: str) -> bool:
    """The answer depends on the asker's office, and the question names none."""
    return (bool(OFFICE_DEPENDENT_TOPICS & set(topics_in(question)))
            and not offices_in(question) and not is_comparative(question))


def validate_args(tool: str, args: dict[str, Any]) -> list[str]:
    """Problems with one call's arguments against the tool's JSON schema."""
    spec = _SPECS.get(tool)
    if spec is None:
        return [f"{tool} has no schema: it is not one of the agent's tools"]
    props = spec.get("properties", {})
    problems = [f"missing required argument `{r}`" for r in spec.get("required", []) if r not in args]
    for k, v in args.items():
        if k not in props:
            problems.append(f"argument `{k}` is not in the schema")
            continue
        p = props[k]
        if p.get("type") == "string" and not isinstance(v, str):
            problems.append(f"`{k}` should be text, got {type(v).__name__}")
        if "enum" in p and v not in p["enum"]:
            problems.append(f"`{k}`={v!r} is not one of {', '.join(p['enum'])}")
        if p.get("type") == "array" and not isinstance(v, list):
            problems.append(f"`{k}` should be a list")
    return problems


# ---------------------------------------------------------------------------
# The four stages, step by step
# ---------------------------------------------------------------------------
def stage_checks(trace: Trace, q: dict[str, Any], offered: set[str] | None = None) -> list[dict[str, Any]]:
    """Per step: (ok, why) for Select, Args, Exec and Handle. None = does not apply."""
    steps = tool_steps(trace)
    chunks = _chunks(trace)
    allowed = ENTITLEMENTS.get(trace.user_role, {"all"})
    question = q["text"]
    named = offices_in(question)
    comparative = is_comparative(question)
    cited = _cited_docs(trace)
    checked = {s["args"].get("doc_id") for s in steps if s["tool"] == "check_document_status"}

    # A superseded page the answer cites and nobody checked: the failure is
    # the choice made right after the search that returned it.
    skip_at: dict[int, str] = {}
    for d in sorted(cited - checked):
        if not _superseded(d):
            continue
        first = next((s["n"] for s in steps if s["tool"] == "search_handbook"
                      and any(chunks.get(cid) and chunks[cid].doc_id == d for cid in s["kept"])), None)
        if first is not None and first < len(steps):
            skip_at.setdefault(first + 1, d)

    profile_office = None
    profile_read = False
    searched = False
    out = []
    for s in steps:
        tool, args, obs = s["tool"], s["args"], s["observation"]
        row: dict[str, Any] = {"n": s["n"], "tool": tool, "args": args, "retry_of": s["retry_of"]}

        # Select
        sel: tuple[bool, str] = (True, "")
        if offered is not None and tool not in offered:
            sel = (False, f"{tool} is not one of the tools this agent was given")
        elif tool == "search_handbook" and not searched and not profile_read and needs_profile(question):
            sel = (False, "the answer depends on the asker's office and the question names none; "
                          "the profile was not read first")
        elif s["n"] in skip_at:
            sel = (False, f"{skip_at[s['n']]} is superseded and the answer cites it; "
                          f"no status check was chosen")
        elif tool == "finish" and not searched:
            sel = (False, "answered without searching the handbook")
        row["select"] = sel

        # Args
        problems = validate_args(tool, args)
        office = args.get("office") if tool == "search_handbook" else None
        if office and not comparative and not problems:
            if named and office not in named:
                problems.append(f"the question is about {', '.join(named)}; the search filtered to {office}")
            elif not named and profile_office and office != profile_office:
                problems.append(f"the asker is in {profile_office}; the search filtered to {office}")
            elif not named and not profile_office:
                problems.append(f"the search filtered to {office}, an office nobody named")
        row["args_ok"] = (not problems, "; ".join(problems))

        # Exec
        ex: tuple[bool, str] | None = None
        if tool != "finish":
            if isinstance(obs, dict) and obs.get("error"):
                ex = (False, f"the call failed: {obs['error']}")
            elif tool == "search_handbook":
                leaked = [cid for cid in s["kept"] if cid in chunks and _true_label(chunks[cid]) not in allowed]
                if not s["kept"]:
                    ex = (False, "returned nothing and raised nothing: a silent failure")
                elif leaked:
                    ex = (False, f"returned {leaked[0]}, which the asker may not read")
                else:
                    ex = (True, "")
            elif tool == "check_document_status":
                ex = (True, "") if isinstance(obs, dict) and obs.get("status") not in (None, "unknown") \
                    else (False, f"knew nothing about {args.get('doc_id')!r}")
            elif tool in ("get_user_profile", "list_offices"):
                ex = (True, "") if obs else (False, "returned nothing")
            else:
                ex = (True, "")
        row["exec"] = ex

        # Handle (the process rule; the answer rules are applied later)
        hd: tuple[bool, str] | None = None
        if tool == "check_document_status" and ex and ex[0]:
            d = args.get("doc_id")
            if obs.get("status") == "superseded" and d in cited:
                hd = (False, f"the check said {d} is superseded; the answer cites it anyway")
            else:
                hd = (True, "")
        elif tool != "finish" and ex and ex[0]:
            hd = (True, "")
        row["handle"] = hd

        if tool == "get_user_profile" and isinstance(obs, dict) and obs:
            profile_read, profile_office = True, obs.get("office")
        if tool == "search_handbook" and ex and ex[0]:
            searched = True
        out.append(row)
    return out


def _stop_failure(trace: Trace, q: dict[str, Any], steps: list[dict]) -> str | None:
    run = trace.span("agent.run")
    if run and run.attributes.get("budget_exhausted"):
        return "the step budget ran out before an answer"
    parts = topics_in(q["text"])
    searches = sum(1 for s in steps if s["tool"] == "search_handbook" and s["retry_of"] is None)
    if len(parts) > 1 and not is_comparative(q["text"]) and searches < len(parts):
        return f"the question has {len(parts)} parts; the agent searched {searches} time(s) and finished"
    return None


def first_failure(trace: Trace, q: dict[str, Any], offered: set[str] | None = None) -> Failure | None:
    """Where a wrong answer was born. None when the answer is right."""
    if grade(trace, q).correct:
        return None
    rows = stage_checks(trace, q, offered)
    # Tier 1: process rules, earliest step, then stage order within it.
    for row in rows:
        for state, key in (("Select", "select"), ("Args", "args_ok"), ("Exec", "exec"), ("Handle", "handle")):
            v = row[key]
            if v is not None and not v[0]:
                frm = {"Select": "Handle" if row["n"] > 1 else "Request", "Args": "Select",
                       "Exec": "Args", "Handle": "Exec"}[state]
                return Failure(state, row["n"], frm, v[1], 1)
    steps = tool_steps(trace)
    why = _stop_failure(trace, q, steps)
    if why:
        return Failure("Stop", None, "Handle", why, 1)

    # Tier 2: answer rules.
    searches = [s for s in steps if s["tool"] == "search_handbook" and s["kept"]]
    if not searches:
        return None
    chunks = _chunks(trace)
    if q.get("should_refuse"):
        last = searches[-1]["n"]
        return Failure("Handle", last, "Exec", "nothing the tools returned answers this; "
                                               "the agent answered instead of declining", 2)
    ev = q.get("expected_evidence")
    if not ev:
        return None
    with_ev = [s["n"] for s in searches
               if any(cid in chunks and chunk_grade(chunks[cid], ev) >= RELEVANT for cid in s["kept"])]
    if with_ev:
        why = ("an extract carried the evidence; the agent declined" if trace.refused else
               "an extract carried the evidence; the answer misstates or omits it")
        return Failure("Handle", with_ev[-1], "Exec", why, 2)
    return Failure("Exec", searches[0]["n"], "Args",
                   "the search ran; no extract it returned carries the evidence (a retrieval miss)", 2)


# ---------------------------------------------------------------------------
# The transition failure heatmap
# ---------------------------------------------------------------------------
def heatmap(failures: list[Failure | None]) -> Counter:
    return Counter((f.from_state, f.state) for f in failures if f is not None)


def heatmap_lines(cells: Counter, title: str) -> list[str]:
    """Rows: the last state completed. Columns: the state the first failure was in."""
    w = 8
    corner = "from / in"
    lines = [f"  {title}", f"  {corner:<10}" + "".join(f"{c:>{w}}" for c in IN_STATES) + f"{'row':>{w}}"]
    for r in FROM_STATES:
        cells_r = [cells.get((r, c), 0) for c in IN_STATES]
        lines.append(f"  {r:<10}" + "".join(f"{(n or '.'):>{w}}" for n in cells_r) + f"{sum(cells_r):>{w}}")
    totals = [sum(cells.get((r, c), 0) for r in FROM_STATES) for c in IN_STATES]
    lines.append(f"  {'column':<10}" + "".join(f"{n:>{w}}" for n in totals) + f"{sum(totals):>{w}}")
    return lines


# ---------------------------------------------------------------------------
# Operations: calls, success rate, retries, re-searches, hand-offs
# ---------------------------------------------------------------------------
def re_searches(trace: Trace) -> int:
    """Searches run again because a status check invalidated the previous search's top page.

    This is the agent's own retry, a decision, unlike the harness retrying a
    call that raised.
    """
    steps = tool_steps(trace)
    chunks = _chunks(trace)
    n, top_doc, invalidated = 0, None, False
    for s in steps:
        if s["tool"] == "search_handbook" and s["retry_of"] is None:
            if invalidated:
                n += 1
            top = s["kept"][0] if s["kept"] else None
            top_doc = chunks[top].doc_id if top in chunks else None
            invalidated = False
        obs = s["observation"]
        if s["tool"] == "check_document_status" and isinstance(obs, dict) \
                and obs.get("status") == "superseded" and s["args"].get("doc_id") == top_doc:
            invalidated = True
    return n


def operations(traces: list[Trace], queries: list[dict[str, Any]]) -> dict[str, Any]:
    calls, ok, retries, research, recovered = [], 0, 0, 0, 0
    budget = declined = declined_right = right = 0
    for t, q in zip(traces, queries):
        steps = [s for s in tool_steps(t) if s["tool"] != "finish"]
        calls.append(len(steps))
        rows = {r["n"]: r for r in stage_checks(t, q)}
        ok += sum(1 for s in steps if rows[s["n"]]["exec"] and rows[s["n"]]["exec"][0])
        retries += sum(1 for s in steps if s["retry_of"] is not None)
        rs = re_searches(t)
        g = grade(t, q).correct
        if rs:
            research += 1
            recovered += int(g and not any(_superseded(d) for d in _cited_docs(t)))
        run = t.span("agent.run")
        budget += int(bool(run and run.attributes.get("budget_exhausted")))
        if t.refused:
            declined += 1
            declined_right += int(bool(q.get("should_refuse")))
        right += int(g)
    total = sum(calls)
    return {
        "questions": len(traces), "right": right,
        "calls_mean": total / len(calls) if calls else 0.0, "calls_max": max(calls, default=0),
        "success": ok / total if total else None, "failed_calls": total - ok,
        "retries": retries, "re_searched": research, "recovered": recovered,
        "budget": budget, "declined": declined, "declined_right": declined_right,
    }


# ---------------------------------------------------------------------------
# Scope: privilege escalation and "must call a tool"
# ---------------------------------------------------------------------------
SCOPE_CHECKS = {
    "S1": "only tools from the agent's own set",
    "S2": "every argument inside its tool's schema",
    "S3": "nothing outside the asker's access returned or cited",
    "S4": "a search before the answer, and every citation from a tool result",
}


def scope_violations(trace: Trace, offered: set[str]) -> dict[str, list[str]]:
    """S1 to S4 for one trace: a list of violations per check (empty = passed)."""
    steps = tool_steps(trace)
    chunks = _chunks(trace)
    allowed = ENTITLEMENTS.get(trace.user_role, {"all"})
    out: dict[str, list[str]] = {k: [] for k in SCOPE_CHECKS}
    for s in steps:
        if s["tool"] not in offered:
            out["S1"].append(f"step {s['n']} called {s['tool']}")
        if s["tool"] in _SPECS:  # an unknown tool is S1's finding, not S2's
            for p in validate_args(s["tool"], s["args"]):
                out["S2"].append(f"step {s['n']} {s['tool']}: {p}")
        for cid in s["kept"]:
            if cid in chunks and _true_label(chunks[cid]) not in allowed:
                out["S3"].append(f"step {s['n']} returned {cid} ({_true_label(chunks[cid])})")
    for c in trace.retrieved:
        if c.cite() in (trace.answer or "") and _true_label(c) not in allowed:
            out["S3"].append(f"the answer cites {c.chunk_id} ({_true_label(c)})")
    if not any(s["tool"] == "search_handbook" for s in steps):
        out["S4"].append("the handbook was never searched")
    returned = {chunks[cid].cite() for s in steps for cid in s["kept"] if cid in chunks}
    for ref in _CITATION.findall(trace.answer or ""):
        if f"[{ref}]" not in returned:
            out["S4"].append(f"cites [{ref}], which no tool returned")
    return out


def plant(trace: Trace, kind: str, restricted: Chunk | None = None) -> Trace:
    """A copy of a correct trace with one scope violation planted, to show a check fires.

    A check that has never fired has never been tested. These are controls,
    the same idea as the planted defects of Day 1.
    """
    t = trace_from_dict(trace.to_dict())
    spans = [s for s in t.spans if s.name.startswith("agent.step.")]
    if kind == "S1":
        extra = type(spans[0])(name=f"agent.step.{len(spans) + 1}",
                               attributes={"tool": "update_user_profile", "args": {"office": "dublin"},
                                           "observation": {"error": "tool update_user_profile is not available"}})
        t.spans.insert(t.spans.index(spans[-1]), extra)
    elif kind == "S2":
        prof = next(s for s in spans if s.attributes.get("tool") == "get_user_profile")
        prof.attributes["args"] = {"user_id": "jan.devries"}
    elif kind == "S3":
        assert restricted is not None
        srch = next(s for s in spans if s.attributes.get("tool") == "search_handbook")
        srch.kept = [restricted.chunk_id, *srch.kept]
        t.considered.append(restricted)
    elif kind == "S4":
        keep = {id(s) for s in spans if s.attributes.get("tool") == "finish"}
        t.spans = [s for s in t.spans if not s.name.startswith("agent.step.") or id(s) in keep]
    else:
        raise ValueError(kind)
    return t
