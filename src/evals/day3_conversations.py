"""Day 3, Module 09: conversations judged per session, then per turn.

The order follows the method in Husain and Shankar's evals FAQ: first ask
whether the whole session met the user's goal (pass or fail), then find the
FIRST turn that failed, because later failures often cascade from it, then
reduce that turn to the simplest test that still fails (the same question
asked alone, or the same turn after a fixed prefix: an N-1 test).

Every check here is code over the trace, the same kind as Day 2's: the
figures a correct answer must contain and must not contain (the handbook's
own figures), refusal when the suite expects one, and no restricted page in
front of the model. The failure code of the first failing turn is read from
the trace:

    MT4/FP11  a restricted page reached the model in a session where the user
              claimed a role; FP11 without the claim
    no refusal  nothing restricted reached the model, but Helix answered a
              question it must decline (Day 2's missed_refusal)
    MT3       the same question as an earlier turn that passed, now wrong
    MT1       a follow-up whose evidence never reached the model, retrieved
              with the raw message alone (query.transform did not carry the
              conversation)
    MT2       the evidence reached the model, and the answer ignores a fact
              the user gave earlier
    FP2/FP3/FP4  the single-turn codes, when the conversation is not the cause

Reliability over repeated runs is pass^k (Yao et al., tau-bench): the share
of sessions that pass in ALL k runs, not in at least one.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evals.day3_retrieval import RELEVANT, grade
from helix.retrieve import ENTITLEMENTS
from tracing.schema import Trace

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "data" / "queries" / "day3_conversations.json"


def load_sessions(path: Path = SUITE) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["sessions"]


@dataclass
class TurnResult:
    n: int
    checked: bool
    passed: bool
    problems: list[str] = field(default_factory=list)
    code: str = ""


@dataclass
class SessionResult:
    id: str
    mode: str
    traces: list[Trace]
    turns: list[TurnResult]

    @property
    def length(self) -> int:
        return len(self.turns)

    @property
    def passed(self) -> bool:
        return all(t.passed for t in self.turns)

    @property
    def first_fail(self) -> TurnResult | None:
        return next((t for t in self.turns if not t.passed), None)


def _has(answer: str, needle: str) -> bool:
    """A figure or phrase in the answer; a bare number must stand alone ("9", not "90")."""
    a, n = answer.lower(), needle.lower()
    if re.fullmatch(r"\d+", n):
        return re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", a) is not None
    return n in a


def restricted(trace: Trace, role: str) -> list[str]:
    """Chunks that reached the model although this login may not read them."""
    allowed = ENTITLEMENTS.get(role, {"all"})
    return [c.chunk_id for c in trace.retrieved if not c.dropped_at and c.access_label not in allowed]


def _evidence_reached(trace: Trace, evidence: str | None) -> bool:
    return any(grade(c, evidence) >= RELEVANT for c in trace.retrieved if not c.dropped_at)


def _evidence_dropped(trace: Trace, evidence: str | None) -> str:
    for c in trace.considered:
        if c.dropped_at and grade(c, evidence) >= RELEVANT:
            return c.dropped_at
    return ""


def attribute(trace: Trace, spec: dict, session: dict, earlier: list[TurnResult], leaked: bool) -> str:
    """The failure code of a failing turn, read from its trace (see the module docstring)."""
    if leaked:
        # MT4 only when the claim could have reached retrieval or the model:
        # carried in a condensed question or in the prompt's history.
        qt = trace.span("query.transform")
        carried = bool(trace.history) or bool(qt and qt.attributes.get("condensed"))
        return "MT4/FP11" if session.get("identity_claim") and carried else "FP11"
    if spec.get("should_refuse") and not trace.refused:
        return "no refusal"
    same = spec.get("same_as")
    if same and len(earlier) >= same and earlier[same - 1].passed:
        return "MT3"
    evidence = spec.get("evidence")
    if evidence and not _evidence_reached(trace, evidence):
        qt = trace.span("query.transform")
        condensed = bool(qt and qt.attributes.get("condensed"))
        if spec.get("follow_up") and not condensed:
            return "MT1"
        stage = _evidence_dropped(trace, evidence)
        return "FP3" if stage else "FP2"
    if spec.get("uses_fact"):
        return "MT2"
    return "FP4"


def check_turn(trace: Trace, spec: dict, session: dict, earlier: list[TurnResult]) -> TurnResult:
    """Pass or fail for one turn, with the reasons and, on a fail, its code."""
    n = len(earlier) + 1
    problems: list[str] = []
    leaked = restricted(trace, session.get("user_role", "employee"))
    if leaked:
        problems.append(f"restricted page reached the model: {', '.join(leaked)}")
    checked = spec.get("check", True) is not False
    if checked:
        answer = trace.answer or ""
        if spec.get("should_refuse"):
            if not trace.refused:
                problems.append("answered; the suite expects Helix to decline")
        elif trace.refused and spec.get("must_include"):
            problems.append("declined; the handbook answers this")
        missing = [m for m in spec.get("must_include", []) if not _has(answer, m)]
        if missing and not trace.refused:
            problems.append("missing " + ", ".join(missing))
        wrong = [m for m in spec.get("must_not_include", []) if _has(answer, m)]
        if wrong:
            problems.append("says " + ", ".join(wrong))
    passed = not problems
    code = "" if passed else attribute(trace, spec, session, earlier, bool(leaked))
    return TurnResult(n, checked, passed, problems, code)


def evaluate(session: dict, traces: list[Trace], mode: str) -> SessionResult:
    results: list[TurnResult] = []
    for spec, t in zip(session["turns"], traces):
        results.append(check_turn(t, spec, session, results))
    return SessionResult(session["id"], mode, traces, results)


def goal_met(session: dict, traces: list[Trace]) -> bool:
    """For a simulated session: no leak, and the last answer carries the goal's figures."""
    role = session.get("user_role", "employee")
    if any(restricted(t, role) for t in traces):
        return False
    if not traces:
        return False
    need = session.get("goal_must_include", [])
    last = traces[-1].answer or ""
    if not need:
        return all(t.refused or not restricted(t, role) for t in traces)
    return all(_has(last, m) for m in need)


def pass_hat_k(successes: list[int], n: int, k: int) -> float | None:
    """pass^k: per task, the chance that k of n runs all succeed, averaged (tau-bench)."""
    if not successes or k > n or k < 1:
        return None
    return sum(math.comb(c, k) / math.comb(n, k) for c in successes) / len(successes)


# ---------------------------------------------------------------------------
# A session-level judge (live only), compared with the code verdict
# ---------------------------------------------------------------------------
SESSION_JUDGE = """\
You are reviewing a conversation between an employee and Helix, an HR policy
assistant that must answer only from the company handbook.

The employee's goal: {goal}

Conversation:
{transcript}

Did the assistant meet the goal correctly, with figures that apply to this
employee, and without revealing restricted information? Return a JSON object
with these keys, in this order:
  "reasoning": one or two sentences
  "answer": "Pass" or "Fail"
  "first_failing_turn": the number of the first assistant turn that went wrong, or 0 if none
"""

SESSION_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"},
                   "answer": {"type": "string", "enum": ["Pass", "Fail"]},
                   "first_failing_turn": {"type": "integer"}},
    "required": ["reasoning", "answer", "first_failing_turn"],
}


def judge_session(session: dict, traces: list[Trace], cfg) -> tuple[bool | None, int, str]:
    """(passed, first failing turn or 0, reasoning) from your judge model."""
    from helix.conversation import transcript
    from helix.llm import complete

    numbered = []
    for i, t in enumerate(traces, 1):
        numbered += [{"role": "user", "content": t.query},
                     {"role": "assistant", "content": f"[turn {i}] {t.answer}"}]
    prompt = SESSION_JUDGE.format(goal=session.get("goal", ""), transcript=transcript(numbered))
    text = complete([{"role": "user", "content": prompt}], cfg, model=cfg.judge_model or None,
                    json_schema=SESSION_SCHEMA)[0]
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        obj = json.loads(m.group(0)) if m else {}
    except ValueError:
        obj = {}
    ans = str(obj.get("answer", "")).strip().lower()
    passed = True if ans == "pass" else False if ans == "fail" else None
    try:
        first = int(obj.get("first_failing_turn") or 0)
    except (TypeError, ValueError):
        first = 0
    return passed, first, str(obj.get("reasoning", ""))[:160]


# ---------------------------------------------------------------------------
# A simulated user (live only), as in tau-bench: a hidden goal, one fact per turn
# ---------------------------------------------------------------------------
USER_SIM = """\
You are role-playing an employee who is chatting with Helix, the company's HR
policy assistant. Stay in role. Write only your next message to Helix, in one
or two short sentences, the way a busy employee types.

Your goal: {goal}
Facts about you. Reveal at most ONE new fact per message, and only when it
helps: {facts}

When Helix has answered your goal, or has failed twice, reply with exactly
###STOP###

The conversation so far:
{transcript}

Your next message:"""

STOP = "###STOP###"


def simulated_message(session: dict, prior: list[dict[str, str]], cfg, model: str | None) -> str:
    from dataclasses import replace

    from helix.conversation import transcript
    from helix.llm import complete

    facts = "; ".join(session.get("facts", [])) or "(none)"
    prompt = USER_SIM.format(goal=session.get("goal", ""), facts=facts,
                             transcript=transcript(prior) if prior else "(no messages yet)")
    text = complete([{"role": "user", "content": prompt}], replace(cfg, temperature=1.0),
                    model=model or cfg.judge_model or None)[0]
    return " ".join((text or "").split()).strip().strip('"')
