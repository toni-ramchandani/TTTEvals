"""Day 4, Module 10: the release gate, three kinds of rule.

A release gate answers one question, SHIP or BLOCK, from rules written down
before the run. The rules come in three kinds, because they protect against
three different things:

floors      absolute, on every question. One failure blocks, however much
            better the rest is. For harms a person must never meet: a page
            they may not read, a superseded policy, an answer its sources do
            not support (charter field 8).
ratchet     paired against the shipped build. A question that passes today
            must still pass, or a named person accepts the regression in
            writing (data/queries/day4_suite_changes.json, "accepted"). For
            quality the instruments measure less well, or where the
            baseline is still poor: it can only move up.
budgets     operational limits on the candidate, relative to the shipped
            build: model tokens (a proxy for cost), the slowest answers
            (p95 latency, live only) and failed model calls.

"Is the candidate better?" is a different question, and Module 11's:
`python run.py compare`. A gate can pass a candidate that is no better, and
block one that is much better.

Change a rule here only after changing it in the charter, and before the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from evals.day4_stats import percentile
from evals.day4_suite import FLOOR_NAMES, FLOORS, Arm, load_changes

#: Charter v2.0 field 8, the worked example. Budgets are relative to the shipped build.
BUDGETS = {
    "tokens": 0.10,          # model tokens over the suite: at most 10 percent more
    "p95_seconds": 0.20,     # live: the p95 answer time at most 20 percent slower, same machine
    "errors": 0,             # failed model calls
}


@dataclass
class Rule:
    kind: str                # "floor" | "ratchet" | "budget"
    name: str
    passed: bool
    detail: str
    cases: list[str] = field(default_factory=list)
    applies: bool = True     # False: cannot be measured in this run (latency offline)


@dataclass
class Decision:
    rules: list[Rule]
    accepted: list[dict] = field(default_factory=list)

    @property
    def ship(self) -> bool:
        return all(r.passed for r in self.rules if r.applies)

    def blocking(self) -> list[Rule]:
        return [r for r in self.rules if r.applies and not r.passed]

    def summary(self) -> str:
        if self.ship:
            return "SHIP: every floor held, nothing regressed without a name on it, within budget"
        why = "; ".join(f"{r.kind} {r.name}: {r.detail}" for r in self.blocking())
        return f"BLOCK: {why}"


def tokens(arm: Arm) -> int:
    return sum((v.prompt_tokens or 0) + (v.completion_tokens or 0) for v in arm.verdicts.values())


def counted(arm: Arm) -> bool:
    """True when every model call in the run reported its token counts."""
    called = [v for v in arm.verdicts.values() if not v.error and not v.guarded]
    return all(v.prompt_tokens is not None for v in called)


def model_calls(arm: Arm) -> int:
    return sum(1 for v in arm.verdicts.values() if not v.error and not v.guarded)


def seconds(arm: Arm) -> list[float]:
    return [v.seconds for v in arm.verdicts.values() if v.seconds is not None]


def active_acceptances(today: date | None = None) -> dict[str, dict]:
    """Accepted regressions whose review date has not passed. An expired one counts again."""
    today = today or date.today()
    out = {}
    for a in load_changes()["accepted"]:
        if not a.get("id") or not a.get("accepted_by") or not a.get("reason"):
            continue                                   # a name and a reason, or it is not accepted
        by = a.get("review_by", "")
        if by and by < today.isoformat():
            continue
        out[a["id"]] = a
    return out


def decide(shipped: Arm, candidate: Arm, accepted: dict[str, dict] | None = None) -> Decision:
    """Apply the three kinds of rule to a candidate, against the shipped build."""
    if shipped.suite_sha != candidate.suite_sha:
        raise ValueError("the two arms ran different suites; a paired gate needs the same questions")
    accepted = active_acceptances() if accepted is None else accepted
    rules: list[Rule] = []

    for f in FLOORS:
        bad = candidate.floor_failures(f)
        rules.append(Rule("floor", f, not bad,
                          "held on every question" if not bad else f"{len(bad)} failing: {', '.join(bad)}",
                          bad))

    before, after = shipped.passed(), candidate.passed()
    broke = [k for k in before if before[k] and not after[k]]
    open_ = [k for k in broke if k not in accepted]
    if not broke:
        detail = "no question that passed on the shipped build fails now"
    elif not open_:
        detail = f"{len(broke)} regressed, each accepted by name: {', '.join(broke)}"
    else:
        detail = f"{len(open_)} regressed with no acceptance: {', '.join(open_)}"
    rules.append(Rule("ratchet", "per question", not open_, detail, open_))

    t0, t1 = tokens(shipped), tokens(candidate)
    if not counted(shipped) or not counted(candidate):
        rules.append(Rule("budget", "model tokens", True,
                          "not measured: the model API reported no token counts", applies=False))
    else:
        limit = t0 * (1 + BUDGETS["tokens"])
        change = (t1 - t0) / t0 if t0 else 0.0
        rules.append(Rule("budget", "model tokens", t1 <= limit,
                          f"{t1:,} vs {t0:,} shipped ({change:+.1%}; limit {BUDGETS['tokens']:+.0%})"))

    s0, s1 = seconds(shipped), seconds(candidate)
    if s0 and s1:
        p0, p1 = percentile(s0, 95), percentile(s1, 95)
        ok = p1 <= p0 * (1 + BUDGETS["p95_seconds"])
        rules.append(Rule("budget", "p95 answer time", ok,
                          f"{p1:.1f}s vs {p0:.1f}s shipped (limit {BUDGETS['p95_seconds']:+.0%})"))
    else:
        rules.append(Rule("budget", "p95 answer time", True,
                          "not measured: the reference run calls no model, so its timings mean nothing",
                          applies=False))

    errs = candidate.errors()
    rules.append(Rule("budget", "failed model calls", len(errs) <= BUDGETS["errors"],
                      "none" if not errs else f"{len(errs)}: {', '.join(errs)}", errs))
    return Decision(rules, [accepted[k] for k in broke if k in accepted])


def lines(decision: Decision) -> list[str]:
    out = []
    for r in decision.rules:
        mark = "n/a " if not r.applies else ("ok  " if r.passed else "FAIL")
        name = FLOOR_NAMES.get(r.name, r.name) if r.kind == "floor" else r.name
        out.append(f"  {mark} {r.kind:<8} {name}")
        out.append(f"       {'':<8} {r.detail}")
    for a in decision.accepted:
        out.append(f"       accepted {a['id']}: by {a['accepted_by']}, \"{a['reason']}\""
                   + (f", review by {a['review_by']}" if a.get("review_by") else ""))
    return out
