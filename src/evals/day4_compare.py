"""Day 4, Modules 10 and 11: comparing two configurations and gating on it.

Everything here depends on the config hash recorded in every trace on Day 1.
Without it, "before" and "after" are two piles of traces with no way to say
what differed, and a regression report becomes an anecdote.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from evals.day1_triage import triage
from tracing.schema import Trace


@dataclass
class Arm:
    name: str
    config_hash: str
    traces: list[Trace]

    def failure_counts(self, evidence: dict[str, str]) -> Counter[str]:
        c: Counter[str] = Counter()
        for t in self.traces:
            c.update(f.code for f in triage(t, evidence.get(t.query)))
        return c


def compare(before: Arm, after: Arm, evidence: dict[str, str]) -> str:
    """A before/after table over the failure-point catalog.

    Deliberately paired: the same queries run through both arms. Unpaired
    comparison needs many more examples to separate two systems from noise,
    which is Module 11's subject.
    """
    assert before.config_hash != after.config_hash, (
        "both arms carry the same config hash; nothing changed, or the hash "
        "is not capturing what you changed"
    )
    b, a = before.failure_counts(evidence), after.failure_counts(evidence)
    codes = sorted(set(b) | set(a))
    width = max((len(c) for c in codes), default=4)

    lines = [
        f"{'code'.ljust(width)}  {before.name:>8}  {after.name:>8}  delta",
        "-" * (width + 30),
    ]
    for code in codes:
        delta = a[code] - b[code]
        arrow = "fixed" if delta < 0 else ("regressed" if delta > 0 else "")
        lines.append(f"{code.ljust(width)}  {b[code]:>8}  {a[code]:>8}  {delta:+d} {arrow}")
    lines.append(
        f"\ntotal findings: {sum(b.values())} -> {sum(a.values())} "
        f"over {len(before.traces)} paired queries"
    )
    return "\n".join(lines)


def gate(before: Arm, after: Arm, evidence: dict[str, str],
         must_not_regress: tuple[str, ...] = ("FP11", "FP8")) -> tuple[bool, str]:
    """A release gate.

    `must_not_regress` is the floor from charter field 8. FP11 is here
    because entitlement leakage is a security defect wearing a quality
    costume, and FP8 because a stale answer carries the company's name.

    EXERCISE (Day 4, Module 10): the thresholds belong to the team, not to
    this function. Agree them before you measure, or you will rationalise
    whatever number you get.
    """
    b, a = before.failure_counts(evidence), after.failure_counts(evidence)
    regressions = [c for c in must_not_regress if a[c] > b[c]]
    if regressions:
        return False, f"blocked: {', '.join(regressions)} regressed"
    improved = sum(b.values()) - sum(a.values())
    return True, f"passed: {improved:+d} fewer findings, no gated regression"
