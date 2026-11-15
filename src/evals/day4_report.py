"""Day 4: the builds the scripts compare, and how they print.

Shared by `day4`, `compare`, `capstone` and the CI gate, so every Day 4
command means the same thing by "shipped" and "candidate".
"""

from __future__ import annotations

from dataclasses import fields, replace

from evals.day4_gate import model_calls, tokens
from evals.day4_stats import p_text, paired, pct, percentile
from evals.day4_suite import FLOORS, Arm, Build, SuiteCase, run_arm
from helix import DEFAULT, REPAIRED
from helix.config import Config
from helix.console import progress


def builds() -> dict[str, Build]:
    """The named builds of Day 4. DEFAULT and REPAIRED follow --offline."""
    return {
        "shipped": Build("shipped", DEFAULT),
        "repaired": Build("repaired", REPAIRED),
        "repaired-current": Build("repaired-current", REPAIRED, ("current-only",)),
        "candidate": Build("candidate", REPAIRED, ("current-only",), "scope-v2"),
    }


#: Module 12: one change each, on top of the shipped build, with the failure it targets.
INTERVENTIONS: list[tuple[str, dict, tuple[str, ...], str, str]] = [
    # name, Config changes, index transforms, guard, what it targets
    ("entitlements", {"enforce_entitlements": True}, (), "", "FP11 access: the filter was off"),
    ("current-only", {}, ("current-only",), "", "FP8/FP9: a superseded page is indexed"),
    ("structural", {"chunk_strategy": "structural", "prepend_heading": True}, (), "",
     "FP10: heading severed from its table"),
    ("budget-900", {"context_token_budget": 900}, (), "", "FP3: evidence cut by the token budget"),
    ("cutoff-5", {"rerank_cutoff": 5}, (), "", "FP2: evidence ranked below the cut-off"),
    ("scope-v1", {}, (), "scope-v1", "missed refusals: nothing tells Helix to decline"),
]

#: Combinations, built up one step at a time.
COMBINATIONS: list[tuple[str, Config, tuple[str, ...], str]] = [
    ("repaired", REPAIRED, (), ""),
    ("+current-only", REPAIRED, ("current-only",), ""),
    ("+scope-v1", REPAIRED, ("current-only",), "scope-v1"),
    ("+scope-v2", REPAIRED, ("current-only",), "scope-v2"),
]


def single(name: str) -> Build:
    for n, changes, transforms, guard, _ in INTERVENTIONS:
        if n == name:
            return Build(n, replace(DEFAULT, **changes), transforms, guard)
    raise KeyError(name)


def parse_changes(text: str, base: Config) -> Config:
    """`top_k=8,rerank_cutoff=5,prepend_heading=true` applied to a Config."""
    if not text:
        return base
    types = {f.name: f.type for f in fields(Config)}
    out = {}
    for part in text.split(","):
        if not part.strip():
            continue
        key, _, raw = part.partition("=")
        key, raw = key.strip(), raw.strip()
        if key not in types or key in ("offline", "judge_model", "generation_model", "embedding_model"):
            allowed = [k for k in types if k not in ("offline", "judge_model", "generation_model", "embedding_model")]
            raise SystemExit(f"--change: unknown setting '{key}'. Choose from: {', '.join(allowed)}")
        current = getattr(base, key)
        if isinstance(current, bool):
            value = raw.lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        else:
            value = raw
        out[key] = value
    return replace(base, **out)


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def short(text: str, n: int = 64) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 3] + "..."


def run(build: Build, cases: list[SuiteCase], live: bool) -> Arm:
    """One arm, with a progress line per question when live."""
    def on_case(i, n, v):
        if live:
            mark = "error" if v.error else ("pass" if v.passed else "FAIL")
            took = f"{v.seconds:.1f}s" if v.seconds is not None else ""
            progress(f"{build.name:<16} {v.case.id:<5} {mark:<5} {took}", i, n)
    return run_arm(build, cases, on_case=on_case)


def arm_row(arm: Arm, base: Arm | None = None) -> str:
    n = len(arm.verdicts)
    passed = sum(arm.passed().values())
    floors = " ".join(f"{len(arm.floor_failures(f)):>4}" for f in FLOORS)
    missed = sum("answered; must decline" in v.behaviour for v in arm.verdicts.values())
    false_ = sum(any(b.startswith("declined") for b in v.behaviour) for v in arm.verdicts.values())
    other = sum(bool([b for b in v.behaviour if b not in ("answered; must decline",)
                      and not b.startswith("declined")]) for v in arm.verdicts.values())
    cols = f"{passed:>3}/{n:<3} {floors}   {missed:>4} {false_:>5} {other:>6}  {tokens(arm):>7,} {model_calls(arm):>5}"
    if base is not None and base is not arm:
        t = paired(base.passed(), arm.passed())
        cols += f"  {len(t.fixed):>3} {len(t.broke):>3}  {p_text(t.p):>7}"
    return cols


ARM_HEADER = (f"{'':<17}{'pass':>7} {'leak':>4} {'stale':>5} {'grnd':>4}   {'miss':>4} {'false':>5} "
              f"{'content':>6}  {'tokens':>7} {'calls':>5}  {'fix':>3} {'brk':>3}  {'p':>7}")


def arm_legend() -> list[str]:
    return [
        "pass     every floor held and every written condition met",
        "leak, stale, grnd   floor failures: access, current policy, groundedness",
        "miss     answered a question it must decline;  false: declined one it should answer",
        "content  a missing or forbidden phrase, figure or citation",
        "tokens   model tokens over the suite (prompt + completion);  calls: model calls",
        "fix, brk questions fixed and broken against the shipped build;  p: exact McNemar",
    ]


def latency_line(arm: Arm) -> str:
    secs = [v.seconds for v in arm.verdicts.values() if v.seconds is not None]
    if not secs:
        return f"{arm.name:<17} not measured offline (the stub calls no model)"
    return (f"{arm.name:<17} p50 {percentile(secs, 50):.1f}s  p95 {percentile(secs, 95):.1f}s  "
            f"max {max(secs):.1f}s  over {len(secs)} answers")


def share(k: int, n: int) -> str:
    return f"{k}/{n} ({pct(k / n if n else None, 0)})"
