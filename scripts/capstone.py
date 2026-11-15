"""Day 4 hands-on, Module 12: evidence-driven improvement, the full cycle.

The gate blocked the Day 1 repair. This finds out why, chooses changes aimed
at each cause, measures them, reads what they broke, and ends with a
decision someone can defend.

  STEP 1  localize: where each failure of the shipped build was born
  STEP 2  one change at a time, each aimed at one cause, against the shipped build
  STEP 3  combine, one step at a time, with the gate's decision at each step
  STEP 4  inspect: what regressed, and what a step lost that an earlier one fixed
  STEP 5  trade-offs: quality, floors, tokens, model calls, latency
  STEP 6  the conclusion, with its evidence and its limits

Your own experiment, on top of the shipped build (or --on repaired):

  python run.py capstone --change top_k=8,rerank_cutoff=5
  python run.py capstone --on repaired --transform current-only --guard scope-v2

Settings for --change are Config fields (src/helix/config.py). Transforms:
current-only. Guards: scope-v1, scope-v2 (src/evals/day4_suite.py).

Live, every build answers every question: the full catalog is about 900
model calls. Use --source day1,architectures (19 questions) or --offline.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from evals.day1_triage import triage  # noqa: E402
from evals.day4_gate import decide, model_calls, tokens  # noqa: E402
from evals.day4_report import (  # noqa: E402
    ARM_HEADER, COMBINATIONS, INTERVENTIONS, arm_legend, arm_row, builds, latency_line, parse_changes, rule, run,
    short, single,
)
from evals.day4_stats import p_text, paired, pct, prob_text  # noqa: E402
from evals.day4_suite import FLOORS, GUARDS, TRANSFORMS, Build, build_suite, select  # noqa: E402
from helix import DEFAULT, REPAIRED  # noqa: E402
from helix.console import banner  # noqa: E402

CAUSE_TO_CHANGE = {
    "FP11": "entitlements", "FP8": "current-only", "FP9": "current-only", "FP10": "structural",
    "FP3": "budget-900", "FP2": "cutoff-5", "spec": "scope-v1",
}
BORN = {
    "FP11": "retrieve: the entitlement filter is off",
    "FP8": "ingestion: a superseded page is still indexed",
    "FP9": "corpus: two versions of one policy disagree",
    "FP10": "ingestion: a heading severed from its table",
    "FP3": "context.assemble: evidence cut by the token budget",
    "FP2": "rerank: evidence ranked below the cut-off",
    "FP4": "generate: the evidence reached the model; the answer is wrong",
    "spec": "generate: nothing tells Helix to decline this",
    "other": "read the trace",
}


def cause(v) -> str:
    """The first failure in one failing trace, most upstream first."""
    if not v.floors.get("access", True):
        return "FP11"
    t = v.trace
    codes = [f.code for f in triage(t, v.case.expected_evidence or None)] if t else []
    if not v.floors.get("current", True):
        return "FP8"
    if "answered; must decline" in v.behaviour:
        return "spec"
    for c in ("FP10", "FP9", "FP2", "FP3"):
        if c in codes:
            return c
    if t and any(b.startswith("declined") for b in v.behaviour):
        stages = {c.dropped_at for c in t.considered if c.dropped_at}
        return "FP3" if "context.assemble" in stages else "FP2" if "rerank" in stages else "other"
    return "FP4" if t and t.retrieved else "other"


def step_1(shipped) -> None:
    rule("STEP 1  Localize: where each failure of the shipped build was born")
    failing = {k: v for k, v in shipped.verdicts.items() if not v.passed}
    by = Counter(cause(v) for v in failing.values())
    print(f"{len(failing)} of {len(shipped.verdicts)} questions fail. The first failure in each trace:\n")
    print(f"{'cause':<6} {'questions':>9}  {'born at':<58} change to try")
    for c, n in by.most_common():
        print(f"{c:<6} {n:>9}  {BORN.get(c, BORN['other']):<58} {CAUSE_TO_CHANGE.get(c, '-')}")
    print("\nFP4 (the right page reached the model, the answer is still wrong) has no change in")
    print("the catalog: it is a generation failure, fixed by the prompt or the model, which the")
    print("offline stub cannot show. 'spec' is a specification failure: the system was never")
    print("told to decline. A prompt can say so, live; a guard in code can say so in both runs.")


def show_table(rows, base, title) -> None:
    print(ARM_HEADER)
    print("-" * len(ARM_HEADER))
    print(f"{base.name:<17}{arm_row(base)}")
    for name, arm, note in rows:
        print(f"{name:<17}{arm_row(arm, base)}" + (f"   {note}" if note else ""))


def step_2(shipped, singles) -> None:
    rule("STEP 2  One change at a time, each aimed at one cause")
    show_table([(n, a, "") for n, a in singles], shipped, "")
    print()
    for n, changes, transforms, guard, target in INTERVENTIONS:
        print(f"  {n:<14} {target}")
    print("\nRead each row against the cause it targets, then against everything else.")
    if DEFAULT.offline:
        print("In the reference run: entitlements clears the access floor and breaks nothing.")
        print("current-only clears the current-policy floor and breaks two questions that")
        print("passed. budget-900 makes things worse and costs about a quarter more tokens.")
        print("scope-v1 fixes most missed refusals, and declines fair questions that say 'salary'")
        print("or 'pay'. structural fixes 11 and breaks 3: not significant alone (p 0.057).")


def step_3(shipped, combos):
    rule("STEP 3  Combine, one step at a time, with the gate at each step")
    show_table([(n, a, "") for n, a in combos], shipped, "")
    print()
    for n, a in combos:
        print(f"  {n:<15} {decide(shipped, a).summary()}")
    return combos[-1][1]


def step_4(shipped, steps) -> None:
    rule("STEP 4  Inspect: what regressed, and what a later step lost")
    final_name, final = steps[-1]
    t = paired(shipped.passed(), final.passed())
    print(f"Against the shipped build, {final_name} broke {len(t.broke)}: {', '.join(t.broke) or 'none'}\n")
    for k in t.broke:
        before, after = shipped.verdicts[k], final.verdicts[k]
        print(f"  {k}  {short(before.case.text, 68)}")
        print(f"    shipped    {short(before.trace.answer, 66)}")
        dropped = Counter(c.dropped_at for c in before.trace.considered if c.dropped_at)
        if before.refused:
            print(f"               passed by declining; dropped before the model: "
                  f"{', '.join(f'{n} at {s}' for s, n in dropped.items()) or 'nothing'}")
        print(f"    {final_name:<10} {short(after.trace.answer, 66)}")
        print(f"               {'; '.join(after.problems())}")
    print("\nA shipped pass that came from declining because the evidence never reached the")
    print("model was luck, not behaviour. Losing it is a regression all the same: the ratchet")
    print("holds it until a named person decides, with the trace in front of them.")
    print("\nWhat each step lost that the step before had fixed:")
    prev_name, prev = "shipped", shipped
    for name, arm in steps:
        lost = [k for k in prev.verdicts if prev.verdicts[k].passed and not arm.verdicts[k].passed]
        gained = [k for k in prev.verdicts if not prev.verdicts[k].passed and arm.verdicts[k].passed]
        print(f"  {prev_name:>13} -> {name:<14} fixed {len(gained):>2}, lost {len(lost):>2}"
              + (f": {', '.join(lost[:10])}" if lost else ""))
        prev_name, prev = name, arm
    print("\nChanges do not add up. Rerun the whole suite after every step.")
    if DEFAULT.offline:
        print("In the reference run: removing the superseded page changes which chunks the")
        print("reranker sees, so some parental leave questions get a different first chunk")
        print("(a5, o3, c2t1, c7t5, c7t6 lost). A guard that declines on a word declines fair")
        print("questions with that word in them (c7t2). And a refusal can hide a wrong answer:")
        print("when scope-v2 stopped declining c1t2 and j3, their answers turned out wrong.")


def step_5(shipped, steps) -> None:
    rule("STEP 5  Trade-offs: quality, floors, cost, latency")
    t0, c0 = tokens(shipped), model_calls(shipped)
    print(f"{'build':<17} {'pass':>6} {'floors':>7} {'tokens':>8} {'vs shipped':>11} {'calls':>6} {'declined':>9}")
    for name, a in [("shipped", shipped), *steps]:
        fl = sum(len(a.floor_failures(f)) for f in FLOORS)
        dec = sum(v.refused for v in a.verdicts.values())
        print(f"{name:<17} {sum(a.passed().values()):>6} {fl:>7} {tokens(a):>8,} "
              f"{(f'{(tokens(a) - t0) / t0:+.1%}' if t0 else 'n/a'):>11} {model_calls(a):>6} {dec:>9}")
    print("\nA guard declines before retrieval, so a declined request costs no model call: fewer")
    print(f"tokens, and no wait. The shipped build made {c0} model calls for {len(shipped.verdicts)} questions.")
    print("\nLatency, live:")
    for name, a in [("shipped", shipped), *steps]:
        print(f"  {latency_line(a).replace(a.name, name, 1)}")


def step_6(shipped, name, final) -> None:
    rule("STEP 6  The conclusion")
    d = decide(shipped, final)
    t = paired(shipped.passed(), final.passed())
    n = len(final.verdicts)
    k0, k1 = sum(shipped.passed().values()), sum(final.passed().values())
    rest = Counter()
    for v in final.verdicts.values():
        if not v.passed:
            rest[cause(v)] += 1
    floors_ok = all(r.passed for r in d.rules if r.kind == "floor")
    better = (f"is better than the shipped build on this suite (p = {p_text(t.p)})" if t.p < 0.05 and
              len(t.fixed) > len(t.broke) else
              f"is not shown to differ from the shipped build on this suite (p = {p_text(t.p)})")
    print(f"Claim      {name} {better}" + (", and holds every floor" if floors_ok else
                                           "; it does not hold every floor"))
    print(f"Evidence   pass {k0}/{n} -> {k1}/{n}; {len(t.fixed)} fixed, {len(t.broke)} broke; "
          f"exact McNemar p = {p_text(t.p)}; P(better) {prob_text(t.prob_better)}")
    print(f"           tokens {tokens(shipped):,} -> {tokens(final):,}; model calls "
          f"{model_calls(shipped)} -> {model_calls(final)}")
    print(f"Gate       {d.summary()}")
    print(f"Open       {len(t.broke)} regression(s) need a named decision: {', '.join(t.broke) or 'none'}")
    print("           still failing, by first cause: " + ", ".join(f"{c} {k}" for c, k in rest.most_common()))
    print("Limits     the reference run's stub cannot test a prompt change or a live model;")
    print("           scope-v2 was narrowed after reading this suite, so the suite now flatters it:")
    print("           check it on traffic it has not seen (python run.py production);")
    print("           85 questions show a large change, not a small one (python run.py compare)")
    print("Decision   " + ("SHIP" if d.ship else
                          "BLOCK until each regression has an owner's decision, then rerun"))
    top = rest.most_common(1)[0][0] if rest else ""
    print(f"Next       {top} ({BORN.get(top, BORN['other'])}); then rerun this suite" if top else
          "Next       widen the suite: promote what production review finds")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--change", default="", help="Config changes, e.g. top_k=8,rerank_cutoff=5")
    ap.add_argument("--transform", default="", help=f"index transform: {', '.join(TRANSFORMS)}")
    ap.add_argument("--guard", default="", help=f"scope guard: {', '.join(GUARDS)}")
    ap.add_argument("--on", default="shipped", choices=["shipped", "repaired"], help="the build your change starts from")
    ap.add_argument("--ids", default="")
    ap.add_argument("--source", default="")
    args = ap.parse_args(argv)
    if args.transform and args.transform not in TRANSFORMS:
        raise SystemExit(f"unknown transform: {args.transform} (choose from {', '.join(TRANSFORMS)})")
    if args.guard and args.guard not in GUARDS:
        raise SystemExit(f"unknown guard: {args.guard} (choose from {', '.join(GUARDS)})")
    banner(DEFAULT)
    live = not DEFAULT.offline
    cases = select(build_suite(), args.ids, args.source)
    shipped = run(builds()["shipped"], cases, live)
    step_1(shipped)

    if args.change or args.transform or args.guard:
        base = DEFAULT if args.on == "shipped" else REPAIRED
        yours = Build("yours", parse_changes(args.change, base),
                      (args.transform,) if args.transform else (), args.guard)
        print(f"\nyours: {yours.describe()}")
        arm = run(yours, cases, live)
        steps = [("yours", arm)]
        rule("STEP 2-3  Your change against the shipped build")
        show_table([("yours", arm, "")], shipped, "")
        print(f"\n  {decide(shipped, arm).summary()}")
    else:
        if live:
            n = len(cases) * (len(INTERVENTIONS) + len(COMBINATIONS))
            print(f"\nThe catalog answers {len(cases)} questions with {len(INTERVENTIONS) + len(COMBINATIONS)} builds:"
                  f" {n} model calls, live.")
        singles = [(n, run(single(n), cases, live)) for n, *_ in INTERVENTIONS]
        step_2(shipped, singles)
        combos = [(n, run(Build(n, cfg, tr, g), cases, live)) for n, cfg, tr, g in COMBINATIONS]
        step_3(shipped, combos)
        steps = combos
    print()
    for line in arm_legend():
        print(f"  {line}")
    step_4(shipped, steps)
    step_5(shipped, steps)
    step_6(shipped, *steps[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
