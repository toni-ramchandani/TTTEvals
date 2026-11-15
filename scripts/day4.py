"""Day 4 hands-on, Module 10: the regression suite and the release gate.

The Improve phase starts with a rule: no change ships on a feeling. This runs
the shipped build and a candidate over the same regression suite, now, and
applies the release gate the charter wrote down before the run.

  STEP 1  the regression suite: 85 questions from Days 1 to 3, each with
          where it came from and the pass condition a person wrote
  STEP 2  both builds answer every question; floors and behaviour per build
  STEP 3  the release gate: floors, a per-question ratchet, budgets
  STEP 4  provenance: what each score must carry, and the gap Day 1 planted
  STEP 5  cost and latency: tokens per build, the cost formula, p50 and p95
  STEP 6  what runs when: the same gate in CI (python run.py ci)

The candidate is the Day 1 repair (REPAIRED). Every answer is made now, live
by default; --offline is the reference run quoted on the slides. Traces go to
data/traces/day4_shipped.jsonl and day4_candidate.jsonl.

Run:  python run.py day4
      python run.py day4 --candidate repaired-current
      python run.py day4 --source day1,architectures       fewer live calls
      python run.py day4 --price-in 0.15 --price-out 0.60  your prices, per million tokens
      python run.py day4 --strict                          exit 1 on BLOCK, for CI
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from evals.day4_gate import BUDGETS, decide, lines, model_calls, tokens  # noqa: E402
from evals.day4_provenance import diff, fingerprint, planted_gap  # noqa: E402
from evals.day4_report import ARM_HEADER, arm_legend, arm_row, builds, latency_line, rule, run, short  # noqa: E402
from evals.day4_suite import FLOOR_NAMES, FLOORS, build_suite, select, suite_sha  # noqa: E402
from helix import DEFAULT  # noqa: E402
from helix.console import banner  # noqa: E402
from tracing import write_jsonl  # noqa: E402

TRACES = ROOT / "data" / "traces"


def step_1(cases) -> None:
    rule("STEP 1  The regression suite: every question Days 1 to 3 wrote a pass condition for")
    by = Counter(c.source for c in cases)
    what = {
        "day1": "the six seed questions: nuggets, or must decline",
        "architectures": "Day 3's 13 questions: must include, must not include, must cite",
        "robustness": "Day 3's behavioural tests: paraphrase, alias, office, noise, injection",
        "synthetic": "Day 2's questions from tuples: answer, or decline",
        "conversation": "Day 3's conversation turns that stand on their own",
        "added": "added by your team after reviewing production (day4_suite_changes.json)",
    }
    for src, n in by.items():
        print(f"  {src:<14} {n:>3}  {what[src]}")
    print(f"  {'total':<14} {len(cases):>3}  suite fingerprint {suite_sha(cases)}")
    refuse = sum(c.should_refuse for c in cases)
    print(f"\n{refuse} of them must be declined; {len(cases) - refuse} must be answered.")
    print("\nEvery question is checked twice over:")
    print("  floors, the same on every question (charter field 8):")
    for f in FLOORS:
        print(f"    {FLOOR_NAMES[f]}")
    print("  behaviour, written per question when it was written: decline or answer,")
    print("    phrases and figures it must or must not contain, a page it must cite")
    ex = next(c for c in cases if c.id == "a7")
    print(f"\nOne question, as the suite holds it ({ex.id}, from {ex.source}):")
    print(f"  text      {ex.text}")
    print(f"  asker     {ex.role}, {ex.office.title()}")
    print(f"  must not  {', '.join(ex.must_not_include)};  must cite {ex.must_cite_doc}")


def step_2(arms) -> None:
    rule("STEP 2  Both builds answer every question, now")
    for a in arms:
        print(f"  {a.name:<16} {a.build.describe()}")
    print("\n" + ARM_HEADER)
    print("-" * len(ARM_HEADER))
    for a in arms:
        print(f"{a.name:<17}{arm_row(a, arms[0])}")
    print()
    for line in arm_legend():
        print(f"  {line}")


def step_3(shipped, candidate):
    rule("STEP 3  The release gate: three kinds of rule, written before the run")
    print("floors   absolute: one failure on any question blocks")
    print("ratchet  a question that passes on the shipped build must still pass,")
    print("         unless a named person accepts the regression in writing")
    print(f"budgets  model tokens at most {BUDGETS['tokens']:+.0%}, p95 answer time at most "
          f"{BUDGETS['p95_seconds']:+.0%} (live), failed calls {BUDGETS['errors']}\n")
    d = decide(shipped, candidate)
    for line in lines(d):
        print(line)
    print(f"\n  DECISION: {d.summary()}")
    for r in d.blocking():
        for cid in r.cases[:4]:
            v = candidate.verdicts[cid]
            print(f"\n  {cid}  {short(v.case.text, 66)}")
            print(f"       {short(v.trace.answer if v.trace else '', 66)}")
            for p in v.problems()[:2]:
                print(f"       {short(p, 66)}")
    better = len([k for k in shipped.verdicts if candidate.passed()[k] and not shipped.passed()[k]])
    print(f"\nThe candidate fixes {better} questions and the gate still blocks it. Whether it is")
    print("better is a different question (python run.py compare). A floor does not trade:")
    print("one employee told the superseded 2022 entitlement is the failure the charter named.")
    return d


def step_4(arms, sha) -> None:
    rule("STEP 4  Provenance: what a score must carry to be compared with another")
    fps = [fingerprint(a.build, sha) for a in arms]
    print(f"{'':<30} {arms[0].name:<30} {arms[1].name}")
    for k in fps[0]:
        a, b = fps[0][k], fps[1].get(k, "")
        mark = "*" if a != b and k != "run date" else " "
        print(f"{mark} {k:<28} {short(a, 29):<30} {short(b, 29)}")
    changed = [k for k, _, _ in diff(*fps)]
    print(f"\n* differs: {', '.join(changed) or 'nothing'}")
    print("\nThe gap Day 1 planted. Edit one instruction in the prompt template, in memory:")
    g = planted_gap(DEFAULT)
    print(f"  {g['edit']}")
    print(f"  config hash      {g['config hash before']} -> {g['config hash after']}   (unchanged)")
    print(f"  prompt version   {g['prompt version before']} -> {g['prompt version after']}             (unchanged)")
    print(f"  template sha     {g['template sha before']} -> {g['template sha after']}   (changed)")
    print("Config.hash() covers the prompt_version label, not the template text. The trace")
    print("schema's comment says it covers the template; it covers the label. Two runs with")
    print("different prompts would carry the same hash. The fingerprint hashes the text.")
    print("The index transform and the scope guard are invisible to the hash for the same")
    print("reason: they are not Config fields. Record them by name, or lose them.")


def step_5(arms, price_in: float | None, price_out: float | None) -> None:
    rule("STEP 5  Cost and latency")
    print(f"{'':<17}{'prompt':>9} {'completion':>11} {'total':>9} {'calls':>6} {'per call':>9}")
    for a in arms:
        p = sum(v.prompt_tokens or 0 for v in a.verdicts.values())
        c = sum(v.completion_tokens or 0 for v in a.verdicts.values())
        calls = model_calls(a)
        print(f"{a.name:<17}{p:>9,} {c:>11,} {tokens(a):>9,} {calls:>6} {tokens(a) / max(calls, 1):>9.0f}")
    if DEFAULT.offline:
        print("\nOffline the stub counts words, not model tokens; the comparison holds, the")
        print("absolute numbers are your model's to measure.")
    print("\ncost per question = (prompt tokens x input price + completion tokens x output price)")
    print("                    / 1,000,000, with prices per million tokens from your provider's page")
    if price_in is not None and price_out is not None:
        for a in arms:
            p = sum(v.prompt_tokens or 0 for v in a.verdicts.values())
            c = sum(v.completion_tokens or 0 for v in a.verdicts.values())
            n = len(a.verdicts)
            usd = (p * price_in + c * price_out) / 1e6
            print(f"  {a.name:<16} {usd:.4f} for the suite, {usd / n * 1000:.3f} per 1,000 questions"
                  f" at {price_in} in, {price_out} out")
    else:
        print("  (add --price-in and --price-out to see it; a local model costs time, not tokens)")
    print("\nLatency, whole request, live:")
    for a in arms:
        print(f"  {latency_line(a)}")
    print("Read the p95, not the mean: one slow answer in twenty is what users remember.")
    print("Compare latency only between runs on the same machine, model and load.")


def step_6() -> None:
    rule("STEP 6  What runs when")
    plan = [
        ("every change to code,", "the suite's floors and ratchet, code checks only:"),
        ("prompt, index or config", "  python run.py ci   (seconds offline, minutes live)"),
        ("nightly", "the same suite live on the pinned model, with the judged metrics"),
        ("weekly", "production sample reviewed; failures promoted into the suite"),
        ("a new model or judge", "everything above, plus the judge re-measured on the anchor set"),
        ("", "  (python run.py evaluators) and attacked (python run.py redteam)"),
    ]
    for when, what in plan:
        print(f"  {when:<26} {what}")
    print("\n.github/workflows/day4-gate.yml runs the reference gate on every pull request.")
    print("A gate that runs only when someone remembers is a report, not a gate.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", default="repaired", choices=[k for k in builds() if k != "shipped"])
    ap.add_argument("--ids", default="", help="comma-separated suite ids")
    ap.add_argument("--source", default="", help="day1, architectures, robustness, synthetic, conversation, added")
    ap.add_argument("--price-in", type=float, default=None, help="input price per million tokens")
    ap.add_argument("--price-out", type=float, default=None, help="output price per million tokens")
    ap.add_argument("--strict", action="store_true", help="exit 1 when the gate blocks")
    args = ap.parse_args(argv)

    banner(DEFAULT)
    live = not DEFAULT.offline
    cases = select(build_suite(), args.ids, args.source)
    step_1(cases)
    b = builds()
    order = [b["shipped"], b[args.candidate]]
    if live:
        print(f"\nThis answers {len(cases)} questions with each build: {2 * len(cases)} model calls, live.")
    arms = [run(x, cases, live) for x in order]
    step_2(arms)
    decision = step_3(*arms)
    step_4(arms, suite_sha(cases))
    step_5(arms, args.price_in, args.price_out)
    step_6()

    TRACES.mkdir(parents=True, exist_ok=True)
    for a, name in zip(arms, ("day4_shipped.jsonl", "day4_candidate.jsonl")):
        write_jsonl([v.trace for v in a.verdicts.values() if v.trace], TRACES / name)
    print(f"\nTraces written to data/traces/day4_shipped.jsonl and day4_candidate.jsonl")
    return 1 if args.strict and not decision.ship else 0


if __name__ == "__main__":
    raise SystemExit(main())
