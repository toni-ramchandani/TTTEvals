"""Day 4 hands-on, Module 11: reading the evidence before deciding.

The same two builds as `python run.py day4`, the same regression suite, read
for what the numbers can and cannot say.

  STEP 1  one pass rate, with its interval (Wilson, not mean plus or minus 2 SE)
  STEP 2  two builds, paired: the questions that changed, an exact McNemar
          test, and the same evidence as a probability
  STEP 3  how many changed questions a decision needs
  STEP 4  where the difference comes from, by source, and how small each part is
  STEP 5  what the instruments add: a checker that misses failures widens the
          interval around the failure rate
  STEP 6  repeatability: the same build twice (live, --repeat 2 or more)
  STEP 7  the decision record, filled with this run's numbers

Run:  python run.py compare
      python run.py compare --candidate candidate
      python run.py compare --repeat 3          live: three runs of the candidate
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from evals.day2_evaluators import bootstrap_interval, corrected_failure_rate  # noqa: E402
from evals.day4_gate import decide  # noqa: E402
from evals.day4_report import builds, rule, run, short  # noqa: E402
from evals.day4_stats import (  # noqa: E402
    Z95, interval, prob_text, mcnemar_exact, min_discordant, p_text, paired, pct, prob_better, smallest_win, wilson,
)
from evals.day4_suite import SOURCES, build_suite, select  # noqa: E402
from helix import DEFAULT  # noqa: E402
from helix.console import banner  # noqa: E402

#: How well each instrument agrees with people, from the Helix charter v1.2
#: worked example (the reference run). Replace with the rates in your charter.
#: (failures people found, of which caught; passes people gave, of which passed)
VALIDATION = {
    "content checker (code)": {"failures": 5, "caught": 4, "passes": 1, "passed": 1,
                               "on": "Day 3 sheet"},
    "entitlement check (code)": {"failures": 3, "caught": 3, "passes": 10, "passed": 10,
                                 "on": "golden set, dev split"},
}


def clt(k: int, n: int) -> tuple[float, float]:
    p = k / n
    half = Z95 * math.sqrt(p * (1 - p) / n)
    return p - half, p + half


def step_1(shipped, cand) -> None:
    rule("STEP 1  One pass rate, with its interval")
    for a in (shipped, cand):
        k, n = sum(a.passed().values()), len(a.verdicts)
        print(f"  {a.name:<16} {interval(k, n)}")
    print("\n95% Wilson score interval. The textbook mean +/- 1.96 standard errors is too")
    print("narrow on small samples (Bowyer, Aitchison and Ivanova, ICML 2025). On 85 questions")
    print("near the middle the two agree; on the six Day 1 questions they do not:")
    d1 = [k for k, v in shipped.verdicts.items() if v.case.source == "day1"]
    if d1:
        k = sum(shipped.verdicts[i].passed for i in d1)
        lo, hi = clt(k, len(d1))
        wl, wh = wilson(k, len(d1))
        print(f"  shipped, Day 1 questions: {k}/{len(d1)}   mean +/- 2 SE [{pct(lo)}, {pct(hi)}]   "
              f"Wilson [{pct(wl)}, {pct(wh)}]")
        if lo < 0 or hi > 1:
            print("  An interval that leaves 0 to 100 percent is the textbook formula failing.")
    k0, k1, n = sum(shipped.passed().values()), sum(cand.passed().values()), len(shipped.verdicts)
    lo0, hi0 = wilson(k0, n)
    lo1, hi1 = wilson(k1, n)
    if lo1 <= hi0:
        print(f"\nThe two intervals overlap ({pct(lo1)} is below {pct(hi0)}). Read as two separate")
        print("samples, the run cannot tell the builds apart. They are not separate samples.")


def step_2(shipped, cand):
    rule("STEP 2  Paired: the same questions, so only the changed ones carry information")
    t = paired(shipped.passed(), cand.passed())
    w = 12
    print(f"{'':<22}{'candidate pass':>{w + 3}}{'candidate fail':>{w + 3}}")
    print(f"{'shipped pass':<22}{len(t.both_pass):>{w + 3}}{len(t.broke):>{w + 3}}   <- broke")
    print(f"{'shipped fail':<22}{len(t.fixed):>{w + 3}}{len(t.both_fail):>{w + 3}}")
    print(f"{'':<22}{'fixed ^':>{w + 3}}")
    changed = len(t.fixed) + len(t.broke)
    print(f"\n{t.n - changed} questions gave the same verdict on both builds and say nothing about")
    print(f"which is better. {changed} changed: {len(t.fixed)} fixed, {len(t.broke)} broke.")
    print(f"\n  exact McNemar, two-sided   p = {p_text(t.p)}")
    print("    if the builds were equally good, a split this lopsided or more would happen")
    print(f"    {p_text(t.p)} of the time")
    print(f"  P(candidate better)        {prob_text(t.prob_better)}")
    print("    uniform prior on the share of changed questions that are fixes; the chance")
    print("    that share is above one half (the Bayesian reading Bowyer et al. recommend)")
    print(f"  difference in pass rate    {100 * (len(t.fixed) - len(t.broke)) / t.n:+.1f} points on these {t.n}")
    print("\nNeither number says the candidate can ship. The gate says that, and it says no:")
    print(f"  {decide(shipped, cand).summary()}")
    return t


def step_3(t) -> None:
    rule("STEP 3  How many changed questions a decision needs")
    m = min_discordant()
    print(f"With fewer than {m} changed questions no result can be significant at 0.05: even")
    print(f"{m - 1} fixes and 0 breaks gives p = {p_text(mcnemar_exact(m - 1, 0))}.\n")
    print(f"{'changed':>8} {'fixes needed':>13} {'of which':>9}")
    for n in (6, 10, 16, 20, 30, 50):
        k = smallest_win(n)
        print(f"{n:>8} {k if k else '-':>13} {('%.0f%%' % (100 * k / n)) if k else '':>9}")
    print("\nA small improvement needs many changed questions. Miller (2024) works the same")
    print("arithmetic for a 3-point gain between two models and arrives at about a thousand")
    print("questions. Eighty-five questions can show a large change, not a small one.")
    print(f"This run: {len(t.fixed)} of {len(t.fixed) + len(t.broke)} changed were fixes.")


def step_4(shipped, cand, t) -> None:
    rule("STEP 4  Where the difference comes from")
    fixed, broke = set(t.fixed), set(t.broke)
    print(f"{'source':<14} {'questions':>9} {'fixed':>6} {'broke':>6} {'shipped pass':>13} {'candidate pass':>15}")
    for src in SOURCES:
        ids = [k for k, v in shipped.verdicts.items() if v.case.source == src]
        if not ids:
            continue
        k0 = sum(shipped.verdicts[i].passed for i in ids)
        k1 = sum(cand.verdicts[i].passed for i in ids)
        print(f"{src:<14} {len(ids):>9} {len(fixed & set(ids)):>6} {len(broke & set(ids)):>6} "
              f"{k0:>6}/{len(ids):<6} {k1:>8}/{len(ids):<6}")
    topics = Counter("parental" if "parental" in v.case.text.lower() or "caregiver" in v.case.text.lower()
                     or "baby" in v.case.text.lower() else "other" for v in shipped.verdicts.values())
    print(f"\n{topics['parental']} of {sum(topics.values())} questions are about parental leave. A pass rate"
          f" is a statement about")
    print("this mix of questions, not about Helix in general. Each row above is a handful of")
    print("questions: read its interval before its percentage.")
    print("\nWhat changed, question by question:")
    for k in t.fixed:
        print(f"  fixed {k:<6} {short(shipped.verdicts[k].problems()[0], 62)}")
    for k in t.broke:
        print(f"  broke {k:<6} {short(cand.verdicts[k].problems()[0], 62)}")


def step_5(shipped, cand) -> None:
    rule("STEP 5  What the instruments add to the uncertainty")
    print("A rate measured by a checker that misses failures is lower than the real rate.")
    print("The charter records how often each instrument agrees with people:\n")
    for name, r in VALIDATION.items():
        lo, hi = wilson(r["caught"], r["failures"])
        print(f"  {name:<26} TPR {r['caught']}/{r['failures']} [{pct(lo, 0)}, {pct(hi, 0)}]   "
              f"TNR {r['passed']}/{r['passes']}   on the {r['on']}")
    content = [k for k, v in shipped.verdicts.items()
               if any(b.startswith(("missing", "says", "does not cite")) for b in v.behaviour)]
    n = len(shipped.verdicts)
    obs = len(content) / n
    r = VALIDATION["content checker (code)"]
    tpr, tnr = r["caught"] / r["failures"], r["passed"] / r["passes"]
    corr = corrected_failure_rate(obs, tpr, tnr)
    ev = [False] * r["caught"] + [True] * (r["failures"] - r["caught"]) + [True] * r["passed"] \
        + [False] * (r["passes"] - r["passed"])
    hu = [False] * r["failures"] + [True] * r["passes"]
    band = bootstrap_interval(ev, hu, obs)
    print(f"\nShipped build, content failures found by the checker: {len(content)}/{n} = {pct(obs)}.")
    print(f"Corrected for its known misses (Rogan and Gladen, as on Day 2): {pct(corr)}.")
    if band:
        print(f"With TPR and TNR resampled from their few labels: [{pct(band[0])}, {pct(band[1])}].")
    print("Six labels behind the checker's rates make the corrected rate a wide range. This is")
    print("why the charter lets weak instruments widen the uncertainty, not decide alone, and")
    print("why the floors use the code checks people confirmed on every labelled failure.")


def step_6(cand_build, cases, repeat: int, live: bool) -> None:
    rule("STEP 6  Repeatability: the same build, the same questions, again")
    if not live:
        print("The reference run is deterministic: the stub gives the same answer every time,")
        print("so a second run changes nothing. That is the stub's property, not a model's.")
        print("Live, run with --repeat 3: the candidate answers the suite three times and every")
        print("question whose verdict changes between runs is listed. A change of one build")
        print("against another smaller than this run-to-run change is noise.")
        return
    if repeat < 2:
        print("Add --repeat 2 (or more) to answer the suite again and count verdict flips.")
        return
    runs = [run(cand_build, cases, live) for _ in range(repeat)]
    flips = [k for k in runs[0].verdicts if len({r.verdicts[k].passed for r in runs}) > 1]
    words = [k for k in runs[0].verdicts if len({(r.verdicts[k].trace.answer if r.verdicts[k].trace else "")
                                                for r in runs}) > 1]
    every = sum(all(r.verdicts[k].passed for r in runs) for k in runs[0].verdicts)
    print(f"{len(words)} of {len(cases)} answers changed wording between runs; {len(flips)} changed verdict.")
    print(f"pass^{repeat} (passed every time): {every}/{len(cases)}")
    for k in flips:
        print(f"  {k:<6} " + "  ".join("pass" if r.verdicts[k].passed else "FAIL" for r in runs))


def step_7(shipped, cand, t) -> None:
    rule("STEP 7  The decision record")
    n = t.n
    k0, k1 = sum(shipped.passed().values()), sum(cand.passed().values())
    d = decide(shipped, cand)
    record = [
        ("Question", f"May {cand.name} replace {shipped.name}?"),
        ("Evidence", f"{n} suite questions answered by both builds, {'reference run' if DEFAULT.offline else 'live'}"),
        ("Result", f"pass {interval(k0, n)} -> {interval(k1, n)}"),
        ("Paired", f"{len(t.fixed)} fixed, {len(t.broke)} broke; exact McNemar p = {p_text(t.p)}; "
                   f"P(better) {prob_text(t.prob_better)}"),
        ("Gate", d.summary()),
        ("Not shown", "real traffic; a second run of a live model; questions outside the suite's mix"),
        ("Decision", "SHIP" if d.ship else "BLOCK, and fix what the gate names (python run.py capstone)"),
        ("Owner", "the person named in charter field 8 for each failing floor"),
    ]
    for k, v in record:
        print(f"  {k:<10} {v}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", default="repaired", choices=[k for k in builds() if k != "shipped"])
    ap.add_argument("--ids", default="")
    ap.add_argument("--source", default="")
    ap.add_argument("--repeat", type=int, default=1, help="live: answer the suite this many times with the candidate")
    args = ap.parse_args(argv)
    banner(DEFAULT)
    live = not DEFAULT.offline
    cases = select(build_suite(), args.ids, args.source)
    b = builds()
    shipped = run(b["shipped"], cases, live)
    cand = run(b[args.candidate], cases, live)
    step_1(shipped, cand)
    t = step_2(shipped, cand)
    step_3(t)
    step_4(shipped, cand, t)
    step_5(shipped, cand)
    step_6(b[args.candidate], cases, args.repeat, live)
    step_7(shipped, cand, t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
