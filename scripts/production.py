"""Day 4 hands-on, Module 10: evaluation after release, on a simulated month.

There is no real traffic in a workshop. This builds four weeks of it from
Day 2's tuples, with the mix, the feedback model and the review budget
written at the top of src/evals/day4_production.py, and prints SIMULATED
beside every number that depends on them.

  STEP 1  the traffic: 1,000 requests a week; the shipped build serves weeks
          1 to 3, repaired + current-only is released in week 4
  STEP 2  online checks on every request: the code checks that need no
          reference answer, with intervals, week by week
  STEP 3  feedback: who clicks, what they complain about, what they cannot see
  STEP 4  human review at a fixed budget: three ways to choose 50 answers
  STEP 5  promotion: failures found in review become suite questions
  STEP 6  the yardstick: in week 4 an instrument was edited too; separate the
          system's change from the instrument's

Run:  python run.py production                  live: answers each distinct question once per build
      python run.py production --offline        the reference run, in seconds
      python run.py production --budget 100     a bigger review budget
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from evals import day4_production as P  # noqa: E402
from evals.day4_provenance import INSTRUMENTS, instrument_sha, sha  # noqa: E402
from evals.day4_report import builds, rule, short  # noqa: E402
from evals.day4_stats import pct, reviews_for, wilson  # noqa: E402
from evals.day4_suite import build_suite  # noqa: E402
from helix import DEFAULT  # noqa: E402
from helix.console import banner, progress  # noqa: E402

import inspect  # noqa: E402


def step_1(live: bool) -> None:
    rule("STEP 1  SIMULATED traffic: four weeks, 1,000 requests a week")
    print("Each request is one of Day 2's tuples (topic x asker x kind of question), in the")
    print("reference wording, drawn with these weights (an assumption, not a measurement):")
    print("  topic  " + ", ".join(f"{k} {v:.0%}" for k, v in P.TOPIC_MIX.items()))
    print("  kind   " + ", ".join(f"{k} {v:.0%}" for k, v in P.KIND_MIX.items()))
    print("  asker  the four Day 2 askers, equally likely")
    print("\nThe shipped build serves weeks 1 to 3. Repaired + current-only is released at the")
    print("start of week 4. Each distinct request is answered once per build"
          + (", live." if live else "."))


def step_2(weeks: dict[int, list], week4_v1: list) -> None:
    rule("STEP 2  Online checks: code, on every request, no reference answer needed")
    print(f"{'week':<6} {'build':<17} {'leak':>26} {'stale':>26}")
    for w, rs in weeks.items():
        r = P.online_rates(rs)
        build = "shipped" if w < 4 else "repaired+current"
        print(f"{w:<6} {build:<17} {P.rate_line(r['leak'], len(rs)):>26} {P.rate_line(r['stale'], len(rs)):>26}")
    print("\n(count, rate, 95% Wilson interval). Ungrounded answers: 0 in every week, which is")
    print("what a stub that copies its sources would give; your model will differ.")
    lo = min(P.online_rates(rs)["leak"] for w, rs in weeks.items() if w < 4)
    hi = max(P.online_rates(rs)["leak"] for w, rs in weeks.items() if w < 4)
    print(f"\nWeeks 1 to 3: same build, same mix, and the leak count moves from {lo} to {hi}.")
    print("That is chance in which requests arrived. A dashboard that alarms on a move this")
    print("size alarms every week. Compare intervals, not points.")
    r4 = P.online_rates(weeks[4])
    r4v1 = P.online_rates(week4_v1)
    print(f"\nWeek 4: leaks {r4['leak']}, and the stale rate jumps to {pct(r4['stale'] / len(weeks[4]))}."
          f" Step 6 explains the jump.")
    print(f"(The same week counted with the charter's stale check: {r4v1['stale']}.)")


def step_3(week: list) -> None:
    rule("STEP 3  SIMULATED feedback: what users tell you, and what they cannot")
    print(f"Feedback model: {P.CLICK_RATE:.0%} of users click at all. Of those who click, the share")
    print("who click thumbs-down, by what happened to them:")
    for k, v in P.THUMBS_DOWN.items():
        print(f"  {k:<15} {v:.0%}")
    clicked = [r for r in week if r.clicked]
    down = [r for r in week if r.thumbs_down]
    truth = Counter(r.truth for r in week)
    got = Counter(r.truth for r in down)
    print(f"\nWeek 3: {len(clicked)} clicks, {len(down)} thumbs-down out of {len(week)} requests.\n")
    print(f"{'what happened':<16} {'requests':>9} {'thumbs-down':>12}")
    for k in ("leak", "missed refusal", "false refusal", "stale", "ungrounded", "ok"):
        print(f"{k:<16} {truth[k]:>9} {got[k]:>12}")
    print("\nThe two failures the charter cares most about, a restricted page shown and a")
    print("superseded policy cited, are the two a user cannot see. Feedback is a signal about")
    print("what users can see; it is not a quality measure, and its silence is not a pass.")


def step_4(week: list, budget: int) -> list:
    rule(f"STEP 4  Human review at a fixed budget: {budget} answers a week")
    true = sum(map(P.refusal_error, week))
    print("What only a reviewer can see here: refusal errors (Helix should have declined and")
    print(f"answered, or the reverse). In week 3 there are {true} of them in {len(week)} requests "
          f"({pct(true / len(week))}), SIMULATED.")
    g = P.strata(week)
    print("\nWhere they are: " + ", ".join(f"{k} {len(v)} requests, {sum(map(P.refusal_error, v))} errors"
                                         for k, v in g.items()))
    print(f"\nThree ways to spend {budget} reviews, each drawn 500 times:\n")
    print(f"{'strategy':<14} {'read':>5} {'errors found':>13} {'estimated rate':>15} {'95% of draws':>17}")
    res = P.compare_strategies(week, budget)
    for s in res:
        print(f"{s.name:<14} {s.reviewed:>5.0f} {s.found:>13.1f} {pct(s.estimate):>15} "
              f"{'[' + pct(s.low) + ', ' + pct(s.high) + ']':>17}")
    print(f"\nTrue rate {pct(true / len(week))}. Random reading finds what is common and estimates the rate")
    print("without bias. Thumbs-down picks itself, always the same few, and its rate describes")
    print("who complains. Stratified oversamples where errors concentrate (Helix declined; a")
    print("user complained), weights each group by its size, and finds about twice as many")
    print("errors to fix; with so few reviews left for the large remainder, its estimate is")
    print("wider. One budget, two jobs: a random slice for the rate, a targeted slice for")
    print("finding failures. Run both, and report the rate from the random slice.")
    for w in (0.10, 0.05):
        print(f"  for +/- {w:.0%} on a rate near {pct(true / len(week), 0)}: about "
              f"{reviews_for(w, true / len(week))} random reviews")
    print("The count of reviews sets the precision, not the share of traffic: 5 percent of")
    print("a quiet week and 5 percent of a busy one are different evidence.")
    return next(s for s in res if s.name == "stratified").one_draw


def step_5(reviewed: list) -> None:
    rule("STEP 5  Promotion: a failure found in production becomes a suite question")
    found = P.distinct_failing(reviewed)
    texts = {c.text for c in build_suite()}
    cands = P.promotion_candidates(found, texts)
    print(f"One stratified draw found {len(found)} distinct failing requests; {len(cands)} are not in the")
    print("suite yet (first five shown). Each needs a pass condition a person writes before it")
    print("joins: what it must say, must not say, or that it must be declined.\n")
    for c in cands:
        print("  " + json.dumps(c, ensure_ascii=False))
    print("\nPaste the finished entries into data/queries/day4_suite_changes.json under")
    print('"cases". No script writes that file: what joins the suite is a decision.')


def step_6(week3: list, week4_v1: list, week4_v2: list) -> None:
    rule("STEP 6  The yardstick: when the instrument changes too")
    print("In week 4 someone also edited the current-policy check, in good faith: 'a page not")
    print("reviewed since 2025 counts as stale'. It sounds stricter. It is a new definition.")
    print(f"  charter check  {sha(inspect.getsource(P.stale))}   cites a superseded page")
    print(f"  edited check   {sha(inspect.getsource(P.stale_by_date))}   cites a page last modified before 2025\n")
    n = len(week3)
    a = P.online_rates(week3)["stale"] / n
    b = P.online_rates(week4_v1)["stale"] / len(week4_v1)
    c = P.online_rates(week4_v2)["stale"] / len(week4_v2)
    d = P.decompose(a, b, c)
    print(f"Dashboard, stale rate: week 3 {pct(a)} -> week 4 {pct(c)}  ({d['total'] * 100:+.1f} points)")
    print(f"  the release (both weeks with the charter check):  {d['system'] * 100:+.1f} points")
    print(f"  the instrument (week 4, edited minus charter):     {d['instrument'] * 100:+.1f} points")
    print("\nThe release removed every stale citation. The edited check flagged every answer")
    print("that cites the current 2024 parental leave page. Read alone, the dashboard says the")
    print("release made things worse. Score the new week with the old instrument, and the new")
    print("instrument on the old week, before attributing a change to either.")
    print("\nInstruments carry fingerprints for this reason (python run.py day4, STEP 4):")
    for name, fn in INSTRUMENTS.items():
        print(f"  {name:<9} {instrument_sha(fn)}  {fn.__module__}.{fn.__name__}")
    print("A changed fingerprint means the instrument's error rates are unknown again: re-measure")
    print("it on the anchor set (the Day 2 golden set: python run.py evaluators) and attack it")
    print("(python run.py redteam) before comparing any number it produces with an old one.")
    print("For a judge, the model name is part of the instrument: a hosted model can change")
    print("behind the same name, so re-measure on a schedule, not only when you change it.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=P.REVIEW_BUDGET, help="answers reviewed per week")
    args = ap.parse_args(argv)
    banner(DEFAULT)
    live = not DEFAULT.offline
    b = builds()
    note = (lambda i, text: progress(short(text, 60), i, 96)) if live else None
    shipped = P.Server(b["shipped"], on_new=note)
    cand = P.Server(b["repaired-current"], on_new=note)
    step_1(live)
    weeks = {w: P.serve_week(w, shipped) for w in (1, 2, 3)}
    weeks[4] = P.serve_week(4, cand, stale_check=P.stale_by_date)
    week4_v1 = P.serve_week(4, cand)
    step_2(weeks, week4_v1)
    step_3(weeks[3])
    reviewed = step_4(weeks[3], args.budget)
    step_5(reviewed)
    step_6(weeks[3], week4_v1, weeks[4])
    print("\nNothing is stored: the month is drawn again, the same way, on every run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
