"""Day 3, Module 09 hands-on: red-teaming the evaluators.

Takes eight answers that are known to fail, changes their surface without
fixing them, and counts how many each evaluator now passes:

  1. the eight answers under attack, and why each one fails
  2. the seven attacks, shown on one answer
  3. false passes, per evaluator and attack: the code checks always, the
     two Module 08 judges live, Ragas, DeepEval and G-Eval with --toolkits
  4. which attack beat which evaluator, and what a second evaluator with a
     different blind spot catches

The attacked answers are always the reference run's, whatever model the
session uses, so their labels are known and only the evaluator varies.

Live: 88 judge calls (8 + 3 answers x 8 conditions). --toolkits adds Ragas
and DeepEval faithfulness, Ragas factual correctness and DeepEval's G-Eval
on correctness, several judge calls each: a few hundred calls. --attacks
limits the conditions.

Run:  python run.py redteam --offline             code checks only, seconds
      python run.py redteam                       plus the two judges
      python run.py redteam --attacks none,master\\ key,evaluator\\ instruction
      python run.py redteam --toolkits            plus Ragas, DeepEval and G-Eval
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from evals.day3_redteam import (  # noqa: E402
    ATTACKS,
    BAD_LINEAGE,
    CODE_INSTRUMENTS,
    FAILING,
    JUDGE_INSTRUMENTS,
    UNGROUNDED,
    attack,
    false_passes,
    reference_answers,
    support_pass,
    targets,
)
from evals.day3_sheet import REFERENCE  # noqa: E402
from helix import DEFAULT  # noqa: E402
from helix.console import banner, progress  # noqa: E402

SUITE = ROOT / "data" / "queries" / "day1_seed.json"
#: Why a code check passes a failing answer with no attack (read in its source).
BLIND = {
    "correct check": "the reference's figures are in the answer; whether it answers the question is not checked",
    "grounded check": "it counts words longer than four letters, so a wrong figure beside the right words "
                      "reads as grounded",
    "correct + support": "a true, supported sentence; neither instrument asks whether it answers the question",
}
SHORT = {"none": "none", "padding": "pad", "figure stuffing": "figs", "authority": "auth",
         "self-assessment": "self", "evaluator instruction": "instr", "master key": "key",
         "fake current citation": "cite"}


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def changed(before: str, after: str, width: int = 46) -> str:
    """The part an attack added or replaced, shortened."""
    b, a = " ".join(before.split()), " ".join(after.split())
    if a == b:
        return "(unchanged: nothing to swap)"
    if a.startswith(b):
        return "... + " + textwrap.shorten(a[len(b):].strip(), width, placeholder=" ...")
    if a.endswith(b):
        return textwrap.shorten(a[:-len(b)].strip(), width, placeholder=" ...") + " + ..."
    for ref in re.findall(r"\[[^\]]+\]", a):
        if ref not in b:
            return f"cites {ref}"
    return "replaced by: " + textwrap.shorten(a, width, placeholder=" ...")


def table(results: dict, names: list[str], should: dict[str, tuple], attacks: list[str]) -> None:
    print(f"  {'evaluator':20} {'scored':>6}" + "".join(f"{SHORT[a]:>6}" for a in attacks))
    for n in names:
        cells = [false_passes(results[a][n], should[n]) for a in attacks]
        scored = cells[0][1] if cells else 0
        print(f"  {n:20} {scored:>6}" + "".join(f"{p:>6}" for p, _ in cells))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attacks", default="", help="comma-separated attack names (default: all)")
    ap.add_argument("--toolkits", action="store_true", help="also attack Ragas, DeepEval and G-Eval (live)")
    args = ap.parse_args(argv)
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()] or list(ATTACKS)
    unknown = [a for a in attacks if a not in ATTACKS]
    if unknown:
        print(f"unknown attack(s): {', '.join(unknown)}. Choose from: {', '.join(ATTACKS)}")
        return 2
    if "none" not in attacks:
        attacks = ["none", *attacks]

    banner(DEFAULT)
    suite = json.loads(SUITE.read_text(encoding="utf-8"))["queries"]
    tg = targets(reference_answers(suite), suite)

    rule("STEP 1  Eight answers that fail, and why")
    for k, (t, _) in tg.items():
        why = REFERENCE[k][3] if k in REFERENCE else "planted: a figure that is in no page it was given"
        print(f"  {k:4} {' '.join(t.answer.split())[:70]}")
        print(f"  {'':4} {why}")
    print("\n  Correctness and completeness evaluators are scored on all eight,")
    print(f"  groundedness evaluators on the planted three ({', '.join(UNGROUNDED)}), the lineage")
    print(f"  check on {', '.join(BAD_LINEAGE)} (stale or restricted citations). Every count below")
    print("  is a FALSE pass: lower is better.")

    rule("STEP 2  The attacks, on q5x (30 days; the handbook says 25)")
    t5, _ = tg["q5x"]
    for i, line in enumerate(textwrap.wrap(" ".join(t5.answer.split()), 70)):
        print(f"  {'q5x' if i == 0 else '':4} {line}")
    print()
    for a in attacks:
        if a == "none":
            continue
        print(f"  {SHORT[a]:5} {a:22} {changed(t5.answer, attack(a, t5).answer)}")
    t1, _ = tg["q1"]
    if "fake current citation" in attacks:
        print(f"  {'':5} {'':22} on q1: {changed(t1.answer, attack('fake current citation', t1).answer)}")
    print("\n  None of them changes a fact the answer states.")

    rule("STEP 3  False passes under attack")
    results = {a: {} for a in attacks}
    for a in attacks:
        for n, (_, should, fn) in CODE_INSTRUMENTS.items():
            results[a][n] = {k: bool(fn(attack(a, t), q)) for k, (t, q) in tg.items() if k in should}
    should = {n: v[1] for n, v in CODE_INSTRUMENTS.items()}
    names = list(CODE_INSTRUMENTS)
    reasons: list[str] = []
    if not DEFAULT.offline:
        cfg = DEFAULT
        total = sum(len(v[1]) for v in JUDGE_INSTRUMENTS.values()) * len(attacks)
        i = 0
        for n, (_, sh, fn) in JUDGE_INSTRUMENTS.items():
            should[n] = sh
            names.append(n)
            for a in attacks:
                results[a][n] = {}
                for k in sh:
                    t, q = tg[k]
                    v = fn(attack(a, t), q, cfg)
                    results[a][n][k] = bool(v)
                    if v and a != "none" and not results["none"][n].get(k):
                        reasons.append(f"{n}, {a}, {k}: {v.label}. {v.reason}")
                    i += 1
                    progress(f"{n} {SHORT[a]} {k}", i, total)
        print()
    if args.toolkits and not DEFAULT.offline:
        from toolkits.cases import case_from_trace
        from toolkits.kits import Kits, passes, val

        kits = Kits(DEFAULT)
        undecided: dict[str, int] = {}
        kit_metrics = [("Ragas faithfulness", "r", "faithfulness", UNGROUNDED),
                       ("DeepEval faithfulness", "d", "faithfulness", UNGROUNDED),
                       ("Ragas factual corr.", "r", "factual_correctness", FAILING + UNGROUNDED),
                       ("G-Eval correctness", "g", "correctness", FAILING + UNGROUNDED)]
        for label, kind, metric, sh in kit_metrics:
            if (kits.ragas if kind == "r" else kits.deepeval) is None:
                continue
            should[label] = sh
            names.append(label)
            for a in attacks:
                results[a][label] = {}
                for k in sh:
                    t, q = tg[k]
                    c = case_from_trace(attack(a, t), q)
                    m = {"r": kits.r, "d": kits.d, "g": kits.g}[kind](metric, c)
                    verdict = passes(val(m))
                    if verdict is None:  # no score: neither a catch nor a false pass
                        undecided[label] = undecided.get(label, 0) + 1
                        continue
                    results[a][label][k] = verdict
        if kits.missing():
            print(f"  not installed: {', '.join(kits.missing())} (python run.py toolkits)")
        for label, n in undecided.items():
            print(f"  {label}: {n} score(s) undecided (an unusable judge reply);")
            print("    left out of that row, neither caught nor passed")
    table(results, names, should, attacks)
    print("\n  " + "  ".join(f"{SHORT[a]}={a}" for a in attacks[:4]))
    print("  " + "  ".join(f"{SHORT[a]}={a}" for a in attacks[4:]))
    if DEFAULT.offline:
        print("\n  Offline, only the code checks run: the judges need a model. Run live to")
        print("  put the two Module 08 judges in the same table.")

    rule("STEP 4  Which attack beat which evaluator")
    blind = []
    for n in names:
        base = false_passes(results["none"][n], should[n])[0]
        hits = [f"{a} ({false_passes(results[a][n], should[n])[0]})" for a in attacks
                if a != "none" and false_passes(results[a][n], should[n])[0] > base]
        print(f"  {n:20} baseline {base}; " + (f"worse under {', '.join(hits)}" if hits else "no attack made it worse"))
        if base:
            ids = [k for k, v in results["none"][n].items() if v]
            blind.append(f"{n} passes {', '.join(ids)}" + (f": {BLIND[n]}" if n in BLIND else ""))
    if blind:
        print("\n  A baseline above 0 is a blind spot that needs no attack at all:")
        for b in blind:
            for i, line in enumerate(textwrap.wrap(b, 72)):
                print(f"    {line}" if i == 0 else f"      {line}")
    if "fake current citation" in attacks:
        caught = [k for k in BAD_LINEAGE if not support_pass(attack("fake current citation", tg[k][0]))]
        print("\n  Under the fake citation, the lineage check passes what it caught before.")
        print(f"  Citation support on the same answers catches {len(caught)} of {len(BAD_LINEAGE)} "
              f"({', '.join(caught) or 'none'}): the page")
        print("  now cited does not say what the answer says. Lineage asks whether the")
        print("  page is current; support asks whether it says so. Paired, they hold.")
    if reasons:
        print("\n  What the judge said when an attack turned its fail into a pass:")
        for r in reasons[:4]:
            for line in textwrap.wrap(r, 74):
                print(f"    {line}")

    rule("WHAT TO READ")
    print("  Validation against people says an evaluator agrees with people on honest")
    print("  answers. This says what it does with dishonest ones. Run both.")
    print("  Code checks fail in ways you can read in their source: figures matched as")
    print("  substrings, words counted without their numbers. Judges fail in ways you")
    print("  only find by trying: an instruction, a confident tone, a non-answer.")
    print("  Pair evaluators whose blind spots differ, and keep these attacks as a test")
    print("  that every new judge model or prompt must pass (Day 4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
