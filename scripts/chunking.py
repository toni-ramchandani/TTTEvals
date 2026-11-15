"""Day 3, Module 07 hands-on: chunking as a variable, measured.

Re-chunks and re-indexes the handbook once per configuration and scores
retrieval on the Day 1 questions with expected evidence: Recall@5 and hit
rate for the first stage, nDCG@5 for the order, and how many questions'
evidence survived to the model under the default 340-token budget.

No answer is generated. Live, each configuration costs one embedding pass
over the handbook plus one embedding per question (about 8 x 6 calls).

Run:  python run.py chunking
      python run.py chunking --offline        the reference run, seconds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from evals.day3_chunking import configs, score_config  # noqa: E402
from helix import DEFAULT  # noqa: E402
from helix.console import banner, progress  # noqa: E402

SUITE = ROOT / "data" / "queries" / "day1_seed.json"


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def f2(x) -> str:
    return "-" if x is None else f"{x:.2f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    suite = json.loads(SUITE.read_text(encoding="utf-8"))["queries"]
    banner(DEFAULT)
    rule("STEP 1  One chunking setting changed at a time, same questions, same judgments")
    print("Relevant: a chunk carrying ALL the expected evidence, judged in each")
    print("configuration's own chunks. A question whose evidence no single chunk")
    print("carries scores 0: the fact was split across chunks.\n")
    grid = configs(DEFAULT)
    results = []
    for i, (label, cfg) in enumerate(grid, 1):
        results.append((label, score_config(cfg, suite)))
        progress(label, i, len(grid))
    print()
    print(f"{'configuration':<32} {'chunks':>6} {'R@5':>5} {'hit':>5} {'nDCG@5':>7} {'reached':>8}  split")
    print("-" * 78)
    for label, r in results:
        print(f"{label:<32} {r['chunks']:>6} {f2(r['recall']):>5} {f2(r['hit']):>5} {f2(r['ndcg']):>7} "
              f"{r['reached']:>4}/{r['asked']:<3}  {', '.join(r['split']) or '-'}")
    best = max(results, key=lambda lr: ((lr[1]['recall'] or 0), (lr[1]['ndcg'] or 0), lr[1]['reached']))
    print(f"\nHighest recall, then nDCG: {best[0]}.")
    print("reached = questions whose relevant chunk survived to the model under the")
    print("default 340-token budget. Bigger chunks can raise recall and still lower")
    print("this number: they fill the budget sooner.")
    rule("WHAT TO READ")
    print("  Which setting finds the evidence (R@5), and which puts it first (nDCG)?")
    print("  Which questions are 'split', and at which sizes? That is a chunk too small.")
    print("  Does the best R@5 also get the evidence to the model? If not, the budget and")
    print("  the chunk size have to be chosen together.")
    print("  Five questions choose nothing for production: run the grid on your own")
    print("  question set, as the Day 2 golden set grows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
