"""Day 4 hands-on, Module 10: the release gate as tests, run by DeepEval.

`python run.py day4` makes the SHIP or BLOCK decision in a script. In a real
team the same decision runs on every change, as tests in CI. This runs
gates/test_release_gate.py, which applies the same three kinds of rule:

  floors    one test per suite question: access, current policy and
            groundedness (code), plus DeepEval faithfulness judged by your
            model when live
  ratchet   one test: nothing that passes on the shipped build fails now
  budgets   one test: tokens, p95 answer time (live), failed model calls

Both builds answer the regression suite now; nothing from an earlier run is
used and nothing is saved. Any failing test blocks the release.

Run:  python run.py ci                                    in the toolkit environment
      python run.py ci --candidate candidate              the capstone's build
      python run.py ci --source day1,architectures        fewer live calls
      python run.py ci --judge openai/gpt-5.4-mini        answers local, judge in the cloud
      python run.py ci --offline                          the reference run, code floors only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from toolkits import env  # noqa: E402

CANDIDATES = ("repaired", "repaired-current", "candidate")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", default="repaired", choices=CANDIDATES)
    ap.add_argument("--source", default="", help="limit the suite, e.g. day1,architectures")
    args, rest = ap.parse_known_args(sys.argv[1:] if argv is None else argv)
    env.prepare()
    env.require("deepeval")
    try:
        import pytest
    except ImportError:
        print("pytest is not installed. Run:  pip install pytest")
        return 1
    from helix import DEFAULT

    os.environ["HELIX_GATE_CANDIDATE"] = args.candidate
    os.environ["HELIX_GATE_SOURCE"] = args.source
    how = "with the offline stub (the reference run)" if DEFAULT.offline else "live, on your model"
    print(f"Release gate: shipped and {args.candidate} answer the suite {how}; then the tests.\n")
    code = pytest.main(["-q", "--tb=line", "-p", "no:cacheprovider", "-p", "no:deepeval",
                        str(ROOT / "gates"), *rest])
    print("\nDECISION: " + ("SHIP (every floor held, nothing regressed, within budget)" if code == 0
                            else "BLOCK (read the failing tests above)"))
    print("Same rules as `python run.py day4`, written as tests so they run on every change.")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
