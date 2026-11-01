"""Day 4 hands-on, Module 10: the release gate as tests, run by DeepEval.

`python run.py day4` makes the SHIP or BLOCK decision in a script. In a real
team the same decision runs on every change, as tests in CI. This runs
gates/test_release_gate.py: the candidate configuration answers every
question live on your model, then one test per question asserts the
charter's floors on that answer. Nothing from an earlier run is used.

  entitlement (FP11) and current policy (FP8)   Day 2's code checks, as DeepEval metrics
  faithfulness                                  DeepEval, judged by your model

A failing test blocks the release. Rerun after any change to the index, the
prompt or the model; in CI the same command runs on every pull request.

Run:  python run.py ci                                    in the toolkit environment
      python run.py ci --judge openai/gpt-5.4-mini          answers local, judge in the cloud
      python run.py ci --offline                          the reference run, code floors only
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolkits import env  # noqa: E402



def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    env.prepare()
    env.require("deepeval")
    try:
        import pytest
    except ImportError:
        print("pytest is not installed. Run:  pip install pytest")
        return 1
    from helix import DEFAULT

    how = "with the offline stub (the reference run)" if DEFAULT.offline else "live, on your model"
    print(f"Release gate: the candidate answers each question {how}, then one test per question.\n")
    code = pytest.main(["-q", "--tb=line", "-p", "no:cacheprovider", "-p", "no:deepeval",
                        str(ROOT / "gates"), *argv])
    print("\nDECISION: " + ("SHIP (every floor held)" if code == 0 else "BLOCK (a floor failed; read the test output)"))
    print("\n`python run.py day4` asks a different question: is anything worse than before?")
    print("This gate asks: does every answer meet the floor? The two can disagree. Fewer")
    print("stale answers than before passes the first and still fails the second while")
    print("one remains. The charter says which question your release gate asks.")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
