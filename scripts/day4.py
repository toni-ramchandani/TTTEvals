"""Day 4 hands-on: compare two configurations and gate the release on it.

The Improve phase, and the point where evaluation stops being a report and
becomes a decision.

Everything here depends on the config hash recorded in every trace on Day 1.
Without it, "before" and "after" are two piles of traces with no way to say what
differed, and a regression report is an anecdote.

The comparison is paired: the same six queries run through both arms. Unpaired
comparison needs far more examples to separate two systems from noise.

Run:  python run.py day4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evals.day4_compare import Arm, compare, gate  # noqa: E402
from helix import DEFAULT, REPAIRED, answer, build_index  # noqa: E402
from helix.console import Stream, banner  # noqa: E402
from tracing import write_jsonl  # noqa: E402

SUITE = ROOT / "data" / "queries" / "day1_seed.json"
BEFORE = ROOT / "data" / "traces" / "day4_before.jsonl"
AFTER = ROOT / "data" / "traces" / "day1_repaired.jsonl"


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def run_arm(cfg, out: Path, queries: list[dict]) -> list:
    index = build_index(cfg)
    traces = []
    for q in queries:
        print(f"  {q['id']}  {q['text'][:70]}")
        stream = Stream("      -> ")
        traces.append(answer(q["text"], index, cfg,
                             user_office=q.get("user_office", "amsterdam"),
                             user_role=q.get("user_role", "employee"),
                             on_token=stream))
        stream.done()
    write_jsonl(traces, out)
    return traces


def main() -> int:
    queries = json.loads(SUITE.read_text(encoding="utf-8"))["queries"]
    evidence = {q["text"]: q.get("expected_evidence") or "" for q in queries}

    banner(DEFAULT)
    rule("STEP 1  Two arms, same queries, one variable set changed")
    print(f"before: {DEFAULT.describe()}")
    print(f"after:  {REPAIRED.describe()}")
    print("\nThe config hash is what makes these comparable. Two arms carrying the")
    print("same hash mean nothing changed, or the hash is not capturing what did.")

    # Both arms run live now, on the same model, so the comparison is between
    # two configurations and nothing else. Nothing from an earlier day is reused.
    print("\nbefore arm:")
    before_traces = run_arm(DEFAULT, BEFORE, queries)
    print("\nafter arm:")
    after_traces = run_arm(REPAIRED, AFTER, queries)

    before = Arm("before", DEFAULT.hash(), before_traces)
    after = Arm("after", REPAIRED.hash(), after_traces)

    rule("STEP 2  Paired comparison over the failure-point catalog")
    print(compare(before, after, evidence))

    rule("STEP 3  The gate")
    print("FP11 and FP8 are the floors: entitlement leakage is a security defect")
    print("wearing a quality costume, and a stale answer carries the company's name.")
    print("Both were agreed before any number was looked at. A threshold written")
    print("after seeing the result is a rationalisation.\n")
    passed, why = gate(before, after, evidence)
    print(f"  DECISION: {'SHIP' if passed else 'BLOCK'} ({why})")

    rule("STEP 4  What would make this a real release gate")
    for line in (
        "run it in CI on every change to the index, the prompt or the model",
        "pin the model version; an unpinned model makes two runs incomparable",
        "  (an Ollama tag like llama3.1:8b can be re-pulled to new weights; pin the digest)",
        "report confidence intervals, not point estimates, on a set this small",
        "sample production traffic into the suite continuously, so it keeps",
        "  resembling what users actually ask",
        "re-run error analysis when the gate passes but users still complain:",
        "  that means the taxonomy has drifted from reality",
    ):
        print(f"  - {line}" if not line.startswith("  ") else f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
