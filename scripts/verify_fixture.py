"""Verify the Day 1 fixture without pytest.

The same assertions as tests/test_day1_fixture.py, runnable anywhere Python
and numpy exist. Run it after changing chunking, retrieval or the stub: if
the seeded failures move, the slides and the course document have to move
with them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evals.day1_triage import triage  # noqa: E402
from helix import DEFAULT, REPAIRED, answer, build_index  # noqa: E402

EXPECTED = {
    "q1": {"FP9", "FP8", "FP3"},
    "q2": {"FP9", "FP8"},
    "q3": {"FP2"},
    "q4": {"FP11"},
    "q5": set(),
    "q6": set(),
}


def main() -> int:
    suite = json.loads((ROOT / "data" / "queries" / "day1_seed.json").read_text())
    index = build_index(DEFAULT)
    failures = 0

    print(f"config  {DEFAULT.describe()}\n")
    for q in suite["queries"]:
        t = answer(q["text"], index, DEFAULT,
                   user_office=q["user_office"], user_role=q["user_role"])
        got = {f.code for f in triage(t, q.get("expected_evidence"))}
        ok = got == EXPECTED[q["id"]]
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {q['id']}  "
              f"expected {sorted(EXPECTED[q['id']]) or '[]'}  got {sorted(got) or '[]'}")

    spans = [s.name for s in t.spans]
    assert spans == ["query.transform", "retrieve", "rerank",
                     "context.assemble", "generate", "postprocess"], spans
    assert DEFAULT.hash() != REPAIRED.hash()
    assert all(c.doc_version and c.last_modified for c in t.retrieved)
    print("\nspans, provenance and config hashing: OK")

    r_index = build_index(REPAIRED)
    q4 = next(q for q in suite["queries"] if q["id"] == "q4")
    r4 = {f.code for f in triage(
        answer(q4["text"], r_index, REPAIRED, user_role=q4["user_role"]),
        q4.get("expected_evidence"))}
    assert "FP11" not in r4, "the repaired config must close the entitlement leak"
    q2 = next(q for q in suite["queries"] if q["id"] == "q2")
    r2 = {f.code for f in triage(
        answer(q2["text"], r_index, REPAIRED, user_role=q2["user_role"]),
        q2.get("expected_evidence"))}
    assert "FP8" in r2, "a config change must not appear to fix a corpus problem"
    print("repaired config behaves as documented: OK")

    print("\nFIXTURE VERIFIED" if not failures else f"\n{failures} MISMATCH(ES)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
