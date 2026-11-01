"""The Day 1 fixture is a teaching artifact, so it is pinned by tests.

If a change to chunking, retrieval or the stub moves which failure points the
six seeded queries produce, these tests fail and the course material has to be
updated with them. That is the point: the slides, the document and the code
must agree, and the only way to keep them agreeing is to assert it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evals.day1_triage import triage  # noqa: E402
from helix import DEFAULT, REPAIRED, answer, build_index  # noqa: E402

SUITE = json.loads((ROOT / "data" / "queries" / "day1_seed.json").read_text())

#: The failure points each seeded query must produce under the default config.
EXPECTED: dict[str, set[str]] = {
    "q1": {"FP9", "FP8", "FP3"},
    "q2": {"FP9", "FP8"},
    "q3": {"FP2"},
    "q4": {"FP11"},
    "q5": set(),
    "q6": set(),
}


@pytest.fixture(scope="module")
def index():
    return build_index(DEFAULT)


def _run(q, index, cfg):
    t = answer(q["text"], index, cfg, user_office=q["user_office"], user_role=q["user_role"])
    return t, {f.code for f in triage(t, q.get("expected_evidence"))}


@pytest.mark.parametrize("q", SUITE["queries"], ids=[q["id"] for q in SUITE["queries"]])
def test_seeded_failures(q, index):
    _, codes = _run(q, index, DEFAULT)
    assert codes == EXPECTED[q["id"]], f"{q['id']}: got {sorted(codes)}"


def test_two_queries_are_clean(index):
    clean = [q["id"] for q in SUITE["queries"] if not _run(q, index, DEFAULT)[1]]
    assert clean == ["q5", "q6"]


def test_every_trace_has_the_six_spans(index):
    t, _ = _run(SUITE["queries"][0], index, DEFAULT)
    assert [s.name for s in t.spans] == [
        "query.transform",
        "retrieve",
        "rerank",
        "context.assemble",
        "generate",
        "postprocess",
    ]


def test_provenance_is_complete(index):
    """Every chunk that reaches the model carries the three load-bearing fields."""
    t, _ = _run(SUITE["queries"][0], index, DEFAULT)
    assert t.config_hash
    for c in t.retrieved:
        assert c.doc_version and c.last_modified


def test_dropped_chunks_are_recorded(index):
    """FP2 and FP3 are only detectable because dropped chunks are kept."""
    t, _ = _run(SUITE["queries"][0], index, DEFAULT)
    dropped = [c for c in t.considered if c.dropped_at]
    assert dropped, "no dropped chunks recorded; FP2/FP3 would be invisible"
    assert all(c.dropped_at in {"retrieve", "rerank", "context.assemble"} for c in dropped)


def test_config_hash_changes_with_configuration():
    assert DEFAULT.hash() != REPAIRED.hash()


def test_repaired_config_fixes_entitlement_leakage():
    """The entitlement switch is the one repair that must be absolute."""
    idx = build_index(REPAIRED)
    q = next(q for q in SUITE["queries"] if q["id"] == "q4")
    _, codes = _run(q, idx, REPAIRED)
    assert "FP11" not in codes


def test_repaired_config_does_not_fix_the_corpus():
    """A configuration change cannot repair a corpus problem.

    The superseded 2022 page is still indexed under every configuration, so
    FP8 survives the repair. This is the Day 4 lesson about which kind of
    intervention a given failure point actually needs.
    """
    idx = build_index(REPAIRED)
    q = next(q for q in SUITE["queries"] if q["id"] == "q2")
    _, codes = _run(q, idx, REPAIRED)
    assert "FP8" in codes


def test_offline_mode_is_deterministic(index):
    q = SUITE["queries"][0]
    a = answer(q["text"], index, DEFAULT).answer
    b = answer(q["text"], index, DEFAULT).answer
    assert a == b
