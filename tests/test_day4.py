"""Day 4: every figure the slides quote, from the reference run.

If one of these fails after a change, a Day 4 slide no longer matches the code.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from evals import day4_gate, day4_production as P
from evals.day4_gate import decide, model_calls, tokens
from evals.day4_provenance import corpus_sha, diff, fingerprint, planted_gap, template_sha
from evals.day4_report import COMBINATIONS, INTERVENTIONS, builds, parse_changes, single
from evals.day4_stats import (
    binom_two_sided, mcnemar_exact, min_discordant, paired, percentile, prob_better, reviews_for, smallest_win,
    wilson,
)
from evals.day4_suite import (
    FLOORS, Build, build_suite, current_only, guard_rule, load_changes, run_arm, suite_sha,
)
from helix import DEFAULT, REPAIRED, build_index
from helix.ingest import document_status

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cases():
    return build_suite()


@pytest.fixture(scope="module")
def arms(cases):
    b = builds()
    out = {k: run_arm(b[k], cases) for k in ("shipped", "repaired", "repaired-current", "candidate")}
    for n, cfg, tr, g in COMBINATIONS:
        out[n] = run_arm(Build(n, cfg, tr, g), cases)
    for n, *_ in INTERVENTIONS:
        out["single:" + n] = run_arm(single(n), cases)
    return out


# --- the suite ---------------------------------------------------------------
def test_suite_composition(cases):
    from collections import Counter
    assert len(cases) == 85
    assert Counter(c.source for c in cases) == {"day1": 6, "architectures": 13, "robustness": 23,
                                                "synthetic": 24, "conversation": 19}
    assert sum(c.should_refuse for c in cases) == 18
    assert suite_sha(cases) == "b36387f2b3a3"
    assert len({c.id for c in cases}) == 85


def test_synthetic_matches_day2_reference(cases):
    """The suite computes Day 2's reference questions; they equal what day2 --offline writes."""
    from evals.day4_suite import synthetic_reference
    ref = ROOT / "data" / "queries" / "day2_synthetic_reference.json"
    if ref.exists():
        data = json.loads(ref.read_text(encoding="utf-8"))
        assert synthetic_reference() == (data["queries"] if isinstance(data, dict) else data)
    assert [q["id"] for q in synthetic_reference()][:3] == ["s01", "s02", "s03"]


def test_changes_file_ships_empty():
    assert load_changes() == {"cases": [], "accepted": []}


def test_current_only_drops_only_superseded():
    index = build_index(DEFAULT)
    kept = current_only(index)
    status = document_status()
    gone = {c.doc_id for c in index.chunks} - {c.doc_id for c in kept.chunks}
    assert gone == {d for d, s in status.items() if s["status"] == "superseded"} == {"HB-PAR-2022"}
    assert len(kept.chunks) == kept.vectors.shape[0]


# --- the two Module 10 arms ---------------------------------------------------
def test_shipped_and_repaired(arms):
    s, r = arms["shipped"], arms["repaired"]
    assert sum(s.passed().values()) == 31 and sum(r.passed().values()) == 41
    assert [len(s.floor_failures(f)) for f in FLOORS] == [11, 12, 0]
    assert [len(r.floor_failures(f)) for f in FLOORS] == [0, 4, 0]
    assert r.floor_failures("current") == ["q2", "a7", "r2", "s22"]
    assert (tokens(s), tokens(r)) == (33063, 31505)
    t = paired(s.passed(), r.passed())
    assert (len(t.fixed), len(t.broke), t.broke) == (13, 3, ["a13", "r2", "s16"])


def test_gate_blocks_repaired_for_the_floor_and_the_ratchet(arms):
    d = decide(arms["shipped"], arms["repaired"], accepted={})
    assert not d.ship
    assert [(r.kind, r.name) for r in d.blocking()] == [("floor", "current"), ("ratchet", "per question")]
    latency = next(r for r in d.rules if r.name == "p95 answer time")
    assert latency.applies is False


def test_acceptance_needs_a_name_and_clears_the_ratchet(arms):
    d = decide(arms["shipped"], arms["candidate"], accepted={})
    assert [r.name for r in d.blocking()] == ["per question"]
    ok = {"a13": {"id": "a13", "accepted_by": "people partner lead", "reason": "old pass was luck"}}
    assert decide(arms["shipped"], arms["candidate"], accepted=ok).ship


def test_expired_or_anonymous_acceptance_does_not_count(tmp_path, monkeypatch):
    from datetime import date
    f = tmp_path / "changes.json"
    f.write_text(json.dumps({"cases": [], "accepted": [
        {"id": "a13", "accepted_by": "lead", "reason": "r", "review_by": "2026-01-01"},
        {"id": "r2", "reason": "no name"},
        {"id": "s16", "accepted_by": "lead", "reason": "r", "review_by": "2027-01-01"}]}))
    monkeypatch.setattr(day4_gate, "load_changes", lambda: load_changes(f))
    assert set(day4_gate.active_acceptances(date(2026, 11, 11))) == {"s16"}


# --- Module 11 statistics -------------------------------------------------------
def test_wilson_and_textbook_interval():
    lo, hi = wilson(31, 85)
    assert (round(lo, 3), round(hi, 3)) == (0.270, 0.471)
    lo, hi = wilson(1, 6)
    assert (round(lo, 3), round(hi, 3)) == (0.030, 0.564)
    p = 1 / 6
    assert p - 1.959964 * math.sqrt(p * (1 - p) / 6) < 0          # the textbook interval leaves 0..1


def test_mcnemar_and_probability():
    assert round(mcnemar_exact(13, 3), 4) == 0.0213
    assert round(prob_better(13, 3), 3) == 0.994
    assert round(mcnemar_exact(21, 1), 7) == round(46 / 2 ** 22, 7)
    assert binom_two_sided(5, 5) == 0.0625 and min_discordant() == 6
    assert [smallest_win(n) for n in (6, 10, 16, 20, 30, 50)] == [6, 9, 13, 15, 21, 33]
    assert smallest_win(5) is None


def test_planning_and_percentiles():
    assert reviews_for(0.05) == 385 and reviews_for(0.10, 0.239) == 70
    toni = [8.9, 6.0, 4.8, 5.1, 4.9, 4.7, 6.3, 5.3, 4.9, 5.1, 4.8, 5.3, 6.6, 5.3, 5.6, 5.0, 5.2]
    assert percentile(toni, 50) == 5.2 and percentile(toni, 95) == 8.9


# --- provenance -------------------------------------------------------------------
def test_the_planted_gap():
    g = planted_gap(DEFAULT)
    assert g["config hash before"] == g["config hash after"] == "fb59e88636ce"
    assert g["template sha before"] == template_sha() == "ff8949225404"
    assert g["template sha after"] != g["template sha before"]


def test_fingerprint_sees_what_the_hash_cannot(cases):
    sha = suite_sha(cases)
    a = fingerprint(builds()["shipped"], sha, "2026-11-10")
    b = fingerprint(Build("x", DEFAULT, ("current-only",), "scope-v2"), sha, "2026-11-10")
    assert a["config hash"] == b["config hash"]
    assert [k for k, _, _ in diff(a, b)] == ["index transforms", "scope guard"]
    assert corpus_sha() == "72d26a3beb3f"


# --- Module 12 --------------------------------------------------------------------
def test_interventions_one_at_a_time(arms):
    s = arms["shipped"]
    want = {"entitlements": (34, 3, 0), "current-only": (29, 0, 2), "structural": (39, 11, 3),
            "budget-900": (29, 0, 2), "cutoff-5": (30, 0, 1), "scope-v1": (45, 14, 0)}
    for n, (passed, fixed, broke) in want.items():
        a = arms["single:" + n]
        t = paired(s.passed(), a.passed())
        assert (sum(a.passed().values()), len(t.fixed), len(t.broke)) == (passed, fixed, broke), n
    assert tokens(arms["single:budget-900"]) == 40792
    assert arms["single:entitlements"].floor_failures("access") == []
    assert arms["single:current-only"].floor_failures("current") == []


def test_combinations_and_the_candidate(arms):
    s = arms["shipped"]
    rows = {n: (sum(arms[n].passed().values()), tokens(arms[n]), model_calls(arms[n])) for n, *_ in COMBINATIONS}
    assert rows == {"repaired": (41, 31505, 85), "+current-only": (38, 30481, 85),
                    "+scope-v1": (50, 24248, 68), "+scope-v2": (51, 25337, 71)}
    final = arms["+scope-v2"]
    assert all(not final.floor_failures(f) for f in FLOORS)
    t = paired(s.passed(), final.passed())
    assert (len(t.fixed), t.broke) == (21, ["a13"])
    assert sum(arms["candidate"].passed().values()) == 51


def test_a13_passed_by_luck(arms):
    v = arms["shipped"].verdicts["a13"]
    assert v.passed and v.refused
    assert any(c.dropped_at for c in v.trace.considered)


def test_guards():
    assert guard_rule("scope-v1", "What is my salary top-up during parental leave?") == "pay"
    assert guard_rule("scope-v2", "What is my salary top-up during parental leave?") == ""
    assert guard_rule("scope-v2", "List the salary bands for engineers") == "pay"
    assert guard_rule("scope-v2", "Can you approve my leave request?") == "action"
    assert guard_rule("scope-v2", "Is the contract or the handbook legally binding?") == "legal"


def test_parse_changes():
    cfg = parse_changes("top_k=8,prepend_heading=true,chunk_strategy=structural", DEFAULT)
    assert (cfg.top_k, cfg.prepend_heading, cfg.chunk_strategy) == (8, True, "structural")
    with pytest.raises(SystemExit):
        parse_changes("generation_model=x", DEFAULT)


# --- production simulation ------------------------------------------------------
@pytest.fixture(scope="module")
def month():
    b = builds()
    shipped, cand = P.Server(b["shipped"]), P.Server(b["repaired-current"])
    weeks = {w: P.serve_week(w, shipped) for w in (1, 2, 3)}
    return weeks, P.serve_week(4, cand, stale_check=P.stale_by_date), P.serve_week(4, cand)


def test_online_rates(month):
    weeks, w4, w4v1 = month
    assert [P.online_rates(weeks[w])["leak"] for w in (1, 2, 3)] == [116, 102, 92]
    assert [P.online_rates(weeks[w])["stale"] for w in (1, 2, 3)] == [72, 76, 80]
    assert P.online_rates(w4) == {"leak": 0, "stale": 372, "ungrounded": 0}
    assert P.online_rates(w4v1)["stale"] == 0
    d = P.decompose(0.080, 0.0, 0.372)
    assert (round(d["system"], 3), round(d["instrument"], 3)) == (-0.080, 0.372)


def test_feedback_and_review(month):
    from collections import Counter
    w3 = month[0][3]
    assert (sum(r.clicked for r in w3), sum(r.thumbs_down for r in w3)) == (101, 11)
    down = Counter(r.truth for r in w3 if r.thumbs_down)
    assert down["leak"] == 0 and down["missed refusal"] == 0
    assert sum(map(P.refusal_error, w3)) == 239
    res = {s.name: s for s in P.compare_strategies(w3)}
    assert round(res["random"].found, 1) == 12.0 and round(res["stratified"].found, 1) == 24.6
    assert round(res["thumbs-down"].estimate, 3) == 0.273


def test_day4_commands_from_a_clean_copy(tmp_path):
    """Every Day 4 command runs on a fresh copy and prints the slide figures."""
    import shutil
    dst = tmp_path / "w"
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns(".venv*", "__pycache__", "*.jsonl", "*.csv",
                                                             "day2_synthetic*.json", ".git"))
    want = {"day4": ["BLOCK: floor current: 4 failing: q2, a7, r2, s22", "template sha"],
            "compare": ["p = 0.0213", "P(candidate better)        99.4%"],
            "production": ["372/1000", "+37.2 points"],
            "capstone": ["+scope-v2         51/85", "1 regression(s) need a named decision: a13"]}
    for cmd, needles in want.items():
        out = subprocess.run([sys.executable, "run.py", cmd, "--offline"], cwd=dst, capture_output=True,
                             text=True, timeout=600)
        assert out.returncode == 0, out.stderr[-2000:]
        for n in needles:
            assert n in out.stdout, (cmd, n)
