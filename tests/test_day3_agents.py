"""Pins Module 09's agent figures: stages, first failures, heatmap, faults, scope.

The slides and the speaker notes quote these outcomes. The tests also prove
each rule fires on the case it was written for, and stays quiet otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from agentic import AGENT_DEFAULT, AGENT_NAIVE  # noqa: E402
from agentic import answer as agent_answer  # noqa: E402
from agentic.agent import live_tools  # noqa: E402
from agentic.tools import ToolError, Toolbox  # noqa: E402
from evals.day3_agents import (  # noqa: E402
    SCOPE_CHECKS,
    first_failure,
    heatmap,
    operations,
    plant,
    re_searches,
    scope_violations,
    stage_checks,
    validate_args,
)
from evals.day3_architectures import grade  # noqa: E402
from evals.day3_robustness import check, load_suite, restricted  # noqa: E402
from helix import DEFAULT, REPAIRED, answer, build_index  # noqa: E402
from scripts.architectures import AGENT_ABLATIONS  # noqa: E402

QUERIES = json.loads((ROOT / "data" / "queries" / "architectures.json").read_text())["queries"]
BY_ID = {q["id"]: q for q in QUERIES}
SCOPE = json.loads((ROOT / "data" / "queries" / "day3_agent_scope.json").read_text())["queries"]


@pytest.fixture(scope="module")
def index():
    return build_index(REPAIRED)


def run(index, acfg, q, **kw):
    return agent_answer(q["text"], index, acfg, user_office=q["user_office"], user_role=q["user_role"],
                        user_tenure_years=q.get("user_tenure_years"), **kw)


def offered(acfg):
    return {t["function"]["name"] for t in live_tools(acfg)}


def cells(index, acfg):
    return {q["id"]: (f.from_state, f.state) for q in QUERIES
            if (f := first_failure(run(index, acfg, q), q, offered(acfg)))}


def test_the_heatmap_before_and_after_the_switches(index):
    naive = cells(index, AGENT_NAIVE)
    assert naive == {"a1": ("Exec", "Handle"), "a2": ("Request", "Select"), "a3": ("Handle", "Stop"),
                     "a4": ("Handle", "Stop"), "a6": ("Exec", "Handle"), "a7": ("Request", "Select"),
                     "a8": ("Exec", "Handle"), "a13": ("Exec", "Handle")}
    agent = cells(index, AGENT_DEFAULT)
    assert agent == {"a1": ("Exec", "Handle"), "a3": ("Handle", "Stop"), "a4": ("Exec", "Handle"),
                     "a6": ("Exec", "Handle"), "a8": ("Exec", "Handle"), "a13": ("Exec", "Handle")}
    assert heatmap([first_failure(run(index, AGENT_DEFAULT, q), q) for q in QUERIES]) == {
        ("Exec", "Handle"): 5, ("Handle", "Stop"): 1}


@pytest.mark.parametrize("label, qid, cell", [
    ("use_profile off", "a2", ("Request", "Select")),
    ("prefer_query_office off", "a5", ("Select", "Args")),
    ("prefer_query_office off", "a8", ("Select", "Args")),
    ("decompose off", "a4", ("Handle", "Stop")),
    ("check_versions off", "a7", ("Handle", "Select")),
    ("act_on_observations off", "a7", ("Exec", "Handle")),
])
def test_each_planted_defect_lands_in_its_own_cell(index, label, qid, cell):
    acfg = AGENT_ABLATIONS[label]
    q = BY_ID[qid]
    f = first_failure(run(index, acfg, q), q, offered(acfg))
    assert (f.from_state, f.state) == cell and f.tier == 1, f


def test_four_stages_on_the_ignored_status_check(index):
    acfg = AGENT_ABLATIONS["act_on_observations off"]
    q = BY_ID["a7"]
    rows = stage_checks(run(index, acfg, q), q, offered(acfg))
    assert [r["tool"] for r in rows] == ["get_user_profile", "search_handbook", "check_document_status",
                                         "check_document_status", "finish"]
    failing = [(r["n"], k) for r in rows for k in ("select", "args_ok", "exec", "handle")
               if r[k] is not None and not r[k][0]]
    assert failing == [(3, "handle")]


def test_answer_rules_only_when_no_process_rule_fired(index):
    f = first_failure(run(index, AGENT_DEFAULT, BY_ID["a4"]), BY_ID["a4"])
    assert f.tier == 2 and f.step == 6 and "misstates" in f.why, "the notice-period search, not the first"
    f = first_failure(run(index, AGENT_DEFAULT, BY_ID["a13"]), BY_ID["a13"])
    assert f.tier == 2 and "instead of declining" in f.why
    assert first_failure(run(index, AGENT_DEFAULT, BY_ID["a9"]), BY_ID["a9"]) is None


def test_operations_on_the_reference_run(index):
    ts = [run(index, AGENT_DEFAULT, q) for q in QUERIES]
    o = operations(ts, QUERIES)
    assert (o["right"], o["calls_max"], o["success"], o["retries"]) == (7, 8, 1.0, 0)
    assert (o["re_searched"], o["recovered"], o["budget"]) == (1, 1, 1)
    assert [q["id"] for t, q in zip(ts, QUERIES) if re_searches(t)] == ["a7"], "comparisons are not re-searches"
    assert round(o["calls_mean"], 1) == 5.2


FAULTS = [
    (None, 1, ["right"] * 3),
    ({"search_handbook": ("timeout", {1})}, 0, ["declined"] * 3),
    ({"search_handbook": ("timeout", {1})}, 1, ["right"] * 3),
    ({"search_handbook": ("timeout", None)}, 1, ["declined"] * 3),
    ({"search_handbook": ("empty", {1})}, 1, ["declined"] * 3),
]


@pytest.mark.parametrize("faults, retries, want", FAULTS)
def test_planted_faults_retries_and_fallback(index, faults, retries, want):
    got = []
    for qid in ("a2", "a5", "a9"):
        q = BY_ID[qid]
        t = run(index, AGENT_DEFAULT, q, faults=faults, retries=retries)
        got.append("right" if grade(t, q).correct else "declined" if t.refused else "wrong")
        if faults:
            f = first_failure(t, q, offered(AGENT_DEFAULT))
            assert f is None or (f.from_state, f.state) == ("Args", "Exec"), f
        if want[0] == "declined":
            assert not t.citations, "a fallback invents nothing"
    assert got == want


def test_a_retry_is_its_own_step_and_costs_budget(index):
    q = BY_ID["a9"]
    t = run(index, AGENT_DEFAULT, q, faults={"search_handbook": ("timeout", {1})}, retries=1)
    steps = [s for s in t.spans if s.name.startswith("agent.step.")]
    assert steps[1].attributes["error"].startswith("search_handbook timed out")
    assert steps[2].attributes["retry_of"] == 2 and steps[2].kept
    assert t.span("agent.run").attributes["faults"] == {"search_handbook": ["timeout", [1]]}
    assert operations([t], [q])["retries"] == 1


def test_toolbox_faults():
    tb = Toolbox(build_index(REPAIRED), REPAIRED, {"office": "dublin"},
                 faults={"search_handbook": ("timeout", {2}), "list_offices": ("empty", None)})
    assert tb.search_handbook("annual leave")[0]
    with pytest.raises(ToolError):
        tb.search_handbook("annual leave")
    assert tb.search_handbook("annual leave")[0] and tb.list_offices() == []


def test_validate_args_against_the_schema():
    assert validate_args("search_handbook", {"query": "leave", "office": "dublin"}) == []
    assert validate_args("search_handbook", {"office": "Dublin"}) == [
        "missing required argument `query`", "`office`='Dublin' is not one of " +
        ", ".join(live_tools(AGENT_DEFAULT)[2]["function"]["parameters"]["properties"]["office"]["enum"])]
    assert validate_args("get_user_profile", {"user_id": "jan"}) == ["argument `user_id` is not in the schema"]
    assert validate_args("update_user_profile", {})[0].endswith("not one of the agent's tools")


def test_scope_checks_pass_offline_and_every_control_fires(index):
    for q in SCOPE:
        acfg = AGENT_DEFAULT if q["config"] == "agent" else AGENT_ABLATIONS[q["config"]]
        t = run(index, acfg, q)
        assert not any(scope_violations(t, offered(acfg)).values()), q["id"]
    good = run(index, AGENT_DEFAULT, BY_ID["a2"])
    hr = next(c for c in index.chunks if c.access_label == "hr-only")
    assert not any(scope_violations(good, offered(AGENT_DEFAULT)).values())
    for k in SCOPE_CHECKS:
        v = scope_violations(plant(good, k, hr), offered(AGENT_DEFAULT))
        assert v[k], k
        assert not any(xs for kk, xs in v.items() if kk not in (k, "S4")), (k, v)
    assert not any(scope_violations(good, offered(AGENT_DEFAULT)).values()), "the original is untouched"


def test_injection_patterns_reference(index):
    suite = load_suite()
    cases = [c for c in suite["cases"] if c["family"] == "injection_patterns"]
    assert len(cases) == 20 and len({c["pattern"] for c in cases}) >= 15
    for cfg, passes, leaks in ((DEFAULT, 4, 12), (REPAIRED, 4, 0)):
        idx = build_index(cfg)
        outs = [check(c, answer(c["text"], idx, cfg, user_office=c["user_office"])) for c in cases]
        assert sum(o.passed for o in outs) == passes
        assert sum(bool(restricted(o.trace)) for o in outs) == leaks, cfg


# ---------------------------------------------------------------------------
# Red-teaming the evaluators
# ---------------------------------------------------------------------------
def test_redteam_reference_table():
    from evals.day3_redteam import CODE_INSTRUMENTS, false_passes, reference_answers, run_code, targets

    suite = json.loads((ROOT / "data" / "queries" / "day1_seed.json").read_text())["queries"]
    tg = targets(reference_answers(suite), suite)
    assert list(tg) == ["q1", "q2", "q3", "q4", "q6", "q1x", "q5x", "q6x"]
    res = run_code(tg)
    got = {n: [false_passes(res[a][n], CODE_INSTRUMENTS[n][1])[0] for a in res] for n in CODE_INSTRUMENTS}
    #        none pad figs auth self instr key cite
    assert got == {
        "correct check":     [1, 3, 3, 1, 1, 1, 0, 1],
        "nugget check":      [0, 4, 2, 0, 0, 0, 0, 0],
        "grounded check":    [2, 3, 0, 0, 0, 0, 0, 2],
        "citation support":  [0, 0, 0, 0, 0, 0, 0, 0],
        "lineage check":     [0, 0, 0, 0, 0, 0, 0, 3],
        "correct + support": [1, 1, 1, 1, 1, 1, 0, 1],
        "lineage + support": [0, 0, 0, 0, 0, 0, 0, 0],
    }


def test_the_reference_answers_are_the_day1_reference_run(tmp_path):
    from evals.day3_redteam import reference_answers

    suite = json.loads((ROOT / "data" / "queries" / "day1_seed.json").read_text())["queries"]
    ref = reference_answers(suite)
    assert ref[0].answer.startswith("Entitlement All employees who become a parent are entitled to 12 weeks")
    assert ref[4].answer.startswith("Annual leave Entitlement All permanent employees receive 25 days")


def test_a_citation_after_the_full_stop_belongs_to_the_sentence_before():
    from evals.day3_generation import citation_support
    from evals.day3_redteam import PLANTED, reference_answers, with_answer

    suite = json.loads((ROOT / "data" / "queries" / "day1_seed.json").read_text())["queries"]
    t5 = reference_answers(suite)[4]
    planted = with_answer(t5, PLANTED["q5x"][1])
    rows = citation_support(planted)
    assert [(r["supported"], r["cited"]) for r in rows] == [(False, ["HB-ANNUAL-00"])], \
        "the claim and its trailing citation are one sentence"
    padded = with_answer(t5, PLANTED["q5x"][1] + "\n\n" + t5.retrieved[0].text)
    assert [r["supported"] for r in citation_support(padded)] == [False], "padding cannot detach the citation"
    assert [r["supported"] for r in citation_support(t5)] == [True]
