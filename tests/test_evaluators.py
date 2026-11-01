"""Day 2 Module 06 evaluators and the toolkit test cases, on the reference run.

No toolkit and no model is needed here: the code checks read the trace, the
workshop's own binary judge and G-Eval code are driven by a scripted model, and
the case adapter is plain Python. The toolkit integration itself is in test_toolkits.py, which
runs only where DeepEval and Ragas are installed.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import evals.day2_evaluators as E
from helix import DEFAULT, answer, build_index
from toolkits import env
from toolkits.cases import cases_from_traces, show

ROOT = Path(__file__).resolve().parents[1]
SUITE = json.loads((ROOT / "data" / "queries" / "day1_seed.json").read_text())["queries"]


_TRACES: dict[str, object] = {}


def cases_trace(case):
    return _TRACES[case.trace_id]


@pytest.fixture(scope="module")
def cases():
    index = build_index(DEFAULT)
    traces = [answer(q["text"], index, DEFAULT, user_office=q["user_office"], user_role=q["user_role"])
              for q in SUITE]
    _TRACES.update({t.trace_id: t for t in traces})
    return {c.id: c for c in cases_from_traces(traces, SUITE)}


def test_cases_carry_what_the_toolkits_and_the_checks_read(cases, tmp_path, monkeypatch):
    q4 = cases["q4"]
    assert q4.input_text.startswith("Asker: employee, Amsterdam office")
    assert "HB-COMP-BANDS-00" in q4.cited_ids
    assert {m["access_label"] for m in q4.context_meta} == {"hr-only", "all"}
    assert q4.should_refuse is True and cases["q1"].should_refuse is False
    assert all(isinstance(t, str) and t for t in q4.contexts)
    # what the model saw, without the trace-only heading path
    assert all(not t.startswith("Engineering bands, Amsterdam office |") for t in q4.contexts)
    monkeypatch.chdir(tmp_path)
    shown = show(cases["q1"], cases_trace(cases["q1"]))
    assert any("DeepEval input, Ragas user_input" in ln for ln in shown)
    assert any("dropped:" in ln and "context.assemble" in " ".join(shown) for ln in shown)
    assert not any(tmp_path.rglob("*")), "showing a case writes nothing"


def test_code_checks_find_the_seeded_failures(cases):
    stale = {k: E.check_stale(c).passed for k, c in cases.items()}
    leak = {k: E.check_leak(c).passed for k, c in cases.items()}
    refusal = {k: E.check_false_refusal(c).passed for k, c in cases.items()}
    missed = {k: E.check_missed_refusal(c).passed for k, c in cases.items()}
    assert [k for k, v in stale.items() if not v] == ["q1", "q2"]
    assert [k for k, v in leak.items() if not v] == ["q4"]
    assert [k for k, v in refusal.items() if not v] == ["q3"]
    assert [k for k, v in missed.items() if not v] == ["q4"]
    assert "superseded by HB-PAR-2024" in E.check_stale(cases["q1"]).reason


def test_rates_count_caught_missed_and_false_alarms():
    # positive = failure present; evaluator False = it failed the trace
    r = E.rates([False, True, False, None], [False, False, True, True])
    assert (r["caught"], r["failures"], r["missed"]) == (1, 2, 1)
    assert (r["passed"], r["passes"], r["false_alarm"], r["undecided"]) == (0, 2, 1, 1)
    assert r["tpr"] == 0.5 and r["tnr"] == 0.0 and r["agreement"] == 0.25
    none = E.rates([True, True], [True, True])
    assert none["tpr"] is None, "no failures to measure TPR on is a gap, not 100%"


def test_corrected_failure_rate_is_rogan_gladen():
    assert E.corrected_failure_rate(0.3, 0.8, 0.9) == pytest.approx(0.2 / 0.7)
    assert E.corrected_failure_rate(0.25, 1.0, 1.0) == 0.25, "a perfect evaluator needs no correction"
    assert E.corrected_failure_rate(0.5, 0.5, 0.5) is None, "a coin carries nothing to correct"
    assert E.corrected_failure_rate(0.01, 0.9, 0.8) == 0.0, "clipped to a rate"
    assert E.corrected_failure_rate(0.3, None, 0.9) is None
    # the two illustrative judges in the Day 2 course document and deck
    assert round(E.corrected_failure_rate(0.25, 0.70, 0.95), 2) == 0.31, "misses failures: under-reports"
    assert round(E.corrected_failure_rate(0.30, 0.95, 0.80), 2) == 0.13, "false alarms: over-reports"
    assert E.corrected_failure_rate(0.3, 0.5, 0.52) is None, "TPR + TNR - 1 below 0.05: too close to a coin"


def _pairs(caught: int, failures: int, passed: int, passes: int):
    """Labelled (evaluator, human) pairs with the given counts; False means fail."""
    human = [False] * failures + [True] * passes
    ev = [False] * caught + [True] * (failures - caught) + [True] * passed + [False] * (passes - passed)
    return ev, human


def test_the_corrected_rate_gets_a_bootstrap_interval():
    # Slide 6.5 continued: 20 failures and 20 passes labelled, as on a small test split.
    misses = _pairs(14, 20, 19, 20)            # TPR 0.70, TNR 0.95
    alarms = _pairs(19, 20, 16, 20)            # TPR 0.95, TNR 0.80
    assert (E.rates(*misses)["tpr"], E.rates(*misses)["tnr"]) == (0.70, 0.95)
    lo, hi = E.bootstrap_interval(*misses, 0.25)
    assert (round(lo, 2), round(hi, 2)) == (0.15, 0.48)
    lo, hi = E.bootstrap_interval(*alarms, 0.30)
    assert (round(lo, 2), round(hi, 2)) == (0.0, 0.29)
    assert E.bootstrap_interval(*misses, 0.25) == E.bootstrap_interval(*misses, 0.25), "seeded, repeatable"
    perfect = _pairs(5, 5, 8, 8)
    assert E.bootstrap_interval(*perfect, 0.2) == pytest.approx((0.2, 0.2)), "a perfect evaluator: nothing to be unsure of"
    assert E.bootstrap_interval(*_pairs(1, 2, 1, 2), 0.5, iterations=200) is not None
    assert E.bootstrap_interval(*_pairs(0, 4, 0, 4), 0.5) is None, "never better than a coin"
    wider = E.bootstrap_interval(*_pairs(7, 10, 19, 20), 0.25)
    assert wider[1] - wider[0] > hi - lo, "fewer labels, wider interval"


def test_our_binary_judge_has_examples_definitions_and_json(cases, monkeypatch):
    calls = []

    def scripted(messages, cfg, on_token=None, model=None, json_schema=None):
        calls.append((messages[-1]["content"], cfg.temperature, json_schema))
        return '{"reasoning": "Company-wide figure; the asker is in Amsterdam.", "answer": "Fail"}', {}

    monkeypatch.setattr(E, "complete", scripted)
    live = replace(DEFAULT, offline=False)
    examples = [E.Example("q2", cases["q2"], False, "company-wide 12 weeks"),
                E.Example("q5", cases["q5"], True, "The figure applies to everyone.")]
    r = E.binary_judge(cases["q1"], E.SCOPE_MODE, examples, live)
    assert r.passed is False and r.reason.startswith("Company-wide")
    prompt, temperature, schema = calls[-1]
    assert temperature == 0.0 and schema == E.JUDGE_SCHEMA
    assert prompt.index("Input 1:") < prompt.index("Input 2:") < prompt.index("Now evaluate this one.")
    assert "Pass: the figure applies" in prompt and "Fail: the answer gives another office's" in prompt
    assert prompt.rstrip().endswith("Your JSON evaluation:")
    assert prompt.count("Asker: employee, Amsterdam office") == 3, "the asker goes in, every time"
    assert E.parse_verdict('Sure. {"reasoning": "ok", "answer": "Pass"}') == (True, "ok")
    assert E.parse_verdict('"answer": "FAIL"')[0] is False
    assert E.parse_verdict("It could pass or fail.")[0] is None
    for mode in E.JUDGED:
        assert set(E.DEFINITIONS[mode]) == {"pass", "fail", "why_pass"}, mode


def test_our_geval_writes_its_steps_live_and_averages_samples(cases, tmp_path, monkeypatch):
    calls = []

    def scripted(messages, cfg, on_token=None, model=None, json_schema=None):
        prompt = messages[-1]["content"]
        calls.append((prompt, cfg.temperature))
        if "Write 3 to 5 short numbered steps" in prompt:
            return "1. Find the asker's office.\n2. Find the figure.\n3. Compare them.", {}
        return ("Score: 2" if len(calls) % 2 else "4"), {}

    monkeypatch.setattr(E, "complete", scripted)
    monkeypatch.chdir(tmp_path)
    live = replace(DEFAULT, offline=False)
    steps = E.geval_steps(E.SCOPE_CRITERIA, live)
    assert steps == ["Find the asker's office.", "Find the figure.", "Compare them."]
    E.geval_steps(E.SCOPE_CRITERIA, live)
    assert len(calls) == 2, "the steps are written live on every run, never loaded"
    calls.clear()
    r = E.geval_score(cases["q1"], E.SCOPE_CRITERIA, steps, live, samples=3)
    assert r.score == pytest.approx((2 + 4 + 2) / 3) and r.passed is False
    assert all(t == 1.0 for _, t in calls), "samples are drawn at temperature 1"
    assert "Asker: employee, Amsterdam office" in calls[-1][0]
    sweep = dict(E.threshold_sweep([r.score], [False]))
    assert sweep[2.5]["caught"] == 0 and sweep[3.5]["caught"] == 1 and sweep[3.5]["tpr"] == 1.0
    assert not any(tmp_path.rglob("*")), "nothing is saved"


def test_prepare_turns_off_telemetry_and_cloud_upload(monkeypatch):
    for key in env.SAFE_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CONFIDENT_API_KEY", "secret")
    notes = env.prepare()
    import os

    assert os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] == "1"
    assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"
    assert os.environ["DEEPEVAL_DISABLE_DOTENV"] == "1"
    assert os.environ["DEEPEVAL_FILE_SYSTEM"] == "READ_ONLY"
    assert "CONFIDENT_API_KEY" not in os.environ and notes


def test_nothing_uses_the_ragas_metrics_with_the_open_cve():
    """CVE-2026-6587 sits in Ragas's multi-modal image loader. Keep it unused."""
    offenders = []
    for p in [*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").rglob("*.py"), *(ROOT / "gates").rglob("*.py")]:
        text = p.read_text(encoding="utf-8")
        if "MultiModal" in text or "multi_modal" in text:
            if p.name != "ragas_metrics.py":
                offenders.append(str(p))
            else:
                assert "MultiModal" not in text.split('"""', 2)[2], "only the docstring may name it"
    assert not offenders


def test_the_toolkits_stay_optional():
    """No core module imports a toolkit at import time."""
    for p in (ROOT / "src").rglob("*.py"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.startswith(("import deepeval", "from deepeval", "import ragas", "from ragas")):
                pytest.fail(f"{p} imports a toolkit at module level")
