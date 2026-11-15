"""DeepEval and Ragas, driven through the workshop's judge, with no network.

These run only where the toolkits are installed (the toolkit environment);
elsewhere they are skipped. A scripted model stands in for the judge: it
reads the JSON schema each toolkit asks for and answers in that shape. That
exercises the real toolkit code (its prompts, its parsing, its scoring) and
the path every call takes: toolkit -> our wrapper -> helix.llm.complete.
Any attempt to open a network connection fails the test.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from toolkits import env

env.prepare()
deepeval = pytest.importorskip("deepeval")
ragas = pytest.importorskip("ragas")

import toolkits.judge as J  # noqa: E402
from helix import DEFAULT, answer, build_index  # noqa: E402
from helix import llm  # noqa: E402
from toolkits import deepeval_metrics as D  # noqa: E402
from toolkits import ragas_metrics as R  # noqa: E402
from toolkits.cases import cases_from_traces, to_deepeval  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUITE = json.loads((ROOT / "data" / "queries" / "day1_seed.json").read_text())["queries"]
LIVE = replace(DEFAULT, offline=False)


def instance(schema: dict, defs: dict | None = None, overrides: dict | None = None):
    """The smallest value that satisfies a JSON schema, with chosen field values."""
    defs = defs if defs is not None else schema.get("$defs", {})
    overrides = overrides or {}
    if "$ref" in schema:
        return instance(defs[schema["$ref"].split("/")[-1]], defs, overrides)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            return instance(next(o for o in schema[key] if o.get("type") != "null"), defs, overrides)
    if "allOf" in schema:
        return instance(schema["allOf"][0], defs, overrides)
    if "enum" in schema:
        return schema["enum"][0]
    t = schema.get("type")
    if t == "object" or "properties" in schema:
        return {k: overrides.get(k, instance(v, defs, overrides)) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return [instance(schema.get("items", {"type": "string"}), defs, overrides)]
    return {"string": "yes", "integer": max(schema.get("minimum", 1), 1), "number": 1.0,
            "boolean": True}.get(t)


@pytest.fixture
def scripted(monkeypatch):
    """Replace the model with a schema-following script; forbid the network."""
    def no_network(*a, **k):
        raise AssertionError("a toolkit tried to reach the network")

    monkeypatch.setattr(llm, "_open", no_network)
    log: list[dict] = []
    rules: list = []

    def complete(messages, cfg, on_token=None, model=None, json_schema=None):
        prompt = messages[-1]["content"]
        log.append({"prompt": prompt, "schema": json_schema, "model": model})
        for rule in rules:
            out = rule(prompt, json_schema)
            if out is not None:
                return out, {}
        if json_schema is None:
            return "4", {}
        return json.dumps(instance(json_schema)), {}

    monkeypatch.setattr(J, "complete", complete)
    return log, rules


@pytest.fixture(scope="module")
def cases():
    index = build_index(DEFAULT)
    traces = [answer(q["text"], index, DEFAULT, user_office=q["user_office"], user_role=q["user_role"])
              for q in SUITE]
    return {c.id: c for c in cases_from_traces(traces, SUITE)}


def test_a_bad_json_reply_gets_one_second_chance(scripted):
    log, rules = scripted
    from pydantic import BaseModel

    class Verdict(BaseModel):
        verdict: str

    replies = iter(["Sure! The verdict is yes.", '```json\n{"verdict": "no"}\n```'])
    rules.append(lambda p, s: next(replies))
    stats = J.JudgeStats()
    assert J.structured("judge this", Verdict, LIVE, stats).verdict == "no"
    assert (stats.calls, stats.repaired, stats.failed) == (2, 1, 0)
    assert "not valid JSON" in log[-1]["prompt"]

    rules[:] = [lambda p, s: "still not json"]
    with pytest.raises(J.JudgeParseError):
        J.structured("judge this", Verdict, LIVE, stats)
    assert stats.failed == 1


def test_every_deepeval_metric_runs_on_the_workshop_judge(scripted, cases):
    log, _ = scripted
    stats = J.JudgeStats()
    judge = J.deepeval_judge(LIVE, stats)
    tc = to_deepeval(cases["q1"])
    metrics = [D.scope_geval(judge, "criteria", ["step one", "step two"]), D.scope_dag(judge),
               *D.rag_metrics(judge).values(), *(D.geval_metric(judge, n) for n in D.GEVAL)]
    for m in metrics:
        r = D.run(m, tc)
        assert r.score is not None, f"{r.metric}: {r.reason}"
    agent = replace(cases["q1"], tools_called=[{"name": "search_handbook", "args": {"query": "leave"}}],
                    expected_tools=["get_user_profile", "search_handbook"])
    tools = D.run(D.agent_metrics(judge)["tool_correctness"], to_deepeval(agent))
    assert tools.score == pytest.approx(0.5), "one of the two expected tools, by name"
    ordered = D.run(D.agent_metrics(judge)["tool_correctness_ordered"], to_deepeval(agent))
    assert ordered.score == pytest.approx(0.5), "a subsequence of length one, of two expected"
    assert D.run(D.agent_metrics(judge)["task_completion"], to_deepeval(agent)).score is not None
    assert D.run(D.agent_metrics(judge)["argument_correctness"], to_deepeval(agent)).score is not None
    assert stats.calls == len(log) and stats.failed == 0
    assert all(entry["model"] == LIVE.generation_model for entry in log), "the judge is the workshop's model"


def test_the_dag_scores_the_path_the_judge_takes(scripted, cases):
    _, rules = scripted

    def verdicts(prompt, schema):
        if schema is None or "verdict" not in json.dumps(schema):
            return None
        figure_question = "Does the question ask for an entitlement" in prompt
        return json.dumps(instance(schema, overrides={"verdict": figure_question, "reason": "scripted"}))

    rules.append(verdicts)
    m = D.run(D.scope_dag(J.deepeval_judge(LIVE)), to_deepeval(cases["q1"]))
    assert m.score == 0.0 and m.passed is False, "figure question, wrong figure: the tree says 0"


def test_code_checks_run_inside_deepevals_test_runner(scripted, cases, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from deepeval import assert_test

    from evals.day2_evaluators import check_leak

    metric = lambda: D.code_check_metric("entitlement (FP11)", check_leak, cases)  # noqa: E731
    assert_test(to_deepeval(cases["q5"]), [metric()], run_async=False)
    with pytest.raises(AssertionError):
        assert_test(to_deepeval(cases["q4"]), [metric()], run_async=False)


def test_every_ragas_metric_runs_on_the_workshop_judge(scripted, cases):
    stats = J.JudgeStats()
    built = R.build(J.ragas_llm(LIVE, stats), J.ragas_embeddings(DEFAULT))
    assert set(built) == set(R.METRICS)
    agent = replace(cases["q1"], tools_called=[{"name": "search_handbook", "args": {"query": "leave"}}],
                    expected_tools=["get_user_profile", "search_handbook"])
    on_agent = ("tool_call_f1", "tool_call_accuracy", "goal_accuracy")
    for name, metric in built.items():
        r = R.score(name, metric, agent if name in on_agent else cases["q1"])
        assert r.score is not None, f"{name}: {r.reason}"
    assert R.score("tool_call_f1", built["tool_call_f1"], agent).score == pytest.approx(0.6667)
    # One call where the reference has two, in another order: the sequence does not align
    assert R.score("tool_call_accuracy", built["tool_call_accuracy"], agent).score == 0.0
    in_order = replace(agent, tools_called=[{"name": "get_user_profile", "args": {}},
                                            {"name": "search_handbook", "args": {"query": "leave"}}])
    assert R.score("tool_call_accuracy", built["tool_call_accuracy"], in_order).score == 1.0
    assert stats.failed == 0


# DeepEval's own evaluate() asks asyncio for an event loop the old way (its code, not ours).
@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_the_toolkits_write_nothing_to_disk(scripted, cases, tmp_path, monkeypatch):
    """Every score is computed live and printed. No toolkit file, cache or id appears."""
    home, work = tmp_path / "home", tmp_path / "work"
    home.mkdir()
    work.mkdir()
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(home))
    for var in ("XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(var, str(home / var.lower()))
    monkeypatch.chdir(work)

    from deepeval import assert_test, evaluate

    from evals.day2_evaluators import check_leak

    judge = J.deepeval_judge(LIVE)
    tc = to_deepeval(cases["q1"])
    for m in [D.scope_geval(judge, "criteria", ["step"]), D.scope_dag(judge), *D.rag_metrics(judge).values()]:
        D.run(m, tc)
    check = lambda: D.code_check_metric("entitlement (FP11)", check_leak, cases)  # noqa: E731
    assert_test(to_deepeval(cases["q5"]), [check()], run_async=False)
    evaluate([to_deepeval(cases["q5"])], [check()])  # by default this saves runs and a result cache
    built = R.build(J.ragas_llm(LIVE), J.ragas_embeddings(DEFAULT))
    for name, metric in built.items():
        R.score(name, metric, cases["q1"])
    import logging

    logging.getLogger("ragas").warning("a Ragas log line, which reads a user id")
    written = [str(p) for p in tmp_path.rglob("*") if p.is_file()]
    assert written == [], f"a toolkit wrote to disk: {written}"


# ---------------------------------------------------------------------------
# Day 3: conversations, noise, and the open box
# ---------------------------------------------------------------------------
def test_conversation_metrics_run_on_the_workshop_judge(scripted):
    from evals.day3_conversations import load_sessions
    from helix import REPAIRED
    from helix.conversation import run as converse
    from toolkits.cases import to_deepeval_conversation, to_ragas_conversation

    s = next(x for x in load_sessions() if x["id"] == "c5")
    traces = converse([t["text"] for t in s["turns"]], build_index(REPAIRED), REPAIRED, "condense",
                      s["user_office"], s["user_role"], "c5")
    stats = J.JudgeStats()
    tc = to_deepeval_conversation(traces, s)
    assert len(tc.turns) == 2 * len(traces) and tc.turns[1].retrieval_context
    for name, metric in D.conversation_metrics(J.deepeval_judge(LIVE, stats)).items():
        r = D.run(metric, tc)
        assert r.score is not None, f"{name}: {r.reason}"
    built = R.build_conversation(J.ragas_llm(LIVE, stats))
    r = R.score_conversation("goal_accuracy", built["goal_accuracy"], to_ragas_conversation(traces))
    assert r.score is not None, r.reason
    assert stats.failed == 0


def test_noise_sensitivity_runs_on_the_workshop_judge(scripted):
    stats = J.JudgeStats()
    m = R.score_noise(R.build_noise(J.ragas_llm(LIVE, stats)), "How many days of annual leave?",
                      "25 days.", "25 days of paid annual leave", ["Staff cafeteria opens at 08:00.",
                                                                   "25 days of paid annual leave."])
    assert m.score is not None, m.reason
    assert stats.failed == 0


def test_show_opens_the_box_on_one_trace(scripted, cases, tmp_path, monkeypatch, capsys):
    import importlib.util

    from tracing import write_jsonl

    index = build_index(DEFAULT)
    traces = [answer(q["text"], index, DEFAULT, user_office=q["user_office"], user_role=q["user_role"])
              for q in SUITE]
    path = tmp_path / "day1.jsonl"
    write_jsonl(traces, path)
    spec = importlib.util.spec_from_file_location("toolkits_script", ROOT / "scripts" / "toolkits.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "TRACES", path)
    monkeypatch.setattr(mod, "DEFAULT", LIVE)
    assert mod.show("q1") == 0
    out = capsys.readouterr().out
    assert "STEP 2  DeepEval faithfulness" in out and "claims (from the answer)" in out
    assert "STEP 3  Ragas faithfulness" in out and "statements:" in out and "verdicts:" in out


def test_the_day3_lens_runs_every_toolkit_metric_on_the_workshop_judge(scripted, tmp_path, monkeypatch, capsys):
    """`toolkits --module 07|08|09` against the real libraries, judged by the script."""
    import scripts.lens as lens
    from toolkits.kits import Kits
    from tracing import write_jsonl

    index = build_index(DEFAULT)
    traces = [answer(q["text"], index, DEFAULT, user_office=q["user_office"], user_role=q["user_role"])
              for q in SUITE]
    path = tmp_path / "day1.jsonl"
    write_jsonl(traces, path)
    monkeypatch.setattr(lens, "TRACES", path)
    monkeypatch.setattr(lens, "DEFAULT", LIVE)
    kits = Kits(LIVE)
    lens.module_07(["q1"], kits)
    lens.module_08(["q1"], kits)
    lens.module_09(list(lens.PARTS), kits)
    out = capsys.readouterr().out
    retrieved = next(ln for ln in out.splitlines() if ln.strip().startswith("retrieved "))
    assert "-" not in retrieved.split()[3:], f"a contextual metric did not score: {retrieved}"
    row = next(ln for ln in out.splitlines() if ln.strip().startswith("correctness "))
    assert row.split()[-2:] != ["-", "-"], f"G-Eval or Ragas correctness did not score: {row}"
    agents = next(ln for ln in out.splitlines() if ln.strip().startswith("a2 empty result"))
    assert agents.split()[-7:].count("-") == 0, f"an agent metric did not score: {agents}"
    # The rule-based tool metrics need no judge, so their values are fixed. The run that
    # failed (an empty search result) called exactly the expected tools in order: every
    # tool-name score is perfect. The right run called more tools: DeepEval's order (a
    # subsequence) still passes it, Ragas's exact order gives 0.
    tools, order, _, _, f1, exact, _ = agents.split()[-7:]
    assert (tools, order, f1, exact) == ("1.00", "1.00", "1.00", "1.00")
    right = next(ln for ln in out.splitlines() if ln.strip().startswith("a2 no fault")).split()[-7:]
    assert (right[0], right[1], right[4], right[5]) == ("1.00", "1.00", "0.80", "0.00")
    assert kits.geval_steps("correctness"), "G-Eval wrote its steps once and kept them"
    assert kits.stats.failed == 0
