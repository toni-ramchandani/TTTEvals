"""The Day 3 toolkit lens: the curriculum map, the toolkits' arithmetic, and the wiring.

These run in the core environment, with no toolkit installed. The toolkit
metrics themselves are exercised against the real libraries in
test_toolkits.py, which runs where DeepEval and Ragas are installed. Here a
stand-in records which metric each lens step asks for, on which test case,
so the wiring is pinned without a model.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from evals.day3_lens import CURRICULUM, NO_TOOLKIT, by_module, stages, toolkit_precision, verdict_string
from evals.day3_retrieval import judgments
from helix import DEFAULT, answer, build_index
from helix.ingest import chunk_corpus

ROOT = Path(__file__).resolve().parents[1]
SUITE = json.loads((ROOT / "data" / "queries" / "day1_seed.json").read_text())["queries"]
OFF = replace(DEFAULT, offline=True)

#: The Day 3 curriculum, as the course outline words it.
CURRICULUM_TEXT = {
    "07": ["Precision@k", "Recall@k", "F1@k", "MRR", "MAP", "nDCG", "Contextual metrics", "reranking"],
    "08": ["Correctness", "Relevance", "Completeness", "Faithfulness", "Context utilization", "Citations",
           "Consistency", "stale evidence"],
    "09": ["Adversarial", "red-teaming", "GraphRAG", "Tool routing", "Retries, fallback, recovery",
           "Trajectories", "Task completion"],
}


def test_every_curriculum_item_has_an_instrument_and_a_toolkit_answer():
    for module, words in CURRICULUM_TEXT.items():
        text = " ".join(i.item for i in by_module(module))
        for w in words:
            assert w.lower() in text.lower(), f"Module {module} map misses {w}"
    for it in CURRICULUM:
        assert it.ours and it.where, it.item
        if it.deepeval == it.geval == it.ragas == "-":
            assert it.item in NO_TOOLKIT, f"{it.item}: no toolkit and no reason given"
    # Each toolkit is used somewhere in each module except where none exists
    for module in ("08", "09"):
        items = by_module(module)
        assert any(i.deepeval != "-" for i in items) and any(i.geval != "-" for i in items)
        assert any(i.ragas != "-" for i in items)


def _ragas_ap(verdicts):
    """Ragas 0.4.3 ContextPrecisionWithReference._calculate_average_precision, as written."""
    denominator = sum(verdicts) + 1e-10
    numerator = sum([(sum(verdicts[: i + 1]) / (i + 1)) * verdicts[i] for i in range(len(verdicts))])
    return numerator / denominator


def _deepeval_ap(verdicts):
    """DeepEval 4.2.8 ContextualPrecisionMetric._calculate_score, as written."""
    if not verdicts:
        return 0
    s, n = 0.0, 0
    for k, rel in enumerate(verdicts, start=1):
        if rel:
            n += 1
            s += (n / k) * rel
    return 0 if n == 0 else s / n


def test_toolkit_precision_is_both_toolkits_arithmetic():
    rng = random.Random(7)
    lists = [[0, 0, 1, 0, 0], [1, 0, 1], [0, 0], [], [1], [0, 1, 1, 0, 1]]
    lists += [[rng.randint(0, 1) for _ in range(rng.randint(1, 8))] for _ in range(50)]
    for v in lists:
        assert toolkit_precision(v) == pytest.approx(_ragas_ap(v), abs=1e-9)
        assert toolkit_precision(v) == pytest.approx(_deepeval_ap(v))
    # It divides by the relevant chunks IN the list: a missed chunk cannot lower it
    assert toolkit_precision([1, 0, 0]) == 1.0


@pytest.fixture(scope="module")
def q1():
    index = build_index(OFF)
    q = SUITE[0]
    t = answer(q["text"], index, OFF, user_office=q["user_office"], user_role=q["user_role"])
    return q, t


def test_q1_stages_with_peoples_verdicts(q1):
    q, t = q1
    st = stages(t, judgments(chunk_corpus(OFF), q["expected_evidence"]))
    got = [(s.name, len(s.chunks), verdict_string(s.verdicts), round(s.precision, 2), s.recall) for s in st]
    assert got == [("retrieved", 5, "no no YES no no", 0.33, 1.0),
                   ("reranked", 3, "no no YES", 0.33, 1.0),
                   ("reached the model", 2, "no no", 0.0, 0.0)]


class FakeKits:
    """Stands in for toolkits.kits.Kits: records each request, returns a score."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object]] = []
        self.deepeval = self.ragas = {"answer_relevancy": object()}
        self.stats = None

    def _m(self, kind, name, case, score=0.5):
        from toolkits.deepeval_metrics import Measured
        self.calls.append((kind, name, case))
        return Measured(name, score, score >= 0.5, "")

    def r(self, name, case):
        return self._m("ragas", name, case)

    def d(self, name, case):
        return self._m("deepeval", name, case)

    def d_keep(self, name, case):
        return self._m("deepeval", name, case), None

    def g(self, criterion, case):
        return self._m("geval", criterion, case)

    def geval_steps(self, criterion):
        return []

    def missing(self):
        return []


@pytest.fixture
def live(monkeypatch, tmp_path):
    """The lens as it runs live, with the stand-in instead of the toolkits."""
    import scripts.lens as lens
    from tracing import write_jsonl

    index = build_index(OFF)
    traces = [answer(q["text"], index, OFF, user_office=q["user_office"], user_role=q["user_role"])
              for q in SUITE]
    path = tmp_path / "day1.jsonl"
    write_jsonl(traces, path)
    monkeypatch.setattr(lens, "TRACES", path)
    monkeypatch.setattr(lens, "DEFAULT", replace(OFF, offline=False))
    return lens


def test_module_07_scores_every_stage_list(live, capsys):
    kits = FakeKits()
    live.module_07(["q1"], kits)
    lists = [len(c.contexts) for kind, name, c in kits.calls if name == "context_precision"]
    assert lists == [5, 3, 2], "retrieved, reranked, reached the model"
    asked = {(kind, name) for kind, name, _ in kits.calls}
    assert asked == {("ragas", "context_precision"), ("ragas", "context_recall"),
                     ("deepeval", "contextual_precision"), ("deepeval", "contextual_recall"),
                     ("deepeval", "contextual_relevancy")}
    out = capsys.readouterr().out
    assert "retrieved             0.33     1.00     0.50     0.50   0.50   0.50    0.50" in out


def test_module_08_asks_each_criterion_of_the_right_case(live, capsys):
    kits = FakeKits()
    live.module_08(["q1"], kits)
    by = {(k, n): c for k, n, c in kits.calls}
    assert set(by) == {("geval", "correctness"), ("ragas", "factual_correctness"),
                       ("deepeval", "answer_relevancy"), ("ragas", "answer_relevancy"),
                       ("geval", "completeness"), ("ragas", "factual_recall"),
                       ("deepeval", "faithfulness"), ("ragas", "faithfulness"),
                       ("ragas", "context_utilization"), ("geval", "citation support"),
                       ("deepeval", "citation_faithfulness")}
    assert by[("geval", "completeness")].reference == "Vital facts: 26 weeks; 130 working days"
    assert by[("ragas", "factual_recall")].reference == by[("geval", "completeness")].reference
    numbered = by[("deepeval", "citation_faithfulness")].answer
    assert "[HB-" not in numbered and ("[1]" in numbered or "[2]" in numbered or "[?]" in numbered)
    assert by[("geval", "correctness")].reference.startswith("130 working days / 26 weeks")
    assert all(c.startswith("[HB-") for c in by[("geval", "citation support")].contexts)
    out = capsys.readouterr().out
    assert "completeness         0 of 2 nuggets" in out
    assert "context utilization  0/2 used, 2 not in ctx" in out
    assert "-> faithful to a wrong or superseded document: fix the corpus" in out


def test_module_09_reads_attacks_graphs_and_agent_runs(live, capsys):
    kits = FakeKits()
    live.module_09(["attacks", "graph", "agents"], kits)
    asked = [(k, n) for k, n, _ in kits.calls]
    assert asked.count(("geval", "correctness")) == 3 + 4, "three attack conditions, four graph runs"
    for metric in ("tool_correctness", "tool_correctness_ordered", "argument_correctness", "task_completion"):
        assert asked.count(("deepeval", metric)) == 4
    for metric in ("tool_call_f1", "tool_call_accuracy", "goal_accuracy"):
        assert asked.count(("ragas", metric)) == 4
    agent_cases = [c for k, n, c in kits.calls if n == "goal_accuracy"]
    empty = agent_cases[-1]
    assert any(t["result"] == "(empty result)" for t in empty.tools_called), "the silent fault is in the case"
    out = capsys.readouterr().out
    assert "a7 stale cited  FAIL Exec->Handle" in out
    assert "a2 empty result FAIL Args->Exec" in out
    assert "a11  graph     FAIL GR2" in out


def test_the_lens_runs_offline_without_toolkits(capsys):
    import scripts.lens as lens

    assert lens.main(["--module", "09", "--part", "agents"]) == 0
    out = capsys.readouterr().out
    assert "Exact order wants one reference path" in out
    assert lens.main(["--module", "09", "--part", "nonsense"]) == 2
