"""The Day 4 release gate, written as tests that DeepEval's runner executes.

    python run.py ci        from the toolkit environment

One test per question in the Day 1 suite. The candidate configuration
(REPAIRED) answers every question live, on your model, when the tests
start, the way CI runs the system on each change. Nothing is read from an
earlier run and nothing is saved. A test fails when any metric on it fails.
The floors come from the charter and were agreed before anyone looked at a
score:

    entitlement (FP11)   Day 2 code check, must pass on every question
    current policy (FP8) Day 2 code check, must pass on every question
    faithfulness         DeepEval, at least HELIX_GATE_FAITHFULNESS (default 0.8),
                         judged by your model; skipped for refusals and offline

Two of the three metrics never call a model. A toolkit's test runner does not
care how a metric decides, and the floors that matter most here are lookups.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from toolkits import env

env.prepare()
pytest.importorskip("deepeval", reason="DeepEval runs in the toolkit environment (requirements-toolkits.txt)")

from deepeval import assert_test  # noqa: E402

from evals.day2_evaluators import check_leak, check_stale  # noqa: E402
from helix import DEFAULT, REPAIRED, answer, build_index  # noqa: E402
from toolkits.cases import cases_from_traces, to_deepeval  # noqa: E402
from toolkits.deepeval_metrics import code_check_metric, rag_metrics  # noqa: E402
from toolkits.judge import deepeval_judge  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "data" / "queries" / "day1_seed.json"
FAITHFULNESS_FLOOR = float(os.getenv("HELIX_GATE_FAITHFULNESS", "0.8"))
CANDIDATE = REPAIRED


def _answer_live() -> dict:
    """The candidate answers the suite now. Kept in memory for this run only."""
    suite = json.loads(SUITE.read_text(encoding="utf-8"))["queries"]
    index = build_index(CANDIDATE)
    traces = [answer(q["text"], index, CANDIDATE, user_office=q.get("user_office", "amsterdam"),
                     user_role=q.get("user_role", "employee")) for q in suite]
    return {c.id: c for c in cases_from_traces(traces, suite)}


CASES = _answer_live()


def gate_metrics(case):
    metrics = [
        code_check_metric("entitlement (FP11)", check_leak, CASES),
        code_check_metric("current policy (FP8)", check_stale, CASES),
    ]
    if not DEFAULT.offline and not case.refused and case.contexts:
        faith = rag_metrics(deepeval_judge(DEFAULT))["faithfulness"]
        faith.threshold = FAITHFULNESS_FLOOR
        metrics.append(faith)
    return metrics


@pytest.mark.parametrize("qid", sorted(CASES))
def test_release_gate(qid: str) -> None:
    case = CASES[qid]
    assert_test(to_deepeval(case), gate_metrics(case), run_async=False)
