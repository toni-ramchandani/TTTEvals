"""The Day 4 release gate, written as tests that DeepEval's runner executes.

    python run.py ci                         from the toolkit environment
    python run.py ci --candidate candidate   another build from the Day 4 catalog

The same three kinds of rule as `python run.py day4`, so the script and CI
can never disagree:

    floors    one test per suite question, three code metrics (access,
              current policy, groundedness by citation support) plus, live,
              DeepEval faithfulness at HELIX_GATE_FAITHFULNESS (default 0.8),
              judged by your model, on answered questions
    ratchet   one test: no question that passes on the shipped build fails
              on the candidate, unless accepted by name in
              data/queries/day4_suite_changes.json
    budgets   one test: model tokens, p95 answer time (live), failed calls

Both builds answer the suite now, when the tests start. Nothing is read from
an earlier run and nothing is saved. HELIX_GATE_CANDIDATE picks the
candidate, HELIX_GATE_SOURCE limits the suite (for example day1,architectures).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from toolkits import env

env.prepare()
pytest.importorskip("deepeval", reason="DeepEval runs in the toolkit environment (requirements-toolkits.txt)")

from deepeval import assert_test  # noqa: E402

from evals.day4_gate import decide  # noqa: E402
from evals.day4_report import builds, run  # noqa: E402
from evals.day4_suite import FLOOR_NAMES, FLOORS, build_suite, select  # noqa: E402
from helix import DEFAULT  # noqa: E402
from toolkits.cases import case_from_trace, to_deepeval  # noqa: E402
from toolkits.deepeval_metrics import code_check_metric, rag_metrics  # noqa: E402
from toolkits.judge import deepeval_judge  # noqa: E402

FAITHFULNESS_FLOOR = float(os.getenv("HELIX_GATE_FAITHFULNESS", "0.8"))
CANDIDATE = os.getenv("HELIX_GATE_CANDIDATE", "repaired")
SOURCE = os.getenv("HELIX_GATE_SOURCE", "")

_CASES = select(build_suite(), source=SOURCE)
_LIVE = not DEFAULT.offline
SHIPPED = run(builds()["shipped"], _CASES, _LIVE)
CAND = run(builds()[CANDIDATE], _CASES, _LIVE)
VERDICTS = CAND.verdicts
CASES = {k: case_from_trace(v.trace, {"id": k, "should_refuse": v.case.should_refuse})
         for k, v in VERDICTS.items() if v.trace is not None}


def _floor_check(floor: str):
    def check(case):
        v = VERDICTS[case.id]
        ok = v.floors.get(floor, False)
        return SimpleNamespace(passed=ok, reason="held" if ok else v.why.get(floor, "failed"))
    return check


def gate_metrics(case):
    metrics = [code_check_metric(FLOOR_NAMES[f], _floor_check(f), CASES) for f in FLOORS]
    if _LIVE and not case.refused and case.contexts:
        faith = rag_metrics(deepeval_judge(DEFAULT))["faithfulness"]
        faith.threshold = FAITHFULNESS_FLOOR
        metrics.append(faith)
    return metrics


@pytest.mark.parametrize("qid", sorted(VERDICTS))
def test_floor(qid: str) -> None:
    v = VERDICTS[qid]
    assert not v.error, f"the model call failed: {v.error}"
    case = CASES[qid]
    assert_test(to_deepeval(case), gate_metrics(case), run_async=False)


def test_ratchet() -> None:
    rule = next(r for r in decide(SHIPPED, CAND).rules if r.kind == "ratchet")
    assert rule.passed, rule.detail


def test_budgets() -> None:
    bad = [f"{r.name}: {r.detail}" for r in decide(SHIPPED, CAND).rules
           if r.kind == "budget" and r.applies and not r.passed]
    assert not bad, "; ".join(bad)
