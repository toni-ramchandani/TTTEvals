"""DeepEval metrics, built for the workshop's failure modes.

Every builder takes the judge from `toolkits.judge.deepeval_judge`, so no
metric falls back to DeepEval's default model (OpenAI). Metrics run in
synchronous mode: one call at a time, in order, which is slower and easy to
follow on a laptop running one local model.

`run` measures one metric on one test case and never raises: a judge that
returns unusable JSON becomes an undecided result with the reason attached,
the same way an unparseable verdict does in Day 3's own judge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .cases import Case


@dataclass
class Measured:
    metric: str
    score: float | None
    passed: bool | None
    reason: str


def run(metric, test_case) -> Measured:
    name = getattr(metric, "name", None) or metric.__class__.__name__
    try:
        metric.measure(test_case, _show_indicator=False)
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        return Measured(name, None, None, f"{type(exc).__name__}: {str(exc)[:160]}")
    score = metric.score
    return Measured(name, None if score is None else float(score), metric.success,
                    (metric.reason or "").strip().replace("\n", " ")[:220])


def _params(*names: str):
    from deepeval.test_case import SingleTurnParams
    return [getattr(SingleTurnParams, n) for n in names]


# ---------------------------------------------------------------------------
# Day 2, Module 06: a judged failure mode, two ways
# ---------------------------------------------------------------------------
def scope_geval(judge, criteria: str, steps: list[str] | None = None, name: str = "right scope"):
    """DeepEval's G-Eval on the same criteria, and the steps written in the same run.

    Through a custom model DeepEval cannot read token probabilities, so it
    uses the score the judge writes, not the probability-weighted one. The
    0.5 threshold is DeepEval's default. Nobody calibrated it for this task.
    """
    from deepeval.metrics import GEval

    return GEval(
        name=name,
        criteria=None if steps else criteria,
        evaluation_steps=steps or None,
        evaluation_params=_params("INPUT", "ACTUAL_OUTPUT"),
        model=judge, threshold=0.5, async_mode=False, verbose_mode=False,
    )


def scope_dag(judge):
    """The same decision as a tree of yes/no questions.

    A DAG metric asks narrow questions in a fixed order and maps each path to
    a score you chose in advance. The judge decides the yes/no answers; the
    tree decides the score. That makes the score repeatable in a way a
    single 0 to 10 rating is not.
    """
    from deepeval.metrics import DAGMetric
    from deepeval.metrics.dag import BinaryJudgementNode, DeepAcyclicGraph

    io = _params("INPUT", "ACTUAL_OUTPUT")
    root = BinaryJudgementNode(
        criteria=("Does the question ask for an entitlement or a figure (days, weeks, an amount, "
                  "a notice period) that can differ by office or by role?"),
        evaluation_params=io,
    )
    right_figure = BinaryJudgementNode(
        criteria=("Does the answer give the figure for the asker's own office, as named on the "
                  "Asker line, and for the role the question states, if it states one? A "
                  "company-wide figure, another office's figure or another role's figure is "
                  "a no. A refusal to answer is a yes."),
        evaluation_params=io,
    )
    root.add_verdict(False, score=10)            # nothing that varies by scope: pass
    root.add_verdict(True, then=right_figure)    # it varies: ask the second question
    right_figure.add_verdict(True, score=10)     # the asker's own figure
    right_figure.add_verdict(False, score=0)     # someone else's figure
    return DAGMetric(name="right scope (DAG)", dag=DeepAcyclicGraph(root_nodes=[root]),
                     model=judge, threshold=0.5, async_mode=False, verbose_mode=False)


# ---------------------------------------------------------------------------
# Day 3: RAG metrics
# ---------------------------------------------------------------------------
def rag_metrics(judge) -> dict[str, object]:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )

    common = dict(model=judge, threshold=0.5, async_mode=False, verbose_mode=False, include_reason=True)
    return {
        "faithfulness": FaithfulnessMetric(**common),
        "answer_relevancy": AnswerRelevancyMetric(**common),
        "contextual_precision": ContextualPrecisionMetric(**common),
        "contextual_recall": ContextualRecallMetric(**common),
    }


# ---------------------------------------------------------------------------
# Day 3, Module 09: agent metrics
# ---------------------------------------------------------------------------
def agent_metrics(judge) -> dict[str, object]:
    from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric

    return {
        # Compares tool names called with the tools the suite expects. No model
        # is asked; the judge is passed only so DeepEval does not build its own.
        "tool_correctness": ToolCorrectnessMetric(model=judge, threshold=0.5, async_mode=False,
                                                  verbose_mode=False),
        "task_completion": TaskCompletionMetric(model=judge, threshold=0.5, async_mode=False,
                                                verbose_mode=False),
    }


# ---------------------------------------------------------------------------
# Day 4: your own code checks inside DeepEval's test runner
# ---------------------------------------------------------------------------
def code_check_metric(name: str, check: Callable[[Case], object], cases: dict[str, Case]):
    """Wrap a Day 2 code check as a DeepEval metric, so CI runs both kinds.

    A toolkit's runner does not care whether a metric calls a model. The
    entitlement check is a dictionary lookup, and it is the floor the Day 4
    gate cares about most.
    """
    from deepeval.metrics import BaseMetric

    class CodeCheck(BaseMetric):
        def __init__(self) -> None:
            self.threshold = 1.0
            self.include_reason = True
            self.async_mode = False
            self.strict_mode = True
            self.evaluation_model = "none (code)"

        def measure(self, test_case, *args, **kwargs) -> float:
            case = cases[test_case.name]
            r = check(case)
            self.score = 1.0 if r.passed else 0.0
            self.reason = r.reason
            self.success = bool(r.passed)
            return self.score

        async def a_measure(self, test_case, *args, **kwargs) -> float:
            return self.measure(test_case)

        def is_successful(self) -> bool:
            return bool(self.success)

        @property
        def __name__(self) -> str:
            return name

    return CodeCheck()
