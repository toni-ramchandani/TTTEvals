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
    """The RAG metrics, each judged by your model.

    contextual precision   average precision over the retrieval context, with
                           the judge deciding which chunks help the expected
                           output. It divides by the relevant chunks IN the
                           list, so it cannot see a chunk retrieval missed.
    contextual recall      share of the expected output's sentences the
                           retrieval context can be attributed to
    contextual relevancy   share of the retrieval context's statements that
                           are relevant to the input (no expected output)
    """
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )

    from deepeval.metrics.community import CitationFaithfulnessMetric

    common = dict(model=judge, threshold=0.5, async_mode=False, verbose_mode=False, include_reason=True)
    return {
        "faithfulness": FaithfulnessMetric(**common),
        "answer_relevancy": AnswerRelevancyMetric(**common),
        "contextual_precision": ContextualPrecisionMetric(**common),
        "contextual_recall": ContextualRecallMetric(**common),
        "contextual_relevancy": ContextualRelevancyMetric(**common),
        # A community metric: does each [N] marker point to the passage that
        # supports its claim? One verdict per answer, 1.0 or 0.0, and it
        # passes only at 1.0. It reads [N] markers, so the answer's chunk-id
        # citations are renumbered first (cases.numbered_citations).
        "citation_faithfulness": CitationFaithfulnessMetric(model=judge, threshold=1.0, async_mode=False,
                                                            verbose_mode=False, include_reason=True),
    }


# ---------------------------------------------------------------------------
# Day 3, Module 08: G-Eval for the criteria no built-in metric covers as asked
# ---------------------------------------------------------------------------
#: Each entry: (criteria, the test-case fields the judge reads). The criteria
#: are written from the Day 2 rubric. G-Eval turns them into steps on the
#: first case and reuses the steps for the rest of the run.
GEVAL = {
    "correctness": (
        "Compare the actual output with the expected output. The answer is correct only if every "
        "figure, unit, condition and office it states matches the expected output for this asker. "
        "A figure for another office, another role or an older policy is wrong. A refusal is wrong "
        "when the expected output gives an answer. Wording and order do not matter.",
        ("INPUT", "ACTUAL_OUTPUT", "EXPECTED_OUTPUT")),
    "completeness": (
        "The expected output lists the vital facts a complete answer must state. Check each vital "
        "fact: is it stated in the actual output, in any wording? Score by the share of vital facts "
        "stated. Facts that are not in the list neither raise nor lower the score.",
        ("INPUT", "ACTUAL_OUTPUT", "EXPECTED_OUTPUT")),
    # DeepEval has a citation metric too (community CitationFaithfulnessMetric,
    # one verdict per answer, [N] markers). This one reads chunk ids and scores
    # the share of cited sentences, as the workshop's own check does.
    "citation support": (
        "Each retrieval context starts with its chunk id in square brackets. For every sentence of "
        "the actual output that cites a chunk id, check that the chunk with that id states what the "
        "sentence says. A sentence is unsupported if its cited chunk does not say it, even when "
        "another chunk does. Score by the share of cited sentences their own chunk supports.",
        ("ACTUAL_OUTPUT", "RETRIEVAL_CONTEXT")),
}


def geval_metric(judge, name: str):
    """DeepEval's G-Eval on one of the criteria above, judged by your model.

    Through a custom model DeepEval cannot read token probabilities, so the
    score is the one the judge writes, normalised to 0..1. The 0.5 threshold
    is DeepEval's default, not a calibration.
    """
    from deepeval.metrics import GEval

    criteria, params = GEVAL[name]
    return GEval(name=f"G-Eval {name}", criteria=criteria, evaluation_params=_params(*params),
                 model=judge, threshold=0.5, async_mode=False, verbose_mode=False)


# ---------------------------------------------------------------------------
# Day 3, Module 09: agent metrics
# ---------------------------------------------------------------------------
def agent_metrics(judge) -> dict[str, object]:
    from deepeval.metrics import ArgumentCorrectnessMetric, TaskCompletionMetric, ToolCorrectnessMetric

    return {
        # Compares tool names called with the tools the suite expects. No model
        # is asked; the judge is passed only so DeepEval does not build its own.
        # Extra calls do not lower it: expected tools found / expected tools.
        "tool_correctness": ToolCorrectnessMetric(model=judge, threshold=0.5, async_mode=False,
                                                  verbose_mode=False),
        # The same, in order: the longest common subsequence of expected and
        # called names, divided by the expected. Extra calls in between are
        # allowed; Ragas's tool call accuracy allows none.
        "tool_correctness_ordered": ToolCorrectnessMetric(model=judge, threshold=0.5, async_mode=False,
                                                          verbose_mode=False, should_consider_ordering=True),
        # The judge reads the input and each call's arguments: were they right
        # for this request? A run with no tool call is undecided here: DeepEval
        # requires tools_called, and a case without calls sends None.
        "argument_correctness": ArgumentCorrectnessMetric(model=judge, threshold=0.5, async_mode=False,
                                                          verbose_mode=False),
        # The judge infers the task and the outcome from the input, the output
        # and the tools called. It has no expected output to compare with.
        "task_completion": TaskCompletionMetric(model=judge, threshold=0.5, async_mode=False,
                                                verbose_mode=False),
    }


# ---------------------------------------------------------------------------
# Day 3, Module 09: conversation metrics
# ---------------------------------------------------------------------------
def conversation_metrics(judge) -> dict[str, object]:
    """Four of DeepEval's multi-turn metrics, each judged by your model.

    knowledge retention     assistant turns that do not forget a fact the user gave
    turn relevancy          assistant turns relevant to the turns before them (sliding window)
    turn faithfulness       assistant turns supported by the context of that turn
    conversation completeness  the user's intentions the conversation satisfied
    """
    from deepeval.metrics import (
        ConversationCompletenessMetric,
        KnowledgeRetentionMetric,
        TurnFaithfulnessMetric,
        TurnRelevancyMetric,
    )

    common = dict(model=judge, threshold=0.5, async_mode=False, verbose_mode=False)
    return {
        "knowledge_retention": KnowledgeRetentionMetric(**common),
        "turn_relevancy": TurnRelevancyMetric(**common),
        "turn_faithfulness": TurnFaithfulnessMetric(**common),
        "conversation_completeness": ConversationCompletenessMetric(**common),
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
