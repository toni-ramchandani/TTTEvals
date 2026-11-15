"""Ragas metrics on workshop cases.

Ragas 0.4 has two metric families. This uses the newer one,
`ragas.metrics.collections`, because it accepts any subclass of Ragas's
small LLM base class (`InstructorBaseRagasLLM`), which is how the judge
stays on helix.llm.

Known issue, accepted with a mitigation: CVE-2026-6587 (an SSRF flaw, CVSS
6.3, every Ragas version from 0.2.3 to 0.4.3, no fix released). The flaw is
in the image loader of the multi-modal faithfulness and relevance metrics,
which fetch any URL found in `retrieved_contexts`. The workshop never builds
those two metrics. `METRICS` below, plus Noise Sensitivity and Agent Goal
Accuracy without a reference for Day 3 (`build_noise`, `build_conversation`),
is the complete list it uses. A test
checks that nothing in the project imports them.

A second advisory, CVE-2025-69872, affects diskcache, which Ragas installs:
reading a cache file runs any pickled code inside it. The workshop never
turns on Ragas's cache (`cache=None` throughout), so no cache file is read.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

from .cases import Case

METRICS = ("faithfulness", "context_precision", "context_recall", "context_utilization",
           "factual_correctness", "factual_recall", "answer_relevancy", "tool_call_f1",
           "tool_call_accuracy", "goal_accuracy")


@dataclass
class Measured:
    metric: str
    score: float | None
    reason: str = ""


def _value(result) -> float | None:
    v = getattr(result, "value", result)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def build(llm, embeddings=None) -> dict[str, object]:
    """Every Ragas metric the workshop uses, on your judge.

    context_precision    average precision over the retrieved contexts, the
                         judge deciding which help the reference answer. It
                         divides by the useful contexts IN the list.
    context_utilization  the same average precision, judged against the
                         response instead: did the answer use what ranked high?
    factual_correctness  claims of response and reference matched, F1
    factual_recall       mode="recall". Read its source before you read the
                         number: it counts the response's claims the
                         reference supports, divided by those plus the
                         reference's claims the response misses. Close to
                         coverage of the reference, not equal to it.
    tool_call_f1         tool names called against the reference, F1 (no model)
    tool_call_accuracy   the same calls in the same order, or 0 (no model)
    goal_accuracy        did the run reach the reference outcome? 1 or 0
    """
    from ragas.metrics.collections import (
        AgentGoalAccuracyWithReference,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        ContextUtilization,
        FactualCorrectness,
        Faithfulness,
        ToolCallAccuracy,
        ToolCallF1,
    )

    built = {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "context_utilization": ContextUtilization(llm=llm),
        "factual_correctness": FactualCorrectness(llm=llm),
        "factual_recall": FactualCorrectness(llm=llm, mode="recall", name="factual_recall"),
        "tool_call_f1": ToolCallF1(),
        "tool_call_accuracy": ToolCallAccuracy(strict_order=True),
        "goal_accuracy": AgentGoalAccuracyWithReference(llm=llm),
    }
    if embeddings is not None:
        built["answer_relevancy"] = AnswerRelevancy(llm=llm, embeddings=embeddings)
    return built


async def _score(name: str, metric, case: Case) -> Measured:
    if name == "faithfulness":
        if not case.contexts:
            return Measured(name, None, "no context reached the model")
        r = await metric.ascore(user_input=case.input_text, response=case.answer,
                                retrieved_contexts=case.contexts)
    elif name == "context_precision":
        if not case.reference or not case.contexts:
            return Measured(name, None, "needs a reference and contexts")
        r = await metric.ascore(user_input=case.input_text, reference=case.reference,
                                retrieved_contexts=case.contexts)
    elif name == "context_recall":
        if not case.reference or not case.contexts:
            return Measured(name, None, "needs a reference and contexts")
        r = await metric.ascore(user_input=case.input_text, retrieved_contexts=case.contexts,
                                reference=case.reference)
    elif name == "context_utilization":
        if not case.answer or not case.contexts:
            return Measured(name, None, "needs an answer and contexts")
        r = await metric.ascore(user_input=case.input_text, response=case.answer,
                                retrieved_contexts=case.contexts)
    elif name in ("factual_correctness", "factual_recall"):
        if not case.reference:
            return Measured(name, None, "needs a reference")
        r = await metric.ascore(response=case.answer, reference=case.reference)
    elif name == "answer_relevancy":
        r = await metric.ascore(user_input=case.input_text, response=case.answer)
    elif name in ("tool_call_f1", "tool_call_accuracy"):
        if not case.expected_tools:
            return Measured(name, None, "the suite names no tools for this question")
        from .cases import to_ragas_messages

        messages, reference = to_ragas_messages(case)
        r = await metric.ascore(user_input=messages, reference_tool_calls=reference)
    elif name == "goal_accuracy":
        if not case.reference:
            return Measured(name, None, "needs a reference outcome")
        from .cases import to_ragas_messages

        messages, _ = to_ragas_messages(case, results=True)
        r = await metric.ascore(user_input=messages, reference=case.reference)
    else:
        raise KeyError(name)
    value = _value(r)
    return Measured(name, value, "" if value is not None else "no statements to score (NaN)")


def build_noise(llm, mode: str = "irrelevant") -> object:
    """Ragas Noise Sensitivity (lower is better).

    mode="irrelevant", the right one for a planted noise chunk: the share of
    the answer's claims that are wrong and supported only by a context that
    does not support the reference. Ragas's default, "relevant", counts wrong
    claims supported by a relevant context instead, and would score an answer
    copied from the noise chunk as perfect.
    """
    from ragas.metrics.collections import NoiseSensitivity

    return NoiseSensitivity(llm=llm, mode=mode)


def score_noise(metric, question: str, answer: str, reference: str, contexts: list[str]) -> Measured:
    """Noise Sensitivity on one answer; failures become an undecided result."""
    async def _go():
        return await metric.ascore(user_input=question, response=answer, reference=reference,
                                   retrieved_contexts=contexts)

    try:
        value = _value(asyncio.run(_go()))
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        return Measured("noise_sensitivity", None, f"{type(exc).__name__}: {str(exc)[:160]}")
    return Measured("noise_sensitivity", value, "" if value is not None else "no claims to score (NaN)")


def build_conversation(llm) -> dict[str, object]:
    """Ragas's binary goal check on a whole conversation (no reference needed).

    The judge infers the user's goal and the end state from the messages, then
    says whether the goal was achieved: 1 or 0.
    """
    from ragas.metrics.collections import AgentGoalAccuracyWithoutReference

    return {"goal_accuracy": AgentGoalAccuracyWithoutReference(llm=llm)}


def score_conversation(name: str, metric, messages) -> Measured:
    """One conversation metric on one session; failures become an undecided result."""
    async def _go():
        return await metric.ascore(user_input=messages)

    try:
        value = _value(asyncio.run(_go()))
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        return Measured(name, None, f"{type(exc).__name__}: {str(exc)[:160]}")
    return Measured(name, value, "" if value is not None else "no score (NaN)")


def score(name: str, metric, case: Case) -> Measured:
    """Score one case with one metric; failures become an undecided result."""
    try:
        return asyncio.run(_score(name, metric, case))
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        return Measured(name, None, f"{type(exc).__name__}: {str(exc)[:160]}")
