"""Ragas metrics on workshop cases.

Ragas 0.4 has two metric families. This uses the newer one,
`ragas.metrics.collections`, because it accepts any object that implements
Ragas's small LLM interface, which is how the judge stays on helix.llm.

Known issue, accepted with a mitigation: CVE-2026-6587 (an SSRF flaw, CVSS
6.3, every Ragas version from 0.2.3 to 0.4.3, no fix released). The flaw is
in the image loader of the multi-modal faithfulness and relevance metrics,
which fetch any URL found in `retrieved_contexts`. The workshop never builds
those two metrics, and `METRICS` below is the complete list it uses. A test
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

METRICS = ("faithfulness", "context_precision", "context_recall", "factual_correctness",
           "answer_relevancy", "tool_call_f1")


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
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        FactualCorrectness,
        Faithfulness,
        ToolCallF1,
    )

    built = {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "factual_correctness": FactualCorrectness(llm=llm),
        "tool_call_f1": ToolCallF1(),
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
    elif name == "factual_correctness":
        if not case.reference:
            return Measured(name, None, "needs a reference")
        r = await metric.ascore(response=case.answer, reference=case.reference)
    elif name == "answer_relevancy":
        r = await metric.ascore(user_input=case.input_text, response=case.answer)
    elif name == "tool_call_f1":
        if not case.expected_tools:
            return Measured(name, None, "the suite names no tools for this question")
        from .cases import to_ragas_messages

        messages, reference = to_ragas_messages(case)
        r = await metric.ascore(user_input=messages, reference_tool_calls=reference)
    else:
        raise KeyError(name)
    value = _value(r)
    return Measured(name, value, "" if value is not None else "no statements to score (NaN)")


def score(name: str, metric, case: Case) -> Measured:
    """Score one case with one metric; failures become an undecided result."""
    try:
        return asyncio.run(_score(name, metric, case))
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        return Measured(name, None, f"{type(exc).__name__}: {str(exc)[:160]}")
