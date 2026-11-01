"""Day 3, Modules 06 and 08: LLM-as-judge, and validating it.

An LLM judge is itself a system that needs evaluation. The circularity
objection is real, and the answer is not to avoid judges but to measure them
against human labels before trusting them. `validate` below is the part teams
skip, and skipping it is how a dashboard ends up confidently wrong.

Groundedness and correctness are scored separately on purpose. They are
independent: an answer can be perfectly faithful to a stale document and
wrong, and it can be correct while ignoring the evidence entirely. Collapsing
them into one "accuracy" number is the most common error in RAG evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from helix.config import Config
from helix.llm import complete
from helix.prompts import render_context
from tracing.schema import Trace

GROUNDEDNESS = """\
You are evaluating whether an answer is supported by the evidence it was given.

EVIDENCE:
{context}

ANSWER:
{answer}

Is every factual claim in the ANSWER supported by the EVIDENCE? Ignore whether
the answer is true in the world; judge only whether the EVIDENCE supports it.

Reply with one word, GROUNDED or UNGROUNDED, then a new line with one short
sentence naming the first unsupported claim if there is one.
"""

CORRECTNESS = """\
You are checking an answer against a reference.

QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}

Does the ANSWER convey the same substantive information as the REFERENCE?
Differences in wording, length or formatting do not matter.

Reply with one word, CORRECT or INCORRECT, then a new line with one short
sentence explaining the difference if there is one.
"""


@dataclass
class Verdict:
    label: str
    reason: str

    @classmethod
    def parse(cls, raw: str, positive: str) -> "Verdict":
        """Find the verdict word wherever the model put it.

        Live models do not always obey "one word, then a new line": some add
        "Verdict:", some bold the word, some explain first. The negative word
        is matched first because UNGROUNDED contains GROUNDED. Anything that
        carries neither word is UNPARSEABLE, which counts as a fail and is
        worth counting in its own right.
        """
        negative = {"GROUNDED": "UNGROUNDED", "CORRECT": "INCORRECT"}[positive]
        text = (raw or "").strip()
        m = re.search(rf"\b({negative}|{positive})\b", text.upper())
        if not m:
            return cls(label="UNPARSEABLE", reason=text[:200])
        reason = (text[m.end():] or text[:m.start()]).strip(" *:.-\n")
        return cls(label=m.group(1), reason=reason.splitlines()[0] if reason else "")

    def __bool__(self) -> bool:
        return self.label in {"GROUNDED", "CORRECT"}


def judge_groundedness(trace: Trace, cfg: Config) -> Verdict:
    """Ask the judge model (HELIX_JUDGE_MODEL, or the answer model) for a verdict."""
    reached = [c for c in trace.retrieved if not c.dropped_at]
    raw = complete(
        [{"role": "user", "content": GROUNDEDNESS.format(
            context=render_context(reached), answer=trace.answer)}],
        cfg, model=cfg.judge_model or None,
    )[0]
    return Verdict.parse(raw, "GROUNDED")


def judge_correctness(trace: Trace, reference: str, cfg: Config) -> Verdict:
    raw = complete(
        [{"role": "user", "content": CORRECTNESS.format(
            question=trace.query, reference=reference, answer=trace.answer)}],
        cfg, model=cfg.judge_model or None,
    )[0]
    return Verdict.parse(raw, "CORRECT")


def quadrant(grounded: bool, correct: bool) -> str:
    """The four cases, and which layer each one sends you to."""
    return {
        (True, True): "working as intended, no intervention",
        (True, False): "faithful to a wrong or superseded document: fix the corpus",
        (False, True): "answered from pretraining, ignored the evidence: fix grounding",
        (False, False): "unsupported and wrong: localise with the trace first",
    }[(grounded, correct)]


def validate(judge_labels: list[bool], human_labels: list[bool]) -> dict[str, float]:
    """Agreement between a judge and the humans it is meant to stand in for.

    EXERCISE (Day 2, Module 06): a judge below roughly 0.8 agreement is not
    yet a measuring instrument. Report true/false positive rates, not just
    accuracy: a judge that calls everything GROUNDED scores well on a corpus
    that is mostly grounded and tells you nothing.
    """
    assert len(judge_labels) == len(human_labels), "label sets must align"
    n = len(human_labels) or 1
    tp = sum(j and h for j, h in zip(judge_labels, human_labels))
    tn = sum((not j) and (not h) for j, h in zip(judge_labels, human_labels))
    fp = sum(j and (not h) for j, h in zip(judge_labels, human_labels))
    fn = sum((not j) and h for j, h in zip(judge_labels, human_labels))
    return {
        "agreement": round((tp + tn) / n, 3),
        "false_positive_rate": round(fp / max(fp + tn, 1), 3),
        "false_negative_rate": round(fn / max(fn + tp, 1), 3),
        "n": n,
    }
