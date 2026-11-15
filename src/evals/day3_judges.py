"""Day 3, Module 08: groundedness and correctness, two judges and two checkers.

An LLM judge is itself a system that needs evaluation. The circularity
objection is real, and the answer is not to avoid judges but to measure them
against human labels before trusting them, on the judge's own criterion:
the groundedness judge against people's grounded labels, the correctness
judge against their correct labels (the Day 3 sheet, `evals.day3_sheet`).
`measure` below reports TPR and TNR with their counts, as Day 2 does. That
step is the one teams skip, and skipping it is how a dashboard ends up
confidently wrong.

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


#: The UMBRELA relevance prompt (Upadhyay et al. 2024, reproducing the Bing
#: assessor of Thomas et al.), as Hands-On RAG for Production prints it. One
#: call per (question, chunk). Day 3 measures it against the relevance the
#: suite's expected evidence defines, before anyone trusts it.
UMBRELA = """\
Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:
0 = represents that the passage has nothing to do with the query,
1 = represents that the passage seems related to the query but does not answer it,
2 = represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and
3 = represents that the passage is dedicated to the query and contains the exact answer.
Important Instructions:
Assign category 1 if the passage is somewhat related to the topic but not completely, category 2 if the passage presents something very important related to the entire topic but also has some extra information, and category 3 if the passage only and entirely refers to the topic.
If none of the above satisfies, give it category 0.
Query: {query}
Passage: {chunk}
Split this problem into steps:
Consider the underlying intent of the search.
Measure how well the content matches a likely intent of the query (M).
Measure how trustworthy the passage is (T).
Consider the aspects above and the relative importance of each, and decide on final score (O). The final score must be an integer value only.
Do not provide any code in the result. Provide each score in the format of a single integer without any reasoning.
"""


def judge_chunk_relevance(query: str, chunk_text: str, cfg: Config) -> int | None:
    """0 to 3 from the judge; the last integer it writes is the final score (O)."""
    raw = complete([{"role": "user", "content": UMBRELA.format(query=query, chunk=chunk_text)}],
                   cfg, model=cfg.judge_model or None)[0]
    found = re.findall(r"\b([0-3])\b", raw or "")
    return int(found[-1]) if found else None


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


# ---------------------------------------------------------------------------
# The cheap checkers: free, instant, brittle. Every judge is compared with them.
# ---------------------------------------------------------------------------
_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def grounded_check(trace: Trace) -> bool:
    """Is the answer's content in the context it was given (word overlap, 75 percent)."""
    reached = [c for c in trace.retrieved if not c.dropped_at]
    if not reached:
        return False
    ctx: set[str] = set()
    for c in reached:
        ctx |= words(c.text) | words(c.heading_path)
    content = [w for w in words(trace.answer) if len(w) > 4]
    if not content:
        return False
    return sum(1 for w in content if w in ctx) / len(content) >= 0.75


def correct_check(trace: Trace, ground_truth: str) -> bool:
    """Does the answer carry the reference's key facts.

    Numbers are the load-bearing part of a policy answer, so every figure in
    the reference must appear. A parental-leave answer that gets one of two
    numbers right is not half correct, it is wrong.
    """
    nums = re.findall(r"\d+", ground_truth or "")
    body = (trace.answer or "").lower()
    if nums:
        return all(n in body for n in nums)
    keys = re.findall(r"[a-z]{6,}", (ground_truth or "").lower())[:3]
    return bool(keys) and all(k in body for k in keys)


# ---------------------------------------------------------------------------
# Validation: a judge against people, on the judge's own criterion
# ---------------------------------------------------------------------------
#: The two criteria Day 3 judges. The Day 3 sheet holds one human label per
#: criterion. Positive means the failure is present, as on Day 2: an
#: ungrounded answer, an incorrect answer.
CRITERIA = {
    "grounded": "every factual claim is supported by the context the model was given "
                "(a refusal makes no claim, so it is grounded)",
    "correct": "the answer gives the reference's substantive answer to the question asked",
    "complete": "the answer contains every vital fact the question needs "
                "(for a question Helix must decline, the vital fact is the refusal)",
}


def measure(instrument_pass: list[bool | None], human_pass: list[bool]) -> dict[str, object]:
    """TPR and TNR with their counts, exactly as Day 2 measures an evaluator."""
    from evals.day2_evaluators import rates

    return rates(instrument_pass, human_pass)
