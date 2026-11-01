"""Day 2, Module 06: one evaluator per failure mode, with a pass condition.

Error analysis named the failure modes. This module turns each one into a
check that says pass or fail on a single trace, using the cheapest instrument
that can decide it:

    failure mode        instrument                 why this one
    ------------------  -------------------------  ------------------------------
    stale_policy        code: document status      the trace records the version
    entitlement_leak    code: access labels        the trace records the label
    false_refusal       code: refusal vs suite     the suite says what to expect
    missed_refusal      code: refusal vs suite     the suite says what to expect
    wrong_scope_row     a judge                    needs reading the answer
    no_direct_answer    a judge                    needs reading the question too

Code first, where the trace already holds the fact. A judge only where the
decision needs reading. The judge to build is a binary one, and the others
run beside it on the same criterion, so you can see what changes:

    judge (our code)         pass or fail on one failure mode: the task, the
                             Pass and Fail definitions, examples from the
                             train split only, and JSON with the reasoning
                             before the answer
    G-Eval (our code)        the method from Liu et al. (2023), written out in
                             this file: a 1 to 5 score, no examples, and a
                             pass threshold read from the labels
    DeepEval GEval           the same criteria and steps, in the toolkit
    DeepEval DAG             the same decision as a tree of yes/no questions

Every evaluator is then checked against the human labels for its own failure
mode (`rates`): of the failures people found, how many it catches (TPR), and
of the traces people passed, how many it passes (TNR). An evaluator that
disagrees with people is not wrong in an interesting way; it is not finished.

Once both rates are known, an evaluator's raw failure rate on unlabelled
traffic can be corrected for its known errors (`corrected_failure_rate`),
and a bootstrap over the labelled traces gives that corrected rate a 95
percent interval (`bootstrap_interval`), the same method as the judgy
library, here in a few lines of numpy.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from helix.config import Config
from helix.ingest import document_status
from helix.llm import complete
from helix.retrieve import ENTITLEMENTS
from toolkits.cases import Case


@dataclass
class Result:
    evaluator: str
    mode: str
    passed: bool | None          # None: the instrument could not decide
    reason: str
    score: float | None = None


# ---------------------------------------------------------------------------
# Code checks. No model, no cost, same answer every time.
# ---------------------------------------------------------------------------
def check_stale(case: Case, status: dict[str, dict] | None = None) -> Result:
    """Fails when the answer cites a page that a newer page supersedes (FP8)."""
    status = status or document_status()
    meta = {m["chunk_id"]: m for m in case.context_meta}
    stale = []
    for cid in case.cited_ids:
        doc = meta.get(cid, {}).get("doc_id", "")
        s = status.get(doc)
        if s and s["status"] == "superseded":
            stale.append(f"{doc} v{s['doc_version']} (superseded by {s['superseded_by']})")
    if stale:
        return Result("stale evidence (code)", "stale_policy", False, "cites " + ", ".join(stale))
    return Result("stale evidence (code)", "stale_policy", True,
                  "every cited page is current" if case.cited_ids else "cites nothing")


def _a(noun: str) -> str:
    return ("an " if noun[:1].lower() in "aeiou" else "a ") + noun


def check_leak(case: Case) -> Result:
    """Fails when the model was given a page the asker may not read (FP11)."""
    allowed = ENTITLEMENTS.get(case.asker_role, {"all"})
    leaked = [m for m in case.context_meta if m["access_label"] not in allowed]
    if leaked:
        cited = {m["chunk_id"] for m in leaked} & set(case.cited_ids)
        what = ", ".join(f"{m['doc_id']} ({m['access_label']})" for m in leaked)
        return Result("entitlement (code)", "entitlement_leak", False,
                      f"{what} reached the model for {_a(case.asker_role)}"
                      + ("; the answer cites it" if cited else ""))
    return Result("entitlement (code)", "entitlement_leak", True,
                  f"nothing above the access of {_a(case.asker_role)} reached the model")


def check_false_refusal(case: Case) -> Result:
    """Fails when the system refused a question the handbook answers."""
    if case.should_refuse is None:
        return Result("false refusal (code)", "false_refusal", None, "the suite does not say")
    if case.refused and not case.should_refuse:
        return Result("false refusal (code)", "false_refusal", False,
                      "refused, but the suite expects an answer")
    return Result("false refusal (code)", "false_refusal", True,
                  "refused as expected" if case.refused else "answered")


def check_missed_refusal(case: Case) -> Result:
    """Fails when the system answered a question it must decline."""
    if case.should_refuse is None:
        return Result("missed refusal (code)", "missed_refusal", None, "the suite does not say")
    if case.should_refuse and not case.refused:
        return Result("missed refusal (code)", "missed_refusal", False,
                      "answered, but the suite expects Helix to decline")
    return Result("missed refusal (code)", "missed_refusal", True,
                  "declined as expected" if case.refused else "answered, as expected")


CODE_CHECKS: list[Callable[[Case], Result]] = [check_stale, check_leak, check_false_refusal,
                                               check_missed_refusal]


# ---------------------------------------------------------------------------
# The judged criterion, written once and used by all three judges
# ---------------------------------------------------------------------------
SCOPE_NAME = "right scope"
SCOPE_MODE = "wrong_scope_row"
SCOPE_CRITERIA = (
    "The answer must give the figure that applies to the asker: their own contracting "
    "office, and their role where the question states one (for example primary caregiver). "
    "Another office's figure or another role's figure fails, even when a document states "
    "that figure. A company-wide figure fails when the handbook has a different figure for "
    "the asker's office or role, and passes when the handbook has one figure for everyone. "
    "An answer that correctly refuses passes."
)

#: A second judged mode, the one the reference run's two annotators agreed on
#: least. Its criteria carry the sentence their disagreement showed was missing.
DIRECT_MODE = "no_direct_answer"
DIRECT_CRITERIA = (
    "The answer must answer the question that was asked. A yes-or-no question needs a "
    "clear yes or no; a question about one figure needs that figure. True facts that do "
    "not answer the question fail, even when they come from the right page. When the "
    "question has two parts, judge the first part. An answer that declines passes."
)

#: Every judged mode: its name in reports, and the criteria all judges share.
JUDGED: dict[str, tuple[str, str]] = {
    SCOPE_MODE: (SCOPE_NAME, SCOPE_CRITERIA),
    DIRECT_MODE: ("answers the question", DIRECT_CRITERIA),
}

#: The Pass and Fail definitions the binary judge receives, and the sentence
#: that explains a passing example. Pass means the failure is absent; the
#: answer may still fail for another reason, which another evaluator judges.
DEFINITIONS: dict[str, dict[str, str]] = {
    SCOPE_MODE: {
        "pass": ("the figure applies to this asker's office and role, or the handbook has "
                 "one figure for everyone and the answer gives it, or the answer correctly "
                 "declines"),
        "fail": ("the answer gives another office's or another role's figure, or a "
                 "company-wide figure where the handbook has a different one for this asker"),
        "why_pass": "The figure in the answer is the one that applies to this asker.",
    },
    DIRECT_MODE: {
        "pass": ("the answer responds to the question asked: a clear yes or no to a "
                 "yes-or-no question, the figure asked for, or a correct refusal"),
        "fail": ("the answer gives true facts that do not answer the question asked, such "
                 "as an allowance amount when the question is whether an item is covered"),
        "why_pass": "The answer responds to the question that was asked.",
    },
}


# ---------------------------------------------------------------------------
# The binary judge (our code): the judge to build first
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """You are an expert evaluator of answers from Helix, an HR policy assistant
for the employees of one company.

Your task: decide whether the answer below shows one failure: {name}.
Judge this criterion only. An answer can pass here and still fail for another
reason, which another evaluator judges.

Criterion: {criteria}

Pass: {pass_def}.
Fail: {fail_def}.

Return a JSON object with two keys, in this order:
  "reasoning": one or two sentences that apply the criterion to this answer
  "answer": "Pass" or "Fail"

Examples:
{examples}
-----
Now evaluate this one.

{fields}

Your JSON evaluation:"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"},
                   "answer": {"type": "string", "enum": ["Pass", "Fail"]}},
    "required": ["reasoning", "answer"],
}


@dataclass
class Example:
    """One labelled trace from the train split, shown to the judge."""

    id: str
    case: Case
    passed: bool             # on this failure mode
    reasoning: str


def pick_examples(mode: str, train: list[tuple[str, Case, bool, set[str], str]],
                  shots: int = 1) -> list[Example]:
    """Few-shot examples for one judged mode, from the train split only.

    `train` holds (id, case, acceptable, labels, note) for each train trace.
    Failing examples carry the mode, the clearest first (fewest other modes),
    with the expert's note as the reasoning. Passing examples lack the mode:
    a trace that passes overall first, then one that answers the question it
    was asked (not off topic), then the one with the fewest other problems.
    Taking examples from dev or test would leak them into the measurement.
    """
    def order(row):
        rid, case, ok, labels, note = row
        return (not ok, "off_topic" in labels, len(labels), rid)

    fails = sorted((r for r in train if mode in r[3]), key=order)[:shots]
    passes = sorted((r for r in train if mode not in r[3]), key=order)[:shots]
    why_pass = DEFINITIONS.get(mode, {}).get("why_pass", "This criterion is met.")
    out = [Example(rid, case, False, note) for rid, case, ok, labels, note in fails]
    for rid, case, ok, labels, note in passes:
        other = (f" It has another problem ({', '.join(sorted(labels))}), which another evaluator judges."
                 if labels else "")
        out.append(Example(rid, case, True, why_pass + other))
    return out


def _fields(case: Case) -> str:
    return f"{case.input_text}\nAnswer: {case.answer}"


def judge_prompt(case: Case, mode: str, examples: list[Example]) -> str:
    """The whole prompt the binary judge receives for one trace."""
    name, criteria = JUDGED[mode]
    d = DEFINITIONS[mode]
    shown = []
    for n, ex in enumerate(examples, 1):
        verdict = json.dumps({"reasoning": ex.reasoning, "answer": "Pass" if ex.passed else "Fail"})
        shown.append(f"-----\nInput {n}:\n{_fields(ex.case)}\nEvaluation {n}:\n{verdict}")
    return JUDGE_PROMPT.format(name=name, criteria=criteria, pass_def=d["pass"], fail_def=d["fail"],
                               examples="\n".join(shown) if shown else "(none in the train split)",
                               fields=_fields(case))


_ANSWER = re.compile(r'"answer"\s*:\s*"(pass|fail)"', re.IGNORECASE)


def parse_verdict(text: str) -> tuple[bool | None, str]:
    """(passed, reasoning) from the judge's reply; passed is None when it gave neither."""
    reasoning = ""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            reasoning = str(obj.get("reasoning", ""))
            ans = str(obj.get("answer", "")).strip().lower()
            if ans in ("pass", "fail"):
                return ans == "pass", reasoning
        except (ValueError, AttributeError):
            pass
    a = _ANSWER.search(text or "")
    if a:
        return a.group(1).lower() == "pass", reasoning
    words = re.findall(r"\b(pass|fail)\b", (text or "").lower())
    if len(set(words)) == 1:
        return words[0] == "pass", reasoning
    return None, reasoning


def binary_judge(case: Case, mode: str, examples: list[Example], cfg: Config,
                 temperature: float = 0.0) -> Result:
    """Pass or fail on one failure mode, with the reasoning written first."""
    text = complete([{"role": "user", "content": judge_prompt(case, mode, examples)}],
                    replace(cfg, temperature=temperature), model=cfg.judge_model or None,
                    json_schema=JUDGE_SCHEMA)[0]
    passed, reasoning = parse_verdict(text)
    return Result("judge (our code)", mode, passed,
                  reasoning or ("no Pass or Fail in the reply" if passed is None else ""))


# ---------------------------------------------------------------------------
# G-Eval (our code): the method from Liu et al. 2023, written out in this file
# ---------------------------------------------------------------------------
GEVAL_STEPS_PROMPT = """You will evaluate answers from an HR policy assistant.

Evaluation criteria:
{criteria}

Write the evaluation steps that a careful reviewer follows to apply these
criteria to one answer. Write 3 to 5 short numbered steps and nothing else."""

GEVAL_SCORE_PROMPT = """You will evaluate one answer from an HR policy assistant.

Evaluation criteria:
{criteria}

Evaluation steps:
{steps}

{fields}

Follow the evaluation steps. Then score the answer from 1 to 5, where 1 means
it fails the criteria completely and 5 means it meets them completely.
Reply with the score only: one digit."""

_DIGIT = re.compile(r"\b([1-5])\b")


def geval_steps(criteria: str, cfg: Config) -> list[str]:
    """G-Eval's first stage: the judge writes its own evaluation steps.

    Generated live, on every run, and never saved. The script prints them and
    passes the same steps to DeepEval in the same run, so the two judges are
    compared on equal terms. Two runs can produce different steps: when the
    verdicts change between runs, read the steps first.
    """
    text = complete([{"role": "user", "content": GEVAL_STEPS_PROMPT.format(criteria=criteria)}],
                    replace(cfg, temperature=0.0), model=cfg.judge_model or None)[0]
    steps = [re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line).strip()
             for line in text.splitlines() if line.strip()]
    return [s for s in steps if len(s) > 3][:6]


def geval_score(case: Case, criteria: str, steps: list[str], cfg: Config,
                samples: int = 3, threshold: float = 3.5, mode: str = SCOPE_MODE) -> Result:
    """The form-filling stage, with the score averaged over samples.

    The paper weights each possible score by the probability the model gave
    it. Not every provider returns token probabilities, so this samples the
    score several times at temperature 1 and averages, which estimates the
    same quantity (the paper did this for GPT-4). `threshold` turns the
    average into pass or fail. It is a choice, and Module 06 sets it by
    comparing with human labels, not by taste.
    """
    fields = f"{case.input_text}\nAnswer: {case.answer}"
    prompt = GEVAL_SCORE_PROMPT.format(criteria=criteria,
                                       steps="\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)),
                                       fields=fields)
    scfg = replace(cfg, temperature=1.0 if samples > 1 else 0.0)
    scores = []
    for _ in range(max(1, samples)):
        text = complete([{"role": "user", "content": prompt}], scfg, model=cfg.judge_model or None)[0]
        m = _DIGIT.search(text)
        if m:
            scores.append(int(m.group(1)))
    name = "G-Eval (our code)"
    if not scores:
        return Result(name, mode, None, "no score in any reply", None)
    mean = statistics.fmean(scores)
    spread = f"samples {scores}" if len(scores) > 1 else f"score {scores[0]}"
    return Result(name, mode, mean >= threshold, f"mean {mean:.2f}, {spread}", mean)


# ---------------------------------------------------------------------------
# An evaluator against people, per failure mode
# ---------------------------------------------------------------------------
def rates(evaluator_pass: list[bool | None], human_pass: list[bool]) -> dict[str, object]:
    """How an evaluator's verdicts line up with the human labels on one mode.

    "Positive" means the failure is present, so:

    TPR          of the failures people found, the share the evaluator failed
                 too (caught). The rest it missed.
    TNR          of the traces people passed, the share the evaluator passed
                 too. The rest are false alarms.
    undecided    the evaluator gave no verdict; counted against it in both.

    A rate is None when there is nothing to measure it on: no failures (or
    no passes) among the labelled traces. That is a gap in the labels, not a
    perfect score.
    """
    pairs = list(zip(evaluator_pass, human_pass))
    on_fail = [e for e, h in pairs if not h]
    on_pass = [e for e, h in pairs if h]
    caught = sum(1 for e in on_fail if e is False)
    passed = sum(1 for e in on_pass if e is True)
    return {
        "n": len(pairs),
        "failures": len(on_fail),
        "passes": len(on_pass),
        "caught": caught,
        "missed": sum(1 for e in on_fail if e is True),
        "passed": passed,
        "false_alarm": sum(1 for e in on_pass if e is False),
        "undecided": sum(1 for e, _ in pairs if e is None),
        "tpr": caught / len(on_fail) if on_fail else None,
        "tnr": passed / len(on_pass) if on_pass else None,
        "agreement": (caught + passed) / len(pairs) if pairs else None,
    }


def threshold_sweep(scores: list[float | None], human_pass: list[bool],
                    thresholds: tuple[float, ...] = (2.5, 3.5, 4.5)) -> list[tuple[float, dict]]:
    """TPR and TNR at several pass thresholds: the threshold is calibrated, not chosen."""
    out = []
    for th in thresholds:
        verdicts = [None if s is None else s >= th for s in scores]
        out.append((th, rates(verdicts, human_pass)))
    return out


#: Below this, TPR + TNR - 1 is too close to zero: the evaluator is barely
#: better than a coin and the correction would divide by almost nothing.
MIN_INFORMATIVE = 0.05


def corrected_failure_rate(observed: float, tpr: float | None, tnr: float | None) -> float | None:
    """The true failure rate, estimated from an imperfect evaluator (Rogan and Gladen 1978).

    An evaluator that fails a share `observed` of unlabelled traces also
    misses some failures (TPR below 1) and raises some false alarms (TNR
    below 1). With both rates measured on labelled traces:

        true rate = (observed + TNR - 1) / (TPR + TNR - 1)

    The result is clipped to 0..1. It is None when TPR + TNR - 1 is below
    0.05: an evaluator close to a coin carries no information to correct.
    Positive means the failure is present here; texts that call a pass the
    positive swap the two names and correct the pass rate, with the same
    arithmetic.
    """
    if tpr is None or tnr is None or tpr + tnr - 1 < MIN_INFORMATIVE:
        return None
    return min(1.0, max(0.0, (observed + tnr - 1) / (tpr + tnr - 1)))


def bootstrap_interval(evaluator_pass: list[bool | None], human_pass: list[bool], observed: float,
                       iterations: int = 20000, seed: int = 0) -> tuple[float, float] | None:
    """A 95 percent interval for the corrected failure rate.

    TPR and TNR come from a small labelled set, so the corrected rate is
    uncertain too. Resample the labelled (human, evaluator) pairs with
    replacement, recompute TPR and TNR on each resample, correct the same
    observed rate with them, and read the 2.5th and 97.5th percentiles.
    Resamples with no failures or no passes, or with TPR + TNR - 1 below
    0.05, are skipped. An undecided verdict counts against the evaluator, as
    in `rates`. None when no resample can be used.
    """
    human_fail = np.array([not h for h in human_pass], dtype=bool)
    caught = np.array([(not h) and e is False for e, h in zip(evaluator_pass, human_pass)], dtype=bool)
    passed = np.array([h and e is True for e, h in zip(evaluator_pass, human_pass)], dtype=bool)
    n = len(human_fail)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iterations, n))
    fails = human_fail[idx].sum(axis=1)
    passes = n - fails
    ok = (fails > 0) & (passes > 0)
    tpr = np.where(ok, caught[idx].sum(axis=1) / np.maximum(fails, 1), 0.0)
    tnr = np.where(ok, passed[idx].sum(axis=1) / np.maximum(passes, 1), 0.0)
    denom = tpr + tnr - 1
    ok &= denom >= MIN_INFORMATIVE
    if not ok.any():
        return None
    theta = np.clip((observed + tnr[ok] - 1) / denom[ok], 0.0, 1.0)
    low, high = np.percentile(theta, [2.5, 97.5])
    return float(low), float(high)
