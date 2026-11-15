"""Day 3, Module 08: what the model did with the context it was given.

Groundedness asks whether the answer stays inside its evidence. These checks
ask the other questions about generation, each one a code check that reads
the trace, so each can be measured against people like any other evaluator:

  completeness       does the answer contain every vital fact ("nugget") the
                     question needs? The nuggets are written down beside each
                     question in the suite, as the expected evidence is.
  context use        for each nugget: was it in the context the model saw,
                     and did the answer use it? In the context but missing
                     from the answer is an omission by the model; missing
                     from the context is a retrieval or assembly failure.
  citation support   does each cited chunk support the sentence it is
                     attached to? The lineage check (in the pipeline) asks
                     whether the cited page is current and permitted; this
                     asks whether it says what the sentence says.
  direct answer      a yes-or-no question gets a yes or a no (or a refusal).
                     This is the relevance failure of q6: true, faithful, and
                     no answer to the question asked.
  key facts          the figures an answer states, and whether it declined,
                     for comparing two answers to one question: the
                     context-swap test and the consistency test.

All of them match words and figures, not meaning. That makes them cheap,
instant and reproducible, and it makes them brittle in known ways: a judge
reads meaning, and is measured the same way.
"""

from __future__ import annotations

import re

from helix import bm25
from helix.llm import is_refusal
from helix.retrieve import ENTITLEMENTS

from tracing.schema import Chunk, Trace

_CITATION = re.compile(r"\[([^\]]+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")
_LEADING_CITATIONS = re.compile(r"^(?:\s*\[[^\]]+\])+")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")
_WORD = re.compile(r"[a-z0-9]+")


def _plain(text: str) -> str:
    """Lowercase, citations removed, whitespace normalised."""
    return " ".join(_CITATION.sub(" ", text or "").lower().split())


def _has(text: str, phrase: str) -> bool:
    """Whole-word match, so a nugget of 26 does not match the year 2026."""
    p = re.escape(" ".join(phrase.lower().split()))
    return re.search(rf"(?<![\w]){p}(?![\w])", text) is not None


# ---------------------------------------------------------------------------
# Completeness: nugget coverage
# ---------------------------------------------------------------------------
def nugget_coverage(answer: str, nuggets) -> dict:
    """Which vital nuggets the answer contains.

    `nuggets` is a list of nuggets, each a list of acceptable phrasings, or
    the word "refusal" for a question the system must decline.
    """
    if nuggets == "refusal":
        ok = is_refusal(answer or "")
        return {"found": [ok], "nuggets": ["a refusal"], "status": "complete" if ok else "missing"}
    text = _plain(answer)
    found = [any(_has(text, ph) for ph in alts) for alts in nuggets or []]
    if not found:
        return {"found": [], "nuggets": [], "status": "no nuggets"}
    status = "complete" if all(found) else "partial" if any(found) else "missing"
    return {"found": found, "nuggets": [alts[0] for alts in nuggets], "status": status}


def complete_check(trace: Trace, nuggets) -> bool | None:
    """Pass when every vital nugget is in the answer. None when the suite has none."""
    cov = nugget_coverage(trace.answer, nuggets)
    return None if cov["status"] == "no nuggets" else cov["status"] == "complete"


# ---------------------------------------------------------------------------
# Context use: was each nugget available, and was it used?
# ---------------------------------------------------------------------------
def context_use(trace: Trace, nuggets) -> list[dict]:
    """For each vital nugget: in the context the model saw? In the answer?"""
    if not nuggets or nuggets == "refusal":
        return []
    reached = [c for c in trace.retrieved if not c.dropped_at]
    ctx = _plain(" ".join(c.text for c in reached))
    ans = _plain(trace.answer)
    out = []
    for alts in nuggets:
        in_ctx = any(_has(ctx, ph) for ph in alts)
        in_ans = any(_has(ans, ph) for ph in alts)
        verdict = ("used" if in_ans else
                   "OMITTED: in the context, not in the answer" if in_ctx else
                   "not in the context: retrieval or assembly lost it")
        out.append({"nugget": alts[0], "in_context": in_ctx, "in_answer": in_ans, "verdict": verdict})
    return out


# ---------------------------------------------------------------------------
# Citation support
# ---------------------------------------------------------------------------
def _cited_ids(sentence: str, known: set[str]) -> list[str]:
    ids = []
    for raw in _CITATION.findall(sentence):
        head = raw.split()[0] if raw.split() else ""
        if head in known:
            ids.append(head)
    return ids


def citation_support(trace: Trace) -> list[dict]:
    """Each cited sentence, the chunks it cites, and whether they support it.

    Supported: every figure in the sentence appears in the cited chunks, and
    at least 60 percent of its content words (longer than four letters) do.
    """
    chunks = {c.chunk_id: c for c in trace.considered}
    known = set(chunks)
    rows = []
    # A citation written after the full stop ("... salary. [HB-X-00 v2] Next.")
    # belongs to the sentence before it, not to the one that follows.
    sentences: list[str] = []
    for part in _SENTENCE.split((trace.answer or "").strip()):
        lead = _LEADING_CITATIONS.match(part)
        if sentences and lead:
            sentences[-1] += " " + lead.group(0).strip()
            part = part[lead.end():]
        if part.strip():
            sentences.append(part)
    for sent in sentences:
        cited = _cited_ids(sent, known)
        if not cited:
            continue
        claim = _plain(sent)
        source = _plain(" ".join(chunks[c].text + " " + chunks[c].heading_path for c in cited))
        nums = _NUMBER.findall(claim)
        missing_nums = [n for n in nums if not _has(source, n)]
        words = [w for w in _WORD.findall(claim) if len(w) > 4]
        share = sum(1 for w in words if w in source.split()) / len(words) if words else 1.0
        ok = not missing_nums and share >= 0.6
        why = ("supported" if ok else
               f"figure(s) {', '.join(missing_nums)} not in the cited chunk" if missing_nums else
               f"only {share:.0%} of its words are in the cited chunk")
        rows.append({"sentence": " ".join(_CITATION.sub("", sent).split()), "cited": cited,
                     "supported": ok, "why": why})
    return rows


def citation_precision(trace: Trace) -> float | None:
    rows = citation_support(trace)
    return sum(r["supported"] for r in rows) / len(rows) if rows else None


# ---------------------------------------------------------------------------
# Direct answer to a yes-or-no question
# ---------------------------------------------------------------------------
_CLOSED = re.compile(r"^\s*(can|could|is|are|am|do|does|did|may|will|should|shall|must|has|have)\b", re.I)
_YESNO = re.compile(
    r"\b(yes|no|you can|you cannot|you can't|you may|you may not|is covered|is not covered|"
    r"are covered|are not covered|is eligible|is not eligible|not allowed|is allowed|you are entitled|"
    r"you are not entitled)\b", re.I)


def is_closed_question(question: str) -> bool:
    return bool(_CLOSED.match(question or ""))


def direct_answer_check(question: str, answer: str) -> bool | None:
    """Pass when a yes-or-no question gets a yes, a no, or a refusal. None if not yes-or-no."""
    if not is_closed_question(question):
        return None
    if is_refusal(answer or ""):
        return True
    return bool(_YESNO.search(_plain(answer)))


# ---------------------------------------------------------------------------
# Key facts: for comparing two answers to the same question
# ---------------------------------------------------------------------------
def key_facts(answer: str) -> tuple[str, frozenset[str]]:
    """('declined' or 'answered', the figures it states). Citations are ignored."""
    if is_refusal(answer or ""):
        return "declined", frozenset()
    return "answered", frozenset(_NUMBER.findall(_plain(answer)))


def same_facts(a: str, b: str) -> bool:
    """Two answers state the same thing: both declined, or the same figures."""
    ka, kb = key_facts(a), key_facts(b)
    if ka[0] != kb[0]:
        return False
    if ka[0] == "declined":
        return True
    if ka[1] or kb[1]:
        return ka[1] == kb[1]
    wa, wb = set(_WORD.findall(_plain(a))), set(_WORD.findall(_plain(b)))
    return bool(wa | wb) and len(wa & wb) / len(wa | wb) >= 0.8


# ---------------------------------------------------------------------------
# The context-swap test
# ---------------------------------------------------------------------------
def unrelated_chunks(chunks: list[Chunk], question: str, k: int = 2, user_role: str = "employee") -> list[Chunk]:
    """The k chunks least related to the question that the asker may read.

    The lowest BM25 scores against the question, one chunk per document, so
    the swapped context is about something else entirely.
    """
    allowed = ENTITLEMENTS.get(user_role, {"all"})
    pool = [c for c in chunks if c.access_label in allowed]
    if not pool:
        return []
    s = bm25.scores([bm25.tokenize(c.text) for c in pool], bm25.tokenize(question))
    order = sorted(range(len(pool)), key=lambda i: (s[i], pool[i].chunk_id))
    out, seen = [], set()
    for i in order:
        c = pool[i]
        if c.doc_id in seen:
            continue
        out.append(c)
        seen.add(c.doc_id)
        if len(out) == k:
            break
    return out


def swap_verdict(original: str, swapped: str) -> str:
    """What the context-swap test shows about one question."""
    if key_facts(swapped)[0] == "declined":
        return "declined on the unrelated context: it reads its context"
    if same_facts(original, swapped) and key_facts(original)[0] == "answered":
        return "SAME ANSWER on unrelated context: it answers from memory, not the context"
    return "answer changed with the context: it reads its context"
