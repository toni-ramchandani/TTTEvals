"""Day 2, Module 04: synthetic questions, from dimensions to tuples to text.

The common mistake is to ask a model for "fifty user questions". It returns
rephrasings of one easy question, and whole kinds of failure never appear.
The fix is a layer of indirection:

    1. dimensions   the ways real questions differ: topic, who asks, how
    2. tuples       one value per dimension, enumerated so every combination
                    can occur, not only the ones a model finds natural
    3. text         one question per tuple, written by your model

The tuple fixes the combination before any wording exists, which forces
variety instead of hoping for it. The dimensions come from the charter: the
input contract (field 4) says a question arrives with an office and a role,
and the not-in-scope list (field 6) says what Helix must decline.

Each question keeps its tuple. Who asks is not only in the wording: the
office and role go to Helix with the question, the way the real system
receives them from login, and the expected behaviour (answer or decline) is
known before anyone reads an answer.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

from helix.config import Config
from helix.llm import complete


@dataclass(frozen=True)
class Asker:
    role: str           # what the retriever's access check reads
    office: str         # the contracting office
    describe: str       # how the question writer is told about this person


#: Three dimensions for Helix. Three varied dimensions are enough to start;
#: add one when error analysis shows a kind of question you never generate.
TOPICS = ["parental leave", "annual leave", "notice period", "home office expenses",
          "remote work", "salary bands"]
ASKERS = [
    Asker("employee", "amsterdam", "an employee in the Amsterdam office"),
    Asker("employee", "dublin", "an employee in the Dublin office"),
    Asker("employee", "berlin", "an employee in the Berlin office"),
    Asker("manager", "amsterdam", "a manager in the Amsterdam office"),
]
SCENARIOS = {
    "clear question": "a specific question the handbook can answer, with the details the answer depends on",
    "ambiguous request": "a short, vague question that leaves out a detail the answer depends on",
    "out of scope": ("a request Helix must decline: legal advice, the meaning of the person's own "
                     "contract, or asking Helix to do something for them"),
    "multi-part": "two related questions in one message",
}
DIMENSIONS: dict[str, list[str]] = {
    "topic": TOPICS,
    "asker": [a.describe for a in ASKERS],
    "scenario": list(SCENARIOS),
}

#: From the charter: pay is not in scope for Helix (field 6), so a question
#: about salary bands is declined whoever asks it.
DECLINED_TOPICS = {"salary bands"}

QUERY_PROMPT = """\
We are writing test questions for Helix, an internal HR assistant. Helix
answers employee questions from the company handbook: parental leave, annual
leave, notice periods, home office expenses, remote work and onboarding.

Write one question that this person might type into Helix.

Topic: {topic}
Who asks: {asker}
Kind of question: {scenario}: {scenario_means}

Write it the way people really type: first person, plain words, no greeting.
Mention the office only if this person would. Return the question only."""


@dataclass(frozen=True)
class Tuple_:
    topic: str
    asker: Asker
    scenario: str

    @property
    def should_refuse(self) -> bool:
        """The expected behaviour, known from the tuple before any answer exists."""
        return self.scenario == "out of scope" or self.topic in DECLINED_TOPICS

    def describe(self) -> str:
        return f"{self.topic} | {self.asker.describe} | {self.scenario}"


def enumerate_tuples(seed: int = 0, n: int | None = None) -> list[Tuple_]:
    """Every combination, shuffled with a fixed seed.

    Enumerating beats asking a model for combinations: it is free, it covers
    every cell, and it cannot drift back to the easy ones. Use the model for
    the step that needs language, not for the step that needs a cross product.
    6 topics x 4 askers x 4 kinds = 96 tuples.
    """
    combos = [Tuple_(t, a, s) for t, a, s in itertools.product(TOPICS, ASKERS, SCENARIOS)]
    random.Random(seed).shuffle(combos)
    return combos[:n] if n else combos


# ---------------------------------------------------------------------------
# The reference run. The offline stub cannot write questions, so the
# reference run fills a template for each tuple instead. The teaching point is
# the indirection, which lives in the tuples, not in the wording.
# ---------------------------------------------------------------------------
#: topic: (clear question, vague question, request Helix must decline)
TOPIC_TEXT: dict[str, tuple[str, str, str]] = {
    "parental leave": ("How many weeks of parental leave do I get as the primary caregiver?",
                       "How much time off do I get for a new baby?",
                       "Can you book my parental leave in the HR portal from 1 March?"),
    "annual leave": ("How many days of annual leave can I carry over into next year?",
                     "Can I keep my unused days?",
                     "My contract says 28 days of leave. Is the contract or the handbook legally binding?"),
    "notice period": ("What is my notice period if I resign after three years?",
                      "How long do I have to stay after I quit?",
                      "Is the notice clause in my contract legally enforceable?"),
    "home office expenses": ("Does the home office allowance cover a laptop stand?",
                             "Can I expense my desk stuff?",
                             "Please approve my claim for a 450 EUR chair."),
    "remote work": ("How many days a week may I work remotely?",
                    "Can I work from home?",
                    "Can you approve my request to work from Spain for two months?"),
    "salary bands": ("What is the salary band for a senior engineer?",
                     "How much could I earn here?",
                     "Is my salary fair compared with the market?"),
}
OTHER_OFFICE = {"amsterdam": "Dublin", "dublin": "Amsterdam", "berlin": "Dublin"}


def render_offline(t: Tuple_) -> str:
    """Deterministic template text, used when no model is available."""
    clear, vague, declined = TOPIC_TEXT[t.topic]
    who = ("I am a manager in the {o} office." if t.asker.role == "manager"
           else "I work in the {o} office.").format(o=t.asker.office.title())
    if t.scenario == "clear question":
        return f"{who} {clear}"
    if t.scenario == "ambiguous request":
        return vague
    if t.scenario == "out of scope":
        return declined
    return f"{who} {clear} And is it the same in the {OTHER_OFFICE[t.asker.office]} office?"


def query_for(t: Tuple_, cfg: Config) -> str:
    """One question for one tuple, written by your model (or the template offline)."""
    if cfg.offline:
        return render_offline(t)
    raw = complete([{"role": "user", "content": QUERY_PROMPT.format(
        topic=t.topic, asker=t.asker.describe, scenario=t.scenario,
        scenario_means=SCENARIOS[t.scenario])}], cfg)[0]
    # Models like to add "Here is a question:" or wrap the question in quotes.
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not ln.lower().startswith(("here is", "here's", "question:"))] or lines
    return (lines[0] if lines else raw).strip().strip('"“”').strip()


def generate(n: int, cfg: Config, seed: int = 0, on_row=None, existing: list[dict] | None = None) -> list[dict]:
    """Tuples first, then one question per tuple, as suite items.

    Each row has what Helix needs to answer it (text, office, role) and what
    an evaluator needs to judge it (should_refuse), plus the tuple, so error
    analysis can count failures per topic, asker and kind of question.
    Repeated wording is skipped: two identical questions add a trace, not
    information.

    With `existing`, the new rows continue that set up to `n` in all: tuples
    already used are skipped, so reading more traffic never replaces the
    questions people have labelled.
    """
    existing = existing or []
    rows: list[dict] = []
    used = {(r.get("topic"), r.get("asker"), r.get("scenario")) for r in existing}
    seen = {r["text"].lower() for r in existing}
    want = n - len(existing)
    for t in enumerate_tuples(seed=seed):
        if len(rows) >= want:
            break
        if (t.topic, t.asker.describe, t.scenario) in used:
            continue
        text = query_for(t, cfg)
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        row = {"id": f"s{len(existing) + len(rows) + 1:02d}", "text": text, "topic": t.topic,
               "asker": t.asker.describe, "user_role": t.asker.role, "user_office": t.asker.office,
               "scenario": t.scenario, "should_refuse": t.should_refuse}
        rows.append(row)
        if on_row:
            on_row(row, len(rows), want)
    return rows
