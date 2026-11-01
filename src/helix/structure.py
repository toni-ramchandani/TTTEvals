"""Reading structure back out of handbook text.

Two of the workshop's three architectures need to know more about a question
than its words: which office it is about, which caregiver role, how many years
of service, which topics it spans, and whether it compares offices.

* GraphRAG uses these at build time, to turn tables into facts, and at query
  time, to link a question to nodes in the graph.
* Agentic RAG uses them at query time, to decide which tools to call and with
  which arguments.

Vanilla Helix uses none of them, which is why the office in the user's profile
is "available but unused" in its query.transform span.

Everything here is deliberately plain: keyword tables and regular expressions.
A production system would use a model for linking and extraction, and every
failure mode those models have (missed aliases, wrong attribution, a qualifier
lost at a chunk boundary) is reproduced here by a rule you can read.
"""

from __future__ import annotations

import re

#: Canonical office name -> every way the handbook or a user refers to it.
#: The first entry is the canonical name, and the only one "exact" linking uses.
OFFICES: dict[str, tuple[str, ...]] = {
    "dublin": ("dublin", "ireland", "irish"),
    "amsterdam": ("amsterdam", "netherlands", "dutch", "holland"),
    "berlin": ("berlin", "germany", "german"),
    "london": ("london", "united kingdom", "uk", "england", "british"),
    "singapore": ("singapore",),
}

#: Topic -> trigger phrases. Order matters only for readability.
TOPICS: dict[str, tuple[str, ...]] = {
    "parental_leave": (
        "parental", "caregiver", "maternity", "paternity", "new parent",
        "parents", "adoption",
    ),
    "notice_period": (
        "resign", "resignation", "notice period", "quit", "leaving the company",
        "what notice",
    ),
    "salary_band": ("salary", "pay band", "compensation", "band for"),
    "annual_leave": ("annual leave", "holiday", "vacation", "carry over", "carry-over"),
    "home_office": ("home office", "laptop", "allowance", "expense", "desk", "chair"),
    "working_hours": ("core hours", "remote", "working hours", "abroad", "another country"),
}

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_TENURE = re.compile(
    r"\b(\d+(?:\.\d+)?|" + "|".join(_NUMBER_WORDS) + r")\s+years?(?:'|’)?\b", re.I
)
_COMPARATIVE = re.compile(
    r"\bwhich office\b|\blongest\b|\bshortest\b|\bhighest\b|\blowest\b|"
    r"\bcompare\b|\bacross (?:all )?offices\b|\bmost generous\b",
    re.I,
)


def _positions(text: str, phrase: str) -> list[int]:
    return [m.start() for m in re.finditer(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", text)]


def offices_in(text: str, aliases: bool = True) -> list[str]:
    """Canonical office names mentioned in ``text``, in order of first mention.

    With ``aliases=False`` only the canonical name counts, so "the Netherlands"
    links to nothing. That single switch is the GR1 demonstration.
    """
    t = text.lower()
    found: list[tuple[int, str]] = []
    for office, names in OFFICES.items():
        names = names if aliases else names[:1]
        hits = [p for n in names for p in _positions(t, n)]
        if hits:
            found.append((min(hits), office))
    return [o for _, o in sorted(found)]


def office_mentions(text: str, aliases: bool = True) -> list[tuple[int, str]]:
    """Every (position, office) mention, in order. Used by the extractor to
    track which office a table row sits under."""
    t = text.lower()
    out: list[tuple[int, str]] = []
    for office, names in OFFICES.items():
        for n in names if aliases else names[:1]:
            out.extend((p, office) for p in _positions(t, n))
    return sorted(out)


def topics_in(text: str) -> list[str]:
    """Topics a question touches, in order of first mention."""
    t = text.lower()
    found: list[tuple[int, str]] = []
    for topic, phrases in TOPICS.items():
        hits = [p for ph in phrases for p in _positions(t, ph)]
        if hits:
            found.append((min(hits), topic))
    return [tp for _, tp in sorted(found)]


def role_in(text: str) -> str | None:
    t = text.lower()
    if "primary" in t:
        return "primary"
    if "secondary" in t:
        return "secondary"
    return None


def tenure_in(text: str) -> float | None:
    m = _TENURE.search(text)
    if not m:
        return None
    raw = m.group(1).lower()
    return float(_NUMBER_WORDS.get(raw, raw))


def is_comparative(text: str) -> bool:
    return bool(_COMPARATIVE.search(text))


def tenure_band(years: float) -> str:
    """The notice-period column a tenure falls into."""
    if years < 2:
        return "Under 2 years' service"
    if years <= 5:
        return "2 to 5 years"
    return "Over 5 years"


def level_in(text: str) -> str | None:
    t = text.lower()
    if "staff engineer" in t:
        return "E5"
    if "senior engineer" in t:
        return "E4"
    if "engineer" in t:
        return "E3"
    return None


# ---------------------------------------------------------------------------
# Markdown tables
# ---------------------------------------------------------------------------
_ROW_BREAK = re.compile(r"\|\s+\|")


def table_rows(text: str) -> list[tuple[int, list[str]]]:
    """Every pipe-delimited row in ``text`` as (position, cells).

    Works on structural chunks, where rows are lines, and on fixed-size chunks,
    where the newlines are gone and rows run together as ``| a | b | | c | d |``.
    Separator rows (``| --- |``) are skipped. Prose that runs into the first or
    last cell of a flattened table is trimmed off.
    """
    flat = text.replace("\n", " ")
    out: list[tuple[int, list[str]]] = []
    start = 0
    for seg in _ROW_BREAK.split(flat):
        pos = flat.find(seg, start)
        start = pos + len(seg)
        if "|" not in seg:
            continue
        cells = [c.strip() for c in seg.split("|")]
        # Leading prose (a heading, a sentence) sits before the first pipe.
        if cells and (len(cells[0].split()) > 4 or cells[0].startswith("#") or cells[0] == ""):
            cells = cells[1:]
        # Trailing prose sits after the last pipe.
        if cells and (cells[-1] == "" or len(cells[-1].split()) > 6):
            cells = cells[:-1]
        if len(cells) < 2 or all(set(c) <= set("-: ") for c in cells):
            continue
        out.append((pos, cells))
    return out
