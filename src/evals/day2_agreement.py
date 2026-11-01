"""Day 2, Module 05: do two people agree, and what becomes the golden set.

A label is only worth measuring an evaluator against if a second careful
person would give the same label. This module answers that question and
turns the answer into a golden set:

    agreement      percent agreement and Cohen's kappa on the pass/fail
                   verdict, and again on each failure mode
    disagreements  the traces two people labelled differently, side by side:
                   each one marks a place where the rubric is not clear yet
    adjudicate     one person decides each disagreement (the "benevolent
                   dictator"), and the rubric gets the sentence that would
                   have prevented it
    split          the golden set, split into train (examples you may show a
                   judge), dev (to improve the judge) and test (looked at once)
    round 2        after the alignment session, the revised rubric is checked
                   on fresh traces that nobody discussed: relabelling the
                   discussed ones measures the discussion, not the rubric

Cohen's kappa (Cohen 1960) corrects percent agreement for the agreement two
people would reach by chance, given how often each says "fail":

    kappa = (p_o - p_e) / (1 - p_e)

When almost everything fails, two people agree often by chance alone, so a
high percent agreement can hide a low kappa. McHugh (2012) reads a kappa
below 0.60 as inadequate agreement; the more lenient bands of Landis and
Koch (1977) call 0.41 to 0.60 "moderate". Aim for 0.60 or more on each
failure mode before its labels are used to validate a judge. With three or
more annotators, Fleiss' kappa (1971) generalises the same idea.

Kappa compares two peers. Against a golden set, where the human label is
taken as right, an evaluator is measured with TPR and TNR instead, because
the question becomes which way it is wrong (day2_evaluators.rates).

Nothing here calls a model. The functions are plain Python over labels.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Label:
    """One person's verdict on one trace."""

    key: str                 # the question text: what the sheets are keyed by
    acceptable: bool
    note: str = ""
    labels: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------
def kappa(a: list[bool], b: list[bool]) -> dict[str, object]:
    """Percent agreement, chance agreement and Cohen's kappa for two raters.

    `a` and `b` are the same items in the same order, True for "fail" (or for
    "this mode is present"). Returns the 2 x 2 counts too, so a script can
    print the table the numbers come from. Kappa is None when chance
    agreement is 1: both people gave one answer to every item, and there is
    nothing to correct for.
    """
    if len(a) != len(b):
        raise ValueError("both raters must label the same items")
    n = len(a)
    both = sum(1 for x, y in zip(a, b) if x and y)
    neither = sum(1 for x, y in zip(a, b) if not x and not y)
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if not x and y)
    if n == 0:
        return {"n": 0, "p_o": None, "p_e": None, "kappa": None,
                "both": 0, "neither": 0, "only_a": 0, "only_b": 0}
    p_o = (both + neither) / n
    pa, pb = (both + only_a) / n, (both + only_b) / n
    p_e = pa * pb + (1 - pa) * (1 - pb)
    k = None if p_e >= 1 else (p_o - p_e) / (1 - p_e)
    return {"n": n, "p_o": p_o, "p_e": p_e, "kappa": k,
            "both": both, "neither": neither, "only_a": only_a, "only_b": only_b}


#: McHugh (2012), Table 3: level of agreement for each range of kappa:
#: 0 to .20 none, .21 to .39 minimal, .40 to .59 weak, .60 to .79 moderate,
#: .80 to .90 strong, above .90 almost perfect.
MCHUGH = [(0.80, "strong"), (0.60, "moderate"), (0.40, "weak"), (0.21, "minimal")]


def reading(k: float | None) -> str:
    """McHugh's name for a kappa value. Below 0.60 she calls agreement inadequate."""
    if k is None:
        return "not defined: one answer for every item"
    if k < 0:
        return "worse than chance"
    if k > 0.90:
        return "almost perfect"
    for floor, name in MCHUGH:
        if k >= floor:
            return name
    return "none"


#: Landis and Koch (1977): the bands most often quoted. They are more lenient
#: than McHugh's: 0.41 is already "moderate".
LANDIS_KOCH = [(0.81, "almost perfect"), (0.61, "substantial"), (0.41, "moderate"),
               (0.21, "fair"), (0.0, "slight")]


def landis_koch(k: float | None) -> str:
    """Landis and Koch's name for a kappa value."""
    if k is None:
        return "not defined"
    if k < 0:
        return "poor"
    for floor, name in LANDIS_KOCH:
        if k >= floor:
            return name
    return "slight"


#: Agreement to reach before the labels are used to validate a judge. Both
#: McHugh's "moderate" and Landis and Koch's "substantial" start about here.
#: A judge measured against labels that two careful people do not share
#: cannot be measured reliably: their agreement is its ceiling.
JUDGE_READY = 0.60


def fleiss_kappa(ratings: list[list[bool]]) -> float | None:
    """Fleiss' kappa for three or more raters (Fleiss 1971).

    `ratings` has one list per item, one entry per rater (True for "fail" or
    "this mode is present"); every item needs the same number of raters, at
    least two. Like Cohen's kappa it compares the agreement observed on each
    item with the agreement the overall label shares would give by chance.
    None when there is nothing to correct for (every rating the same).
    """
    if not ratings:
        return None
    n = len(ratings[0])
    if n < 2 or any(len(r) != n for r in ratings):
        raise ValueError("every item needs the same number of raters, at least two")
    big_n = len(ratings)
    yes = [sum(1 for v in r if v) for r in ratings]
    p_yes = sum(yes) / (big_n * n)
    p_e = p_yes ** 2 + (1 - p_yes) ** 2
    p_items = [(y * y + (n - y) * (n - y) - n) / (n * (n - 1)) for y in yes]
    p_bar = sum(p_items) / big_n
    if p_e >= 1:
        return None
    return (p_bar - p_e) / (1 - p_e)


def per_mode(a: list[Label], b: list[Label], modes: list[str]) -> dict[str, dict]:
    """Kappa on each failure mode: is the mode present in this trace, yes or no?"""
    out = {}
    for m in modes:
        out[m] = kappa([m in x.labels for x in a], [m in y.labels for y in b])
    return out


def disagreements(a: list[Label], b: list[Label]) -> list[tuple[Label, Label, str]]:
    """Traces labelled differently: on the verdict, or on the failure modes."""
    out = []
    for x, y in zip(a, b):
        if x.acceptable != y.acceptable:
            out.append((x, y, "verdict"))
        elif x.labels != y.labels:
            out.append((x, y, "modes"))
    return out


# ---------------------------------------------------------------------------
# The golden set
# ---------------------------------------------------------------------------
@dataclass
class GoldenRow:
    id: str
    source: str
    asker: str
    query: str
    answer: str
    acceptable: bool
    labels: set[str]
    note: str
    decided: str             # "agreed", or who decided the disagreement
    split: str = ""
    reason: str = ""         # why, when a person decided a disagreement
    round: int = 1           # 1: the traces of the first alignment session; 2: labelled after it


#: Train holds the examples you may show a judge in its prompt; dev is where
#: you improve the judge; test is looked at once, at the end.
TRAIN_MIN = 3      # a mode or the passes need this many traces to spare one for train


def _stable(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def assign_splits(rows: list[GoldenRow], seed: int = 0) -> list[GoldenRow]:
    """Train, dev and test, so that a judge has examples and a fair test.

    1. Train first. A judge prompt needs a clear failing example of its mode
       and a clear passing one, taken from train only. So every failure mode
       seen at least three times gives train one trace (the one with the
       fewest other modes), and so do the passes. Three is the least that
       still leaves one trace each for dev and test.
    2. The rest is dealt out by its rarest label, dev and test in turn.
    3. A last pass makes sure that every mode seen at least twice has a trace
       in dev and one in test.

    With few labels, train ends near 15 to 20 percent and dev and test near
    40 to 45 percent each. The seed fixes the order, so the split is the same
    on every run.
    """
    def modes_of(r: GoldenRow) -> set[str]:
        return r.labels or {"pass"}

    counts: dict[str, int] = {}
    for r in rows:
        for m in modes_of(r):
            counts[m] = counts.get(m, 0) + 1
    for r in rows:
        r.split = ""

    left = dict(counts)          # traces of each mode still free for dev and test
    for m in sorted(counts, key=lambda m: (counts[m], m)):
        if counts[m] < TRAIN_MIN:
            continue
        mine = [r for r in rows if m in modes_of(r)]
        if any(r.split == "train" for r in mine):
            continue
        # Never take a trace that would leave any of its modes fewer than two
        # traces for dev and test.
        fit = [r for r in mine if all(left[k] - 1 >= 2 for k in modes_of(r))]
        if not fit:
            continue
        clearest = min(fit, key=lambda r: (len(modes_of(r)), _stable(r.query, seed)))
        clearest.split = "train"
        for k in modes_of(clearest):
            left[k] -= 1

    def group(r: GoldenRow) -> str:
        return min(modes_of(r), key=lambda m: (counts[m], m))

    rest = sorted((r for r in rows if r.split != "train"),
                  key=lambda r: (counts[group(r)], group(r), _stable(r.query, seed)))
    for i, r in enumerate(rest):
        r.split = ("dev", "test")[i % 2]

    # A mode seen at least twice must reach both dev and test. Where the deal
    # missed one, move a trace of that mode across, or swap it with a trace
    # in the split it needs, whichever leaves fewer gaps. Train is not touched:
    # those traces are the examples a judge needs.
    def gaps() -> list[tuple[str, str]]:
        out = []
        for m in sorted(counts, key=lambda m: (counts[m], m)):
            if counts[m] < 2:
                continue
            where = {r.split for r in rows if m in modes_of(r)}
            out += [(m, need) for need in ("dev", "test") if need not in where]
        return out

    for _ in range(50):
        missing = gaps()
        if not missing:
            break
        m, need = missing[0]
        other = "test" if need == "dev" else "dev"
        best, best_gaps = None, len(missing)
        for x in [r for r in rows if m in modes_of(r) and r.split == other]:
            x.split = need
            if len(gaps()) < best_gaps:
                best, best_gaps = ("move", x, None), len(gaps())
            for y in [r for r in rows if r.split == need and r is not x]:
                y.split = other
                if len(gaps()) < best_gaps:
                    best, best_gaps = ("swap", x, y), len(gaps())
                y.split = need
            x.split = other
        if best is None:
            break
        _, x, y = best
        x.split = need
        if y is not None:
            y.split = other
    return rows


def next_round(previous: list[GoldenRow]) -> int:
    """The round a trace joins when it first enters the golden set.

    Round 1 is the set of the first alignment session. Every trace labelled
    after that joins round 2, where the revised rubric is checked on traces
    nobody has discussed yet.
    """
    return 2 if previous else 1


GOLDEN_COLUMNS = ["id", "source", "asker", "query", "answer", "acceptable", "labels",
                  "note", "decided", "reason", "round", "split"]


def write_golden(rows: list[GoldenRow], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=GOLDEN_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({"id": r.id, "source": r.source, "asker": r.asker, "query": r.query,
                        "answer": r.answer, "acceptable": "yes" if r.acceptable else "no",
                        "labels": ", ".join(sorted(r.labels)), "note": r.note,
                        "decided": r.decided, "reason": r.reason, "round": r.round,
                        "split": r.split})
    return path


def read_golden(path: Path) -> list[GoldenRow]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return [GoldenRow(id=r["id"], source=r["source"], asker=r["asker"], query=r["query"],
                          answer=r["answer"], acceptable=r["acceptable"].strip().lower() == "yes",
                          labels={x.strip() for x in r["labels"].split(",") if x.strip()},
                          note=r["note"], decided=r["decided"], split=r["split"],
                          reason=r.get("reason") or "", round=int(r.get("round") or 1))
                for r in csv.DictReader(fh)]
