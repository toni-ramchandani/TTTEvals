"""Day 3, Module 08: the labels that Day 3's judges are validated against.

A judge is validated against human labels for its OWN criterion. Day 2's
golden set says whether each answer is acceptable overall, which mixes in
leaks, scope and refusals. So Day 3 asks three narrower questions about the
six Day 1 answers, one column each:

    grounded   is every factual claim supported by the context the model was
               given? (a refusal makes no claim, so it is grounded)
    correct    does the answer give the reference's substantive answer to the
               question asked?
    complete   does it contain every vital fact the question needs? (for a
               question Helix must decline, the vital fact is the refusal)

The sheet is data/annotations/day3_sheet.csv, a CSV you fill in VS Code or
Excel (y or n in each column). Like the Day 2 sheets, each row carries a
fingerprint of the answer it judged; when an answer changes, the old row
moves to the bottom as "earlier answer" and no longer counts. Nothing a
person wrote is deleted.

The reference run (--offline) never reads or writes that sheet. Its labels
are annotator A's, below, written for the reference answers only.
"""

from __future__ import annotations

import csv
from pathlib import Path

from evals.day2_sheets import ANNOTATIONS, answer_hash, read_rows, to_bool
from helix import DEFAULT
from tracing.schema import Trace

SHEET = ANNOTATIONS / "day3_sheet.csv"
COLUMNS = ["id", "query", "answer", "grounded", "correct", "complete", "note", "answer_hash"]

#: Annotator A on the six reference answers: (grounded, correct, complete, note).
REFERENCE = {
    "q1": (True, False, False, "faithful to the 2022 page; the 2024 page gives 130 days for Amsterdam"),
    "q2": (True, False, False, "faithful to the 2022 page; the answer depends on the office"),
    "q3": (True, False, False, "a refusal makes no claim; but the handbook has the Dublin row"),
    "q4": (True, False, False, "copied from the bands page it was given; it should have refused"),
    "q5": (True, True, True, "25 days, as the reference says"),
    "q6": (True, False, False, "true sentence, but it never says yes, and leaves out that a stand is covered"),
}

LABELS = ("grounded", "correct", "complete")


def _rows(path: Path) -> list[dict]:
    return read_rows(path)


def _yn(raw: str | None) -> bool | None:
    raw = (raw or "").strip()
    return None if not raw else to_bool(raw)


def labels(traces: list[Trace], suite: list[dict], path: Path = SHEET
           ) -> tuple[dict[str, dict[str, bool | None]], str]:
    """{question id: {"grounded": bool|None, "correct": bool|None}} and where they came from."""
    ids = [q["id"] for q in suite]
    if DEFAULT.offline:
        out = {i: dict(zip(LABELS, REFERENCE[i][:3])) for i in ids if i in REFERENCE}
        return out, "the reference run's labels (annotator A)"
    now = {t.query: answer_hash(t.answer) for t in traces}
    out = {}
    for r in _rows(path):
        if r.get("id") in ids and r.get("query") in now and r.get("answer_hash") == now[r["query"]]:
            got = {k: _yn(r.get(k)) for k in LABELS}
            if any(v is not None for v in got.values()):
                out[r["id"]] = got
    return out, f"your Day 3 sheet ({path.relative_to(path.parents[2])})"


def write(traces: list[Trace], suite: list[dict], path: Path = SHEET) -> Path:
    """One row per current answer, keeping whatever a person already wrote.

    A row whose answer has changed keeps its labels but moves to the bottom
    as "earlier answer", where it no longer counts.
    """
    old = _rows(path)
    current = {}
    retired = [r for r in old if r.get("id") == "earlier answer"]
    by_q = {r.get("query"): r for r in old if r.get("id") != "earlier answer"}
    for q, t in zip(suite, traces):
        h = answer_hash(t.answer)
        prev = by_q.get(t.query)
        if prev and prev.get("answer_hash") not in (h, "", None) and any(
                (prev.get(k) or "").strip() for k in (*LABELS, "note")):
            retired.append({**prev, "id": "earlier answer"})
            prev = None
        current[q["id"]] = {
            "id": q["id"], "query": t.query, "answer": " ".join(t.answer.split()),
            **{k: (prev or {}).get(k, "") for k in LABELS},
            "note": (prev or {}).get("note", ""), "answer_hash": h,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in current.values():
            w.writerow(row)
        for row in retired:
            w.writerow(row)
    return path


def missing(traces: list[Trace], suite: list[dict], path: Path = SHEET) -> list[str]:
    """Question ids whose current answer still lacks a label in any of the three columns."""
    got, _ = labels(traces, suite, path)
    return [q["id"] for q in suite if any(got.get(q["id"], {}).get(k) is None for k in LABELS)]


def acceptable_labels(suite: list[dict], traces: list[Trace]) -> tuple[list[bool], str]:
    """The overall verdict people gave each answer (Day 2), for `metrics`.

    The golden set first, then your Day 2 sheet, both only on these exact
    answers. Offline, annotator A. Otherwise the suite's expected outcomes,
    and the output says so.
    """
    from evals.day2_agreement import read_golden
    from evals.day2_sheets import GOLDEN, notes_for

    golden = {r.query: r for r in read_golden(GOLDEN)}
    if traces and all(t.query in golden and golden[t.query].answer == t.answer for t in traces):
        return [golden[t.query].acceptable for t in traces], "the Day 2 golden set"
    notes = notes_for(traces)
    if traces and all((notes.get(t.query, {}).get("acceptable") or "").strip() for t in traces):
        return [to_bool(notes[t.query]["acceptable"]) for t in traces], "your Day 2 verdicts"
    if DEFAULT.offline:
        from evals.day2_examples import EXAMPLE_A

        if all(t.query in EXAMPLE_A for t in traces):
            return [EXAMPLE_A[t.query][0] for t in traces], "the reference run's labels (annotator A)"
    return ([q.get("expected_failure") in (None, "", "none") for q in suite],
            "the suite's expected outcomes (label these answers on Day 2 to use your own)")
