"""Day 2: the traces people read, and the sheets where their judgments live.

Every Day 2 command reads the same list of traces, in the same order:

    the six Day 1 answers     data/traces/day1.jsonl, questions in day1_seed.json
    the synthetic answers     data/traces/day2_synthetic.jsonl, questions in
                              data/queries/day2_synthetic.json

and one annotation sheet per person:

    data/annotations/day2_sheet.csv          yours (the final say)
    data/annotations/day2_sheet_<name>.csv   anyone else who labels the same traces

The reference run (--offline) keeps its own copies of everything Day 2
writes, with "_reference" in the name. Checking the slide figures offline
therefore never touches the questions you labelled live, or your sheets.

A sheet is a CSV you can also fill in VS Code or Excel. Each row holds one
answer and what a person wrote about it. A label judges one answer, so each
row also carries a fingerprint of that answer (answer_hash): when the answer
to a question changes, the old note moves to the bottom of the sheet as
"earlier answer" and the new answer waits for a fresh judgment. Nothing a
person wrote is ever deleted by a script.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from tracing import read_jsonl
from tracing.schema import Trace

from helix import DEFAULT

ROOT = Path(__file__).resolve().parents[2]
REF = "_reference" if DEFAULT.offline else ""
DAY1_TRACES = ROOT / "data" / "traces" / "day1.jsonl"
DAY1_SUITE = ROOT / "data" / "queries" / "day1_seed.json"
SYN_TRACES = ROOT / "data" / "traces" / f"day2_synthetic{REF}.jsonl"
SYN_SUITE = ROOT / "data" / "queries" / f"day2_synthetic{REF}.json"
ANNOTATIONS = ROOT / "data" / "annotations"
SHEET = ANNOTATIONS / f"day2_sheet{REF}.csv"
GOLDEN = ANNOTATIONS / f"day2_golden{REF}.csv"
LABELS = ANNOTATIONS / f"day2_labels{REF}.csv"
COLUMNS = ["id", "source", "asker", "query", "answer", "cited", "acceptable", "open_code", "labels",
           "answer_hash"]


@dataclass
class Item:
    """One trace to read, with the suite entry that says what should happen."""

    id: str
    source: str          # "day1" or "synthetic"
    trace: Trace
    suite: dict

    @property
    def query(self) -> str:
        return self.trace.query

    @property
    def asker(self) -> str:
        return f"{self.trace.user_role}, {self.trace.user_office.title()}"


def load_suite(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["queries"] if isinstance(data, dict) else data


def load_items(include_synthetic: bool = True) -> list[Item]:
    """Day 1 traces first, then the synthetic ones, each paired with its suite entry."""
    items: list[Item] = []
    for traces_path, suite_path, source in ((DAY1_TRACES, DAY1_SUITE, "day1"),
                                            (SYN_TRACES, SYN_SUITE, "synthetic")):
        if source == "synthetic" and not include_synthetic:
            break
        if not traces_path.exists():
            continue
        suite = {q["text"]: q for q in load_suite(suite_path)}
        for i, t in enumerate(read_jsonl(traces_path), 1):
            q = suite.get(t.query, {})
            items.append(Item(q.get("id") or f"{source[0]}{i:02d}", source, t, q))
    return items


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------
def sheet_path(annotator: str | None = None) -> Path:
    """Your sheet, or a second person's: day2_sheet_<name>.csv."""
    if not annotator:
        return SHEET
    slug = re.sub(r"[^a-z0-9]+", "-", annotator.lower()).strip("-") or "b"
    return ANNOTATIONS / f"day2_sheet{REF}_{slug}.csv"


def other_sheets() -> dict[str, Path]:
    """Every second annotator's sheet, by name."""
    prefix = f"day2_sheet{REF}_"
    found = {}
    for p in sorted(ANNOTATIONS.glob(f"{prefix}*.csv")):
        name = p.stem.removeprefix(prefix)
        if REF or not name.startswith("reference"):
            found[name] = p
    return found


#: Rows kept in a sheet only so that nothing a person wrote is lost: a note
#: on a trace that no longer exists, or on an answer that has since changed.
RETIRED = ("no trace", "earlier answer")


def answer_hash(answer: str) -> str:
    """A short fingerprint of an answer, so a label stays tied to the answer it judged.

    Whitespace is ignored. The leading "h" keeps spreadsheet programs from
    reading the value as a number.
    """
    return "h" + hashlib.sha256(" ".join(answer.split()).encode()).hexdigest()[:10]


def read_sheet_all(path: Path) -> list[dict]:
    """Every row of a sheet, the retired ones included."""
    return _rows(path)


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_sheet(path: Path = SHEET) -> dict[str, dict]:
    """The notes on the current rows of a sheet, keyed by question text."""
    return {r["query"]: r for r in _rows(path) if r.get("source") not in RETIRED}


def notes_for(traces: list[Trace], path: Path = SHEET) -> dict[str, dict]:
    """The notes that belong to these exact answers: same question, same answer.

    A label judges one answer. When the answer to a question has changed
    since (Day 1 run again, another model), the old label says nothing about
    the new answer, so it is left out here.
    """
    now = {t.query: answer_hash(t.answer) for t in traces}
    return {q: r for q, r in read_sheet(path).items()
            if q in now and has_input(r) and (r.get("answer_hash") or now[q]) == now[q]}


def has_input(row: dict) -> bool:
    return any((row.get(k) or "").strip() for k in ("acceptable", "open_code", "labels"))


def write_sheet(path: Path, items: list[Item], notes: dict[str, dict]) -> Path:
    """One row per trace, with whatever a person has written about that answer.

    Nothing a person wrote is deleted. A note on a trace that is no longer in
    the list (after `day2 --new`, say) moves to the bottom as "no trace"; a
    note on an answer that has since changed moves there as "earlier answer",
    and the new answer gets an empty row to judge.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = [r for r in _rows(path) if r.get("source") in RETIRED and has_input(r)]
    now = {it.query: answer_hash(it.trace.answer) for it in items}
    rows = []
    for it in items:
        old = notes.get(it.query, {})
        if has_input(old) and (old.get("answer_hash") or now[it.query]) != now[it.query]:
            keep.append({**old, "query": it.query, "source": "earlier answer"})
            old = {}
        rows.append({"id": it.id, "source": it.source, "asker": it.asker, "query": it.query,
                     "answer": it.trace.answer, "cited": "; ".join(it.trace.citations),
                     "acceptable": old.get("acceptable", ""), "open_code": old.get("open_code", ""),
                     "labels": old.get("labels", ""), "answer_hash": now[it.query]})
    for query, row in notes.items():
        if query not in now and has_input(row) and row.get("source") not in RETIRED:
            keep.append({**row, "query": query, "source": "no trace"})
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for row in [*rows, *keep]:
            w.writerow({k: row.get(k, "") for k in COLUMNS})
    return path


def to_bool(raw: str) -> bool:
    return (raw or "").strip().lower() in {"1", "y", "yes", "true", "ok", "pass"}


def labels_of(row: dict) -> set[str]:
    return {x.strip() for x in (row.get("labels") or "").split(",") if x.strip()}
