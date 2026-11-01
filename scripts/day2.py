"""Day 2 hands-on H2, Module 04: error analysis.

The Analyze phase, in this order:

  STEP 1  more traces: synthetic questions from dimensions and tuples, run
          through Helix with the asker's office and role
  STEP 2  open coding: read each trace and note the FIRST thing that went wrong
  STEP 3  axial coding: group the notes into named failure modes
  STEP 4  label every trace with those modes, count, and check saturation

Steps 2 to 4 are human work, and the script treats them that way:

  python run.py day2 --annotate   shows each trace and asks for your verdict
                                  (acceptable or not) and your note
  python run.py day2 --label      after axial coding: the failure-mode keys
                                  for each trace you failed
  (or)                            fill data/annotations/day2_sheet.csv in
                                  VS Code or Excel and run `python run.py day2`

Both stop when you type q and carry on later where you left off. Until you
have written notes, the example notes of the reference run stand in for the
reference run's questions, and the script says so. In step 3 your model
proposes clusters for your notes. That proposal is a draft: you rename,
merge and split it.

A second person (Module 05, Exercise 5.A) labels the same traces in a sheet
of their own, without seeing yours:

  python run.py day2 --annotate --annotator sam
  python run.py day2 --label --annotator sam
  python run.py agreement

The synthetic questions are written once and kept, because your labels
belong to their answers. When your models or settings change, Helix answers
the same questions again, and a note on an old answer stays at the bottom of
the sheet as "earlier answer". New questions only when you ask:

  python run.py day2 --n 30       30 questions in all: adds to the ones you have
                                  (default: 14 live, so 20 traces with Day 1's
                                  six; 24 offline)
  python run.py day2 --new        a new set (old notes stay in the sheet)

Your own system: fill the same columns (query, answer, acceptable,
open_code, labels) for your own questions and answers, then

  python run.py day2 --count my_sheet.csv     counts and saturation
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from evals.day2_error_analysis import (  # noqa: E402
    Annotation,
    failures_by,
    label_matrix,
    prevalence,
    propose_taxonomy,
    saturation,
)
from evals.day2_examples import ANNOTATORS, EXAMPLES, MODES  # noqa: E402
from evals.day2_sheets import (  # noqa: E402
    DAY1_TRACES,
    LABELS,
    SYN_SUITE,
    SYN_TRACES,
    Item,
    answer_hash,
    labels_of,
    load_items,
    load_suite,
    read_sheet,
    read_sheet_all,
    sheet_path,
    to_bool,
    write_sheet,
)
from evals.day2_synthetic import DIMENSIONS, enumerate_tuples, generate  # noqa: E402
from helix import DEFAULT, answer, build_index  # noqa: E402
from helix.console import Stream, banner, progress, traces_note  # noqa: E402
from tracing import read_jsonl, write_jsonl  # noqa: E402


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


# ---------------------------------------------------------------------------
# STEP 1: synthetic questions, written once and kept; answers kept per question
# ---------------------------------------------------------------------------
def save_questions(rows: list[dict], who: str) -> None:
    SYN_SUITE.parent.mkdir(parents=True, exist_ok=True)
    SYN_SUITE.write_text(json.dumps({
        "suite": "day2_synthetic",
        "note": ("Written by `python run.py day2` from dimension tuples. Delete any question "
                 "that is off target, then run day2 again."),
        "written_by": who,
        "dimensions": DIMENSIONS,
        "queries": rows,
    }, indent=2), encoding="utf-8")


def write_questions(n: int, existing: list[dict] | None = None) -> list[dict]:
    print("Three dimensions, from the charter's input contract and its not-in-scope list:")
    for name, values in DIMENSIONS.items():
        print(textwrap.fill(", ".join(values), 78, initial_indent=f"  {name:<9} ",
                            subsequent_indent=" " * 12))
    tuples = enumerate_tuples(seed=0)
    print(f"\n{len(tuples)} tuples: every combination, shuffled with a fixed seed. Asking a model")
    print("for questions directly returns rephrasings of the same easy question; the")
    print("tuple fixes the combination before any wording exists.\n")
    for t in tuples[:5]:
        print(f"  {t.describe()}")
    print(f"  ... {len(tuples) - 5} more\n")
    who = "the templates of the reference run" if DEFAULT.offline else DEFAULT.generation_model
    print(f"One question per tuple, written by {who}:")
    existing = existing or []
    rows = existing + generate(n, DEFAULT, seed=0, existing=existing,
                               on_row=lambda r, i, total: progress(f"{r['scenario']:<18} {r['text'][:80]}",
                                                                   i, total))
    save_questions(rows, who)
    print(f"\nwritten to {SYN_SUITE.relative_to(ROOT)}")
    return rows


def step_1_grow_the_dataset(n: int, new: bool, may_write: bool, more: int | None = None) -> None:
    rule("STEP 1  More traces: synthetic questions from dimensions and tuples")
    rows = load_suite(SYN_SUITE)
    if not may_write:
        print(f"Labelling the {len(rows)} synthetic questions that exist." if rows else
              "No synthetic questions yet. Run `python run.py day2` first.")
        return
    if new or not rows:
        print("Writing synthetic questions, because " + ("you asked for new ones (--new)." if rows
                                                         else "there are none yet.") + "\n")
        rows = write_questions(n)
    elif more and more > len(rows):
        print(f"Adding {more - len(rows)} questions to the {len(rows)} you have, from tuples not used yet.")
        print("The questions you labelled stay as they are.\n")
        rows = write_questions(more, existing=rows)
    else:
        print(f"Using the {len(rows)} synthetic questions in {SYN_SUITE.relative_to(ROOT)}. `--new` writes")
        print("a fresh set; deleting a question from the file drops it.")

    # An answer is kept while its question and the system that answered it are
    # unchanged, because the labels people wrote belong to that answer.
    old = {}
    if SYN_TRACES.exists():
        old = {t.query: t for t in read_jsonl(SYN_TRACES) if t.config_hash == DEFAULT.hash()}
    todo = [r for r in rows if r["text"] not in old]
    if not todo and len(old) == len(rows):
        print(f"Their {len(rows)} answers are from this system, so they are kept.")
        return
    why = "new questions" if not old else "your models or settings changed, or the questions did"
    print(f"\nHelix answers {len(todo)} of them ({why}), with the asker's office and role, as")
    print("login would give them. Before you label, read the questions: delete any that is")
    print("off target from the file and run day2 again. Every later step inherits them.")
    index = build_index(DEFAULT)
    done = {}
    for i, r in enumerate(todo, 1):
        t = answer(r["text"], index, DEFAULT, user_office=r["user_office"], user_role=r["user_role"])
        done[r["text"]] = t
        progress(t.answer[:80].replace("\n", " "), i, len(todo), "(declined)" if t.refused else "")
    traces = [old.get(r["text"]) or done[r["text"]] for r in rows]
    path = write_jsonl(traces, SYN_TRACES)
    print(f"\n{len(traces)} traces -> {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# STEP 2: open coding, in the terminal or in the sheet
# ---------------------------------------------------------------------------
def show(it: Item, position: int, total: int) -> None:
    t = it.trace
    expected = {True: "decline", False: "answer"}.get(it.suite.get("should_refuse"), "not stated")
    print("-" * 78)
    print(f"[{position}/{total}]  {it.id}  asked by {it.asker}   the suite expects Helix to {expected}")
    print(f"question  {t.query}")
    print(f"answer    {t.answer}")
    print(f"cited     {'; '.join(t.citations) or 'nothing'}")
    for c in t.retrieved:
        if not c.dropped_at:
            print(f"  read    {c.cite()} {c.access_label:<7} {c.text[:58].replace(chr(10), ' ')}...")


def annotate_live(items: list[Item], notes: dict[str, dict], path: Path, redo: bool) -> None:
    """Verdict and first-failure note for each trace, saved after every answer."""
    todo = [it for it in items if redo or not notes.get(it.query, {}).get("acceptable", "").strip()]
    if not todo:
        print("Every trace already has a verdict. --redo goes through them again.\n")
        return
    print(f"{len(todo)} traces to read. For each: is the answer acceptable (y/n), and what")
    print("is the FIRST thing that went wrong? Enter keeps an earlier answer, q stops.\n")
    for i, it in enumerate(todo, 1):
        old = notes.get(it.query, {})
        show(it, i, len(todo))
        ok = input(f"acceptable? [y/n]{' (was ' + old['acceptable'] + ')' if old.get('acceptable') else ''} ").strip()
        if ok.lower() == "q":
            break
        note = input("first failure, in your words (Enter if none): ").strip()
        if note.lower() == "q":
            break
        notes[it.query] = {**old, "acceptable": ok or old.get("acceptable", ""),
                           "open_code": note or old.get("open_code", ""),
                           "answer_hash": answer_hash(it.trace.answer)}
        write_sheet(path, items, notes)
    print()


def label_live(items: list[Item], notes: dict[str, dict], path: Path, modes: dict[str, str]) -> None:
    """After axial coding: the failure-mode keys for every trace judged unacceptable."""
    failing = [it for it in items
               if notes.get(it.query, {}).get("acceptable", "").strip()
               and not to_bool(notes[it.query]["acceptable"])]
    if not failing:
        print("No failing traces with a verdict yet. Run --annotate first.\n")
        return
    keys = list(modes)
    print("Failure modes (type numbers or keys, comma-separated; a new key adds a mode;")
    print("Enter keeps the current labels; q stops):")
    for n, k in enumerate(keys, 1):
        print(f"  {n:>2}  {k:<18} {modes[k]}")
    for i, it in enumerate(failing, 1):
        row = notes[it.query]
        show(it, i, len(failing))
        print(f"your note   {row.get('open_code') or '-'}")
        raw = input(f"labels [{row.get('labels') or 'none'}]: ").strip()
        if raw.lower() == "q":
            break
        if raw:
            chosen = []
            for part in (p.strip() for p in raw.split(",") if p.strip()):
                key = keys[int(part) - 1] if part.isdigit() and 0 < int(part) <= len(keys) else part
                if key not in modes:
                    modes[key] = "(your new mode)"
                    keys.append(key)
                chosen.append(key)
            row["labels"] = ", ".join(dict.fromkeys(chosen))
            write_sheet(path, items, notes)
    print()


def annotations_from(items: list[Item], notes: dict[str, dict]) -> list[Annotation]:
    return [Annotation(trace_id=it.trace.trace_id, query=it.query,
                       open_code=notes[it.query].get("open_code", ""),
                       acceptable=to_bool(notes[it.query]["acceptable"]),
                       labels=labels_of(notes[it.query]))
            for it in items if notes.get(it.query, {}).get("acceptable", "").strip()]


def step_2_open_coding(items: list[Item], args, path: Path) -> tuple[list[Annotation], bool]:
    rule("STEP 2  Open coding: read every trace, note the first thing that went wrong")
    # Refresh first: a note on an answer that has changed since moves to the
    # bottom of the sheet, and the new answer waits for a fresh judgment.
    write_sheet(path, items, read_sheet(path))
    notes = read_sheet(path)
    earlier = sum(1 for r in read_sheet_all(path) if r.get("source") == "earlier answer")
    if earlier:
        print(f"{earlier} notes judged answers that have changed since; they are kept at the")
        print("bottom of the sheet, and the new answers wait for your judgment.\n")
    if args.annotate and sys.stdin.isatty():
        annotate_live(items, notes, path, args.redo)
    if args.label and sys.stdin.isatty():
        modes = dict(MODES)
        for row in notes.values():
            for k in labels_of(row):
                modes.setdefault(k, "(your mode)")
        label_live(items, notes, path, modes)
    write_sheet(path, items, notes)
    print(f"{len(items)} traces: {sum(it.source == 'day1' for it in items)} from Day 1, "
          f"{sum(it.source == 'synthetic' for it in items)} synthetic. Sheet: {path.relative_to(ROOT)}\n")

    anns = annotations_from(items, notes)
    yours = bool(anns)
    if not yours and not DEFAULT.offline:
        # Live, nobody sees example notes before writing their own: they
        # describe the reference answers, and reading them first anchors you.
        print("Nobody has written notes in this sheet yet. Read every trace first:")
        print(f"  python run.py day2 --annotate{' --annotator ' + args.annotator if args.annotator else ''}")
        print("or fill the acceptable and open_code columns of the sheet, then run again.")
        return [], False
    if not yours:
        example = EXAMPLES["B" if args.annotator else "A"]
        matched = [it for it in items if it.query in example]
        print("Nobody has written notes in this sheet yet. Run with --annotate, or fill the")
        print("acceptable and open_code columns, then run again.")
        if not matched:
            print("The example notes describe the reference run's questions, and none of them")
            print("are here, so there is nothing to count yet.")
            return [], False
        who = "B" if args.annotator else "A"
        print(f"Until then, annotator {who}'s example notes stand in for the {len(matched)} reference-run")
        print(f"questions. {who} is {ANNOTATORS[who]}.\n")
        anns = [Annotation(trace_id=it.trace.trace_id, query=it.query, open_code=example[it.query][1],
                           acceptable=example[it.query][0], labels=set(example[it.query][2]))
                for it in matched]
    else:
        print(f"Your notes on {len(anns)} of {len(items)} traces.\n")
    ids = {it.query: it.id for it in items}
    for a in anns:
        print(f"  {ids.get(a.query, '?'):<4} {'ok  ' if a.acceptable else 'FAIL'} {a.open_code[:66]}")
    bad = sum(1 for a in anns if not a.acceptable)
    print(f"\n{bad} of {len(anns)} traces judged unacceptable.")
    if len(anns) < 30:
        print("Label at least 30 yourself before you trust a pattern; about 100 is a good start.")
    return anns, yours


# ---------------------------------------------------------------------------
# STEP 3: axial coding
# ---------------------------------------------------------------------------
def step_3_axial_coding(anns: list[Annotation], yours: bool) -> None:
    rule("STEP 3  Axial coding: group the notes into named failure modes")
    failing = [a for a in anns if not a.acceptable]
    if not DEFAULT.offline and failing:
        print("Your model's proposed groups for the failing notes. A draft, never a")
        print("decision: it does not know how your system works, so it merges things that")
        print("need different fixes and splits things that need the same one.\n")
        out = Stream("  ", keep_newlines=True)
        propose_taxonomy(failing, DEFAULT, on_token=out)
        out.done()
        print()
    print("The failure modes kept after reading the reference run, in the order they")
    print("first appeared. Two modes stay separate when their fixes differ: off_topic")
    print("is a retrieval problem (the right page never arrived), no_direct_answer a")
    print("generation problem (the right page arrived and the answer missed the point).\n")
    for key, name in MODES.items():
        print(f"  {key:<18} {name}")
    if yours:
        print("\nName your own modes from your notes. Then label every failing trace:")
        print("  python run.py day2 --label")
    print("\nA model may help group notes you have already written. It must not propose")
    print("the modes before anyone has read a trace: its priors run to generic names")
    print("(\"hallucination\", \"tone\") and away from what is specific to your system.")


# ---------------------------------------------------------------------------
# STEP 4: count, saturation, where failures pile up
# ---------------------------------------------------------------------------
def step_4_count(anns: list[Annotation], items: list[Item]) -> None:
    rule("STEP 4  Label, count, and check saturation")
    if not any(a.labels for a in anns):
        print("No failure-mode labels yet: run `python run.py day2 --label`, or add keys to")
        print("the labels column of the sheet (comma-separated).")
        return
    prev = prevalence(anns)
    modes = list(dict.fromkeys([*[m for m in MODES if m in prev], *prev]))
    print(f"{'failure mode':<20} {'traces':>7} {'share':>7}")
    print("-" * 36)
    for m in sorted(modes, key=lambda m: -prev.get(m, 0)):
        print(f"{m:<20} {round(prev[m] * len(anns)):>7} {prev[m]:>7.0%}")
    print(f"\nShares are of {len(anns)} labelled traces; a trace can carry several modes.")

    print("\nSaturation: new failure modes per 10 traces, in the order you read them")
    for first, last, new in saturation(anns, 10):
        print(f"  traces {first:>3} to {last:<3} {len(new)} new   {', '.join(new)}")
    print("When a block of fresh traces adds nothing new, stop reading this kind of")
    print("traffic. A new kind (another office, another topic) can start the count again.")

    syn = {it.query: it.suite for it in items if it.source == "synthetic" and it.suite}
    if syn and any(a.query in syn for a in anns):
        for dim in ("scenario", "topic"):
            counts = failures_by(anns, {q: s.get(dim, "") for q, s in syn.items()})
            print(f"\nFailing synthetic traces by {dim}:")
            for value, (bad, total) in sorted(counts.items(), key=lambda kv: -kv[1][0] / kv[1][1]):
                print(f"  {value:<22} {bad} of {total}")
    path = label_matrix(anns, modes, LABELS)
    print(f"\nlabel matrix written to {path.relative_to(ROOT)}")
    print("\nThe counts decide which failure mode gets an evaluator first. Before that,")
    print("a second person labels the same traces (Exercise 5.A):")
    print("  python run.py day2 --annotate --annotator NAME   then   python run.py agreement")


def count_own_sheet(path: Path) -> int:
    """Counts and saturation for a sheet about your own system, not Helix.

    The sheet needs a question column (query or question), acceptable,
    open_code and labels, one row per answer, in the order you read them.
    """
    import csv

    if not path.exists():
        print(f"no such file: {path}")
        return 1
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("acceptable") or "").strip()]
    anns = [Annotation(trace_id=str(i), query=r.get("query") or r.get("question") or str(i),
                       open_code=r.get("open_code", ""), acceptable=to_bool(r["acceptable"]),
                       labels=labels_of(r)) for i, r in enumerate(rows, 1)]
    rule(f"Your own system: {path.name}")
    if not anns:
        print("No rows with a verdict yet. Fill the acceptable column (y/n), open_code and labels.")
        return 0
    bad = sum(1 for a in anns if not a.acceptable)
    print(f"{len(anns)} answers judged, {bad} unacceptable.")
    if not any(a.labels for a in anns):
        print("No failure-mode labels yet: group your notes into modes, then fill the labels column.")
        return 0
    prev = prevalence(anns)
    print(f"\n{'failure mode':<24} {'answers':>8} {'share':>7}")
    print("-" * 41)
    for m, share in sorted(prev.items(), key=lambda kv: -kv[1]):
        print(f"{m:<24} {round(share * len(anns)):>8} {share:>7.0%}")
    print("\nSaturation: new failure modes per 10 answers, in the order you read them")
    for first, last, new in saturation(anns, 10):
        print(f"  answers {first:>3} to {last:<3} {len(new)} new   {', '.join(new)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotate", action="store_true", help="verdict and note for each trace, in the terminal")
    ap.add_argument("--label", action="store_true", help="failure-mode keys for each failing trace")
    ap.add_argument("--annotator", default="", help="a second person's name: their own sheet, blind to yours")
    ap.add_argument("--redo", action="store_true", help="with --annotate: go through traces you already judged")
    ap.add_argument("--new", action="store_true", help="write new synthetic questions")
    ap.add_argument("--n", type=int, default=None, help="how many synthetic questions in all (adds to yours)")
    ap.add_argument("--count", default="", help="counts and saturation for a sheet about your own system")
    args = ap.parse_args(argv)

    if args.count:
        return count_own_sheet(Path(args.count))

    if not DAY1_TRACES.exists():
        print("run `python run.py day1` first to produce data/traces/day1.jsonl")
        return 1
    banner(DEFAULT)
    traces_note(read_jsonl(DAY1_TRACES), DEFAULT)
    n = args.n or (24 if DEFAULT.offline else 14)
    # A second annotator labels the traces that exist; they never replace them.
    step_1_grow_the_dataset(n, args.new, may_write=not args.annotator, more=args.n)
    items = load_items()
    if SYN_TRACES.exists():
        traces_note(read_jsonl(SYN_TRACES), DEFAULT, made_by="python run.py day2")
    path = sheet_path(args.annotator)
    if args.annotator:
        print(f"\nAnnotator {args.annotator}: your own sheet, {path.relative_to(ROOT)}. Nothing from")
        print("anyone else's sheet is shown, so your labels are your own.")
    anns, yours = step_2_open_coding(items, args, path)
    if not anns:
        return 0
    step_3_axial_coding(anns, yours)
    if not args.annotator:
        step_4_count(anns, items)
    else:
        print("\nWhen both of you are done:  python run.py agreement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
