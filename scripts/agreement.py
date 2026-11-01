"""Day 2 Exercise 5.A, Module 05: do two people agree, and what goes into the golden set.

Two people label the same traces without seeing each other's sheet. This
compares the two sheets and turns them into the golden set that Module 06
measures every evaluator against.

  STEP 1  the two sheets, and the traces both people labelled
  STEP 2  the verdict: percent agreement, chance agreement, Cohen's kappa
  STEP 3  each failure mode, the same way
  STEP 4  the alignment session: the disagreements, one failure mode at a
          time, with the questions that turn each one into a rubric sentence
  STEP 5  one person decides, with a reason; the golden set, split into
          train, dev and test

Run:  python run.py agreement                  your sheet against the other one
      python run.py agreement --with sam       if more than one other sheet exists
      python run.py agreement --adjudicate     decide each disagreement, and say why
      python run.py agreement --final sam      the other person has the final say

Round 2. After the alignment session, both people label fresh traces with
the revised rubric (`python run.py day2 --n 30` adds questions). The next
run of this command reports kappa on those fresh traces separately: that is
the number that says whether the rubric now works. Relabelling the traces
you discussed measures the discussion, not the rubric.

Three or more annotators: with several other sheets and no --with, this
prints Fleiss' kappa across everyone, then asks for one partner to compare
side by side.

Your sheet is data/annotations/day2_sheet.csv; the other person's was written
by `python run.py day2 --annotate --annotator NAME`. Until both exist, the
example annotators A and B of the reference run stand in for the reference
run's questions. Nothing here calls a model, and no one should ask a model
for help while labelling: two people who lean on the same model agree with
its habits, not with each other.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from evals.day2_agreement import (  # noqa: E402
    JUDGE_READY,
    GoldenRow,
    Label,
    assign_splits,
    disagreements,
    fleiss_kappa,
    kappa,
    landis_koch,
    next_round,
    per_mode,
    read_golden,
    reading,
    write_golden,
)
from evals.day2_examples import ANNOTATORS, EXAMPLES, MODES  # noqa: E402
from evals.day2_sheets import (  # noqa: E402
    GOLDEN,
    SHEET,
    Item,
    labels_of,
    load_items,
    notes_for,
    other_sheets,
    to_bool,
)
from helix import DEFAULT  # noqa: E402
from helix.console import banner  # noqa: E402


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def labels_from_sheet(items: list[Item], path: Path) -> dict[str, Label]:
    """A person's labels on these exact answers (a label on a changed answer is left out)."""
    notes = notes_for([it.trace for it in items], path)
    return {q: Label(q, to_bool(r["acceptable"]), r.get("open_code", ""), labels_of(r))
            for q, r in notes.items() if (r.get("acceptable") or "").strip()}


def labels_from_example(items: list[Item], who: str) -> dict[str, Label]:
    ex = EXAMPLES[who]
    return {it.query: Label(it.query, ex[it.query][0], ex[it.query][1], set(ex[it.query][2]))
            for it in items if it.query in ex}


def verdict(lab: Label) -> str:
    return "pass" if lab.acceptable else "FAIL"


def pct(x: float | None) -> str:
    return "  -  " if x is None else f"{x:.0%}"


def num(x: float | None) -> str:
    return "  -  " if x is None else f"{x:+.2f}"


def wrap(text: str, indent: int = 14, width: int = 78) -> str:
    return textwrap.fill(text, width=width, initial_indent="", subsequent_indent=" " * indent)


def ask_final(x: Label, y: Label, a_name: str, b_name: str) -> tuple[bool, set[str], str, str] | None:
    raw = input(f"  final label? [1] {a_name}'s  [2] {b_name}'s  [3] a new one  [q] stop: ").strip().lower()
    if raw == "q":
        return None
    if raw == "2":
        ok, labs, who = y.acceptable, set(y.labels), b_name
    elif raw == "3":
        ok = to_bool(input("  acceptable? [y/n] "))
        labs = {s.strip() for s in input("  labels (comma-separated): ").split(",") if s.strip()}
        who = "adjudicated"
    else:
        ok, labs, who = x.acceptable, set(x.labels), a_name
    why = input("  why, in one line (it goes into the rubric): ").strip()
    return ok, labs, who, why


def kappa_line(k: dict) -> str:
    return (f"{num(k['kappa'])}   Landis and Koch: {landis_koch(k['kappa'])} · "
            f"McHugh: {reading(k['kappa'])}")


def at_issue(x: Label, y: Label) -> list[str]:
    """The failure modes a disagreement is about, rarest reading first."""
    if x.acceptable != y.acceptable:
        return sorted(x.labels if not x.acceptable else y.labels) or ["(no mode named)"]
    return sorted(x.labels ^ y.labels)


def several(items: list[Item], a: dict[str, Label], others: dict[str, Path]) -> int:
    """Fleiss' kappa for you and every other sheet, on the traces all of you labelled."""
    sheets = {"you": a, **{name: labels_from_sheet(items, path) for name, path in others.items()}}
    shared = [it for it in items if all(it.query in lab for lab in sheets.values())]
    print(f"{len(sheets)} annotators: {', '.join(sheets)}. Traces labelled by all: {len(shared)}.")
    if len(shared) < 2:
        print("Too few to compare. Label the same traces in every sheet.")
        return 0
    verdicts = [[not lab[it.query].acceptable for lab in sheets.values()] for it in shared]
    fk = fleiss_kappa(verdicts)
    print(f"\n  Fleiss' kappa on the verdict   {num(fk)}   Landis and Koch: {landis_koch(fk)}")
    modes = sorted({m for lab in sheets.values() for it in shared for m in lab[it.query].labels})
    for m in modes:
        mk = fleiss_kappa([[m in lab[it.query].labels for lab in sheets.values()] for it in shared])
        print(f"  {m:<28} {num(mk)}   {landis_koch(mk)}")
    print("\nFleiss' kappa is Cohen's kappa for three or more people. For the")
    print("disagreements side by side and the golden set, compare one pair at a time:")
    print(f"  python run.py agreement --with {next(iter(others))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with", dest="other", default="", help="the other annotator's name")
    ap.add_argument("--final", default="", help="who decides disagreements (default: you)")
    ap.add_argument("--adjudicate", action="store_true", help="decide each disagreement in the terminal, with a reason")
    ap.add_argument("--seed", type=int, default=0, help="seed for the train/dev/test split")
    args = ap.parse_args(argv)

    items = load_items()
    if not items:
        print("run `python run.py day1` and `python run.py day2` first")
        return 1
    banner(DEFAULT)

    rule("STEP 1  Two people, the same traces, labelled without seeing each other")
    a_name, a = "you", labels_from_sheet(items, SHEET)
    others = other_sheets()
    if args.other:
        key = args.other.lower()
        match = {k: p for k, p in others.items() if k == key or k.startswith(key)}
        if not match:
            print(f"No sheet for {args.other}. Sheets found: {', '.join(others) or 'none'}")
            return 1
        others = match
    b_name, b = "", {}
    if len(others) > 1:
        if a:
            return several(items, a, others)
        print(f"Several other sheets ({', '.join(others)}): choose one with --with NAME.")
        return 1
    if others:
        b_name, path = next(iter(others.items()))
        b = labels_from_sheet(items, path)
    example = not a or not b
    if example:
        if a and not b:
            print(f"Your sheet has {len(a)} labelled traces, and no second sheet exists yet. Ask a")
            print("partner to label the same traces, without seeing yours:")
            print("  python run.py day2 --annotate --annotator NAME\n")
        a_name, a = "A", labels_from_example(items, "A")
        b_name, b = "B", labels_from_example(items, "B")
        if not a:
            print("Two sheets with labels are needed: yours, and a second person's from")
            print("  python run.py day2 --annotate --annotator NAME")
            print("The example annotators label only the reference run's questions, which")
            print("are not in your traces. Label first, then run this again.")
            return 0
        print("Until two people have labelled the same traces, two example annotators")
        print("stand in, on the reference run's questions:")
        for who in ("A", "B"):
            print(f"  {who}  {ANNOTATORS[who]}")
        if not DEFAULT.offline:
            print("Their labels describe the reference answers, not your live ones.")
    else:
        print(f"  you    {SHEET.relative_to(ROOT)}")
        print(f"  {b_name:<6} {others[b_name].relative_to(ROOT)}")
    shared = [it for it in items if it.query in a and it.query in b]
    print(f"\nTraces labelled by both: {len(shared)} of {len(items)}.")
    if len(shared) < 2:
        print("Too few to compare. Label more traces in both sheets.")
        return 0
    la = [a[it.query] for it in shared]
    lb = [b[it.query] for it in shared]
    ids = {it.query: it.id for it in items}
    answers = {it.query: it.trace.answer for it in items}
    # Round 2: traces labelled after the first golden set was written. They
    # were never discussed, so they test the revised rubric honestly.
    previous = {r.query: r for r in read_golden(GOLDEN)
                if r.query in answers and r.answer == answers[r.query]}
    fresh = [i for i, it in enumerate(shared) if previous and it.query not in previous]

    rule("STEP 2  The verdict: do they agree on pass and fail?")
    k = kappa([not x.acceptable for x in la], [not y.acceptable for y in lb])
    print(f"{'':<14}{b_name + ' FAIL':>12}{b_name + ' pass':>12}")
    print(f"{a_name + ' FAIL':<14}{k['both']:>12}{k['only_a']:>12}")
    print(f"{a_name + ' pass':<14}{k['only_b']:>12}{k['neither']:>12}\n")
    print(f"  percent agreement   p_o = {pct(k['p_o'])}   ({k['both'] + k['neither']} of {k['n']} traces)")
    print(f"  chance agreement    p_e = {pct(k['p_e'])}   two people guessing at these same fail rates")
    print(f"  Cohen's kappa       (p_o - p_e) / (1 - p_e) = {kappa_line(k)}")
    print("\nPercent agreement flatters a set where most traces fail: two people agree")
    print("often by chance alone. Kappa counts only the agreement above chance.")
    ready = k["kappa"] is not None and k["kappa"] >= JUDGE_READY
    print(f"Ready to validate a judge against (kappa {JUDGE_READY:.2f} or more)? "
          f"{'yes' if ready else 'not yet'}. A judge")
    print("cannot agree with these labels more reliably than the two of you agree with")
    print("each other." + ("" if ready else " Below 0.60, do not label more traces: fix the rubric first."))
    if fresh:
        kf = kappa([not la[i].acceptable for i in fresh], [not lb[i].acceptable for i in fresh])
        print(f"\nROUND 2: the revised rubric on {len(fresh)} fresh traces nobody discussed")
        print(f"  percent agreement   p_o = {pct(kf['p_o'])}   chance p_e = {pct(kf['p_e'])}")
        print(f"  Cohen's kappa       {kappa_line(kf)}")
        print("This is the number that says whether the rubric now works.")

    rule("STEP 3  Each failure mode: do they agree it is present?")
    modes = list(dict.fromkeys([*MODES, *sorted({m for x in [*la, *lb] for m in x.labels})]))
    modes = [m for m in modes if any(m in x.labels for x in [*la, *lb])]
    table = per_mode(la, lb, modes)
    print(f"{'failure mode':<20}{a_name:>5}{b_name:>5}{'both':>5}{'agree':>7}{'kappa':>7}"
          f"   Landis and Koch / McHugh")
    print("-" * 78)
    for m in sorted(modes, key=lambda m: (table[m]["kappa"] is not None, table[m]["kappa"] or 0)):
        t = table[m]
        print(f"{m:<20}{t['both'] + t['only_a']:>5}{t['both'] + t['only_b']:>5}{t['both']:>5}"
              f"{pct(t['p_o']):>7}{num(t['kappa']):>7}   {landis_koch(t['kappa'])} / {reading(t['kappa'])}")
    weak = [m for m in modes if table[m]["kappa"] is not None and table[m]["kappa"] < JUDGE_READY]
    if weak:
        print("\n" + textwrap.fill(f"Below 0.60: {', '.join(weak)}. These definitions need sharper "
                                  "words before anyone builds an evaluator for them. Read the "
                                  "disagreements first.", 78))
    if fresh:
        tf = per_mode([la[i] for i in fresh], [lb[i] for i in fresh], modes)
        print(f"\nROUND 2, the same on the {len(fresh)} fresh traces:")
        for m in modes:
            if tf[m]["kappa"] is not None or tf[m]["both"] + tf[m]["only_a"] + tf[m]["only_b"]:
                print(f"  {m:<20}{num(tf[m]['kappa']):>7}   {landis_koch(tf[m]['kappa'])}")

    rule("STEP 4  The alignment session: one failure mode at a time")
    diffs = disagreements(la, lb)
    if not diffs:
        print("None. Either the rubric is clear, or the two of you talked while labelling.")
    groups: dict[str, list[tuple[Label, Label, str]]] = {}
    for x, y, kind in diffs:
        modes_here = at_issue(x, y)
        lowest = min(modes_here, key=lambda m: (table.get(m, {}).get("kappa") is not None,
                                                table.get(m, {}).get("kappa") or 0, m))
        groups.setdefault(lowest, []).append((x, y, kind))
    order = sorted(groups, key=lambda m: (table.get(m, {}).get("kappa") is not None,
                                          table.get(m, {}).get("kappa") or 0, m))
    for m in order:
        cases = groups[m]
        verdicts_split = sum(1 for _, _, kind in cases if kind == "verdict")
        print(f"\n{m}: {len(cases)} disagreement{'s' if len(cases) != 1 else ''} "
              f"({verdicts_split} on the verdict, {len(cases) - verdicts_split} on the modes only)")
        for x, y, kind in cases:
            print(f"  {ids[x.key]:<4} {x.key[:70]}")
            print(f"    {a_name:<6} {verdict(x)}  " + wrap(f"{x.note} [{', '.join(sorted(x.labels)) or '-'}]", 17))
            print(f"    {b_name:<6} {verdict(y)}  " + wrap(f"{y.note} [{', '.join(sorted(y.labels)) or '-'}]", 17))
    split_verdicts = [(x, y) for x, y, kind in diffs if kind == "verdict"]
    if split_verdicts:
        seen = Counter(m for x, y in split_verdicts for m in (x.labels if not x.acceptable else y.labels))
        print(f"\n{len(split_verdicts)} verdicts split. The failure modes the failing person saw in them:")
        for m, c in seen.most_common():
            print(f"  {m:<20} {c}")
    if diffs:
        print("\nTake one failure mode at a time, lowest kappa first, about five minutes a case.")
        print("For each case: what did each of you read? Which words in the rubric allowed")
        print("both readings? What sentence would make the next person agree? Then fix the")
        print("rubric in one of five ways: clearer wording, a new example, a decision rule,")
        print("a split into two criteria, or a tie-break rule for a case that stays")
        print("ambiguous. For example:")
        print('  "An answer that does not say yes or no to a yes-or-no question fails."')
        print('  "A request Helix must decline fails when answered, even with useful facts."')
        print("Talk about the rubric, not about who was right, and do not vote. Write down")
        print("every change and why. Relabel the traces each sentence decides; the person")
        print("with the final say decides the rest. Then check the new rubric on fresh")
        print("traces (round 2): `python run.py day2 --n 30`, both label the new ones, run")
        print("this again.")

    rule("STEP 5  One person decides; the golden set")
    final_is_b = bool(args.final) and args.final.lower() in (b_name.lower(), "b")
    final_name = b_name if final_is_b else a_name
    print(f"Final say: {final_name}. A domain expert makes the final decision on each")
    print("disagreement and records the reason. For most small and medium teams, one")
    print("such expert, the definitive voice on quality, is the most effective approach.")
    decided: dict[str, tuple[bool, set[str], str, str]] = {}
    if args.adjudicate and sys.stdin.isatty() and diffs:
        print()
        for x, y, kind in diffs:
            print(f"{ids[x.key]:<4} {x.key[:72]}")
            print(f"  {a_name}: {verdict(x)} {', '.join(sorted(x.labels))}   {b_name}: {verdict(y)} "
                  f"{', '.join(sorted(y.labels))}")
            choice = ask_final(x, y, a_name, b_name)
            if choice is None:
                break
            decided[x.key] = choice

    final = b if final_is_b else a
    rows: list[GoldenRow] = []
    for it in items:
        if it.query not in final:
            continue
        lab = final[it.query]
        other = (a if final_is_b else b).get(it.query)
        why = ""
        if it.query in decided:
            ok, labs, who, why = decided[it.query]
        else:
            ok, labs = lab.acceptable, set(lab.labels)
            if other is None:
                who = f"{final_name} only"
            elif other.acceptable == lab.acceptable and other.labels == lab.labels:
                who = "agreed"
            else:
                who = final_name
        before = previous.get(it.query)
        if before and not why and before.decided == who:
            why = before.reason
        rows.append(GoldenRow(it.id, it.source, it.asker, it.query, it.trace.answer, ok, labs,
                              lab.note, who, reason=why,
                              round=before.round if before else next_round(list(previous.values()))))
    # A trace keeps its split once it has one: moving a test trace into train
    # would leak it into a judge's prompt. Only new traces are dealt out.
    new_rows = [r for r in rows if r.query not in previous]
    for r in rows:
        if r.query in previous:
            r.split = previous[r.query].split
    assign_splits(new_rows, seed=args.seed)
    rounds = Counter(r.round for r in rows)
    if len(rounds) > 1:
        print("\n" + ", ".join(f"round {n}: {c} traces" for n, c in sorted(rounds.items()))
              + ". Old traces keep their split; new ones are dealt out on their own.")
    print(f"\n{len(rows)} traces in the golden set: "
          + ", ".join(f"{c} {w}" for w, c in Counter(r.decided for r in rows).most_common()) + ".\n")
    print(f"{'split':<7}{'traces':>8}{'pass':>6}{'fail':>6}   what it is for")
    uses = {"train": "examples you may show a judge in its prompt",
            "dev": "where you improve a judge and choose its threshold",
            "test": "looked at once, at the end, for the numbers you report"}
    for s in ("train", "dev", "test"):
        part = [r for r in rows if r.split == s]
        print(f"{s:<7}{len(part):>8}{sum(r.acceptable for r in part):>6}"
              f"{sum(not r.acceptable for r in part):>6}   {uses[s]}")
    print("\nTrain holds one clear example of each failure mode seen three times or more,")
    print("and of the passes, for a judge's prompt. Dev and test share the rest, so every")
    print("mode seen twice reaches both. Aim for 30 to 50 passes and 30 to 50 fails in")
    print("each of dev and test; a judge measured on a handful of traces has not been")
    print("measured. Once the rubric holds, one expert can label the rest, up to the 100")
    print("to 200 labels a judged failure mode needs.")
    if example and not DEFAULT.offline:
        print("\nNo golden set written: these are the example labels, which describe the")
        print("reference answers. Label with a partner to make yours.")
        return 0
    out = write_golden(rows, GOLDEN)
    print(f"\ngolden set written to {out.relative_to(ROOT)}")
    print("Next: python run.py evaluators   (every evaluator against this set, on dev)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
