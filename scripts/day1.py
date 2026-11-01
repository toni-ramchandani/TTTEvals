"""Day 1, hands-on H1: instrument a pipeline, then read its traces.

Runs the six seeded questions through Helix, writes one trace per question,
and walks through reading them, step by step:

  1. build the index
  2. look at one chunk before looking at any answer
  3. run the six questions (you are not told which ones fail)
  4. open one trace: what each span kept and what it dropped
  5. triage with the trace alone (blind)
  6. triage with ground truth
  7. your triage worksheet (written to data/annotations/)
  8. where failures are born
  9. optional: what a configuration change can and cannot fix

No evaluation library is used. You instrument before you measure.

Run:   python run.py day1                 everything, start to finish
       python run.py day1 --pause         stop after each step (press Enter)
       python run.py day1 --compare       also run step 9
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evals.day1_triage import outcome_is_suspect, report, signals, triage  # noqa: E402
from evals.taxonomy import BORN_BEFORE_GENERATION, CATALOG  # noqa: E402
from helix import DEFAULT, REPAIRED, answer, build_index  # noqa: E402
from helix.console import Stream, banner  # noqa: E402
from tracing import write_jsonl  # noqa: E402

SUITE = ROOT / "data" / "queries" / "day1_seed.json"
TRACES = ROOT / "data" / "traces" / "day1.jsonl"
WORKSHEET = ROOT / "data" / "annotations" / "day1_triage_worksheet.csv"

PAUSE = False


def step(title: str) -> None:
    if PAUSE:
        input("\n  press Enter for the next step ")
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    global PAUSE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pause", action="store_true", help="wait for Enter between steps")
    ap.add_argument("--compare", action="store_true", help="also compare against the repaired configuration")
    args = ap.parse_args(argv)
    PAUSE = args.pause and sys.stdin.isatty()

    cfg = DEFAULT
    print(f"config   {cfg.describe()}")
    banner(cfg)

    # 1 -----------------------------------------------------------------------
    step("STEP 1  Build the index")
    print("Six build-time stages run here: source, parse, chunk, enrich, embed, index.")
    print("They run once, before any question is asked. Only the index crosses into")
    print("request time, which is why re-chunking is a rebuild, not a deploy.\n")
    index = build_index(cfg)
    print(f"{len(index)} chunks from {len({c.doc_id for c in index.chunks})} documents")
    for c in index.chunks[:3]:
        print(f"\n  {c.chunk_id}  v{c.doc_version}  {c.last_modified}  access={c.access_label}")
        print(f"    {c.text[:100]}...")

    # 2 -----------------------------------------------------------------------
    step("STEP 2  Look at one chunk before you look at any answer")
    severed = next(c for c in index.chunks if c.chunk_id == "HB-PAR-2024-01")
    print(severed.text)
    print("\nThis one chunk holds Ireland's figures, then the Amsterdam heading, then")
    print("Amsterdam's figures. Nothing in it says which numbers belong to which office.")
    print("The qualifier and the fact were separated at chunking time, three stages")
    print("before generation. That is FP10, visible without running a single query.")

    # 3 -----------------------------------------------------------------------
    step("STEP 3  Run the six seeded questions")
    print("You are not told which ones fail. Read each answer and write down your guess.")
    if not cfg.offline:
        print("Your model is answering live; the answers appear as it writes them.")
    print()
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    traces = []
    for q in suite["queries"]:
        print(f"{q['id']}  {q['text']}")
        out = Stream()
        t = answer(q["text"], index, cfg, user_office=q["user_office"], user_role=q["user_role"],
                   on_token=out)
        out.done(f"cited {len(t.citations)}" + (", refused" if t.refused else ""))
        traces.append(t)
        print()
    write_jsonl(traces, TRACES)
    print(f"traces written to {TRACES.relative_to(ROOT)} (Day 2 starts from this file)")
    from toolkits.cases import cases_from_traces, show

    sara = cases_from_traces(traces[:1], suite["queries"][:1])[0]
    print("\nSara's trace, as the test case DeepEval and Ragas read (built in memory, not saved):")
    print("\n".join(show(sara, traces[0])))

    # 4 -----------------------------------------------------------------------
    step("STEP 4  Open one trace")
    print("Six spans. For each, what it KEPT and what it DROPPED. The dropped set is the")
    print("half of the trace most teams never log, and FP2 and FP3 live entirely in it.\n")
    t = traces[0]
    print(f"query    {t.query}")
    print(f"config   {t.config_hash}\n")
    for s in t.spans:
        print(f"{s.name:<18}")
        if s.kept:
            print(f"    kept     {', '.join(s.kept)}")
        if s.dropped:
            print(f"    dropped  {', '.join(s.dropped)}")
    print("\nwhat reached the model:")
    for c in t.retrieved:
        print(f"    {c.cite()}  {c.text[:70].replace(chr(10), ' ')}...")
    print(f"\nanswer   {t.answer[:150]}")

    # 5 -----------------------------------------------------------------------
    step("STEP 5  Triage with the trace alone (blind)")
    print("Four failure points can be proven from a trace alone, because provenance makes")
    print("them checkable: FP8 stale, FP9 contradictory, FP10 severed, FP11 entitlement.\n")
    for q, t in zip(suite["queries"], traces):
        found = triage(t)
        print(f"{q['id']}  {t.query[:68]}")
        print(f"    outcome   {outcome_is_suspect(t) or 'answered with a citation'}")
        for sig in signals(t):
            print(f"    signal    {sig}")
        print(f"    findings  {[f.code for f in found] or 'none from the trace alone'}")

    # 6 -----------------------------------------------------------------------
    step("STEP 6  Triage with ground truth")
    print("Now each question gets a snippet of the passage that SHOULD have answered it.")
    print("FP2 and FP3 become decidable. A trace shows where a failure could have been")
    print("born; only a judgment about the right answer says whether it was. Supplying")
    print("that judgment at scale is what Day 2 is.\n")
    evidence = {q["text"]: q["expected_evidence"] for q in suite["queries"] if q.get("expected_evidence")}
    print(report(traces, evidence))

    # 7 -----------------------------------------------------------------------
    step("STEP 7  Your triage worksheet")
    # What you already wrote in the last three columns is kept: a rerun
    # refreshes the answers, never your triage.
    yours: dict[str, dict] = {}
    if WORKSHEET.exists():
        with WORKSHEET.open(encoding="utf-8", newline="") as fh:
            yours = {r["question"]: r for r in csv.DictReader(fh)}
    hand = ["failure_point", "span_where_born", "field_that_proves_it"]
    WORKSHEET.parent.mkdir(parents=True, exist_ok=True)
    with WORKSHEET.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["query_id", "question", "answer", *hand])
        for q, t in zip(suite["queries"], traces):
            old = yours.get(q["text"], {})
            w.writerow([q["id"], q["text"], t.answer[:200], *[old.get(h, "") for h in hand]])
    kept = sum(1 for r in yours.values() if any((r.get(h) or "").strip() for h in hand))
    print(f"Written to {WORKSHEET.relative_to(ROOT)}. Open it in VS Code or Excel and fill")
    print("the last three columns by hand, from the trace, not from the detector output.")
    if kept:
        print(f"Your earlier triage on {kept} questions is kept; the answer column is new.")
    print("\nThen one line for your charter's risk register:")
    print("  Of these failure points, the ones my own logging would miss today are ...")

    # 8 -----------------------------------------------------------------------
    step("STEP 8  Where failures are born")
    for fp in CATALOG:
        mark = " " if fp.born_at == "generate" else "*"
        print(f"  {mark} {fp.code:<5} {fp.name:<30} born at {fp.born_at}")
    print(f"\n  * born before the model is called: {len(BORN_BEFORE_GENERATION)} of {len(CATALOG)}")

    # 9 -----------------------------------------------------------------------
    if args.compare:
        step("STEP 9  What a configuration change can and cannot fix")
        from evals.day4_compare import Arm, compare

        r_index = build_index(REPAIRED)
        r_traces = []
        for q in suite["queries"]:
            print(f"{q['id']}  {q['text']}")
            out = Stream()
            r_traces.append(answer(q["text"], r_index, REPAIRED, user_office=q["user_office"],
                                   user_role=q["user_role"], on_token=out))
            out.done()
        print()
        print(f"before: {DEFAULT.describe()}")
        print(f"after:  {REPAIRED.describe()}\n")
        print(compare(Arm("before", DEFAULT.hash(), traces), Arm("after", REPAIRED.hash(), r_traces), evidence))
        print("\nThe stale 2022 page is still indexed, so FP8 survives:")
        print("you cannot fix a corpus problem with a configuration change.")

    print("\nDone. Next: python run.py day2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
