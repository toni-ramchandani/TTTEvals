"""Day 2 hands-on H3, Module 06: an evaluator for each failure mode, checked against people.

Error analysis named the failure modes and counted them. Exercise 5.A turned
two people's labels into a golden set. This turns each mode into an
evaluator with a pass condition, and measures every evaluator against the
golden set on its own mode.

  STEP 1  the modes, and the cheapest instrument for each
  STEP 2  code checks: stale evidence, entitlement, false and missed refusal
  STEP 3  one judged mode, four judges, from the same criteria:
            judge (our code): pass or fail, with Pass and Fail definitions,
              examples from the train split only, reasoning before the answer
            G-Eval (our code): the method written out in day2_evaluators.py,
              a 1 to 5 score from steps the judge writes live
            DeepEval GEval, same criteria and the same steps, in the same run
            DeepEval DAG, the same decision as yes/no questions (right scope)
  STEP 4  every evaluator against people: TPR and TNR on the dev split, and
          the G-Eval pass threshold read from the labels
  STEP 5  from the raw failure rate to a corrected one, with a 95 percent
          bootstrap interval

The human labels come from the golden set (`python run.py agreement`), dev
split unless you choose another; else from your sheet; else the example
labels of the reference run. Train traces are never judged: they are the
examples in the judge's prompt. Every judge call is live, on the models you
name. Nothing is saved: the G-Eval steps are written by your judge on every
run, printed, and used by both G-Eval judges in that run.

Run:  python run.py evaluators                        dev split, judged mode: right scope
      python run.py evaluators --mode no_direct_answer
      python run.py evaluators --split test           once, at the end
      python run.py evaluators --repeat 3             does the judge's label hold across runs?
      python run.py evaluators --judge openai/gpt-5.4-mini   answers local, judge in the cloud
      python run.py evaluators --offline              code checks, and the judge prompt it would send
DeepEval runs only in the toolkit environment (requirements-toolkits.txt).
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from evals.day2_agreement import read_golden  # noqa: E402
from evals.day2_evaluators import (  # noqa: E402
    CODE_CHECKS,
    JUDGED,
    SCOPE_MODE,
    binary_judge,
    bootstrap_interval,
    corrected_failure_rate,
    geval_score,
    geval_steps,
    judge_prompt,
    pick_examples,
    rates,
    threshold_sweep,
)
from evals.day2_examples import EXAMPLES, MODES  # noqa: E402
from evals.day2_sheets import (  # noqa: E402
    DAY1_TRACES,
    GOLDEN,
    SHEET,
    SYN_TRACES,
    Item,
    labels_of,
    load_items,
    notes_for,
    to_bool,
)
from helix import DEFAULT  # noqa: E402
from helix.console import banner, progress, traces_note  # noqa: E402
from toolkits import env  # noqa: E402
from toolkits.cases import cases_from_traces  # noqa: E402
from tracing import read_jsonl  # noqa: E402

INSTRUMENT = {
    "stale_policy": "code: is every cited page current?",
    "entitlement_leak": "code: did a page above the asker's access reach the model?",
    "false_refusal": "code: declined, while the suite expects an answer?",
    "missed_refusal": "code: answered, while the suite expects Helix to decline?",
    "wrong_scope_row": "judge: is the figure the one for this asker's office and role?",
    "no_direct_answer": "judge: does the answer answer the question asked?",
}


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def mark(passed: bool | None) -> str:
    return {True: "pass", False: "FAIL", None: "  ? "}[passed]


def pct(x: float | None) -> str:
    return "  - " if x is None else f"{x:.0%}"


@dataclass
class Truth:
    acceptable: bool
    labels: set[str]
    split: str           # "train", "dev", "test", or "" when there is no golden set
    note: str


def human_truth(items: list[Item]) -> tuple[dict[str, Truth], str]:
    """{question: Truth} for every labelled trace, and where the labels came from."""
    current = {it.query: it for it in items}
    golden = read_golden(GOLDEN)
    if golden:
        fresh = [r for r in golden if r.query in current and r.answer == current[r.query].trace.answer]
        stale = len(golden) - len(fresh)
        source = "the golden set"
        if stale:
            source += (f"; {stale} labels left out because their answers changed since "
                       "(label again, then rerun agreement)")
        return {r.query: Truth(r.acceptable, r.labels, r.split, r.note) for r in fresh}, source
    notes = notes_for([it.trace for it in items], SHEET)
    mine = {q: Truth(to_bool(r["acceptable"]), labels_of(r), "", r.get("open_code", ""))
            for q, r in notes.items() if (r.get("acceptable") or "").strip()}
    if mine:
        return mine, "your sheet (no golden set yet, so no train, dev or test split)"
    ex = {q: Truth(ok, set(labs), "", note) for q, (ok, note, labs) in EXAMPLES["A"].items() if q in current}
    note = "" if DEFAULT.offline else "; they describe the reference answers, not your live ones"
    return ex, f"the example labels of annotator A (no golden set yet{note})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default=SCOPE_MODE, choices=sorted(JUDGED), help="the judged failure mode")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"],
                    help="which golden-set split to measure on (default dev)")
    ap.add_argument("--samples", type=int, default=3, help="G-Eval score samples per trace (default 3)")
    ap.add_argument("--shots", type=int, default=1,
                    help="failing and passing train examples in the judge's prompt (default 1 each)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run the binary judge N times at temperature 0.7 to see whether labels flip")
    args = ap.parse_args(argv)

    if not DAY1_TRACES.exists():
        print("run `python run.py day1` first to produce data/traces/day1.jsonl")
        return 1
    cfg = DEFAULT
    banner(cfg)
    traces_note(read_jsonl(DAY1_TRACES), cfg)
    if SYN_TRACES.exists():
        traces_note(read_jsonl(SYN_TRACES), cfg, made_by="python run.py day2")
    items = load_items()
    cases = cases_from_traces([it.trace for it in items], [it.suite for it in items])
    truth, source = human_truth(items)
    has_splits = any(t.split for t in truth.values())
    train_q = {q for q, t in truth.items() if t.split == "train"}
    if has_splits and args.split != "all":
        measured_q = {q for q, t in truth.items() if t.split == args.split}
    else:
        measured_q = {q for q in truth if q not in train_q}
    if args.split == "test":
        print("\nTEST SPLIT. Look at these numbers once, at the end, and report them. If you")
        print("change a judge after reading them, they become dev numbers.")

    rule("STEP 1  One evaluator per failure mode, cheapest instrument first")
    where = f"{args.split} split, " if has_splits else ""
    print(f"Human labels: {source}; measured on {where}{len(measured_q)} traces.\n")
    every = [truth[q] for q in measured_q]
    for mode, how in INSTRUMENT.items():
        share = sum(1 for t in every if mode in t.labels) / max(1, len(every))
        print(f"  {mode:<18} {share:>5.0%}  {how}")
    print("\nCode wherever the trace or the suite already records the fact. A judge only")
    print("where deciding needs reading, because a judge costs calls and must be checked")
    print("against people before anyone trusts it.")
    rest = [m for m in MODES if m not in INSTRUMENT]
    if rest:
        print(f"No evaluator yet: {', '.join(rest)}. Write criteria for one of them in")
        print("day2_evaluators.JUDGED and DEFINITIONS, and run it with --mode.")

    rule("STEP 2  Code checks: no model, the same answer every run")
    verdicts: dict[str, tuple[str, list[bool | None]]] = {}
    why: dict[str, list[str]] = {}
    for check in CODE_CHECKS:
        rs = [check(c) for c in cases]
        verdicts[rs[0].evaluator] = (rs[0].mode, [r.passed for r in rs])
        why[rs[0].evaluator] = [r.reason for r in rs]
        failing = [(it, r) for it, r in zip(items, rs) if r.passed is False]
        print(f"{rs[0].evaluator:<26} fails {len(failing)} of {len(rs)} traces")
        for it, r in failing[:8]:
            print(f"    {it.id:<4} {r.reason}")
        if len(failing) > 8:
            print(f"    ... {len(failing) - 8} more")

    scores: list[float | None] = [None] * len(items)
    name, criteria = JUDGED[args.mode]
    rule(f"STEP 3  One judged mode, four judges: {name} ({args.mode})")
    print("Criteria, shared by every judge:")
    print(textwrap.fill(criteria, 78, initial_indent="  ", subsequent_indent="  ") + "\n")
    index = {it.query: i for i, it in enumerate(items)}
    train = [(items[index[q]].id, cases[index[q]], truth[q].acceptable, truth[q].labels, truth[q].note)
             for q in sorted(train_q, key=lambda q: index[q])]
    examples = pick_examples(args.mode, train, shots=args.shots)
    if examples:
        print("Examples in the judge's prompt, from the train split only:")
        for ex in examples:
            print(f"  {ex.id:<4} {'Pass' if ex.passed else 'Fail':<5} {ex.reasoning[:66]}")
        if not any(not ex.passed for ex in examples) or not any(ex.passed for ex in examples):
            print("  Train has no " + ("failing" if all(ex.passed for ex in examples) else "passing")
                  + " example of this mode. Label more traces, then run agreement again.")
    else:
        print("No train split yet, so the judge has no examples: the most common mistake")
        print("with a judge. Run `python run.py agreement` first to make the golden set.")
    judged_idx = [i for i, it in enumerate(items) if it.query not in train_q]
    assert not {ex.id for ex in examples} & {items[i].id for i in judged_idx}, "an example would be judged"
    measured_idx = [i for i in judged_idx if items[i].query in measured_q]
    if cfg.offline:
        if measured_idx:
            first = measured_idx[0]
            print(f"\nThe prompt the binary judge receives, here for {items[first].id}:\n")
            for line in judge_prompt(cases[first], args.mode, examples).splitlines():
                for part in textwrap.wrap(line, 74, subsequent_indent="  ") or [""]:
                    print("  | " + part)
        print("\nJudges need a live model, and nothing is stored to show instead. Run")
        print("without --offline to judge with your models.")
    else:
        print(f"\nJudging {len(judged_idx)} traces (every trace except the train examples):\n")
        ours: list[bool | None] = [None] * len(items)
        for n, i in enumerate(judged_idx, 1):
            r = binary_judge(cases[i], args.mode, examples, cfg)
            ours[i] = r.passed
            why.setdefault("judge (our code)", [""] * len(items))[i] = r.reason
            progress(f"judge (our code)     {items[i].id:<4} {mark(r.passed)}  {r.reason[:50]}", n, len(judged_idx))
        verdicts["judge (our code)"] = (args.mode, ours)
        if args.repeat > 1:
            flips = []
            for n, i in enumerate(measured_idx, 1):
                seen = {ours[i]}
                for _ in range(args.repeat - 1):
                    seen.add(binary_judge(cases[i], args.mode, examples, cfg, temperature=0.7).passed)
                if len(seen) > 1:
                    flips.append(items[i].id)
                progress(f"repeat x{args.repeat}            {items[i].id:<4} {'flips' if len(seen) > 1 else 'holds'}",
                         n, len(measured_idx))
            print(f"\n  The label flips on {len(flips)} of {len(measured_idx)} traces across {args.repeat} runs"
                  + (f": {', '.join(flips)}." if flips else "."))
            print("  A label that flips marks a trace the criteria do not decide yet: add a")
            print("  sentence or an example, not more runs.")
        steps = geval_steps(criteria, cfg)
        print(f"\nG-Eval evaluation steps, written just now by {cfg.judge_model or cfg.generation_model}:")
        for i, st in enumerate(steps, 1):
            print(f"  {i}. {st}")
        print()
        geval: list[bool | None] = [None] * len(items)
        for n, i in enumerate(judged_idx, 1):
            r = geval_score(cases[i], criteria, steps, cfg, samples=args.samples, mode=args.mode)
            geval[i], scores[i] = r.passed, r.score
            why.setdefault("G-Eval (our code)", [""] * len(items))[i] = r.reason
            progress(f"G-Eval (our code)    {items[i].id:<4} {mark(r.passed)}  {r.reason}", n, len(judged_idx))
        verdicts["G-Eval (our code)"] = (args.mode, geval)
        if env.installed("deepeval"):
            env.prepare()
            from toolkits import deepeval_metrics as D
            from toolkits.cases import to_deepeval
            from toolkits.judge import JudgeStats, deepeval_judge

            stats = JudgeStats()
            judge = deepeval_judge(cfg, stats)
            builders = [("DeepEval GEval", lambda: D.scope_geval(judge, criteria, steps, name=name))]
            if args.mode == SCOPE_MODE:
                builders.append(("DeepEval DAG", lambda: D.scope_dag(judge)))
            for label, build in builders:
                out: list[bool | None] = [None] * len(items)
                for n, i in enumerate(judged_idx, 1):
                    m = D.run(build(), to_deepeval(cases[i]))
                    out[i] = m.passed
                    why.setdefault(label, [""] * len(items))[i] = f"score {m.score}: {m.reason}"
                    score = "?" if m.score is None else f"{m.score:.2f}"
                    progress(f"{label:<20} {items[i].id:<4} {mark(m.passed)}  score {score}  {m.reason[:50]}",
                             n, len(judged_idx))
                verdicts[label] = (args.mode, out)
            print(f"\n  DeepEval judge: {stats.line()}")
        else:
            print("\n  DeepEval is not installed in this environment, so only our own judges ran.")
            print("  The toolkit environment is set up in requirements-toolkits.txt.")

    rule(f"STEP 4  Every evaluator against people, on its own failure mode ({args.split})")
    print("Positive means the failure is present. TPR: of the failures people found, the")
    print("share the evaluator caught. TNR: of the traces people passed on that mode, the")
    print("share the evaluator passed. Each rate shows its count, because a rate on two")
    print("traces is not a measurement. (Kappa is for two people who are peers; against")
    print("a golden set, the question is which way the evaluator is wrong.)\n")
    print(f"{'evaluator':<26} {'mode':<18} {'TPR':>11} {'TNR':>11} {'undecided':>10}")
    print("-" * 78)
    scored = [i for i, it in enumerate(items) if it.query in measured_q]
    measured: dict[str, dict] = {}
    pairs: dict[str, tuple[list[bool | None], list[bool]]] = {}
    for ev, (mode, vs) in verdicts.items():
        human = [mode not in truth[items[i].query].labels for i in scored]
        mine = [vs[i] for i in scored]
        r = rates(mine, human)
        measured[ev], pairs[ev] = r, (mine, human)
        tpr = f"{pct(r['tpr'])} {r['caught']}/{r['failures']}"
        tnr = f"{pct(r['tnr'])} {r['passed']}/{r['passes']}"
        print(f"{ev:<26} {mode:<18} {tpr:>11} {tnr:>11} {r['undecided']:>10}")
    for ev, reasons in why.items():
        mode, vs = verdicts[ev]
        off = [i for i in scored if vs[i] is not None and vs[i] != (mode not in truth[items[i].query].labels)]
        if off:
            print(f"\n{ev} disagrees with people on {len(off)} of {len(scored)}. Read them before you")
            print("change anything: the label can be wrong too.")
            for i in off:
                said = "passed" if vs[i] else "failed"
                print(f"  {items[i].id:<4} people {'passed' if vs[i] is False else 'failed'}, it {said}: "
                      f"{reasons[i][:60]}")
    if any(scores[i] is not None for i in scored):
        human = [args.mode not in truth[items[i].query].labels for i in scored]
        print("\nThe G-Eval pass threshold, read from the labels rather than chosen:")
        for th, r in threshold_sweep([scores[i] for i in scored], human):
            print(f"  pass if mean score >= {th}:  TPR {pct(r['tpr'])}, TNR {pct(r['tnr'])}")
        print("Pick the threshold on dev. Then run --split test once, and report that.")
    print("\nRefine a judge on dev until TPR and TNR are both above about 90 percent:")
    print("sharpen the Pass and Fail definitions, swap in clearer train examples, read")
    print("each disagreement. If it stalls: a stronger judge model, a narrower criterion")
    print("split in two, or better train examples. Then freeze it and run --split test.")

    rule("STEP 5  From the raw failure rate to a corrected one, with its uncertainty")
    print("An evaluator run on traces nobody labelled reports a raw failure rate. Its")
    print("known misses and false alarms bias that rate, and TPR and TNR correct it:")
    print("  true rate = (raw rate + TNR - 1) / (TPR + TNR - 1)      Rogan and Gladen")
    print("TPR and TNR rest on few labels, so a bootstrap over those labels gives the")
    print("corrected rate a 95 percent interval.\n")
    unlabelled = [i for i, it in enumerate(items) if it.query not in truth]
    others = [i for i, it in enumerate(items) if it.query in truth and it.query not in train_q
              and it.query not in measured_q]
    show_people = False
    if unlabelled:
        target, what = unlabelled, f"the {len(unlabelled)} traces nobody labelled"
    elif others:
        target = others
        what = f"the {len(others)} labelled traces outside the {args.split} split"
        show_people = args.split == "test"
    else:
        target, what = scored, f"the same {len(scored)} traces (no other traces to correct)"
        show_people = True
    print(f"Raw rate on {what}.")
    if not show_people and target is others:
        print("Their human labels stay closed until you run --split test, so the people")
        print("column is empty.")
    print()
    print(f"{'evaluator':<24} {'raw':>5} {'corrected':>10} {'95% interval':>14} {'people':>7}   traces")
    print("-" * 78)
    for ev, (mode, vs) in verdicts.items():
        decided = [vs[i] for i in target if vs[i] is not None]
        if not decided:
            continue
        raw = sum(1 for v in decided if v is False) / len(decided)
        m = measured[ev]
        fixed = corrected_failure_rate(raw, m["tpr"], m["tnr"])
        ci = bootstrap_interval(*pairs[ev], raw)
        band = "     -" if ci is None or fixed is None else f"{ci[0]:.0%} to {ci[1]:.0%}"
        people = None
        if show_people:
            ppl = [mode in truth[items[i].query].labels for i in target if items[i].query in truth]
            people = sum(ppl) / len(ppl) if ppl else None
        print(f"{ev:<24} {pct(raw):>5} {pct(fixed):>10} {band:>14} {pct(people):>7}   {len(decided)}")
    print("\nDecide with the interval, not the point. If the floor for a failure is 10")
    print("percent and the upper bound is above it, the evidence does not show the floor")
    print("is met yet: label more traces, or improve the evaluator. A blank corrected rate")
    print("means TPR or TNR could not be measured, or TPR + TNR - 1 is below 0.05: the")
    print("evaluator is close to a coin. Judge errors mostly widen the interval; they")
    print("move the corrected rate much less.")
    print("\nCode checks agree with careful people because they read facts the trace")
    print("records; when one disagrees, suspect the label first. The judged modes are the")
    print("ones people disagreed on in Exercise 5.A, and a judge inherits that")
    print("disagreement: sharpen the rubric before the prompt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
