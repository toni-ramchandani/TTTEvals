"""Day 3 hands-on: component metrics, and the grounded-versus-correct split.

The Measure phase. Day 2 named the failure modes; this turns them into numbers,
and keeps retrieval quality separate from generation quality so that a score
tells you which layer to go and fix.

Two parts:

  retrieval    rank-aware metrics over the retrieve span, against the chunks a
               human marked relevant. Meaningless until somebody has decided
               what "relevant" means, which is the Day 1 charter's job.
  grounding    groundedness and correctness scored separately, then crossed.
               The interesting cell is grounded-and-wrong: the model did its
               job and the corpus did not.

Two instruments score every answer, side by side:

  checker      a cheap deterministic rule (word overlap for groundedness, the
               reference's numbers for correctness). Free, instant, brittle.
  LLM judge    your judge model (HELIX_JUDGE_MODEL, or the answer model) reading
               the evidence and the answer, through `evals.day3_judges`.

Step 3 validates both against human accept/reject calls on these exact
answers: the Day 2 golden set, else your Day 2 sheet, else (reference run)
annotator A's labels, else the suite's expected outcomes. Neither
instrument is trusted until it agrees with people. In the reference run
(--offline) only the checker runs, since there is no model to judge with.

Run:  python run.py day3
      python run.py day3 --judge anthropic/claude-sonnet-5-5
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evals.day3_judges import judge_correctness, judge_groundedness, quadrant, validate  # noqa: E402
from evals.day3_retrieval import (  # noqa: E402
    context_recall_from_dropped,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from helix import DEFAULT  # noqa: E402
from helix.console import banner, progress, traces_note  # noqa: E402
from tracing import read_jsonl  # noqa: E402

TRACES = ROOT / "data" / "traces" / "day1.jsonl"
SUITE = ROOT / "data" / "queries" / "day1_seed.json"

_WORD = re.compile(r"[a-z0-9]+")


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def relevant_ids(trace, evidence: str) -> set[str]:
    """Chunks a human would mark relevant: those carrying the expected evidence.

    The evidence strings live in the query suite beside each question. Writing
    them is the expensive part of building an evaluation set, and the reason
    to start with reference-free methods until you can afford it.
    """
    needles = [p.strip().lower() for p in (evidence or "").split("|") if p.strip()]
    if not needles:
        return set()
    # A chunk counts as relevant when it carries most of the expected evidence.
    # Requiring all of it would mark a chunk irrelevant for missing one token,
    # which is exactly the brittleness that makes reference-based scoring
    # expensive to maintain.
    need = max(1, (len(needles) + 1) // 2)
    out = set()
    for c in trace.retrieved:
        body = (c.text + " " + c.heading_path).lower()
        if sum(1 for n in needles if n in body) >= need:
            out.add(c.chunk_id)
    return out


def grounded_check(trace) -> bool:
    """The cheap groundedness checker: is the answer's content in the context it was given."""
    reached = [c for c in trace.retrieved if not c.dropped_at]
    if not reached:
        return False
    ctx = set()
    for c in reached:
        ctx |= words(c.text) | words(c.heading_path)
    content = [w for w in words(trace.answer) if len(w) > 4]
    if not content:
        return False
    return sum(1 for w in content if w in ctx) / len(content) >= 0.75


def correct_check(trace, ground_truth: str) -> bool:
    """The cheap correctness checker: does the answer carry the reference's key facts."""
    nums = re.findall(r"\d+", ground_truth)
    body = trace.answer.lower()
    # Numbers are the load-bearing part of a policy answer. Requiring every
    # figure from the reference is strict on purpose: a parental-leave answer
    # that gets one of two numbers right is not half correct, it is wrong.
    if nums:
        return all(n in body for n in nums)
    keys = re.findall(r"[a-z]{6,}", ground_truth.lower())[:3]
    return bool(keys) and all(k in body for k in keys)


def human_labels(suite: list[dict], traces: list) -> tuple[list[bool], str]:
    """Your Day 2 verdicts on these exact answers, else the suite's expected outcomes.

    The golden set comes first (Exercise 5.A), then your Day 2 sheet. A
    verdict counts only if it judged the same answer: run Day 1 again and the
    old verdicts no longer describe the new answers.
    """
    from evals.day2_agreement import read_golden
    from evals.day2_sheets import GOLDEN, notes_for, to_bool

    golden = {r.query: r for r in read_golden(GOLDEN)}
    if all(t.query in golden and golden[t.query].answer == t.answer for t in traces):
        return [golden[t.query].acceptable for t in traces], "the Day 2 golden set"
    notes = notes_for(traces)
    if all((notes.get(t.query, {}).get("acceptable") or "").strip() for t in traces):
        return [to_bool(notes[t.query]["acceptable"]) for t in traces], "your Day 2 verdicts"
    if DEFAULT.offline:
        from evals.day2_examples import EXAMPLE_A

        if all(t.query in EXAMPLE_A for t in traces):
            return [EXAMPLE_A[t.query][0] for t in traces], "the reference run's labels (annotator A)"
    return [q.get("expected_failure") in (None, "", "none") for q in suite], \
        "the suite's expected outcomes (label these answers on Day 2 to use your own)"


def main() -> int:
    if not TRACES.exists():
        print("run `python run.py day1` first to produce data/traces/day1.jsonl")
        return 1
    traces = read_jsonl(TRACES)
    suite = json.loads(SUITE.read_text(encoding="utf-8"))["queries"]
    cfg = DEFAULT
    banner(cfg)
    traces_note(traces, cfg)

    rule("STEP 1  Retrieval metrics, against human-marked relevant chunks")
    print("Rank-aware metrics say nothing about whether the answer was good. They")
    print("say whether the evidence arrived, and where in the ranking it landed.\n")
    print(f"{'id':<4} {'P@3':>6} {'R@5':>6} {'nDCG@5':>8} {'retrieved':>10} {'dropped':>9}  where")
    print("-" * 78)
    rel_map: dict[str, set[str]] = {}
    for q, t in zip(suite, traces):
        rel = relevant_ids(t, q.get("expected_evidence", ""))
        rel_map[t.trace_id] = rel
        if not rel:
            print(f"{q['id']:<4} {'-':>6} {'-':>6} {'-':>8} {'-':>10} {'-':>9}  no chunk carries the expected evidence")
            continue
        gains = {cid: 1.0 for cid in rel}
        d = context_recall_from_dropped(t, rel)
        where = ", ".join(f"{k}={v}" for k, v in d.items() if v) or "all reached the model"
        print(
            f"{q['id']:<4} {precision_at_k(t, rel, 3):>6.2f} {recall_at_k(t, rel, 5):>6.2f} "
            f"{ndcg_at_k(t, gains, 5):>8.2f} {len(rel):>10} "
            f"{sum(1 for c in t.retrieved if c.dropped_at):>9}  {where}"
        )
    print(f"\nMRR across the suite: {mrr(traces, rel_map):.3f}")

    rule("STEP 2  Groundedness and correctness, crossed")
    print("Scored separately on purpose. Collapsing them into one accuracy number")
    print("is the most common error in RAG evaluation, and it hides the worst case.\n")
    live = not cfg.offline
    check_g, check_c, judge_g, judge_c, reasons = [], [], [], [], []
    if live:
        print(f"Judging with {cfg.judge_model or cfg.generation_model}, two calls per answer:")
        for i, (q, t) in enumerate(zip(suite, traces), 1):
            vg = judge_groundedness(t, cfg)
            vc = judge_correctness(t, q.get("ground_truth", ""), cfg)
            judge_g.append(bool(vg))
            judge_c.append(bool(vc))
            reasons.append((q["id"], vg, vc))
            progress(f"{q['id']}  {vg.label:<12} {vc.label}", i, len(suite))
        print()
    for q, t in zip(suite, traces):
        check_g.append(grounded_check(t))
        check_c.append(correct_check(t, q.get("ground_truth", "")))

    if live:
        print(f"{'':4} {'grounded':^19} {'correct':^19}")
        print(f"{'id':<4} {'checker':>9} {'judge':>9} {'checker':>9} {'judge':>9}  verdict (judge)")
        print("-" * 78)
        for i, q in enumerate(suite):
            print(f"{q['id']:<4} {str(check_g[i]):>9} {str(judge_g[i]):>9} {str(check_c[i]):>9} "
                  f"{str(judge_c[i]):>9}  {quadrant(judge_g[i], judge_c[i])}")
        print("\nWhat the judge said, in its own words:")
        for qid, vg, vc in reasons:
            if not vg or not vc:
                why = vc.reason if not vc else vg.reason
                print(f"  {qid}  {why[:100] or '(no reason given)'}")
    else:
        print(f"{'id':<4} {'grounded':>9} {'correct':>8}  verdict")
        print("-" * 78)
        for i, q in enumerate(suite):
            print(f"{q['id']:<4} {str(check_g[i]):>9} {str(check_c[i]):>8}  {quadrant(check_g[i], check_c[i])}")

    rule("STEP 3  Validate the judge before trusting it")
    human, source = human_labels(suite, traces)
    print("A judge is itself a system that needs evaluating. This compares each")
    print("correctness instrument against human accept/reject calls, here")
    print(f"{source}.\n")
    instruments = [("checker", check_c)] + ([("LLM judge", judge_c)] if live else [])
    for name, labels in instruments:
        scores = validate(labels, human)
        print(f"  {name}")
        for k, v in scores.items():
            print(f"    {k:<24} {v}")
    print("\nAnything much below 0.8 agreement is not yet a measuring instrument.")
    print("Fix the judge, or go back and sharpen the rubric it is applying.")
    if live:
        print("Six traces is far too few to trust either number; it shows the method.")
        print("Rerun with another --judge and compare: judges disagree more than people expect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
