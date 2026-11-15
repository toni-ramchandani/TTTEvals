"""Check the toolkit environment: DeepEval and Ragas, safely set up.

Run this once in the toolkit environment before Day 2 (the setup steps are
in requirements-toolkits.txt). It prints, in order:

  1. what is installed, against the pinned versions
  2. the safety settings: telemetry off, no Confident AI upload, no LiteLLM
  3. one live metric from each toolkit on one Day 1 trace, judged by your
     model through helix.llm, and how often the judge's JSON needed repair

With --module 07, 08 or 09 it runs the Day 3 toolkit lens instead: every
curriculum item of that module, the workshop's own instrument beside
DeepEval, G-Eval and Ragas on the same trace (scripts/lens.py).

With --show ID it opens the box on one Day 1 trace instead (Module 08):
the test case both toolkits read, then DeepEval faithfulness and Ragas
faithfulness on the same answer, each with what the judge was asked to
return at every step and what came back. Nothing is saved.

Run:  python run.py toolkits
      python run.py toolkits --judge openai/gpt-5.4-mini
      python run.py toolkits --show q1
      python run.py toolkits --module 08
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from helix import DEFAULT  # noqa: E402
from helix.console import banner  # noqa: E402
from toolkits import env  # noqa: E402

TRACES = ROOT / "data" / "traces" / "day1.jsonl"
SUITE = ROOT / "data" / "queries" / "day1_seed.json"


def _short(text: str, n: int = 70) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def show(qid: str) -> int:
    """Open the box: one metric from each toolkit on one trace, step by step."""
    from toolkits.cases import cases_from_traces, show as show_case
    from tracing import read_jsonl

    if not TRACES.exists():
        print("Run `python run.py day1` first; --show opens one of its traces.")
        return 1
    suite = json.loads(SUITE.read_text(encoding="utf-8"))["queries"]
    traces = read_jsonl(TRACES)
    cases = cases_from_traces(traces, suite)
    pick = [(c, t) for c, t in zip(cases, traces) if c.id == qid]
    if not pick:
        print(f"No trace with id {qid}. Choose one of: {', '.join(c.id for c in cases)}")
        return 1
    case, trace = pick[0]
    print(f"\nSTEP 1  The test case, built in memory from trace {case.id}")
    print("-" * 78)
    print("\n".join(show_case(case, trace)))
    if DEFAULT.offline:
        print("\nOffline reference run: no judge model, so the metrics do not run. Drop")
        print("--offline to watch them work on your model.")
        return 0
    missing = [d for d in env.PINNED if not env.installed(d)]
    if missing:
        print(f"\nNot installed here: {', '.join(missing)}.\n")
        print(env.INSTALL_HINT)
        return 1
    from toolkits import deepeval_metrics as D
    from toolkits import ragas_metrics as R
    from toolkits.cases import to_deepeval
    from toolkits.judge import JudgeStats, deepeval_judge, ragas_llm

    print("\nSTEP 2  DeepEval faithfulness: truths from the context, claims from the answer,")
    print("        then one verdict per claim. Faithful = does not contradict the context.")
    print("-" * 78)
    st = JudgeStats()
    metric = D.rag_metrics(deepeval_judge(DEFAULT, st))["faithfulness"]
    m = D.run(metric, to_deepeval(case))
    for label, items in (("truths (from the context)", getattr(metric, "truths", None) or []),
                         ("claims (from the answer)", getattr(metric, "claims", None) or [])):
        print(f"  {label}: {len(items)}")
        for x in items[:6]:
            print(f"    - {_short(x)}")
    verdicts = getattr(metric, "verdicts", None) or []
    print(f"  verdicts: {len(verdicts)}")
    for v in verdicts:
        print(f"    {getattr(v, 'verdict', '?'):<4} {_short(getattr(v, 'reason', '') or '', 66)}")
    print(f"  score {m.score if m.score is not None else 'undecided'}  (yes and idk count as faithful,"
          " unless penalize_ambiguous_claims=True)")
    print(f"  judge: {st.line()}")

    print("\nSTEP 3  Ragas faithfulness: statements from the answer, then one verdict per")
    print("        statement. Faithful = can be inferred from the context.")
    print("-" * 78)
    st2 = JudgeStats()
    r = R.score("faithfulness", R.build(ragas_llm(DEFAULT, st2))["faithfulness"], case)
    for name, obj in st2.log:
        if hasattr(obj, "statements") and obj.statements and isinstance(obj.statements[0], str):
            print(f"  statements: {len(obj.statements)}")
            for x in obj.statements[:6]:
                print(f"    - {_short(x)}")
        elif hasattr(obj, "statements"):
            print(f"  verdicts: {len(obj.statements)}")
            for v in obj.statements:
                print(f"    {getattr(v, 'verdict', '?'):<4} {_short(getattr(v, 'reason', ''), 66)}")
    print(f"  score {r.score if r.score is not None else 'undecided'}  {r.reason}")
    print(f"  judge: {st2.line()}")
    print("\nSame answer, same context, two definitions of faithful. Neither reads the")
    print("page's version or its access label: those are in the trace, and only Day 2's")
    print("code checks read them.")
    return 0


def main() -> int:
    if "--module" in sys.argv[1:]:
        from scripts.lens import main as lens

        return lens(sys.argv[1:])
    if "--show" in sys.argv[1:]:
        args = sys.argv[1:]
        i = args.index("--show")
        if i + 1 >= len(args):
            print("usage: python run.py toolkits --show ID   (an id from day1_seed.json, e.g. q1)")
            return 2
        for note in env.prepare():
            print(f"note     {note}")
        banner(DEFAULT)
        return show(args[i + 1])
    for note in env.prepare():
        print(f"note     {note}")
    print(f"python   {sys.version.split()[0]}")
    banner(DEFAULT)

    print("\nTOOLKITS AND SAFETY SETTINGS")
    print("-" * 78)
    for item, state, note in env.status():
        print(f"  {item:<28} {state:<10} {note}")
    print("\n  Ragas 0.4.3 carries CVE-2026-6587 (SSRF in its multi-modal image loader,")
    print("  no fix released). The workshop never uses the multi-modal metrics.")

    missing = [d for d in env.PINNED if not env.installed(d)]
    if missing:
        print(f"\nNot installed here: {', '.join(missing)}.\n")
        print(env.INSTALL_HINT)
        return 1
    if DEFAULT.offline:
        print("\nOffline reference run: no model, so no live metric. Drop --offline to test one.")
        return 0
    if not TRACES.exists():
        print("\nRun `python run.py day1` first; the live check scores one of its traces.")
        return 1

    from toolkits import deepeval_metrics as D
    from toolkits import ragas_metrics as R
    from toolkits.cases import cases_from_traces, to_deepeval
    from toolkits.judge import JudgeStats, deepeval_judge, ragas_llm
    from tracing import read_jsonl

    suite = json.loads(SUITE.read_text(encoding="utf-8"))["queries"]
    cases = cases_from_traces(read_jsonl(TRACES), suite)
    case = next((c for c in cases if c.id == "q5"), cases[0])
    print(f"\nLIVE CHECK on {case.id}: {case.question}")
    print("-" * 78)
    stats = JudgeStats()
    d = D.run(D.rag_metrics(deepeval_judge(DEFAULT, stats))["faithfulness"], to_deepeval(case))
    print(f"  DeepEval faithfulness   {d.score if d.score is not None else 'undecided'}   {d.reason[:80]}")
    r = R.score("faithfulness", R.build(ragas_llm(DEFAULT, stats))["faithfulness"], case)
    print(f"  Ragas faithfulness      {r.score if r.score is not None else 'undecided'}   {r.reason[:80]}")
    print(f"\n  judge: {stats.line()}")
    if d.score is None or r.score is None:
        print("\nTOOLKITS CHECK FAILED  a metric could not be scored; the reason is above.")
        print("A small local judge often returns broken JSON. Try --judge with a larger model.")
        return 1
    print("\nTOOLKITS CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
