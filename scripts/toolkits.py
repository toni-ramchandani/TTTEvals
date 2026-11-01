"""Check the toolkit environment: DeepEval and Ragas, safely set up.

Run this once in the toolkit environment before Day 2 (the setup steps are
in requirements-toolkits.txt). It prints, in order:

  1. what is installed, against the pinned versions
  2. the safety settings: telemetry off, no Confident AI upload, no LiteLLM
  3. one live metric from each toolkit on one Day 1 trace, judged by your
     model through helix.llm, and how often the judge's JSON needed repair

Run:  python run.py toolkits
      python run.py toolkits --judge openai/gpt-5.4-mini
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helix import DEFAULT  # noqa: E402
from helix.console import banner  # noqa: E402
from toolkits import env  # noqa: E402

TRACES = ROOT / "data" / "traces" / "day1.jsonl"
SUITE = ROOT / "data" / "queries" / "day1_seed.json"


def main() -> int:
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
