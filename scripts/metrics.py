"""Day 3 hands-on: our metrics beside Ragas and DeepEval, on the same traces.

Day 1 named the shortcut: install a toolkit, run its default metrics, report
"faithfulness 0.82". This runs that shortcut on traces whose failures are
already known, and puts every toolkit number beside our own instruments and
the human verdict.

  STEP 1  retrieval: our rank metrics and the trace's drop record, beside
          Ragas context precision/recall and DeepEval contextual precision/recall
  STEP 2  faithfulness four ways (word-overlap checker, our judge, Ragas,
          DeepEval), beside correctness and the human verdict
  STEP 3  what a high faithfulness score includes: the unacceptable answers
          each instrument passes
  STEP 3b answer relevancy, both toolkits (only with --only relevancy: more calls)
  STEP 4  agents: tool use and task completion, DeepEval and Ragas beside our
          trajectory metrics (needs `python run.py architectures` first)

Every judge call is live, on the models you name; nothing is saved or
replayed. With --offline only the instruments that need no model run.

Run:  python run.py metrics
      python run.py metrics --judge anthropic/claude-sonnet-5-5    answers local, judge in the cloud
      python run.py metrics --only faithfulness --repeat 3   score spread (Module 11)
      python run.py metrics --ids q1,q4                      fewer calls
      python run.py metrics --only relevancy                 answer relevancy
      python run.py metrics --offline                        our metrics that need no model
Ragas and DeepEval run only in the toolkit environment (requirements-toolkits.txt).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from evals.day3_judges import judge_groundedness  # noqa: E402
from evals.day3_retrieval import context_recall_from_dropped, precision_at_k, recall_at_k  # noqa: E402
from helix import DEFAULT  # noqa: E402
from helix.console import banner, traces_note  # noqa: E402
from scripts.day3 import correct_check, grounded_check, human_labels, relevant_ids  # noqa: E402
from toolkits import env  # noqa: E402
from toolkits.cases import cases_from_traces  # noqa: E402
from tracing import read_jsonl  # noqa: E402

TRACES = ROOT / "data" / "traces" / "day1.jsonl"
SUITE = ROOT / "data" / "queries" / "day1_seed.json"
ARCH_SUITE = ROOT / "data" / "queries" / "architectures.json"
AGENT_TRACES = ROOT / "data" / "traces" / "arch_agent.jsonl"
PASS = 0.5  # DeepEval's default threshold, used for both toolkits' scores here


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return f"{v:.2f}"


class Kits:
    """Whichever toolkits are installed, with one judge each, counting calls."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.deepeval = self.ragas = None
        self.stats = None
        if cfg.offline:
            return
        env.prepare()
        from toolkits.judge import JudgeStats

        self.stats = JudgeStats()
        if env.installed("deepeval"):
            from toolkits import deepeval_metrics as D
            from toolkits.judge import deepeval_judge

            self.D, self.deepeval = D, deepeval_judge(cfg, self.stats)
        if env.installed("ragas"):
            from toolkits import ragas_metrics as R
            from toolkits.judge import ragas_embeddings, ragas_llm

            self.R = R
            self.ragas = R.build(ragas_llm(cfg, self.stats), ragas_embeddings(cfg))

    def d(self, metric_name: str, case):
        if not self.deepeval:
            return None
        from toolkits.cases import to_deepeval

        builders = {**self.D.rag_metrics(self.deepeval), **self.D.agent_metrics(self.deepeval)}
        return self.D.run(builders[metric_name], to_deepeval(case))

    def r(self, metric_name: str, case):
        if not self.ragas:
            return None
        return self.R.score(metric_name, self.ragas[metric_name], case)

    def missing(self) -> list[str]:
        return [n for n, v in (("DeepEval", self.deepeval), ("Ragas", self.ragas)) if v is None]


def val(m) -> float | None:
    return None if m is None else m.score


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="retrieval,faithfulness,agent",
                    help="comma-separated steps: retrieval, faithfulness, relevancy, agent")
    ap.add_argument("--ids", default="", help="comma-separated question ids, e.g. q1,q4")
    ap.add_argument("--repeat", type=int, default=1, help="score toolkit faithfulness N times")
    args = ap.parse_args(argv)
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    if not TRACES.exists():
        print("run `python run.py day1` first to produce data/traces/day1.jsonl")
        return 1
    cfg = DEFAULT
    banner(cfg)
    traces = read_jsonl(TRACES)
    traces_note(traces, cfg)
    suite = json.loads(SUITE.read_text(encoding="utf-8"))["queries"]
    human, source = human_labels(suite, traces)
    pairs = list(zip(suite, traces, cases_from_traces(traces, suite), human))
    if args.ids:
        keep = {x.strip() for x in args.ids.split(",")}
        pairs = [p for p in pairs if p[0]["id"] in keep]
    kits = Kits(cfg)
    if cfg.offline:
        print("\nOffline: only the instruments that need no model run. Judge and toolkit")
        print("columns stay empty; nothing is stored, so there is nothing to show instead.")
    elif kits.missing():
        print(f"\nNot installed here: {', '.join(kits.missing())}. Those columns stay empty.")
        print("The toolkit environment is set up in requirements-toolkits.txt.")

    if "retrieval" in only:
        rule("STEP 1  Retrieval: did the evidence arrive, and where was it lost?")
        print("Ours: rank metrics against the chunks carrying the expected evidence, and")
        print("the stage that dropped them. Toolkits: a judge reads each context and")
        print("decides whether it is useful for the reference answer.\n")
        print(f"{'id':<4} {'P@3':>5} {'R@5':>5}  {'lost at (trace)':<22} {'Ragas P':>8} {'Ragas R':>8} "
              f"{'DeepEval P':>11} {'DeepEval R':>11}")
        print("-" * 78)
        for q, t, c, _ in pairs:
            rel = relevant_ids(t, q.get("expected_evidence", ""))
            p3 = precision_at_k(t, rel, 3) if rel else None
            r5 = recall_at_k(t, rel, 5) if rel else None
            lost = context_recall_from_dropped(t, set(c.reference_context_ids))
            where = ", ".join(f"{k} {v}" for k, v in lost.items()) or ("-" if not c.reference_context_ids else "nothing lost")
            live = not cfg.offline
            rp = val(kits.r("context_precision", c)) if live else None
            rr = val(kits.r("context_recall", c)) if live else None
            dp = val(kits.d("contextual_precision", c)) if live else None
            dr = val(kits.d("contextual_recall", c)) if live else None
            print(f"{q['id']:<4} {fmt(p3):>5} {fmt(r5):>5}  {where:<22} {fmt(rp):>8} {fmt(rr):>8} "
                  f"{fmt(dp):>11} {fmt(dr):>11}")
        print("\nOnly the trace can say where evidence was lost. A toolkit score that is")
        print("low tells you that something is missing, not which stage dropped it.")

    if "faithfulness" in only:
        rule("STEP 2  Faithfulness, four implementations, beside correctness and people")
        print("checker: share of answer words found in the context. judge: our Day 3")
        print("GROUNDED/UNGROUNDED prompt. Ragas: share of the answer's claims the context")
        print("supports. DeepEval: share of claims that do not contradict the context.\n")
        print(f"{'id':<4} {'people':>7} {'checker':>8} {'judge':>6} {'Ragas':>6} {'DeepEval':>9} "
              f"{'correct':>8} {'Ragas factual':>14}")
        print("-" * 78)
        faith: dict[str, list] = {"checker": [], "judge": [], "Ragas": [], "DeepEval": []}
        spread: dict[str, list[str]] = {}
        for q, t, c, ok in pairs:
            checker = grounded_check(t)
            judge = bool(judge_groundedness(t, cfg)) if not cfg.offline else None
            live = not cfg.offline
            ragas_runs, deep_runs = [], []
            for _ in range(max(1, args.repeat) if live else 1):
                ragas_runs.append(val(kits.r("faithfulness", c)) if live else None)
                deep_runs.append(val(kits.d("faithfulness", c)) if live else None)
            ragas = ragas_runs[0]
            deep = deep_runs[0]
            factual = val(kits.r("factual_correctness", c)) if live else None
            correct = correct_check(t, q.get("ground_truth", ""))
            if args.repeat > 1 and live:
                spread[q["id"]] = [f"Ragas {[fmt(x) for x in ragas_runs]}", f"DeepEval {[fmt(x) for x in deep_runs]}"]
            for k, v in (("checker", checker), ("judge", judge), ("Ragas", ragas), ("DeepEval", deep)):
                faith[k].append((v, ok))
            print(f"{q['id']:<4} {'ok' if ok else 'FAIL':>7} {fmt(checker):>8} {fmt(judge):>6} {fmt(ragas):>6} "
                  f"{fmt(deep):>9} {fmt(correct):>8} {fmt(factual):>14}")
        print(f"\npeople: {source}")
        if spread:
            print("\nThe same judge, the same trace, scored again (Module 11):")
            for qid, s in spread.items():
                print(f"  {qid}  " + "   ".join(s))

        rule("STEP 3  What a high faithfulness score includes")
        print("Faithfulness asks one question: does the context support the answer? It")
        print("cannot see that the context is a superseded page, or a page this asker")
        print("may not read. Count the answers people rejected that each one passes:\n")
        print(f"{'instrument':<10} {'mean, all':>10} {'mean, rejected':>15} {'rejected but passed':>21}")
        print("-" * 60)
        for k, vals in faith.items():
            got = [(1.0 if v is True else 0.0 if v is False else v, ok) for v, ok in vals if v is not None]
            if not got:
                print(f"{k:<10} {'-':>10} {'-':>15} {'-':>21}")
                continue
            rejected = [v for v, ok in got if not ok]
            passed_bad = sum(1 for v in rejected if v >= PASS)
            mean_all = sum(v for v, _ in got) / len(got)
            mean_bad = sum(rejected) / len(rejected) if rejected else None
            print(f"{k:<10} {mean_all:>10.2f} {fmt(mean_bad):>15} {f'{passed_bad} of {len(rejected)}':>21}")
        print("\nA faithful answer can be wrong or forbidden. In this suite q1 and q2 invite")
        print("the superseded 2022 page and q4 the salary bands. The checks that catch those")
        print("read the trace: the page's version and its access label. Day 2's code")
        print("checks, not a toolkit default.")

    if "relevancy" in only:
        rule("STEP 3b  Answer relevancy: does the answer address the question?")
        print("Both toolkits ask the judge whether the answer is on topic. A leaked salary")
        print("band is a very relevant answer to a salary question.\n")
        print(f"{'id':<4} {'people':>7} {'Ragas relevancy':>16} {'DeepEval relevancy':>19}")
        print("-" * 50)
        for q, t, c, ok in pairs:
            live = not cfg.offline
            rr = val(kits.r("answer_relevancy", c)) if live else None
            dr = val(kits.d("answer_relevancy", c)) if live else None
            print(f"{q['id']:<4} {'ok' if ok else 'FAIL':>7} {fmt(rr):>16} {fmt(dr):>19}")

    if "agent" in only:
        rule("STEP 4  Agents: the trajectory, scored by three instruments")
        if not AGENT_TRACES.exists():
            print("Run `python run.py architectures` first; it writes data/traces/arch_agent.jsonl.")
        else:
            from evals.day3_architectures import grade, trajectory

            aq = json.loads(ARCH_SUITE.read_text(encoding="utf-8"))["queries"]
            atr = read_jsonl(AGENT_TRACES)
            acases = cases_from_traces(atr, aq)
            print("tool recall (ours): share of the suite's expected tools the agent called.")
            print("DeepEval tool correctness: the same idea, names only, computed by DeepEval.")
            print("Ragas tool-call F1: precision and recall over tool names together.")
            print("task completion: a judge reads the run and decides if the task was done.\n")
            print(f"{'id':<4} {'correct':>8} {'tool recall':>12} {'DE tools':>9} {'Ragas F1':>9} {'DE task':>8}  called")
            print("-" * 78)
            for q, t, c in zip(aq, atr, acases):
                live = not cfg.offline
                g, tr = grade(t, q), trajectory(t, q)
                dt = val(kits.d("tool_correctness", c)) if live else None
                rf = val(kits.r("tool_call_f1", c)) if live else None
                dc = val(kits.d("task_completion", c)) if live else None
                called = ",".join(x["name"].replace("search_handbook", "search") for x in c.tools_called)
                print(f"{q['id']:<4} {fmt(g.correct):>8} {fmt(tr['tool_recall']):>12} {fmt(dt):>9} {fmt(rf):>9} "
                      f"{fmt(dc):>8}  {called[:28]}")
            print("\nTool scores reward calling the right tools. They do not see whether the")
            print("agent acted on what a tool returned (AG5): that needs the observation in")
            print("the trace, which is what Day 3's own trajectory check reads.")

    if kits.stats is not None and kits.stats.calls:
        print(f"\ntoolkit judge: {kits.stats.line()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
