"""Day 3 hands-on: one question set, three architectures, one set of instruments.

Runs thirteen questions through:

  helix-default   vanilla RAG as Day 1 shipped it
  helix-repaired  vanilla RAG with the three Day 1 defects fixed
  agent-naive     a tool loop that searches once and stops
  agent           the tool loop with every switch on
  graph-naive     GraphRAG with every switch off
  graph           GraphRAG with every switch on

then prints:

  1. a correctness matrix, with the failure code for every wrong answer
  2. trajectory metrics for the agent, scored separately from correctness
  3. graph metrics: linking, filtering, fallback, provenance
  4. one-switch ablations: turn one switch off, see which question breaks and
     which code the detector assigns. The Day 1 method again: change one
     variable at a time and watch the failure move.

Every trace is written to data/traces/arch_<system>.jsonl in the same schema as
Day 1, so the Day 2 open-coding sheet and the Day 3 retrieval metrics run over
them unchanged.

It opens with a short look at the systems before measuring anything: what the
graph's extractor wrote down, and one question answered three ways.

Live, every answer comes from your model and a progress line prints as each
one finishes. The six systems over thirteen questions is roughly 200 model
calls, a few minutes on a laptop GPU or a cloud model. The ablations in step 5
multiply that by thirteen, so live they run only when asked (--ablations).
The reference run (--offline) always includes them.

Run:  python run.py architectures                  the six systems, live
      python run.py architectures --only agent
      python run.py architectures --only graph
      python run.py architectures --ablations      also step 5 (slow live)
      python run.py architectures --ids a1,a6,a12  a few questions only
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic import AGENT_DEFAULT, AGENT_NAIVE  # noqa: E402
from agentic import answer as agent_answer  # noqa: E402
from evals.day3_architectures import attribute, grade, graph_metrics, trajectory  # noqa: E402
from evals.taxonomy import BY_CODE  # noqa: E402
from graphrag import GRAPH_DEFAULT, GRAPH_NAIVE, GraphBuildConfig  # noqa: E402
from graphrag import answer as graph_answer  # noqa: E402
from graphrag import build as graph_build  # noqa: E402
from helix import DEFAULT, REPAIRED, build_index  # noqa: E402
from helix import answer as helix_answer  # noqa: E402
from helix.console import banner, progress, step_printer  # noqa: E402
from tracing import write_jsonl  # noqa: E402

SUITE = ROOT / "data" / "queries" / "architectures.json"
OUT = ROOT / "data" / "traces"


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


class Systems:
    """Builds each index and graph once, then answers with any configuration."""

    def __init__(self) -> None:
        self.index = {"default": build_index(DEFAULT), "repaired": build_index(REPAIRED)}
        self._graphs: dict[str, object] = {}

    def graph(self, gcfg):
        key = gcfg.build.hash()
        if key not in self._graphs:
            self._graphs[key] = graph_build(gcfg)
        return self._graphs[key]

    def helix(self, cfg, q):
        idx = self.index["repaired" if cfg is REPAIRED else "default"]
        return helix_answer(q["text"], idx, cfg, user_office=q["user_office"], user_role=q["user_role"])

    def agent(self, acfg, q, on_step=None):
        return agent_answer(q["text"], self.index["repaired"], acfg, user_office=q["user_office"],
                            user_role=q["user_role"], user_tenure_years=q.get("user_tenure_years"),
                            on_step=on_step)

    def graphrag(self, gcfg, q):
        return graph_answer(q["text"], self.graph(gcfg), gcfg, index=self.index["repaired"],
                            user_office=q["user_office"], user_role=q["user_role"],
                            user_tenure_years=q.get("user_tenure_years"))


def run_system(sys_: Systems, name: str, queries: list[dict]):
    kind, cfg = SYSTEMS[name]
    fn = {"helix": sys_.helix, "agent": sys_.agent, "graph": sys_.graphrag}[kind]
    graph = sys_.graph(cfg) if kind == "graph" else None
    live = not DEFAULT.offline
    if live:
        print(f"\n  {name}")
    traces, results = [], []
    for i, q in enumerate(queries, 1):
        t = fn(cfg, q)
        g, codes = grade(t, q), attribute(t, q, graph=graph)
        traces.append(t)
        results.append((t, g, codes))
        if live:
            run = t.span("agent.run")
            steps = f"{run.attributes['steps_used']} steps  " if run else ""
            verdict = "right" if g.correct else "wrong " + "/".join(codes)
            progress(f"{q['id']:4} {verdict:<14} {steps}{t.answer[:60].replace(chr(10), ' ')}", i, len(queries))
    write_jsonl(traces, OUT / f"arch_{name}.jsonl")
    return results


SYSTEMS = {
    "helix-default": ("helix", DEFAULT),
    "helix-repaired": ("helix", REPAIRED),
    "agent-naive": ("agent", AGENT_NAIVE),
    "agent": ("agent", AGENT_DEFAULT),
    "graph-naive": ("graph", GRAPH_NAIVE),
    "graph": ("graph", GRAPH_DEFAULT),
}

AGENT_ABLATIONS = {
    "use_profile off": replace(AGENT_DEFAULT, use_profile=False),
    "prefer_query_office off": replace(AGENT_DEFAULT, prefer_query_office=False),
    "decompose off": replace(AGENT_DEFAULT, decompose=False),
    "check_versions off": replace(AGENT_DEFAULT, check_versions=False),
    "act_on_observations off": replace(AGENT_DEFAULT, act_on_observations=False),
    "max_steps 8 -> 16": replace(AGENT_DEFAULT, max_steps=16),
}

GRAPH_ABLATIONS = {
    "linking exact": replace(GRAPH_DEFAULT, linking="exact"),
    "use_user_profile off": replace(GRAPH_DEFAULT, use_user_profile=False),
    "build unit=chunk": replace(GRAPH_DEFAULT, build=GraphBuildConfig(unit="chunk")),
    "build carry_access_label off": replace(GRAPH_DEFAULT, build=GraphBuildConfig(carry_access_label=False)),
    "respect_supersedes off": replace(GRAPH_DEFAULT, respect_supersedes=False),
    "fallback_to_text off": replace(GRAPH_DEFAULT, fallback_to_text=False),
    "hops 2 -> 1": replace(GRAPH_DEFAULT, hops=1),
}


def matrix(all_results: dict[str, list], queries: list[dict]) -> None:
    names = list(all_results)
    width = 15
    print(f"{'':5}" + "".join(f"{n:>{width}}" for n in names))
    for i, q in enumerate(queries):
        row = f"{q['id']:5}"
        for n in names:
            _, g, codes = all_results[n][i]
            cell = "ok" if g.correct else "/".join(codes)
            row += f"{cell:>{width}}"
        print(row)
    print(f"{'right':5}" + "".join(
        f"{sum(g.correct for _, g, _ in all_results[n]):>{width - 3}}/{len(queries):<2}" for n in names))


def ablate(sys_: Systems, kind: str, base_results: list, variants: dict, queries: list[dict]) -> None:
    fn = sys_.agent if kind == "agent" else sys_.graphrag
    for label, cfg in variants.items():
        changes = []
        graph = sys_.graph(cfg) if kind == "graph" else None
        for (t0, g0, c0), q in zip(base_results, queries):
            t = fn(cfg, q)
            g = grade(t, q)
            c = attribute(t, q, graph=graph)
            if g.correct != g0.correct or c != c0:
                before = "ok" if g0.correct else "/".join(c0)
                after = "ok" if g.correct else "/".join(c)
                changes.append(f"{q['id']} {before}->{after}")
        print(f"  {label:30} {', '.join(changes) if changes else 'no change on this set'}")


def look(sys_: Systems) -> None:
    """Read before you measure: the graph's facts, and one question three ways."""
    rule("STEP 0a  What the graph's extractor wrote down")
    g = sys_.graph(GRAPH_DEFAULT)
    print(f"  {g.stats()}\n")
    for f in list(g.facts.values())[:5]:
        print(f"  {f.fact_id:32} {f.access_label:8} {f.statement}")
    print("  ...")
    print("\n  Every fact carries its document, version, date, access label and source chunk,")
    print("  attached by code at build time. The same extractor over Day 1's fixed-size")
    print("  chunks writes some rows down against no office at all:\n")
    g_chunk = sys_.graph(GRAPH_NAIVE)
    for f in g_chunk.facts.values():
        if f.doc_id == "HB-PAR-2024" and f.office is None:
            print(f"  {f.fact_id:32} {f.statement}")
    print("\n  That is FP10, severed context, re-born at build time as GR2.")

    rule("STEP 0b  One question, three architectures")
    q = {"text": "How many weeks of parental leave do I get as the secondary caregiver?",
         "user_office": "berlin", "user_role": "employee", "user_tenure_years": 4}
    print(f"  asker: employee in Berlin.  question: {q['text']}")
    print("  ground truth: 8 weeks (Berlin, secondary caregiver)\n")
    t_h = sys_.helix(REPAIRED, q)
    if not DEFAULT.offline:
        print("  the agent, live, one line per tool call as the model makes it:")
    t_a = sys_.agent(AGENT_DEFAULT, q, on_step=None if DEFAULT.offline else step_printer("    "))
    t_g = sys_.graphrag(GRAPH_DEFAULT, q)
    if not DEFAULT.offline:
        print()
    for name, t in (("vanilla", t_h), ("agent", t_a), ("graph", t_g)):
        print(f"  {name:8} {t.answer[:110].replace(chr(10), ' ')}")
    print("\n  The agent's path, one span per tool call:")
    for s in t_a.spans:
        if s.name.startswith("agent.step."):
            obs = s.attributes.get("observation")
            obs = obs if isinstance(obs, (dict, str)) else ", ".join(obs or [])
            print(f"    {s.name:14} {s.attributes['tool']:22} {s.attributes['args']}")
            if obs:
                print(f"    {'':14} -> {str(obs)[:90]}")
    print("\n  The graph's path:")
    for name in ("graph.link", "graph.traverse", "graph.filter", "graph.rank"):
        s = t_g.span(name)
        print(f"    {name:15} kept {s.kept[:3]}  dropped {s.dropped[:3]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["agent", "graph"], default=None)
    ap.add_argument("--ablations", action="store_true", help="run step 5 live (always on offline)")
    ap.add_argument("--ids", default="", help="comma-separated question ids, e.g. a1,a6")
    args = ap.parse_args(argv)

    banner(DEFAULT)
    queries = json.loads(SUITE.read_text(encoding="utf-8"))["queries"]
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",")}
        queries = [q for q in queries if q["id"] in wanted]
    sys_ = Systems()
    names = [n for n in SYSTEMS if args.only is None or SYSTEMS[n][0] in ("helix", args.only)]

    look(sys_)

    rule(f"STEP 1  {len(queries)} questions, each chosen because some architecture fails it")
    for q in queries:
        print(f"  {q['id']:4} {q['kind']:48} {q['text'][:60]}")
    print()
    for n in names:
        cfg = SYSTEMS[n][1]
        print(f"  {n:15} {cfg.describe()}")

    results = {n: run_system(sys_, n, queries) for n in names}

    rule("STEP 2  Correctness, with the code for every wrong answer")
    matrix(results, queries)
    used = sorted({c for r in results.values() for _, _, cs in r for c in cs})
    print()
    for c in used:
        print(f"  {c:5} {BY_CODE[c].name:32} born at {BY_CODE[c].born_at}")

    for agent_name in [n for n in ("agent-naive", "agent") if n in results]:
        rule(f"STEP 3  {agent_name}: the path, scored separately from the answer")
        print(f"  {'id':4} {'right':6} {'steps':>5} {'searches':>8} {'tool recall':>11} "
              f"{'versions checked':>16}  sequence")
        for (t, g, _), q in zip(results[agent_name], queries):
            tr = trajectory(t, q)
            seq = t.span("agent.run").attributes["tool_sequence"]
            short = " ".join({"get_user_profile": "profile", "list_offices": "offices",
                              "search_handbook": "search", "check_document_status": "status",
                              "finish": "finish"}[s] for s in seq)
            vc = "-" if tr["version_checked_share"] is None else f"{tr['version_checked_share']:.0%}"
            print(f"  {q['id']:4} {'yes' if g.correct else 'no':6} {tr['steps_used']:>5} {tr['searches']:>8} "
                  f"{tr['tool_recall']:>11.0%} {vc:>16}  {short}")
        good_path_bad_answer, bad_path_good_answer = [], []
        for (t, g, codes), q in zip(results[agent_name], queries):
            tr = trajectory(t, q)
            path_ok = tr["tool_recall"] >= 1 and tr["version_checked_share"] in (None, 1)
            if path_ok and not g.correct and not any(c.startswith("AG") for c in codes):
                good_path_bad_answer.append(f"{q['id']} ({'/'.join(codes)})")
            if g.correct and tr["tool_recall"] < 1:
                bad_path_good_answer.append(q["id"])
        if good_path_bad_answer:
            print(f"\n  Read {', '.join(good_path_bad_answer)}.")
            print("  Every expected tool was called and every cited page was checked, and the")
            print("  answer is still wrong. The code says where instead: FP4 is the generator")
            print("  misreading a good extract, FP1 a gap in the handbook itself. Score only")
            print("  the trajectory and you would ship it.")
        if bad_path_good_answer:
            print(f"\n  Read {', '.join(bad_path_good_answer)}.")
            print("  The answer is right and tool recall is below 100%: a tool the question")
            print("  needed was never called, and the search happened to land anyway. Right")
            print("  answer, wrong path. Score only the answer and you would trust it.")

    if "graph" in results:
        rule("STEP 4  The graph's link, filter and fallback")
        print(f"  {'id':4} {'right':6} {'linked':18} {'named':18} {'profile':14} {'facts':>5} "
              f"{'stale out':>9} {'acl out':>7} {'fallback':>8}")
        for (t, g, _), q in zip(results["graph"], queries):
            m = graph_metrics(t)
            print(f"  {q['id']:4} {'yes' if g.correct else 'no':6} {','.join(m['offices_linked']) or '-':18} "
                  f"{','.join(m['offices_named']) or '-':18} {','.join(m['profile_used']) or '-':14} "
                  f"{m['facts_in_context']:>5} {m['dropped_superseded']:>9} {m['blocked_by_entitlement']:>7} "
                  f"{'yes' if m['fallback'] else '-':>8}")

    if DEFAULT.offline or args.ablations:
        if not DEFAULT.offline:
            print("\n  Live ablations: a change can also be the model answering differently on a")
            print("  rerun. Rerun the base system once to see how much moves with no switch changed.")
        if "agent" in results:
            rule("STEP 5a  Agent ablations: one switch off, everything else on")
            ablate(sys_, "agent", results["agent"], AGENT_ABLATIONS, queries)
        if "graph" in results:
            rule("STEP 5b  Graph ablations: one switch off, everything else on")
            ablate(sys_, "graph", results["graph"], GRAPH_ABLATIONS, queries)
    else:
        rule("STEP 5  Ablations")
        print("  Skipped in a live run (13 x the calls). Run with --ablations, ideally on")
        print("  a few questions:  python run.py architectures --only agent --ablations --ids a1,a6,a12")

    rule("WHAT TO TAKE INTO DAY 4")
    print("  No architecture wins every row. Each fixes failures the others have and")
    print("  adds its own, and each new failure is born somewhere vanilla RAG does not")
    print("  have: a tool choice, a tool argument, an extraction rule, a link. The")
    print("  trace schema is what lets one set of evaluators attribute all three.")
    print(f"\n  Traces written to {OUT.relative_to(ROOT)}/arch_*.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
