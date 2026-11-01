"""Ask Helix one question, through any of the three architectures, and read the trace.

The quickest way to explore the systems: type a question, see the answer from
vanilla RAG, the agent and the graph side by side, and the spans behind each.

Run:   python run.py ask "How many weeks of parental leave do I get?"
       python run.py ask "..." --office berlin --years 4
       python run.py ask "..." --system agent --trace
       python run.py ask "..." --config default      (vanilla as Day 1 shipped it)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic import AGENT_DEFAULT  # noqa: E402
from agentic import answer as agent_answer  # noqa: E402
from graphrag import GRAPH_DEFAULT  # noqa: E402
from graphrag import answer as graph_answer  # noqa: E402
from graphrag import build as build_graph  # noqa: E402
from helix import DEFAULT, REPAIRED, build_index  # noqa: E402
from helix import answer as helix_answer  # noqa: E402
from helix.console import Stream, banner, step_printer  # noqa: E402
from helix.structure import OFFICES  # noqa: E402


def show_trace(t) -> None:
    for s in t.spans:
        a = s.attributes
        line = f"    {s.name:<20}"
        if "tool" in a:
            line += f" {a['tool']} {a.get('args', '')}"
        print(line)
        if s.kept:
            print(f"        kept     {', '.join(s.kept[:6])}{' ...' if len(s.kept) > 6 else ''}")
        if s.dropped:
            print(f"        dropped  {', '.join(s.dropped[:6])}{' ...' if len(s.dropped) > 6 else ''}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--system", choices=["helix", "agent", "graph", "all"], default="all")
    ap.add_argument("--office", choices=list(OFFICES), default="amsterdam", help="the asker's office")
    ap.add_argument("--role", choices=["employee", "manager", "people-team"], default="employee")
    ap.add_argument("--years", type=float, default=3, help="the asker's years of service")
    ap.add_argument("--config", choices=["repaired", "default"], default="repaired",
                    help="vanilla Helix configuration (the agent and graph always use repaired)")
    ap.add_argument("--trace", action="store_true", help="print every span")
    args = ap.parse_args(argv)

    who = dict(user_office=args.office, user_role=args.role)
    banner(REPAIRED)
    print(f"\nquestion  {args.question}")
    print(f"asker     {args.role} in {args.office.title()}, {args.years:g} years' service\n")

    repaired = build_index(REPAIRED)
    live = not REPAIRED.offline

    def report(name, t) -> None:
        if t.span("agent.run"):
            run = t.span("agent.run").attributes
            print(f"    steps: {run['steps_used']}  {' -> '.join(run['tool_sequence'])}")
        if t.span("graph.link"):
            link = t.span("graph.link").attributes
            print(f"    linked: offices={link['offices_linked']} topics={link['topics_linked']}")
        print(f"    cited:  {', '.join(t.citations) or 'nothing'}{'   (refused)' if t.refused else ''}")
        if args.trace:
            show_trace(t)
        print()

    if args.system in ("helix", "all"):
        cfg = REPAIRED if args.config == "repaired" else DEFAULT
        idx = repaired if cfg is REPAIRED else build_index(DEFAULT)
        print(f"[vanilla ({args.config})]")
        out = Stream("    ")
        t = helix_answer(args.question, idx, cfg, on_token=out, **who)
        out.done()
        report("vanilla", t)
    if args.system in ("agent", "all"):
        print("[agent]")
        t = agent_answer(args.question, repaired, AGENT_DEFAULT, user_tenure_years=args.years,
                         on_step=step_printer("    ") if live else None, **who)
        print(f"    {t.answer.replace(chr(10), ' ')}")
        report("agent", t)
    if args.system in ("graph", "all"):
        print("[graph]")
        out = Stream("    ")
        t = graph_answer(args.question, build_graph(GRAPH_DEFAULT), GRAPH_DEFAULT, index=repaired,
                         user_tenure_years=args.years, on_token=out, **who)
        out.done()
        report("graph", t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
