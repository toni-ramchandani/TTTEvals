"""Day 3, Module 09 hands-on: the agent's path, one tool call at a time.

The architectures run scored the agent's answers and its trajectory. This
run asks where each wrong answer was born:

  1. one trace, four stages per tool call (Select, Args, Exec, Handle)
  2. the first failure in every wrong trace, with the rule that found it
  3. the transition failure heatmap, before and after the agent's switches,
     and one planted defect at a time to show each lands in its own cell
  4. operations: tool calls per question, tool success rate, retries,
     re-searches, budget refusals and hand-offs
  5. planted tool faults: does a retry recover, does the fallback decline
     instead of inventing, and what can a retry not see?
  6. scope: only its own tools, arguments inside their schema, nothing
     outside the asker's access, and a search before every answer

Live, about 250 model calls: 13 questions through two agents, 15 runs with
planted faults, 4 scope questions. --ids runs a few questions only, --only
runs one part. The planted-defect step (3b) runs live only with --ablations.

Run:  python run.py agents --offline            the reference run, seconds
      python run.py agents --ids a2,a7          a small live run
      python run.py agents --only faults        steps 5 only
      python run.py agents --only scope         step 6 only
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from agentic import AGENT_DEFAULT, AGENT_NAIVE  # noqa: E402
from agentic import answer as agent_answer  # noqa: E402
from agentic.agent import live_tools  # noqa: E402
from evals.day3_agents import (  # noqa: E402
    SCOPE_CHECKS,
    first_failure,
    heatmap,
    heatmap_lines,
    operations,
    plant,
    scope_violations,
    stage_checks,
)
from evals.day3_architectures import attribute, grade  # noqa: E402
from helix import DEFAULT, REPAIRED, build_index  # noqa: E402
from helix.console import banner, progress  # noqa: E402

SUITE = ROOT / "data" / "queries" / "architectures.json"
SCOPE = ROOT / "data" / "queries" / "day3_agent_scope.json"
FAULT_IDS = ("a2", "a5", "a9")
FAULTS = [
    ("no fault", None, 1),
    ("1st search times out, no retry", {"search_handbook": ("timeout", {1})}, 0),
    ("1st search times out, 1 retry", {"search_handbook": ("timeout", {1})}, 1),
    ("every search times out, 1 retry", {"search_handbook": ("timeout", None)}, 1),
    ("1st search returns nothing, 1 retry", {"search_handbook": ("empty", {1})}, 1),
]
SHORT = {"get_user_profile": "profile", "list_offices": "offices", "search_handbook": "search",
         "check_document_status": "status", "finish": "finish"}


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def offered(acfg) -> set[str]:
    return {t["function"]["name"] for t in live_tools(acfg)}


def configs() -> dict:
    from scripts.architectures import AGENT_ABLATIONS
    return {"agent-naive": AGENT_NAIVE, "agent": AGENT_DEFAULT, **AGENT_ABLATIONS}


class Runner:
    def __init__(self) -> None:
        self.index = build_index(REPAIRED)
        self.made: list[tuple] = []   # (trace, question, offered set), for the scope step

    def run(self, acfg, q, **kw):
        t = agent_answer(q["text"], self.index, acfg, user_office=q["user_office"], user_role=q["user_role"],
                         user_tenure_years=q.get("user_tenure_years"), **kw)
        self.made.append((t, q, offered(acfg)))
        return t

    def suite(self, label: str, acfg, queries: list[dict]) -> list:
        out = []
        for i, q in enumerate(queries, 1):
            out.append(self.run(acfg, q))
            if not DEFAULT.offline:
                progress(f"{label} {q['id']}", i, len(queries))
        return out


def mark(v) -> str:
    return "-" if v is None else ("ok" if v[0] else "FAIL")


def cell(f) -> str:
    return "-" if f is None else f"{f.from_state} -> {f.state}"


def show_stages(t, q, acfg) -> None:
    rows = stage_checks(t, q, offered(acfg))
    print(f"  {'step':>4}  {'tool':8} {'arguments':34} {'Select':>6} {'Args':>5} {'Exec':>5} {'Handle':>6}")
    for r in rows:
        args = " ".join((", ".join(f"{k}={v}" for k, v in r["args"].items()) or "-").split())
        print(f"  {r['n']:>4}  {SHORT.get(r['tool'], r['tool']):8} {args[:34]:34} {mark(r['select']):>6} "
              f"{mark(r['args_ok']):>5} {mark(r['exec']):>5} {mark(r['handle']):>6}")
    f = first_failure(t, q, offered(acfg))
    print(f"\n  answer: {t.answer[:90].replace(chr(10), ' ')}")
    if f:
        where = f"step {f.step}" if f.step else "the end of the run"
        print(f"  first failure: {f.state} at {where}, from {f.from_state}.")
        for line in textwrap.wrap(f.why + ".", 74):
            print(f"  {line}")
    else:
        print("  right answer, no failure to locate.")


def step_paths(r: Runner, queries: list[dict], ablations: bool) -> None:
    by_id = {q["id"]: q for q in queries}
    cfgs = configs()

    rule("STEP 1  One trace, four stages per tool call")
    q7 = by_id.get("a7") or queries[0]
    variant = cfgs["act_on_observations off"]
    print(f"  {q7['id']}: {q7['text']}")
    print("  agent with one switch off (act_on_observations): it checks each page's")
    print("  status and does not act on the answer. Read each row left to right.\n")
    show_stages(r.run(variant, q7), q7, variant)
    print("\n  Select: right tool now?  Args: valid and right?  Exec: did the call work?")
    print("  Handle: did the agent use what came back? '-' = the stage does not apply.")

    rule("STEP 2  The first failure in every wrong trace")
    runs = {name: r.suite(name, cfgs[name], queries) for name in ("agent-naive", "agent")}
    fails = {}
    print(f"  {'id':4} {'agent':12} {'codes':9} {'first failure':18} {'step':>4} {'rule':>5}")
    for name, traces in runs.items():
        acfg = cfgs[name]
        fails[name] = [first_failure(t, q, offered(acfg)) for t, q in zip(traces, queries)]
        for t, q, f in zip(traces, queries, fails[name]):
            if f is None:
                continue
            print(f"  {q['id']:4} {name:12} {'/'.join(attribute(t, q)):9} {cell(f):18} "
                  f"{f.step or '-':>4} {f.tier:>5}")
            for line in textwrap.wrap(f.why, 70):
                print(f"  {'':5} {line}")
    print("\n  rule 1: a process rule read from the trace alone. rule 2: an answer rule,")
    print("  used only when no process rule fired; it needs the expected evidence.")

    rule("STEP 3  The transition failure heatmap: before and after the switches")
    for name in runs:
        print()
        for line in heatmap_lines(heatmap(fails[name]), f"{name}: {sum(f is not None for f in fails[name])} "
                                                       f"wrong of {len(queries)}"):
            print(line)
    print("\n  Rows: the last state completed. Columns: the state the first failure was in.")
    print("  Column totals say which state fails most; the row says after what.")

    if DEFAULT.offline or ablations:
        rule("STEP 3b  One planted defect at a time: does each land in its own cell?")
        base = dict(zip([q["id"] for q in queries], fails["agent"]))
        for label, acfg in cfgs.items():
            if label in ("agent-naive", "agent"):
                continue
            moved = []
            for q in queries:
                t = r.run(acfg, q)
                f = first_failure(t, q, offered(acfg))
                f0 = base[q["id"]]
                if cell(f) != cell(f0):
                    moved.append(f"{q['id']} {cell(f)}")
            print(f"  {label:26} {', '.join(moved) if moved else 'no change on this set'}")
        print("\n  A '-' is a question the defect fixed: a3 fits the budget once a check is gone.")
    else:
        rule("STEP 3b  Planted defects")
        print("  Skipped live (6 x 13 runs). Add --ablations, ideally with --ids.")

    rule("STEP 4  Operations: what the path cost and how often a call worked")
    print(f"  {'agent':12} {'right':>6} {'calls/q':>8} {'max':>4} {'tool ok':>8} {'retries':>8} "
          f"{'re-search':>10} {'budget':>7} {'declined':>9}")
    for name, traces in runs.items():
        o = operations(traces, queries)
        ok = "-" if o["success"] is None else f"{o['success']:.0%}"
        rs = f"{o['re_searched']} ({o['recovered']} ok)"
        dec = f"{o['declined']} ({o['declined_right']} ok)"
        print(f"  {name:12} {o['right']:>3}/{o['questions']:<2} {o['calls_mean']:>8.1f} {o['calls_max']:>4} "
              f"{ok:>8} {o['retries']:>8} {rs:>10} {o['budget']:>7} {dec:>9}")
    print("\n  calls/q: tool calls per question, finish not counted. tool ok: calls that")
    print("  worked (Exec passed). retries: calls the harness ran again after an error.")
    print("  re-search: the agent searched again after a status check invalidated the")
    print("  top page, and (ok) how many of those then answered right. budget: runs")
    print("  that ran out of steps. declined: hand-offs to HR, (ok) where a decline")
    print("  was the right answer.")


def step_faults(r: Runner, queries: list[dict]) -> None:
    rule("STEP 5  Planted tool faults: retries, recovery and fallback")
    qs = [q for q in queries if q["id"] in FAULT_IDS] or queries[:3]
    print(f"  The agent with every switch on, on {', '.join(q['id'] for q in qs)}: all three are right with")
    print("  no fault in the reference run. One failure planted per row.\n")
    print(f"  {'condition':38} " + "".join(f"{q['id']:>9}" for q in qs) + f"{'tool ok':>9}  first failure")
    for label, faults, retries in FAULTS:
        traces = [r.run(AGENT_DEFAULT, q, faults=faults, retries=retries) for q in qs]
        o = operations(traces, qs)
        verdicts = []
        for t, q in zip(traces, qs):
            verdicts.append("right" if grade(t, q).correct else "declined" if t.refused else "WRONG")
        ff = {cell(first_failure(t, q, offered(AGENT_DEFAULT))) for t, q in zip(traces, qs)} - {"-"}
        ok = "-" if o["success"] is None else f"{o['success']:.0%}"
        print(f"  {label:38} " + "".join(f"{v:>9}" for v in verdicts) + f"{ok:>9}  {', '.join(sorted(ff)) or '-'}")
        if label.startswith("every"):
            invented = [q["id"] for t, q in zip(traces, qs) if not t.refused or t.citations]
            fallback = "declined, nothing invented" if not invented else f"INVENTED an answer: {', '.join(invented)}"
            print(f"  {'':38} fallback under an outage: {fallback}")
    print("\n  A retry recovers a call that raised. It cannot see a call that returned")
    print("  nothing: that one needs a check on the output (the Exec rule above), or a")
    print("  policy that searches again when a search comes back empty.")


def step_scope(r: Runner) -> None:
    rule("STEP 6  Scope: privilege escalation and 'must call a tool'")
    cfgs = configs()
    scope = json.loads(SCOPE.read_text(encoding="utf-8"))["queries"]
    print(f"  {'id':4} {'tests':6} {'question':58} {'answer':>7}")
    for q in scope:
        acfg = AGENT_DEFAULT if q["config"] == "agent" else cfgs[q["config"]]
        t = r.run(acfg, q)
        v = scope_violations(t, offered(acfg))
        bad = [k for k, xs in v.items() if xs]
        verdict = "-" if q.get("should_refuse") is None else "right" if grade(t, q).correct else "WRONG"
        print(f"  {q['id']:4} {q['invites'][:2]:6} {q['text'][:58]:58} {verdict:>7}"
              f"  {'violates ' + ', '.join(bad) if bad else 'in scope'}")
        print(f"  {'':11} answer: {' '.join(t.answer.split())[:64]}")
        for k in bad:
            for x in v[k][:2]:
                print(f"  {'':11} {k}: {x}")

    print("\n  In scope is not the same as right: S1 to S4 read the path, and the answer")
    print("  is graded as in step 2 (p1 and p2 should decline, p3 should say 25 days).")
    print(f"\n  Every trace this run made ({len(r.made)}), checked:")
    totals = {k: 0 for k in SCOPE_CHECKS}
    for t, _, off in r.made:
        for k, xs in scope_violations(t, off).items():
            totals[k] += bool(xs)
    for k, desc in SCOPE_CHECKS.items():
        print(f"    {k}  {desc:75} {totals[k]:>3} violating")

    print("\n  Controls: one violation planted in a copy of a right trace. A check that")
    print("  has never fired has never been tested.")
    by_id = {q["id"]: q for q in json.loads(SUITE.read_text(encoding="utf-8"))["queries"]}
    good = r.run(AGENT_DEFAULT, by_id["a2"])
    restricted = next(c for c in r.index.chunks if c.access_label not in ("all",))
    for k in SCOPE_CHECKS:
        t = plant(good, k, restricted)
        fired = bool(scope_violations(t, offered(AGENT_DEFAULT))[k])
        print(f"    {k}  planted, the check {'fires' if fired else 'DOES NOT FIRE: the check is broken'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["paths", "faults", "scope"], default=None)
    ap.add_argument("--ids", default="", help="comma-separated question ids, e.g. a2,a7")
    ap.add_argument("--ablations", action="store_true", help="run step 3b live (always on offline)")
    args = ap.parse_args(argv)

    banner(DEFAULT)
    queries = json.loads(SUITE.read_text(encoding="utf-8"))["queries"]
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",")}
        queries = [q for q in queries if q["id"] in wanted]
    r = Runner()
    if args.only in (None, "paths"):
        step_paths(r, queries, args.ablations)
    if args.only in (None, "faults"):
        step_faults(r, queries)
    if args.only in (None, "scope"):
        step_scope(r)

    rule("WHAT TO READ")
    print("  The heatmap's hotspot is where to start reading traces, not the cause:")
    print("  open the traces in that cell and read the step before the failing one.")
    print("  After a fix, run it again. The cell you fixed should empty, and a new one")
    print("  may fill: that is the next thing to fix.")
    print("  A retry is not recovery until the answer is right; a fallback is right")
    print("  when it declines and invents nothing.")
    print("  Offline, the planner cannot leave its scope, so S1 to S4 pass; the controls")
    print("  prove the checks work. Live, run step 6 on every model you consider.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
