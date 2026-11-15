"""Day 3 hands-on, Module 09: conversations, judged per session, then per turn.

Sara does not ask one question. She follows up. This runs eight scripted
sessions (2 to 6 turns, data/queries/day3_conversations.json) through Helix
in three history modes and evaluates them in the order the method asks:

  STEP 1  sessions: pass or fail, and the first turn that failed, per mode
            last      each message alone
            window    the last turns in the prompt, retrieval on the raw message
            condense  a standalone question for retrieval, the turns in the prompt
  STEP 2  the first failing turns: what failed, its code (MT1 to MT4, or a
          single-turn FP code), and the query retrieval actually used
  STEP 3  results by session length
  STEP 4  reduce each conversation failure to the simplest test that still
          fails: an N-1 test (the first turns fixed, the failing turn
          generated again in every mode) and the same turn asked alone
  STEP 5  live only: a session judge beside the code verdict; with
          --toolkits, DeepEval's and Ragas's conversation metrics too
  STEP 6  live only, with --repeat k: pass^k, the share of sessions that pass
          in ALL k runs

The sessions run on the repaired configuration by default, so Day 1's
single-turn failures do not hide the conversation's own. --config default
runs them on Day 1's Helix instead.

Every call is live on your models; offline is the reference run. Traces go
to data/traces/day3_sessions.jsonl. Nothing else is written.

Run:  python run.py conversations
      python run.py conversations --ids c5,c8
      python run.py conversations --config default
      python run.py conversations --toolkits          DeepEval and Ragas beside ours (toolkit env)
      python run.py conversations --repeat 3          pass^3 (overnight)
      python run.py conversations --simulate          a simulated user plays each goal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from evals.day3_conversations import (  # noqa: E402
    STOP,
    evaluate,
    goal_met,
    judge_session,
    load_sessions,
    pass_hat_k,
    simulated_message,
)
from evals.taxonomy import BY_CODE  # noqa: E402
from helix import DEFAULT, REPAIRED, build_index  # noqa: E402
from helix.config import Config  # noqa: E402
from helix.console import banner, progress  # noqa: E402
from helix.conversation import HISTORY_MODES, messages_of, run, turn  # noqa: E402
from tracing import write_jsonl  # noqa: E402

OUT = ROOT / "data" / "traces" / "day3_sessions.jsonl"
MAX_SIM_TURNS = 6


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def cell(r) -> str:
    if r.passed:
        return "pass"
    f = r.first_fail
    return f"FAIL t{f.n} {f.code}"


def pick_config(name: str) -> Config:
    """The repaired arm (Day 1's fixes on) or Day 1's Helix, on the models set now."""
    return REPAIRED if name == "repaired" else DEFAULT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", default="", help="comma-separated session ids, e.g. c5,c8")
    ap.add_argument("--config", default="repaired", choices=["repaired", "default"])
    ap.add_argument("--repeat", type=int, default=1, help="run every session k times for pass^k (live)")
    ap.add_argument("--toolkits", action="store_true", help="DeepEval and Ragas conversation metrics (live)")
    ap.add_argument("--simulate", action="store_true", help="a simulated user plays each goal (live)")
    ap.add_argument("--user-model", default="", help="the model that plays the user (default: the judge)")
    args = ap.parse_args(argv)

    cfg = pick_config(args.config)
    banner(cfg)
    sessions = load_sessions()
    if args.ids:
        keep = {x.strip() for x in args.ids.split(",")}
        sessions = [s for s in sessions if s["id"] in keep]
    if not sessions:
        print("no session with those ids")
        return 1
    print(f"config   {args.config}: {cfg.describe()}")
    index = build_index(cfg)
    live = not cfg.offline

    rule("STEP 1  Sessions first: did the whole conversation pass, and where did it break?")
    print("Each cell: pass, or FAIL with the FIRST failing turn and its code.\n")
    print(f"{'id':<4} {'turns':>5}  {'last':<18} {'window':<18} {'condense':<18}")
    print("-" * 78)
    results: dict[str, dict] = {}
    all_traces = []
    total = len(sessions) * len(HISTORY_MODES)
    done = 0
    for s in sessions:
        results[s["id"]] = {}
        msgs = [t["text"] for t in s["turns"]]
        for mode in HISTORY_MODES:
            traces = run(msgs, index, cfg, mode, s["user_office"], s["user_role"], f"{s['id']}-{mode}")
            results[s["id"]][mode] = evaluate(s, traces, mode)
            all_traces += traces
            done += 1
            if live:
                progress(f"{s['id']} {mode}", done, total)
        r = results[s["id"]]
        print(f"{s['id']:<4} {len(s['turns']):>5}  {cell(r['last']):<18} {cell(r['window']):<18} "
              f"{cell(r['condense']):<18}")
    passed = {m: sum(results[s["id"]][m].passed for s in sessions) for m in HISTORY_MODES}
    print("-" * 78)
    print(f"{'pass':<4} {'':>5}  " + " ".join(f"{f'{passed[m]} of {len(sessions)}':<18}" for m in HISTORY_MODES))
    mt = {m: sum(1 for s in sessions if (results[s["id"]][m].first_fail or None) and
                 results[s["id"]][m].first_fail.code.startswith("MT")) for m in HISTORY_MODES}
    print(f"{'MT':<4} {'':>5}  " + " ".join(f"{f'{mt[m]} MT codes':<18}" for m in HISTORY_MODES))

    rule("STEP 2  The first failing turn of each session, read from its trace")
    for s in sessions:
        groups: dict[tuple, list[str]] = {}
        for mode in HISTORY_MODES:
            f = results[s["id"]][mode].first_fail
            if f is not None:
                t = results[s["id"]][mode].traces[f.n - 1]
                q = t.span("query.transform").attributes.get("rewritten", t.query)
                groups.setdefault((f.n, f.code, t.answer, q), []).append(mode)
        for (n, code, ans, q), modes in groups.items():
            head = code.split("/")[0]
            name = BY_CODE[head].name if head in BY_CODE else "missed refusal (Day 2)"
            t = results[s["id"]][modes[0]].traces[n - 1]
            f = results[s["id"]][modes[0]].first_fail
            print(f"{s['id']}  {', '.join(modes)}: turn {n}  {code}  {name}")
            print(f"    user      {t.query[:66]}")
            if q != t.query:
                print(f"    retrieval {q[:66]}")
            print(f"    answer    {' '.join((ans or '').split())[:66]}")
            print(f"    why       {'; '.join(f.problems)[:66]}")
    print("\nFirst failures only: a later turn can fail because an earlier one did, so")
    print("annotate the first, fix it, and look again.")

    rule("STEP 3  By length: do longer sessions fail more often?")
    lengths = sorted({len(s["turns"]) for s in sessions})
    print(f"{'turns':>5} {'sessions':>9}  " + " ".join(f"{m:>10}" for m in HISTORY_MODES))
    print("-" * 50)
    for n in lengths:
        group = [s for s in sessions if len(s["turns"]) == n]
        cells = [f"{sum(results[s['id']][m].passed for s in group)} pass" for m in HISTORY_MODES]
        print(f"{n:>5} {len(group):>9}  " + " ".join(f"{c:>10}" for c in cells))
    print("\nEight sessions show the method, not a trend. Laban et al. (2025) measured")
    print("the trend on 200,000 simulated conversations: performance fell 39 percent.")

    rule("STEP 4  Reduce each conversation failure to the simplest test that still fails")
    print("N-1: the turns before the failing one replayed exactly as in the `last` run,")
    print("then the failing turn answered again in each mode. Alone: the same turn")
    print("written as one standalone question, with no conversation at all.\n")
    print(f"{'id':<4} {'turn':>4}  {'code (last)':<12} {'N-1 last':>9} {'N-1 window':>11} {'N-1 condense':>13} {'alone':>7}")
    print("-" * 78)
    shown = 0
    for s in sessions:
        r = results[s["id"]]["last"]
        f = r.first_fail
        if f is None or not f.code.startswith("MT"):
            continue
        k = f.n
        prefix = messages_of(r.traces[: k - 1])
        spec = s["turns"][k - 1]
        verdicts = []
        for mode in HISTORY_MODES:
            tr = run([spec["text"]], index, cfg, mode, s["user_office"], s["user_role"], f"{s['id']}-n1-{mode}",
                     prefix=prefix)
            rr = evaluate({**s, "turns": s["turns"][k - 1:k]}, tr, mode)
            verdicts.append("pass" if rr.passed else "FAIL")
        alone_spec = {**spec, "text": spec.get("standalone", spec["text"]), "follow_up": False, "uses_fact": None}
        tr = run([alone_spec["text"]], index, cfg, "last", s["user_office"], s["user_role"], f"{s['id']}-alone")
        alone = evaluate({**s, "turns": [alone_spec]}, tr, "last")
        verdicts.append("pass" if alone.passed else "FAIL")
        print(f"{s['id']:<4} {k:>4}  {f.code:<12} {verdicts[0]:>9} {verdicts[1]:>11} {verdicts[2]:>13} {verdicts[3]:>7}")
        shown += 1
    if not shown:
        print("  no conversation code (MT) among the first failures in `last` mode")
    print("\nAlone passes: the conversation is the cause, so fix how history is carried.")
    print("Alone fails too: the problem is retrieval or the handbook, not the dialogue.")

    if live:
        rule("STEP 5  The same sessions, judged another way (condense mode)")
        print(f"{'id':<4} {'code':>5} {'judge':>6} {'first turn (code, judge)':>26}  judge's reason")
        print("-" * 78)
        agree = 0
        for s in sessions:
            r = results[s["id"]]["condense"]
            ok, first, why = judge_session(s, r.traces, cfg)
            code_first = r.first_fail.n if r.first_fail else 0
            agree += (ok == r.passed)
            print(f"{s['id']:<4} {'pass' if r.passed else 'FAIL':>5} "
                  f"{'-' if ok is None else ('pass' if ok else 'FAIL'):>6} "
                  f"{f'{code_first}, {first}':>26}  {why[:30]}")
        print(f"\nThe judge agrees with the code on {agree} of {len(sessions)} sessions. Before you trust it,")
        print("label sessions yourself and measure it, as on Day 2.")
        if args.toolkits:
            from toolkits import env
            from toolkits.judge import JudgeStats

            env.prepare()
            stats = JudgeStats()
            print("\nToolkits on the condense sessions (scores 0 to 1; DeepEval passes at 0.5):\n")
            print(f"{'id':<4} {'code':>5} {'DE retention':>13} {'DE relevancy':>13} {'DE faithful':>12} "
                  f"{'DE complete':>12} {'Ragas goal':>11}")
            print("-" * 78)
            de = ra = None
            if env.installed("deepeval"):
                from toolkits import deepeval_metrics as D
                from toolkits.cases import to_deepeval_conversation
                from toolkits.judge import deepeval_judge

                de = (D, deepeval_judge(cfg, stats))
            if env.installed("ragas"):
                from toolkits import ragas_metrics as R
                from toolkits.cases import to_ragas_conversation
                from toolkits.judge import ragas_llm

                ra = (R, R.build_conversation(ragas_llm(cfg, stats)))
            for s in sessions:
                r = results[s["id"]]["condense"]
                vals = []
                if de:
                    D, judge = de
                    tc = to_deepeval_conversation(r.traces, s)
                    for name, metric in D.conversation_metrics(judge).items():
                        vals.append(D.run(metric, tc).score)
                else:
                    vals += [None] * 4
                if ra:
                    R, built = ra
                    vals.append(R.score_conversation("goal_accuracy", built["goal_accuracy"],
                                                     to_ragas_conversation(r.traces)).score)
                else:
                    vals.append(None)
                f = ["-" if v is None else f"{v:.2f}" for v in vals]
                print(f"{s['id']:<4} {'pass' if r.passed else 'FAIL':>5} {f[0]:>13} {f[1]:>13} {f[2]:>12} "
                      f"{f[3]:>12} {f[4]:>11}")
            if not de and not ra:
                print("Neither toolkit is installed here (see requirements-toolkits.txt).")
            print(f"\ntoolkit judge: {stats.line()}")
            print("Read the rows where a toolkit score is high and the code says FAIL: a wrong")
            print("figure can be relevant, retained and faithful to a stale page.")

    if args.repeat > 1:
        rule(f"STEP 6  Reliability: pass^{args.repeat} over {args.repeat} runs of every session")
        if not live:
            print("Offline the stub answers the same way every time, so pass^k equals the")
            print("pass rate. Run it live: a model at temperature 0 can still vary.")
        else:
            print(f"{'mode':<10} {'pass^1':>8} {f'pass^{args.repeat}':>8}")
            print("-" * 30)
            for mode in HISTORY_MODES:
                succ = []
                for s in sessions:
                    c = 1 if results[s["id"]][mode].passed else 0
                    for _ in range(args.repeat - 1):
                        tr = run([t["text"] for t in s["turns"]], index, cfg, mode, s["user_office"],
                                 s["user_role"], f"{s['id']}-{mode}-rep")
                        c += evaluate(s, tr, mode).passed
                    succ.append(c)
                p1 = pass_hat_k(succ, args.repeat, 1)
                pk = pass_hat_k(succ, args.repeat, args.repeat)
                print(f"{mode:<10} {p1:>8.2f} {pk:>8.2f}")
            print("\npass^1 is the average pass rate. pass^k counts a session only if every run")
            print("passed. The gap between them is the unreliability a single run hides.")

    if args.simulate:
        rule("SIMULATED USERS  A model plays the employee, one fact per message")
        if not live:
            print("Needs a model: run without --offline. The scripted sessions above stand in.")
        else:
            model = args.user_model or cfg.judge_model or cfg.generation_model
            print(f"user model {model}, at most {MAX_SIM_TURNS} messages, condense mode\n")
            for s in sessions:
                prior: list[dict[str, str]] = []
                traces = []
                for n in range(1, MAX_SIM_TURNS + 1):
                    msg = simulated_message(s, prior, cfg, args.user_model or None)
                    if not msg or STOP in msg:
                        break
                    t = turn(msg, prior, index, cfg, "condense", s["user_office"], s["user_role"],
                             f"{s['id']}-sim", n)
                    traces.append(t)
                    prior += [{"role": "user", "content": msg}, {"role": "assistant", "content": t.answer}]
                all_traces += traces
                ok = goal_met(s, traces)
                print(f"{s['id']}  {len(traces)} turns  goal {'met' if ok else 'NOT met'}")
                for t in traces:
                    print(f"    user  {' '.join(t.query.split())[:70]}")
                    print(f"    helix {' '.join((t.answer or '').split())[:70]}")
            print("\nA simulated user can leak its goal or never stop. Read the transcripts")
            print("before you count anything.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(all_traces, OUT)
    print(f"\nTraces written to {OUT.relative_to(ROOT)} ({len(all_traces)} turns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
