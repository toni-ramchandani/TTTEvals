"""Day 3 through the toolkits: each curriculum item, ours beside DeepEval, G-Eval and Ragas.

Each Day 3 module ends with this lens. It prints, for every item the
curriculum names, the workshop's own instrument and the DeepEval, G-Eval and
Ragas metric that measures the same thing (or a dash, with the reason), then
runs the toolkit side live on a question or two, so the same trace is read
by all of them at once.

  --module 07  contextual metrics on each stage of one ranking: retrieved,
               reranked, reached the model. People's verdicts beside the
               judge's, with the arithmetic both toolkits use.
  --module 08  correctness, relevance, completeness, faithfulness, context
               utilization and citations, each four ways; then what the
               scores say about an answer faithful to a superseded page.
  --module 09  the toolkits on attacks, a planted page and noise, graph
               answers, and agent runs with planted faults.

Every judge call is live, on your judge model; nothing is saved. Offline, it
prints the map and our own columns; the toolkit columns, even the three
rule-based tool metrics, run only in the toolkit environment, live.

Run:  python run.py toolkits --module 07                 q1, about 40 judge calls
      python run.py toolkits --module 08                 q1, about 25 judge calls
      python run.py toolkits --module 08 --ids q1,q6
      python run.py toolkits --module 09                 four parts, about 90 calls
      python run.py toolkits --module 09 --part agents   one part
      python run.py toolkits --module 07 --offline       the map and people's column
DeepEval and Ragas run only in the toolkit environment (requirements-toolkits.txt).
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from evals.day3_lens import NO_TOOLKIT, by_module, stages, verdict_string  # noqa: E402
from helix import DEFAULT, REPAIRED, build_index  # noqa: E402
from helix import answer as helix_answer  # noqa: E402
from helix.console import banner  # noqa: E402
from toolkits.cases import case_from_trace, numbered_citations, with_contexts  # noqa: E402
from toolkits.kits import Kits, val  # noqa: E402

TRACES = ROOT / "data" / "traces" / "day1.jsonl"
SUITE = ROOT / "data" / "queries" / "day1_seed.json"
ARCH_SUITE = ROOT / "data" / "queries" / "architectures.json"
PARTS = ("attacks", "adversarial", "graph", "agents")
PASS = 0.5  # DeepEval's default threshold; nobody calibrated it for this task


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "pass" if v else "FAIL"
    return f"{v:.2f}"


def short(text: str, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 3] + "..."


def wrapped(label: str, text: str, width: int = 78, indent: int = 4) -> None:
    lines = textwrap.wrap(text, width - indent - len(label) - 1) or ["-"]
    print(" " * indent + label + " " + lines[0])
    for ln in lines[1:]:
        print(" " * (indent + len(label) + 1) + ln)


def show_map(module: str) -> None:
    rule(f"MODULE {module}: each curriculum item, ours beside the toolkits")
    for it in by_module(module):
        print(f"\n  {it.item}  (slides {it.where})")
        wrapped("ours     ", it.ours)
        wrapped("DeepEval ", it.deepeval)
        wrapped("G-Eval   ", it.geval)
        wrapped("Ragas    ", it.ragas)
        if it.item in NO_TOOLKIT:
            wrapped("note:", NO_TOOLKIT[it.item])


def day1(ids: list[str]):
    """The Day 1 traces (your model's answers) with their suite items."""
    from tracing import read_jsonl

    from helix.console import traces_note

    suite = json.loads(SUITE.read_text(encoding="utf-8"))["queries"]
    if not TRACES.exists():
        return suite, []
    by_q = {q["text"]: q for q in suite}
    traces = read_jsonl(TRACES)
    traces_note(traces, DEFAULT)
    pairs = [(by_q.get(t.query), t) for t in traces]
    return suite, [(q, t) for q, t in pairs if q and q["id"] in ids]


def missing_note(kits: Kits) -> None:
    if DEFAULT.offline:
        print("\nOffline: the toolkit columns run only live, in the toolkit environment.")
        print("Drop --offline there to fill them.")
    elif kits.missing():
        print(f"\nNot installed here: {', '.join(kits.missing())}. Those columns stay empty.")
        print("The toolkit environment is set up in requirements-toolkits.txt.")


# ---------------------------------------------------------------------------
# Module 07
# ---------------------------------------------------------------------------
def module_07(ids: list[str], kits: Kits) -> None:
    from evals.day3_retrieval import judgments
    from helix.ingest import chunk_corpus

    _, pairs = day1(ids)
    if not pairs:
        print("\nRun `python run.py day1` first; the lens reads its traces.")
        return
    corpus = chunk_corpus(DEFAULT)
    live = not DEFAULT.offline
    for q, t in pairs:
        grades = judgments(corpus, q.get("expected_evidence"))
        st = stages(t, grades)
        base = case_from_trace(t, q)
        rule(f"STEP 1  {q['id']}: one ranking, three lists, people's verdicts")
        print(f"  {short(q['text'], 72)}")
        print(f"  reference: {short(q.get('ground_truth', ''), 64)}\n")
        print(f"  {'stage':<18} {'chunks':>6}  people's verdicts, in rank order")
        for s in st:
            print(f"  {s.name:<18} {len(s.chunks):>6}  {verdict_string(s.verdicts)}")

        rule(f"STEP 2  {q['id']}: the toolkits' contextual metrics on each list")
        print("  P = context precision: average precision over the list, the toolkits'")
        print("  own arithmetic. people P: that arithmetic on people's verdicts. people R:")
        print("  the share of the relevant chunks in the list. The toolkits' recall reads")
        print("  the reference's claims instead: can the list support each of them?\n")
        print(f"  {'stage':<17} {'people P':>8} {'people R':>8} {'Ragas P':>8} {'Ragas R':>8} "
              f"{'DE P':>6} {'DE R':>6} {'DE rel':>7}")
        print("  " + "-" * 74)
        judged = []
        for s in st:
            c = with_contexts(base, s.chunks)
            rp = val(kits.r("context_precision", c)) if live else None
            rr = val(kits.r("context_recall", c)) if live else None
            dp, metric = kits.d_keep("contextual_precision", c) if live else (None, None)
            dr = val(kits.d("contextual_recall", c)) if live else None
            drel = val(kits.d("contextual_relevancy", c)) if live else None
            judged.append((s, metric))
            print(f"  {s.name:<17} {fmt(s.precision):>8} {fmt(s.recall):>8} {fmt(rp):>8} {fmt(rr):>8} "
                  f"{fmt(val(dp)):>6} {fmt(dr):>6} {fmt(drel):>7}")

        if live and any(m is not None for _, m in judged):
            rule(f"STEP 3  {q['id']}: where DeepEval's judge and people disagree")
            print(f"  {'stage':<17} {'chunk':<20} {'people':>6} {'judge':>5}  judge's reason")
            for s, m in judged:
                verdicts = getattr(m, "verdicts", None) or []
                for c, pv, jv in zip(s.chunks, s.verdicts, verdicts):
                    j = str(getattr(jv.verdict, "value", jv.verdict)).lower()
                    flag = "" if (j == "yes") == bool(pv) else "  <-"
                    print(f"  {s.name:<17} {c.chunk_id:<20} {'YES' if pv else 'no':>6} {j:>5}  "
                          f"{short(getattr(jv, 'reason', ''), 18)}{flag}")
    missing_note(kits)
    rule("WHAT TO READ")
    print("  Both toolkits compute the same average precision; the judge only decides")
    print("  which chunks count. Where its verdicts match people's, the toolkit column")
    print("  equals the people column. Where they differ, the difference is the judge.")
    print("  Their precision divides by the relevant chunks in the list, so a chunk")
    print("  retrieval missed cannot lower it: read recall beside it. And no toolkit")
    print("  score names the stage that lost the evidence. The drop record does.")


# ---------------------------------------------------------------------------
# Module 08
# ---------------------------------------------------------------------------
def module_08(ids: list[str], kits: Kits) -> None:
    from evals.day3_generation import (
        citation_support,
        context_use,
        direct_answer_check,
        nugget_coverage,
    )
    from evals.day3_judges import correct_check, grounded_check, quadrant
    from evals.day3_redteam import lineage_pass

    _, pairs = day1(ids)
    if not pairs:
        print("\nRun `python run.py day1` first; the lens reads its traces.")
        return
    live = not DEFAULT.offline
    for q, t in pairs:
        base = case_from_trace(t, q)
        nuggets = q.get("nuggets")
        reached = [c for c in t.retrieved if not c.dropped_at]
        full = nuggets if isinstance(nuggets, list) else []
        vital = replace(base, reference="Vital facts: " + "; ".join(n[0] for n in full)) if full else base
        cited = with_contexts(base, reached, labelled=True)
        numbered = numbered_citations(base)

        cov = nugget_coverage(t.answer, nuggets)
        use = context_use(t, nuggets)
        sup = citation_support(t)
        direct = direct_answer_check(q["text"], t.answer)
        grounded, correct = grounded_check(t), correct_check(t, q.get("ground_truth", ""))
        used = sum(1 for u in use if u["in_answer"])
        omitted = sum(1 for u in use if u["in_context"] and not u["in_answer"])
        lost = sum(1 for u in use if not u["in_context"] and not u["in_answer"])
        ours = {
            "correctness": fmt(correct),
            "relevance": "not yes/no" if direct is None else fmt(direct),
            "completeness": (f"{sum(cov['found'])} of {len(cov['found'])} nuggets"
                             if cov["found"] else "-"),
            "faithfulness": fmt(grounded),
            "context utilization": ((f"{used}/{len(use)} used, {lost} not in ctx" if lost else
                                     f"{used}/{len(use)} used, {omitted} omitted") if use else "-"),
            "citations": (f"{sum(r['supported'] for r in sup)} of {len(sup)} supported"
                          if sup else "no citation"),
            "consistency": "--only consistency",
        }
        cells = {k: {"DeepEval": None, "G-Eval": None, "Ragas": None} for k in ours}
        if live:
            cells["correctness"]["G-Eval"] = val(kits.g("correctness", base))
            cells["correctness"]["Ragas"] = val(kits.r("factual_correctness", base))
            cells["relevance"]["DeepEval"] = val(kits.d("answer_relevancy", base))
            cells["relevance"]["Ragas"] = (val(kits.r("answer_relevancy", base))
                                           if kits.ragas and "answer_relevancy" in kits.ragas else None)
            cells["completeness"]["G-Eval"] = val(kits.g("completeness", vital)) if full else None
            cells["completeness"]["Ragas"] = val(kits.r("factual_recall", vital)) if full else None
            cells["faithfulness"]["DeepEval"] = val(kits.d("faithfulness", base))
            cells["faithfulness"]["Ragas"] = val(kits.r("faithfulness", base))
            cells["context utilization"]["Ragas"] = val(kits.r("context_utilization", base))
            cells["citations"]["G-Eval"] = val(kits.g("citation support", cited)) if sup else None
            cells["citations"]["DeepEval"] = val(kits.d("citation_faithfulness", numbered)) if sup else None

        rule(f"STEP 1  {q['id']}: each criterion, four ways")
        print(f"  {short(q['text'], 72)}")
        print(f"  answer: {short(t.answer, 68)}")
        print(f"  reference: {short(q.get('ground_truth', ''), 65)}\n")
        print(f"  {'criterion':<20} {'ours':<22} {'DeepEval':>9} {'G-Eval':>7} {'Ragas':>7}")
        print("  " + "-" * 68)
        for k, o in ours.items():
            c = cells[k]
            print(f"  {k:<20} {o:<22} {fmt(c['DeepEval']):>9} {fmt(c['G-Eval']):>7} {fmt(c['Ragas']):>7}")
        print("\n  Toolkit scores run 0 to 1 and pass at 0.5, DeepEval's default. Ragas")
        print("  correctness and completeness are factual correctness, F1 and recall mode;")
        print("  both completeness cells read the suite's vital facts. DeepEval citations is")
        print("  citation faithfulness: one verdict, 1 or 0, on the answer with [N] markers.")

        rule(f"STEP 2  {q['id']}: unsupported, or faithful to the wrong evidence?")
        lin = lineage_pass(t)
        print(f"  ours: grounded {fmt(grounded)}, correct {fmt(correct)}, lineage {fmt(lin)}")
        print(f"        -> {quadrant(grounded, correct)}")
        if live:
            f_scores = [x for x in (cells["faithfulness"]["DeepEval"], cells["faithfulness"]["Ragas"]) if x is not None]
            c_scores = [x for x in (cells["correctness"]["G-Eval"], cells["correctness"]["Ragas"]) if x is not None]
            if f_scores and c_scores:
                faithful = min(f_scores) >= PASS
                right = max(c_scores) >= PASS
                print(f"  toolkits: faithfulness {'passes' if faithful else 'fails'}, "
                      f"correctness {'passes' if right else 'fails'}")
                if faithful and not right:
                    print("  -> faithful and wrong: the evidence itself is wrong. Only lineage says")
                    print("     whether it is superseded or restricted; no toolkit reads the version.")
                elif not faithful and not right:
                    print("  -> unsupported and wrong: the answer went beyond its context.")

        if live and kits.deepeval:
            rule("STEP 3  The G-Eval steps your judge wrote (read them; they are the rubric)")
            for crit in ("correctness", "completeness", "citation support"):
                steps = kits.geval_steps(crit)
                if steps:
                    print(f"  {crit}:")
                    for i, s in enumerate(steps[:5], 1):
                        wrapped(f"{i}.", s, indent=4)
    missing_note(kits)
    rule("WHAT TO READ")
    print("  Each toolkit metric answers its own question. Faithfulness asks whether the")
    print("  context supports the answer; correctness asks whether the answer matches the")
    print("  reference. An answer faithful to a superseded page passes the first and")
    print("  fails the second. DeepEval's citation metric gives one verdict per answer;")
    print("  the G-Eval written here scores each cited sentence. No single answer can")
    print("  show consistency. Every toolkit score is an instrument like ours: measure")
    print("  it against people (8.15) before you trust it.")


# ---------------------------------------------------------------------------
# Module 09
# ---------------------------------------------------------------------------
def offline_cfg(cfg):
    return replace(cfg, offline=True)


def part_attacks(kits: Kits, live: bool) -> None:
    from evals.day3_judges import correct_check
    from evals.day3_redteam import attack, reference_answers, support_pass, targets

    rule("PART 1  Evaluator red-teaming: q5x (30 days; the handbook says 25)")
    suite = json.loads(SUITE.read_text(encoding="utf-8"))["queries"]
    t5, q5 = targets(reference_answers(suite), suite)["q5x"]
    print(f"  answer: {short(t5.answer, 68)}")
    print("  It is wrong and its figure is in no page: every pass below is a false pass.\n")
    print(f"  {'attack':<22} {'checker':>8} {'support':>8} {'G-Eval corr':>12} {'Ragas fact':>11} "
          f"{'DE faith':>9}")
    print("  " + "-" * 74)
    for a in ("none", "evaluator instruction", "master key"):
        t = attack(a, t5)
        c = case_from_trace(t, q5)
        g = val(kits.g("correctness", c)) if live else None
        rf = val(kits.r("factual_correctness", c)) if live else None
        df = val(kits.d("faithfulness", c)) if live else None
        print(f"  {a:<22} {fmt(correct_check(t, q5.get('ground_truth', ''))):>8} {fmt(support_pass(t)):>8} "
              f"{fmt(g):>12} {fmt(rf):>11} {fmt(df):>9}")
    print("\n  A toolkit score of 0.5 or more is a pass. The full table, every attack on")
    print("  eight answers: python run.py redteam --toolkits")


def part_adversarial(kits: Kits, live: bool) -> None:
    from evals.day3_robustness import check, load_suite, noise_chunk, planted_chunk, with_planted

    rule("PART 2  Adversarial inputs: a noise chunk and a planted page")
    rs = load_suite()
    cfg = offline_cfg(DEFAULT)
    index = build_index(cfg)
    n1 = next(c for c in rs["cases"] if c["id"] == "n1")
    tn = helix_answer(n1["text"], index, cfg, user_office=n1.get("user_office", "amsterdam"),
                      user_role=n1.get("user_role", "employee"), extra_context=[noise_chunk(rs)])
    planted = planted_chunk(rs)
    j1 = next(c for c in rs["cases"] if c["id"] == "j1")
    tj = helix_answer(j1["text"], with_planted(index, planted, cfg), cfg,
                      user_office=j1.get("user_office", "amsterdam"), user_role=j1.get("user_role", "employee"))
    print(f"  n1 (noise chunk added): {short(tn.answer, 50)}")
    noise = None
    if live and kits.ragas:
        from toolkits import ragas_metrics as R
        from toolkits.judge import ragas_llm

        m = R.score_noise(R.build_noise(ragas_llm(DEFAULT, kits.stats)), n1["text"], tn.answer,
                          n1.get("reference", ""), [c.text for c in tn.retrieved if not c.dropped_at])
        noise = m.score
    print(f"    ours {fmt(check(n1, tn).passed)}   Ragas noise sensitivity {fmt(noise)} (lower is better)")
    ref = "130 working days / 26 weeks, primary caregiver, Amsterdam office (HB-PAR-2024 v4)"
    cj = replace(case_from_trace(tj), reference=ref)
    print(f"\n  j1 (planted page in the index): {short(tj.answer, 44)}")
    ours = check(j1, tj, planted.chunk_id)
    rf = val(kits.r("faithfulness", cj)) if live else None
    df = val(kits.d("faithfulness", cj)) if live else None
    g = val(kits.g("correctness", cj)) if live else None
    print(f"    ours {fmt(ours.passed)}   Ragas faithfulness {fmt(rf)}   DeepEval faithfulness {fmt(df)}")
    print(f"    G-Eval correctness {fmt(g)}   why ours fails: {short('; '.join(ours.problems), 30)}")
    print("\n  The planted answer follows its context, so faithfulness can pass it. The")
    print("  citation check says the cited chunk is not a handbook page.")


def part_graph(kits: Kits, live: bool) -> None:
    from evals.day3_architectures import attribute, grade
    from graphrag import GRAPH_DEFAULT
    from graphrag import answer as graph_answer
    from graphrag import build as graph_build

    rule("PART 3  GraphRAG: the same metrics read graph facts and text chunks")
    aq = {q["id"]: q for q in json.loads(ARCH_SUITE.read_text(encoding="utf-8"))["queries"]}
    rcfg = offline_cfg(REPAIRED)
    gcfg = replace(GRAPH_DEFAULT, base=rcfg)
    index, graph = build_index(rcfg), graph_build(gcfg)
    print(f"  {'id':<4} {'system':<8} {'ours':>5} {'codes':<9} {'G-Eval corr':>12} {'Ragas faith':>12}  answer")
    print("  " + "-" * 74)
    for qid in ("a4", "a11"):
        q = aq[qid]
        kw = dict(user_office=q["user_office"], user_role=q["user_role"])
        runs = [("vanilla", helix_answer(q["text"], index, rcfg, **kw), None),
                ("graph", graph_answer(q["text"], graph, gcfg, index=index,
                                       user_tenure_years=q.get("user_tenure_years"), **kw), graph)]
        for name, t, g_ in runs:
            ok = grade(t, q).correct
            codes = "-" if ok else "/".join(attribute(t, q, graph=g_)) or "?"
            c = case_from_trace(t, q)
            g = val(kits.g("correctness", c)) if live else None
            rf = val(kits.r("faithfulness", c)) if live else None
            print(f"  {qid:<4} {name:<8} {fmt(ok):>5} {codes[:9]:<9} {fmt(g):>12} {fmt(rf):>12}  "
                  f"{short(t.answer, 15)}")
    print("\n  a4 needs several constraints at once; a11 is the question no extraction")
    print("  rule covers. A graph answer can be faithful to its facts and still wrong.")


def part_agents(kits: Kits, live: bool) -> None:
    from agentic import AGENT_DEFAULT
    from agentic import answer as agent_answer
    from agentic.agent import live_tools
    from evals.day3_agents import first_failure
    from evals.day3_architectures import grade

    rule("PART 4  Agents: tool routing, trajectories, faults, task completion")
    aq = {q["id"]: q for q in json.loads(ARCH_SUITE.read_text(encoding="utf-8"))["queries"]}
    base = offline_cfg(REPAIRED)
    index = build_index(base)
    agent = replace(AGENT_DEFAULT, base=base)
    runs = [
        ("a7", "stale cited", replace(agent, act_on_observations=False), {}),
        ("a2", "no fault", agent, {}),
        ("a2", "retried", agent, {"faults": {"search_handbook": ("timeout", {1})}, "retries": 1}),
        ("a2", "empty result", agent, {"faults": {"search_handbook": ("empty", {1})}, "retries": 1}),
    ]
    rows = []
    for qid, label, acfg, kw in runs:
        q = aq[qid]
        t = agent_answer(q["text"], index, acfg, user_office=q["user_office"], user_role=q["user_role"],
                         user_tenure_years=q.get("user_tenure_years"), **kw)
        offered = {x["function"]["name"] for x in live_tools(acfg)}
        f = first_failure(t, q, offered)
        c = case_from_trace(t, q)
        cells = {
            "tools": val(kits.d("tool_correctness", c)) if live else None,
            "order": val(kits.d("tool_correctness_ordered", c)) if live else None,
            "args": val(kits.d("argument_correctness", c)) if live else None,
            "task": val(kits.d("task_completion", c)) if live else None,
            "F1": val(kits.r("tool_call_f1", c)) if live else None,
            "exact": val(kits.r("tool_call_accuracy", c)) if live else None,
            "goal": val(kits.r("goal_accuracy", c)) if live else None,
        }
        rows.append((qid, label, grade(t, q).correct, f, cells, t))
    print(f"  {'':<34}{'DeepEval':^24}{'Ragas':^18}")
    print(f"  {'run':<15} {'ours':>4} {'first failure':<13}" + "".join(f"{k:>6}" for k in rows[0][4]))
    print("  " + "-" * 76)
    for qid, label, ok, f, cells, _ in rows:
        ff = "-" if f is None else f"{f.from_state[:6]}->{f.state}"
        print(f"  {qid + ' ' + label:<15} {fmt(ok):>4} {ff:<13}" + "".join(f"{fmt(v):>6}" for v in cells.values()))
    print("\n  DeepEval: tool correctness, the same in order (a subsequence: extra calls")
    print("  allowed), argument correctness, task completion. Ragas: tool call F1, tool")
    print("  call accuracy (the exact sequence), goal accuracy (against the reference).")
    print("  The four tool-name scores read names only; the stage check reads what each")
    print("  tool returned. Exact order wants one reference path: the suite lists the")
    print("  tools a question needs, so a right run with more calls scores 0 there.")


def module_09(parts: list[str], kits: Kits) -> None:
    live = not DEFAULT.offline
    for p in parts:
        {"attacks": part_attacks, "adversarial": part_adversarial, "graph": part_graph,
         "agents": part_agents}[p](kits, live)
    missing_note(kits)
    rule("WHAT TO READ")
    print("  The failures Module 09 plants hide from the scores that read only names")
    print("  or text: an instruction aimed at the judge, a page's origin, an extraction")
    print("  rule, an empty tool result. Pair each toolkit score with the check that")
    print("  reads the trace, and attack every judge you use, toolkit judges included.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", required=True, choices=["07", "08", "09"])
    ap.add_argument("--ids", default="", help="Day 1 question ids for 07 and 08 (default q1)")
    ap.add_argument("--part", default="", help="for 09, comma-separated: " + ", ".join(PARTS))
    args = ap.parse_args(argv)
    ids = [x.strip() for x in args.ids.split(",") if x.strip()] or ["q1"]
    parts = [x.strip() for x in args.part.split(",") if x.strip()] or list(PARTS)
    bad = [p for p in parts if p not in PARTS]
    if bad:
        print(f"unknown part: {', '.join(bad)} (choose from {', '.join(PARTS)})")
        return 2
    banner(DEFAULT)
    show_map(args.module)
    kits = Kits(DEFAULT)
    if args.module == "07":
        module_07(ids, kits)
    elif args.module == "08":
        module_08(ids, kits)
    else:
        module_09(parts, kits)
    if kits.stats is not None and kits.stats.calls:
        print(f"\ntoolkit judge: {kits.stats.line()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
