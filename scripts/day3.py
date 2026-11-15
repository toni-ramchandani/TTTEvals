"""Day 3 hands-on, Modules 07 and 08: retrieval, generation, judges against people.

The Measure phase. Day 2 named the failure modes; this turns them into
numbers, and keeps retrieval quality separate from generation quality so that
a score tells you which layer to go and fix.

  STEP 1  retrieval (Module 07): an ingestion check for superseded documents;
          rank metrics at k = 5 on the ranking the `retrieve` span recorded,
          relevance judged over the whole corpus; each stage on its own; where
          each relevant chunk ended up; retrieval success against answer
          correctness. No model call.
  STEP 2  grounding (Module 08): groundedness and correctness scored
          separately, then crossed, by a cheap checker and by your judge
          model; and what the lineage check found in the citations.
  STEP 3  generation (Module 08): completeness by vital nuggets, context use
          (in the context but left out of the answer), citation support, and
          the direct-answer check on yes-or-no questions. No model call.
  STEP 4  validation (Module 08): every instrument against people, on its own
          criterion, as TPR and TNR with their counts (positive = the failure
          is present, as on Day 2). The labels are the Day 3 sheet:
          data/annotations/day3_sheet.csv, with grounded, correct and complete
          columns. In the toolkit environment, Ragas and DeepEval faithfulness
          are measured too.
  STEP 5  relevance (Module 08): the direct-answer check against the Day 2
          golden set's no_direct_answer labels, on every labelled trace that
          asks a yes-or-no question. No model call.

Three more steps run only when you name them, because they call the model:

  swap         the context-swap test: each question answered again on two
               unrelated chunks; the same answer means the model is not
               reading its context (6 calls).
  consistency  each question answered N times (--runs, default 3); do the
               figures stay the same? (6 x N calls)
  chunkjudge   an LLM judges each retrieved chunk's relevance on a 0 to 3
               scale (UMBRELA), measured against the suite's relevance
               (25 calls; offline it prints the prompt).

The reference run (--offline) uses annotator A's labels and runs only the
instruments that need no model.

Run:  python run.py day3
      python run.py day3 --only retrieval       seconds, no model call
      python run.py day3 --only grounding,generation
      python run.py day3 --only validation
      python run.py day3 --only swap
      python run.py day3 --only consistency --runs 5
      python run.py day3 --only chunkjudge
      python run.py day3 --judge anthropic/claude-sonnet-5-5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from evals import day3_sheet  # noqa: E402
from evals.day3_generation import (  # noqa: E402
    citation_support,
    complete_check,
    context_use,
    direct_answer_check,
    is_closed_question,
    key_facts,
    nugget_coverage,
    same_facts,
    swap_verdict,
    unrelated_chunks,
)
from evals.day3_judges import (  # noqa: E402
    CRITERIA,
    UMBRELA,
    correct_check,
    grounded_check,
    judge_chunk_relevance,
    judge_correctness,
    judge_groundedness,
    measure,
    quadrant,
)
from evals.day3_retrieval import mean, retrieval_row, retrieval_vs_answer, stale_versions  # noqa: E402
from helix import DEFAULT  # noqa: E402
from helix.console import banner, progress, traces_note  # noqa: E402
from helix.ingest import chunk_corpus, document_status  # noqa: E402
from tracing import read_jsonl  # noqa: E402

TRACES = ROOT / "data" / "traces" / "day1.jsonl"
SUITE = ROOT / "data" / "queries" / "day1_seed.json"
STEPS = ("retrieval", "grounding", "generation", "validation", "relevance")
EXTRA = ("swap", "consistency", "chunkjudge")


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def f2(x) -> str:
    return "-" if x is None else f"{x:.2f}"


def pct(x) -> str:
    return "  -" if x is None else f"{x:.0%}"


# ---------------------------------------------------------------------------
# STEP 1
# ---------------------------------------------------------------------------
def step_retrieval(suite, traces, cfg) -> None:
    rule("STEP 1  Retrieval: did the evidence arrive, and where was it lost?")
    corpus = chunk_corpus(cfg)
    known = {c.chunk_id for c in corpus}
    touched = {c.chunk_id for t in traces for c in t.considered}
    if not touched <= known:
        print("These traces were made with other chunking settings, so relevance is judged")
        print("over the chunks each trace touched, not the whole corpus. Recall may look")
        print("higher than it is. Run `python run.py day1` again with the settings now.\n")

    stale = stale_versions(corpus, document_status())
    print("Ingestion first: is any superseded document still in the index?")
    if stale:
        for r in stale:
            print(f"  {r['doc_id']} v{r['version']} ({r['last_modified']}) is superseded by "
                  f"{r['superseded_by']}, and {r['chunks']} of its chunks are still indexed.")
        print("  Two versions of one policy in the index: the old one can be retrieved and\n"
              "  quoted as current. Remove or version it at ingestion.\n")
    else:
        print("  no: every indexed document is current.\n")

    print("Ranking: the order the retrieve span recorded (top 5). Relevant: a chunk")
    print("carrying ALL the expected evidence, searched over the whole corpus. nDCG")
    print("also gives a smaller gain to a chunk carrying most of it. Everything at k = 5.\n")
    print(f"{'id':<4} {'rank':>4} {'hit':>4} {'P@5':>5} {'R@5':>5} {'F1@5':>5} {'RR':>5} {'AP':>5} {'nDCG':>5}  "
          "where the evidence went")
    print("-" * 78)
    rows, hits = {}, {}
    for q, t in zip(suite, traces):
        universe = corpus if touched <= known else t.considered
        row = retrieval_row(t, universe, q.get("expected_evidence"))
        if not row["relevant"]:
            why = ("no evidence to find: Helix must refuse (Day 2's entitlement check)"
                   if q.get("should_refuse") else "no chunk in the corpus carries the expected evidence")
            print(f"{q['id']:<4} {'-':>4} {'-':>4} {'-':>5} {'-':>5} {'-':>5} {'-':>5} {'-':>5} {'-':>5}  {why}")
            continue
        rows[q["id"]] = row
        hits[q["id"]] = bool(row["hit"])
        rank = ",".join(str(r) for r in row["ranks"]) or "-"
        went = "; ".join(sorted(set(row["fate"].values())))
        print(f"{q['id']:<4} {rank:>4} {int(row['hit']):>4} {f2(row['p']):>5} {f2(row['r']):>5} {f2(row['f1']):>5} "
              f"{f2(row['rr']):>5} {f2(row['ap']):>5} {f2(row['ndcg']):>5}  {went}")
    rs = list(rows.values())
    print(f"\nOver the {len(rs)} questions with evidence to find:  hit rate {f2(mean([r['hit'] for r in rs]))}"
          f"   MRR {f2(mean([r['rr'] for r in rs]))}   MAP {f2(mean([r['ap'] for r in rs]))}"
          f"   mean R@5 {f2(mean([r['r'] for r in rs]))}   mean F1@5 {f2(mean([r['f1'] for r in rs]))}")
    print("With one relevant chunk, P@5 cannot pass 0.20, so F1@5 stays low even when the")
    print("ranking is perfect. Read precision beside how many chunks are relevant.")

    print("\nEach stage on its own. R@5 above is the first stage's score: did it FIND the")
    print("evidence? The reranker's job is order: nDCG on its cutoff, before and after it")
    print("ran. Context assembly then decides what the model sees.\n")
    print(f"{'id':<4} {'nDCG@3 before rerank':>21} {'after rerank':>13}  reached the model")
    print("-" * 78)
    for qid, row in rows.items():
        st = row["stages"]
        moved = ""
        if st["ndcg_before"] is not None and st["ndcg_after"] is not None:
            d = st["ndcg_after"] - st["ndcg_before"]
            moved = "   rerank helped" if d > 1e-9 else "   rerank hurt" if d < -1e-9 else ""
        print(f"{qid:<4} {f2(st['ndcg_before']):>21} {f2(st['ndcg_after']):>13}  "
              f"{'yes' if st['reached'] else 'NO'}{moved}")

    print("\nRelevant chunks, and what happened to each:")
    for qid, row in rows.items():
        for cid, where in row["fate"].items():
            print(f"  {qid:<4} {cid:<18} {where}")

    correct = {q["id"]: correct_check(t, q.get("ground_truth", "")) for q, t in zip(suite, traces) if q["id"] in rows}
    cells = retrieval_vs_answer(hits, correct)
    print("\nRetrieval success against answer correctness (correctness checker):")
    print(f"  retrieved, answer right   {', '.join(cells[(True, True)]) or '-'}")
    print(f"  retrieved, answer WRONG   {', '.join(cells[(True, False)]) or '-'}"
          "   -> the evidence was found: look at context assembly and generation")
    print(f"  missed,    answer WRONG   {', '.join(cells[(False, False)]) or '-'}   -> fix retrieval first")
    print(f"  missed,    answer right   {', '.join(cells[(False, True)]) or '-'}"
          "   -> right by chance, or from the model's own memory")
    print("\nR@5 says whether retrieval found the evidence. Only the trace's drop record")
    print("says which later stage threw it away: a chunk can be ranked 3rd by retrieval")
    print("and still never reach the model. To see how chunking changes all of this:")
    print("  python run.py chunking")


# ---------------------------------------------------------------------------
# STEP 2
# ---------------------------------------------------------------------------
def judge_all(suite, traces, cfg) -> tuple[list[bool], list[bool], list[tuple]]:
    print(f"Judging with {cfg.judge_model or cfg.generation_model}, two calls per answer:")
    jg, jc, why = [], [], []
    for i, (q, t) in enumerate(zip(suite, traces), 1):
        vg = judge_groundedness(t, cfg)
        vc = judge_correctness(t, q.get("ground_truth", ""), cfg)
        jg.append(bool(vg))
        jc.append(bool(vc))
        why.append((q["id"], vg, vc))
        progress(f"{q['id']}  {vg.label:<12} {vc.label}", i, len(suite))
    print()
    return jg, jc, why


def step_grounding(suite, traces, cfg, judged) -> None:
    rule("STEP 2  Groundedness and correctness, crossed")
    print("Scored separately on purpose. Collapsing them into one accuracy number")
    print("is the most common error in RAG evaluation, and it hides the worst case.\n")
    cg = [grounded_check(t) for t in traces]
    cc = [correct_check(t, q.get("ground_truth", "")) for q, t in zip(suite, traces)]
    live = judged is not None
    if live:
        jg, jc, why = judged
        print(f"{'':4} {'grounded':^19} {'correct':^19}")
        print(f"{'id':<4} {'checker':>9} {'judge':>9} {'checker':>9} {'judge':>9}  verdict (judge)")
        print("-" * 78)
        for i, q in enumerate(suite):
            print(f"{q['id']:<4} {str(cg[i]):>9} {str(jg[i]):>9} {str(cc[i]):>9} {str(jc[i]):>9}  "
                  f"{quadrant(jg[i], jc[i])}")
        print("\nWhat the judge said, in its own words:")
        for qid, vg, vc in why:
            if not vg or not vc:
                reason = vc.reason if not vc else vg.reason
                print(f"  {qid}  {reason[:100] or '(no reason given)'}")
    else:
        print(f"{'id':<4} {'grounded':>9} {'correct':>8}  verdict (checker)")
        print("-" * 78)
        for i, q in enumerate(suite):
            print(f"{q['id']:<4} {str(cg[i]):>9} {str(cc[i]):>8}  {quadrant(cg[i], cc[i])}")
    print("\nThe citation check (lineage): each citation in the context, current, permitted.")
    found = False
    for q, t in zip(suite, traces):
        pp = t.span("postprocess")
        problems = (pp.attributes.get("citation_problems") if pp else None)
        if problems is None:
            print(f"  {q['id']:<4} not checked: this trace is older than the check (run day1 again)")
            found = True
        elif problems:
            print(f"  {q['id']:<4} {'; '.join(problems)}")
            found = True
    if not found:
        print("  no problem found in any citation")


# ---------------------------------------------------------------------------
# STEP 3
# ---------------------------------------------------------------------------
def step_generation(suite, traces, cfg) -> None:
    rule("STEP 3  Generation: what did the model do with the context it was given?")
    print("Vital nuggets: the facts each answer needs, written beside each question in")
    print("the suite. Each is looked for in the answer and in the context the model saw.\n")
    print(f"{'id':<4} {'complete':<9} nugget, and what happened to it")
    print("-" * 78)
    omitted, upstream = [], []
    for q, t in zip(suite, traces):
        cov = nugget_coverage(t.answer, q.get("nuggets"))
        uses = context_use(t, q.get("nuggets"))
        if not uses:
            print(f"{q['id']:<4} {cov['status']:<9} {cov['nuggets'][0] if cov['nuggets'] else '-'}: "
                  f"{'present' if all(cov['found']) else 'MISSING'}")
            continue
        for i, u in enumerate(uses):
            head = f"{q['id']:<4} {cov['status']:<9}" if i == 0 else " " * 14
            print(f"{head} {u['nugget']!r}: {u['verdict']}")
            if u["verdict"].startswith("OMITTED"):
                omitted.append(f"{q['id']} ({u['nugget']})")
            elif not u["in_answer"]:
                upstream.append(f"{q['id']} ({u['nugget']})")
    print(f"\nIn the context and left out of the answer: {', '.join(omitted) or 'none'}")
    print("  -> the model did not use what it was given: a generation fault (an omission).")
    print(f"Never in the context: {', '.join(upstream) or 'none'}")
    print("  -> retrieval or context assembly lost it (STEP 1 says which).")

    print("\nCitation support: does the cited chunk say what the sentence says? (figures")
    print("must match; at least 60 percent of the content words must appear)")
    rows_all = 0
    for q, t in zip(suite, traces):
        rows = citation_support(t)
        rows_all += len(rows)
        if not rows:
            continue
        ok = sum(r["supported"] for r in rows)
        bad = [r for r in rows if not r["supported"]]
        print(f"  {q['id']:<4} {ok}/{len(rows)} supported" +
              ("" if not bad else "   " + "; ".join(f"{', '.join(r['cited'])}: {r['why']}" for r in bad)))
    if not rows_all:
        print("  no answer cites a chunk")
    print("  Support and lineage are different checks: q1 cites a chunk that says exactly")
    print("  what the sentence says, from a page that HR replaced (STEP 2).")

    print("\nDirect answer: a yes-or-no question gets a yes, a no, or a refusal.")
    any_closed = False
    for q, t in zip(suite, traces):
        v = direct_answer_check(q["text"], t.answer)
        if v is None:
            continue
        any_closed = True
        print(f"  {q['id']:<4} {'pass' if v else 'FAIL: true and faithful, but no yes or no'}  {q['text']}")
    if not any_closed:
        print("  no yes-or-no question in this suite")


# ---------------------------------------------------------------------------
# STEP 4
# ---------------------------------------------------------------------------
def step_validation(suite, traces, cfg, judged) -> None:
    rule("STEP 4  Every instrument against people, on its own criterion")
    labels, source = day3_sheet.labels(traces, suite)
    gaps = day3_sheet.missing(traces, suite) if not cfg.offline else []
    if gaps:
        path = day3_sheet.write(traces, suite)
        print(f"The Day 3 sheet needs a label for every answer first ({len(gaps)} missing: {', '.join(gaps)}).")
        print(f"Open {path.relative_to(ROOT)} in VS Code or Excel and fill three columns,")
        print("y or n, for each of the six answers:\n")
        for k, v in CRITERIA.items():
            print(f"  {k:<9} {v}")
        print("\nLabel on your own, before you look at any instrument's verdict. Then run:")
        print("  python run.py day3 --only validation")
        return
    print(f"People: {source}.")
    print("Positive means the failure is present, as on Day 2: TPR is the share of the")
    print("failures people found that the instrument caught; TNR is the share of the")
    print("answers people passed that it passed too. Each rate shows its count.\n")
    ids = [q["id"] for q in suite]
    instruments: dict[str, list[tuple[str, list]]] = {
        "grounded": [("checker", [grounded_check(t) for t in traces])],
        "correct": [("checker", [correct_check(t, q.get("ground_truth", "")) for q, t in zip(suite, traces)])],
        "complete": [("nugget check", [complete_check(t, q.get("nuggets")) for q, t in zip(suite, traces)])],
    }
    if judged is not None:
        jg, jc, _ = judged
        instruments["grounded"].append(("our judge", jg))
        instruments["correct"].append(("our judge", jc))
        from toolkits.cases import cases_from_traces
        from toolkits.kits import Kits, passes, val

        kits = Kits(cfg)
        if kits.ragas or kits.deepeval:
            cases = cases_from_traces(traces, suite)
            if kits.ragas:
                instruments["grounded"].append(
                    ("Ragas faithfulness", [passes(val(kits.r("faithfulness", c))) for c in cases]))
            if kits.deepeval:
                instruments["grounded"].append(
                    ("DeepEval faithfulness", [passes(val(kits.d("faithfulness", c))) for c in cases]))
            print(f"(toolkit scores pass at 0.5, DeepEval's default; {kits.stats.line()})\n")
    print(f"{'criterion':<10} {'instrument':<22} {'TPR':>11} {'TNR':>11} {'undecided':>10}")
    print("-" * 78)
    off: list[str] = []
    for crit, insts in instruments.items():
        human = [labels.get(i, {}).get(crit) for i in ids]
        keep = [k for k, h in enumerate(human) if h is not None]
        hp = [human[k] for k in keep]
        for name, verdicts in insts:
            vs = [verdicts[k] for k in keep]
            r = measure(vs, hp)
            tpr = f"{pct(r['tpr'])} {r['caught']}/{r['failures']}"
            tnr = f"{pct(r['tnr'])} {r['passed']}/{r['passes']}"
            print(f"{crit:<10} {name:<22} {tpr:>11} {tnr:>11} {r['undecided']:>10}")
            for k, v, h in zip(keep, vs, hp):
                if v is not None and v != h:
                    off.append(f"  {ids[k]:<4} {crit:<9} people {'pass' if h else 'FAIL'}, {name} {'pass' if v else 'FAIL'}")
    if off:
        print("\nWhere an instrument and people disagree (read these before changing anything):")
        print("\n".join(off))
    print("\nA rate on zero failures or zero passes is blank: there was nothing to measure")
    print("it on. Six answers teach the method, not the measurement. The stop rule from")
    print("Day 2 holds: near 90 percent TPR and TNR on enough labelled traces.")


# ---------------------------------------------------------------------------
# STEP 5 and the steps you name
# ---------------------------------------------------------------------------
def step_relevance(cfg) -> None:
    rule("STEP 5  The direct-answer check against people: the Day 2 golden set")
    from evals.day2_sheets import load_items
    from scripts.evaluators import human_truth

    items = load_items()
    truth, source = human_truth(items)
    rows = [(it, truth[it.query]) for it in items if it.query in truth and is_closed_question(it.query)]
    if not rows:
        print("No labelled trace asks a yes-or-no question yet. The check needs the Day 2")
        print("labels: run `python run.py day2` and `python run.py agreement` first.")
        return
    verdicts = [direct_answer_check(it.query, it.trace.answer) for it, _ in rows]
    human = ["no_direct_answer" not in tr.labels for _, tr in rows]
    r = measure(verdicts, human)
    print(f"People: {source}. {len(rows)} labelled traces ask a yes-or-no question;")
    print("the failure is Day 2's no_direct_answer mode. A code check, no model call.\n")
    print(f"{'instrument':<24} {'TPR':>11} {'TNR':>11}")
    print("-" * 50)
    print(f"{'direct-answer check':<24} {pct(r['tpr'])} {r['caught']}/{r['failures']:<4} "
          f"{pct(r['tnr'])} {r['passed']}/{r['passes']}")
    off = [(it, v, h) for (it, _), v, h in zip(rows, verdicts, human) if v is not None and v != h]
    if off:
        print("\nWhere the check and people disagree:")
        for it, v, h in off:
            print(f"  {it.id:<4} people {'pass' if h else 'FAIL'}, check {'pass' if v else 'FAIL'}  "
                  f"{it.query[:60]}")
    print("\nA word-matching check sees 'yes' and 'no'. It cannot see a yes that answers a")
    print("different question: Day 2's judge for this mode reads meaning.")


def step_swap(suite, traces, cfg) -> None:
    rule("Context swap: the same question, answered on context about something else")
    print("Each question is answered again with its context replaced by two unrelated")
    print("chunks. An answer that does not change is not coming from the context.\n")
    from helix.llm import complete
    from helix.prompts import build_messages

    corpus = chunk_corpus(cfg)
    for i, (q, t) in enumerate(zip(suite, traces), 1):
        swapped = unrelated_chunks(corpus, q["text"], 2, q.get("user_role", "employee"))
        text, _ = complete(build_messages(q["text"], swapped), cfg)
        verdict = swap_verdict(t.answer, text)
        print(f"{q['id']:<4} context now {', '.join(c.chunk_id for c in swapped)}")
        print(f"     before: {' '.join(t.answer.split())[:66]}")
        print(f"     after:  {' '.join(text.split())[:66]}")
        print(f"     {verdict}")
    print("\nIf an answer survives the swap, the fix is usually the prompt (answer only from")
    print("the documents; say so when they do not contain the answer), not the retriever.")
    if cfg.offline:
        print("The reference stub copies a fact from its first chunk, so it always changes.")
        print("Run this live: your model may know some answers without any context.")


def step_consistency(suite, cfg, runs: int) -> None:
    rule(f"Consistency: each question answered {runs} times, the same settings")
    print("Wording may vary. The figures, and whether it declined, should not.\n")
    from helix import answer, build_index

    index = build_index(cfg)
    stable = 0
    for i, q in enumerate(suite, 1):
        outs = [answer(q["text"], index, cfg, user_office=q["user_office"], user_role=q["user_role"]).answer
                for _ in range(runs)]
        groups: list[str] = []
        for a in outs:
            if not any(same_facts(a, g) for g in groups):
                groups.append(a)
        stable += len(groups) == 1
        facts = ["declined" if key_facts(g)[0] == "declined" else ",".join(sorted(key_facts(g)[1])) or "no figure"
                 for g in groups]
        print(f"{q['id']:<4} {'consistent' if len(groups) == 1 else f'{len(groups)} DIFFERENT answers'}"
              f"   figures: {' | '.join(facts)}")
    print(f"\n{stable} of {len(suite)} questions gave the same facts every time.")
    if cfg.offline:
        print("The reference stub is deterministic, so it is always consistent. Run this live:")
        print("temperature 0 makes a model more repeatable, not perfectly repeatable.")


def step_chunkjudge(suite, traces, cfg) -> None:
    rule("An LLM judges chunk relevance (UMBRELA, 0 to 3), measured against the suite")
    from evals.day3_retrieval import judgments, ranking

    corpus = chunk_corpus(cfg)
    by_id = {c.chunk_id: c for c in corpus}
    if cfg.offline:
        q = suite[0]
        first = by_id.get(ranking(traces[0])[0])
        print("Offline there is no judge to measure. This is the prompt it would get for")
        print(f"{q['id']} and the first chunk retrieval ranked:\n")
        print(UMBRELA.format(query=q["text"], chunk=first.text if first else ""))
        print("Run it live: python run.py day3 --only chunkjudge   (25 judge calls)")
        return
    pairs = []
    for q, t in zip(suite, traces):
        if not q.get("expected_evidence"):
            continue
        grades = judgments(corpus, q.get("expected_evidence"))
        for cid in ranking(t)[:5]:
            if cid in by_id:
                pairs.append((q, cid, grades.get(cid, 0)))
    print(f"Judging {len(pairs)} (question, chunk) pairs with {cfg.judge_model or cfg.generation_model}:")
    scores = []
    for i, (q, cid, g) in enumerate(pairs, 1):
        scores.append(judge_chunk_relevance(q["text"], by_id[cid].text, cfg))
        progress(f"{q['id']} {cid}", i, len(pairs))
    print()
    rel = [(g >= 2, sc) for (_, _, g), sc in zip(pairs, scores)]
    yes = [sc for r, sc in rel if r]
    no = [sc for r, sc in rel if not r]
    caught = sum(1 for sc in yes if sc is not None and sc >= 2)
    rejected = sum(1 for sc in no if sc is not None and sc < 2)
    unread = sum(1 for sc in scores if sc is None)
    print("Here a positive is a RELEVANT chunk (the suite's expected evidence, all of it).\n")
    print(f"  of {len(yes)} chunks the suite calls relevant, the judge scored {caught} at 2 or 3")
    print(f"  of {len(no)} chunks the suite calls not relevant, the judge scored {rejected} at 0 or 1")
    print(f"  replies with no score: {unread}")
    print("\nThe judge needs no hand-made relevance labels, which is why it scales. It also")
    print("cannot compute recall: it only ever sees what retrieval returned.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="",
                    help="comma-separated steps: " + ", ".join(STEPS + EXTRA))
    ap.add_argument("--runs", type=int, default=3, help="answers per question for the consistency step")
    args = ap.parse_args(argv)
    only = {s.strip() for s in args.only.split(",") if s.strip()} or set(STEPS)
    unknown = only - set(STEPS) - set(EXTRA)
    if unknown:
        print(f"unknown step: {', '.join(sorted(unknown))} (choose from {', '.join(STEPS + EXTRA)})")
        return 2
    if not TRACES.exists():
        print("run `python run.py day1` first to produce data/traces/day1.jsonl")
        return 1
    traces = read_jsonl(TRACES)
    suite = json.loads(SUITE.read_text(encoding="utf-8"))["queries"]
    cfg = DEFAULT
    banner(cfg)
    traces_note(traces, cfg)

    if "retrieval" in only:
        step_retrieval(suite, traces, cfg)
    judged = None
    needs_judge = (only & {"grounding", "validation"}) and not cfg.offline
    if "grounding" in only:
        if needs_judge:
            print()
            judged = judge_all(suite, traces, cfg)
        step_grounding(suite, traces, cfg, judged)
    if "generation" in only:
        step_generation(suite, traces, cfg)
    if "validation" in only:
        labels_ready = cfg.offline or not day3_sheet.missing(traces, suite)
        if needs_judge and judged is None and labels_ready:
            print()
            judged = judge_all(suite, traces, cfg)
        step_validation(suite, traces, cfg, judged)
    if "relevance" in only:
        step_relevance(cfg)
    if "swap" in only:
        step_swap(suite, traces, cfg)
    if "consistency" in only:
        step_consistency(suite, cfg, max(2, args.runs))
    if "chunkjudge" in only:
        step_chunkjudge(suite, traces, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
