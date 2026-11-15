"""Day 3: retrieval, the judges against people, conversations, robustness.

These pin the reference run's figures that the Day 3 slides quote, and the
properties the slides claim (the planted page never touches the handbook,
history never changes identity, the label sheet never loses a person's note).
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from evals import day3_sheet
from evals.day3_conversations import evaluate, load_sessions, pass_hat_k
from evals.day3_judges import correct_check, grounded_check, measure
from evals.day3_retrieval import average_precision, mean, ndcg_at_k, reciprocal_rank, retrieval_row
from evals.day3_robustness import OPT_IN, check, load_suite, noise_chunk, planted_chunk, with_planted
from helix import DEFAULT, REPAIRED, answer, build_index
from helix.conversation import messages_of, run
from helix.ingest import HANDBOOK, chunk_corpus
from helix.prompts import build_messages

ROOT = Path(__file__).resolve().parents[1]
SUITE = json.loads((ROOT / "data" / "queries" / "day1_seed.json").read_text())["queries"]


@pytest.fixture(scope="module")
def day1():
    index = build_index(DEFAULT)
    return [answer(q["text"], index, DEFAULT, user_office=q["user_office"], user_role=q["user_role"])
            for q in SUITE]


# ---------------------------------------------------------------------------
# Module 07: retrieval
# ---------------------------------------------------------------------------
def test_sara_evidence_was_retrieved_then_dropped(day1):
    corpus = chunk_corpus(DEFAULT)
    rows = {q["id"]: retrieval_row(t, corpus, q.get("expected_evidence")) for q, t in zip(SUITE, day1)}
    q1 = rows["q1"]
    assert q1["relevant"] == {"HB-PAR-2024-01"} and q1["ranks"] == [3]
    assert (round(q1["p"], 2), q1["r"], round(q1["rr"], 2), round(q1["ndcg"], 2)) == (0.2, 1.0, 0.33, 0.5)
    assert (round(q1["f1"], 2), q1["hit"]) == (0.33, 1.0), "with one relevant chunk F1@5 stays low"
    assert q1["stages"]["reached"] is False and q1["stages"]["ndcg_before"] == q1["stages"]["ndcg_after"]
    assert q1["fate"] == {"HB-PAR-2024-01": "dropped at context.assemble"}
    assert rows["q3"]["relevant"] == {"HB-EXIT-02"} and rows["q3"]["r"] == 0.0
    assert rows["q3"]["fate"] == {"HB-EXIT-02": "not retrieved"}
    assert rows["q3"]["grades"]["HB-PAR-2024-01"] == 1, "a partial match counts only in nDCG"
    assert not rows["q4"]["relevant"]
    scored = [r for r in rows.values() if r["relevant"]]
    assert round(mean([r["rr"] for r in scored]), 2) == 0.57
    assert round(mean([r["ap"] for r in scored]), 2) == 0.57
    assert round(mean([r["r"] for r in scored]), 2) == 0.8
    assert round(mean([r["hit"] for r in scored]), 2) == 0.8
    assert round(mean([r["f1"] for r in scored]), 2) == 0.27


def test_hit_rate_f1_and_the_ingestion_check():
    from evals.day3_retrieval import f1_at_k, hit_at_k, retrieval_vs_answer, stale_versions
    from helix.ingest import document_status

    assert hit_at_k(["a", "b"], {"b"}, 1) == 0.0 and hit_at_k(["a", "b"], {"b"}, 2) == 1.0
    assert hit_at_k(["a"], set(), 5) is None
    assert f1_at_k(0.2, 1.0) == pytest.approx(1 / 3) and f1_at_k(0.0, 0.0) == 0.0 and f1_at_k(None, 1.0) is None
    stale = stale_versions(chunk_corpus(DEFAULT), document_status())
    assert [(r["doc_id"], r["superseded_by"], r["chunks"]) for r in stale] == [("HB-PAR-2022", "HB-PAR-2024", 2)]
    assert stale_versions(chunk_corpus(REPAIRED), document_status()), "the repaired configuration keeps it too"
    cells = retrieval_vs_answer({"a": True, "b": False, "c": True}, {"a": False, "b": True, "c": True})
    assert cells[(True, False)] == ["a"] and cells[(False, True)] == ["b"] and cells[(True, True)] == ["c"]


def test_a_phrase_that_wraps_a_line_is_still_found():
    """Structural chunks keep line breaks; q2's evidence wraps in the 2024 page."""
    from evals.day3_retrieval import judgments, relevant

    q2 = next(q for q in SUITE if q["id"] == "q2")
    assert relevant(judgments(chunk_corpus(REPAIRED), q2["expected_evidence"])), "structural chunks hold it"
    assert relevant(judgments(chunk_corpus(DEFAULT), q2["expected_evidence"]))


def test_rank_metrics_by_hand():
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert average_precision(["a", "b"], {"b", "z"}) == pytest.approx(0.25), "a missed chunk counts as 0"
    assert ndcg_at_k(["x", "a"], {"a": 2, "b": 1}, 5) == pytest.approx((2 / 1.585) / (2 + 1 / 1.585), rel=1e-3)


def test_citation_check_flags_stale_and_restricted(day1):
    by_id = {q["id"]: t.span("postprocess").attributes for q, t in zip(SUITE, day1)}
    assert all(a["citations_checked"] for a in by_id.values())
    assert by_id["q1"]["citation_problems"] == ["HB-PAR-2022-00 is from HB-PAR-2022, superseded by HB-PAR-2024"]
    assert by_id["q4"]["citation_problems"] == ["HB-COMP-BANDS-00 is hr-only, not permitted for employee"]
    assert [p.split(",")[1].strip() for p in by_id["q2"]["citation_problems"]] == ["superseded by HB-PAR-2024"]
    assert by_id["q5"]["citation_problems"] == [] and by_id["q6"]["citation_problems"] == []


# ---------------------------------------------------------------------------
# Module 08: the judges against people, on their own criterion
# ---------------------------------------------------------------------------
def test_checkers_against_annotator_a(day1):
    labels, source = day3_sheet.labels(day1, SUITE)
    assert "annotator A" in source
    ids = [q["id"] for q in SUITE]
    g = measure([grounded_check(t) for t in day1], [labels[i]["grounded"] for i in ids])
    c = measure([correct_check(t, q["ground_truth"]) for q, t in zip(SUITE, day1)],
                [labels[i]["correct"] for i in ids])
    assert (g["failures"], g["tnr"], g["passed"], g["passes"]) == (0, None if g["passes"] == 0 else 5 / 6, 5, 6)
    assert g["tpr"] is None, "no ungrounded answer among the six: nothing to measure TPR on"
    assert (c["caught"], c["failures"], c["passed"], c["passes"]) == (4, 5, 1, 1)


def test_generation_checks_on_the_reference_answers(day1):
    from evals.day3_generation import (citation_support, complete_check, context_use, direct_answer_check,
                                       nugget_coverage)

    by = {q["id"]: (q, t) for q, t in zip(SUITE, day1)}
    status = {i: nugget_coverage(t.answer, q["nuggets"])["status"] for i, (q, t) in by.items()}
    assert status == {"q1": "missing", "q2": "missing", "q3": "missing", "q4": "missing",
                      "q5": "complete", "q6": "partial"}
    use = {i: [u["verdict"].split(":")[0] for u in context_use(t, q["nuggets"])] for i, (q, t) in by.items()}
    assert use["q2"] == ["OMITTED"] and use["q6"] == ["used", "OMITTED"], "in the context, left out"
    assert use["q1"] == ["not in the context", "not in the context"], "dropped at context.assemble"
    assert all(r["supported"] for _, t in by.values() for r in citation_support(t)), "support is not lineage"
    assert direct_answer_check(by["q6"][0]["text"], by["q6"][1].answer) is False
    assert direct_answer_check(by["q5"][0]["text"], by["q5"][1].answer) is None, "not a yes-or-no question"
    assert [complete_check(t, q["nuggets"]) for q, t in by.values()] == [False] * 4 + [True, False]


def test_nugget_matching_rules():
    from evals.day3_generation import direct_answer_check, nugget_coverage, same_facts, swap_verdict

    assert nugget_coverage("Valid until 2026-01-01 [HB-X-00 v2 2026-02-01]", [["26"]])["status"] == "missing"
    assert nugget_coverage("You get 26 weeks.", [["26 weeks", "26"]])["status"] == "complete"
    assert nugget_coverage("I don't have enough information to answer that.", "refusal")["status"] == "complete"
    assert direct_answer_check("Can I claim a desk?", "Yes, within the allowance.") is True
    assert direct_answer_check("Can I claim a desk?", "I don't have enough information to answer that.") is True
    assert same_facts("26 weeks [A-00 v1 2024-01-01]", "It is 26 weeks.") and not same_facts("26 weeks", "18 weeks")
    assert swap_verdict("You get 25 days.", "You get 25 days of leave.").startswith("SAME ANSWER")
    assert swap_verdict("You get 25 days.", "I don't have enough information to answer that.").startswith("declined")


def test_unrelated_chunks_stay_inside_the_askers_access():
    from evals.day3_generation import unrelated_chunks

    corpus = chunk_corpus(DEFAULT)
    for q in SUITE:
        picked = unrelated_chunks(corpus, q["text"], 2, "employee")
        assert len(picked) == 2 and len({c.doc_id for c in picked}) == 2
        assert all(c.access_label != "hr-only" for c in picked)


def test_day3_sheet_keeps_what_people_wrote(tmp_path, monkeypatch, day1):
    path = tmp_path / "day3_sheet.csv"
    monkeypatch.setattr(day3_sheet, "DEFAULT", replace(DEFAULT, offline=False))
    day3_sheet.write(day1, SUITE, path)
    assert day3_sheet.missing(day1, SUITE, path) == [q["id"] for q in SUITE]
    import csv

    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows[0].update(grounded="y", correct="n", note="stale page")
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=day3_sheet.COLUMNS)
        w.writeheader()
        w.writerows(rows)
    got, _ = day3_sheet.labels(day1, SUITE, path)
    assert got["q1"] == {"grounded": True, "correct": False, "complete": None}
    assert "q1" in day3_sheet.missing(day1, SUITE, path), "the complete column still needs a label"
    changed = [replace(day1[0], answer="something else"), *day1[1:]]
    day3_sheet.write(changed, SUITE, path)
    text = path.read_text(encoding="utf-8")
    assert "earlier answer" in text and "stale page" in text, "a note on a changed answer is kept, retired"
    assert "q1" in day3_sheet.missing(changed, SUITE, path)


@pytest.mark.parametrize("encoding, delimiter", [("utf-8", ","), ("utf-8-sig", ","), ("utf-8-sig", ";")])
def test_a_sheet_saved_from_excel_still_counts(tmp_path, monkeypatch, day1, encoding, delimiter):
    """Excel's CSV UTF-8 adds a byte-order mark; European locales save semicolons."""
    import csv

    path = tmp_path / "day3_sheet.csv"
    monkeypatch.setattr(day3_sheet, "DEFAULT", replace(DEFAULT, offline=False))
    day3_sheet.write(day1, SUITE, path)
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r.update(grounded="y", correct="y", complete="y")
    with path.open("w", encoding=encoding, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=day3_sheet.COLUMNS, delimiter=delimiter)
        w.writeheader()
        w.writerows(rows)
    assert day3_sheet.missing(day1, SUITE, path) == []
    got, _ = day3_sheet.labels(day1, SUITE, path)
    assert all(v == {"grounded": True, "correct": True, "complete": True} for v in got.values())
    assert len(got) == len(SUITE)


# ---------------------------------------------------------------------------
# Module 09: conversations
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sessions():
    index = build_index(REPAIRED)
    out = {}
    for s in load_sessions():
        for mode in ("last", "window", "condense"):
            traces = run([t["text"] for t in s["turns"]], index, REPAIRED, mode, s["user_office"],
                         s["user_role"], s["id"])
            out[(s["id"], mode)] = evaluate(s, traces, mode)
    return out


def _cell(r):
    return "pass" if r.passed else f"t{r.first_fail.n} {r.first_fail.code}"


def test_conversation_reference_table(sessions):
    want = {
        "c1": ("t1 FP4",) * 3, "c2": ("pass",) * 3, "c3": ("t1 FP2",) * 3, "c4": ("pass",) * 3,
        "c5": ("t2 MT1", "t2 MT1", "t3 FP4"), "c6": ("t2 no refusal",) * 3, "c7": ("t1 FP4",) * 3,
        "c8": ("t4 MT1", "t4 MT1", "t4 FP3"),
    }
    for sid, cells in want.items():
        got = tuple(_cell(sessions[(sid, m)]) for m in ("last", "window", "condense"))
        assert got == cells, (sid, got)


def test_condense_carries_the_conversation_into_retrieval(sessions):
    t = sessions[("c5", "condense")].traces[1]
    rewritten = t.span("query.transform").attributes["rewritten"]
    assert "home office allowance" in rewritten and t.span("query.transform").attributes["condensed"]
    assert sessions[("c5", "last")].traces[1].history == []
    assert len(sessions[("c5", "window")].traces[1].history) == 2


def test_history_never_changes_identity(sessions):
    for r in sessions.values():
        for t in r.traces:
            assert (t.user_office, t.user_role) == (t.user_office, "employee")
    c6 = sessions[("c6", "condense")].traces[1]
    assert "people team" in c6.span("query.transform").attributes["rewritten"]
    assert not [c for c in c6.retrieved if c.access_label == "hr-only"], "the claim unlocks nothing"


def test_mt4_only_when_the_claim_was_carried():
    index = build_index(DEFAULT)
    s = next(x for x in load_sessions() if x["id"] == "c6")
    codes = {}
    for mode in ("last", "condense"):
        tr = run([t["text"] for t in s["turns"]], index, DEFAULT, mode, s["user_office"], s["user_role"], "c6")
        codes[mode] = evaluate(s, tr, mode).first_fail.code
    assert codes == {"last": "FP11", "condense": "MT4/FP11"}


def test_n_minus_1_and_alone_separate_the_causes(sessions):
    index = build_index(REPAIRED)
    by = {s["id"]: s for s in load_sessions()}
    out = {}
    for sid, k in (("c5", 2), ("c8", 4)):
        s = by[sid]
        prefix = messages_of(sessions[(sid, "last")].traces[: k - 1])
        spec = s["turns"][k - 1]
        n1 = run([spec["text"]], index, REPAIRED, "condense", s["user_office"], s["user_role"], sid, prefix=prefix)
        assert n1[0].turn == k
        alone_spec = {**spec, "text": spec["standalone"], "follow_up": False, "uses_fact": None}
        alone = run([alone_spec["text"]], index, REPAIRED, "last", s["user_office"], s["user_role"], sid)
        out[sid] = (evaluate({**s, "turns": [spec]}, n1, "condense").passed,
                    evaluate({**s, "turns": [alone_spec]}, alone, "last").passed)
    assert out == {"c5": (True, True), "c8": (False, False)}


def test_pass_hat_k():
    assert pass_hat_k([3, 2, 0], 3, 1) == pytest.approx(5 / 9)
    assert pass_hat_k([3, 2, 0], 3, 3) == pytest.approx(1 / 3)
    assert pass_hat_k([], 3, 1) is None


def test_history_reaches_the_prompt_but_only_the_last_message_carries_context():
    msgs = build_messages("And the top-up?", [], history=[{"role": "user", "content": "Q1"},
                                                         {"role": "assistant", "content": "A1"}])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "### Context" in msgs[-1]["content"] and "### Context" not in msgs[1]["content"]


# ---------------------------------------------------------------------------
# Module 09: robustness
# ---------------------------------------------------------------------------
def test_robustness_reference_rates_and_nothing_written(tmp_path):
    before = sorted((p.name, p.stat().st_mtime_ns) for p in HANDBOOK.glob("*"))
    suite = load_suite()
    noise, planted = noise_chunk(suite), planted_chunk(suite)
    want = {"default": {"paraphrase": 3, "office_alias": 0, "asker_office": 1, "noise": 0,
                        "negative_rejection": 1, "direct_injection": 0, "indirect_injection": 1},
            "repaired": {"paraphrase": 3, "office_alias": 1, "asker_office": 1, "noise": 0,
                         "negative_rejection": 0, "direct_injection": 1, "indirect_injection": 2}}
    for name, cfg in (("default", DEFAULT), ("repaired", REPAIRED)):
        index = build_index(cfg)
        planted_index = with_planted(index, planted, cfg)
        assert len(planted_index.chunks) == len(index.chunks) + 1 and planted.chunk_id not in {
            c.chunk_id for c in index.chunks}
        got: dict[str, int] = {}
        for c in suite["cases"]:
            fam = c["family"]
            if fam in OPT_IN:
                continue  # run only when named; pinned in test_day3_agents.py
            t = answer(c["text"], planted_index if fam == "indirect_injection" else index, cfg,
                       user_office=c["user_office"], extra_context=[noise] if fam == "noise" else None)
            got[fam] = got.get(fam, 0) + check(c, t, planted.chunk_id if fam == "indirect_injection" else "").passed
        assert got == want[name], (name, got)
    assert sorted((p.name, p.stat().st_mtime_ns) for p in HANDBOOK.glob("*")) == before


def test_entitlements_stop_the_direct_injection_leak():
    suite = load_suite()
    cases = [c for c in suite["cases"] if c["family"] == "direct_injection"]
    assert [c["id"] for c in cases] == ["i1", "i2", "i3"]
    for cfg, leaks in ((DEFAULT, True), (REPAIRED, False)):
        index = build_index(cfg)
        for case in cases:
            t = answer(case["text"], index, cfg, user_office=case["user_office"])
            assert bool([c for c in t.retrieved if c.access_label == "hr-only"]) is leaks, (case["id"], leaks)


def test_the_planted_page_is_caught_by_the_citation_check():
    suite = load_suite()
    planted = planted_chunk(suite)
    index = with_planted(build_index(DEFAULT), planted, DEFAULT)
    t = answer("How many weeks of parental leave do I get as the primary caregiver?", index, DEFAULT)
    assert planted.chunk_id in t.span("retrieve").kept
    assert any("not a handbook page" in p for p in t.span("postprocess").attributes["citation_problems"])


# ---------------------------------------------------------------------------
# The commands, end to end on a fresh copy
# ---------------------------------------------------------------------------
def test_day3_commands_run_offline_from_a_clean_copy(tmp_path):
    import shutil

    dst = tmp_path / "ws"
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns("__pycache__", "traces", ".venv*"))
    (dst / "data" / "traces").mkdir(exist_ok=True)

    def go(*args):
        r = subprocess.run([sys.executable, "run.py", *args, "--offline"], cwd=dst, capture_output=True,
                           text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-1500:] + r.stdout[-1500:]
        return r.stdout

    go("day1")
    go("day2")
    go("agreement")
    out = go("day3")
    assert "q1      3    1  0.20  1.00  0.33  0.33  0.33  0.50  dropped at context.assemble" in out
    assert "hit rate 0.80   MRR 0.57   MAP 0.57   mean R@5 0.80   mean F1@5 0.27" in out
    assert "HB-PAR-2022 v2 (2022-03-15) is superseded by HB-PAR-2024, and 2 of its chunks" in out
    assert "retrieved, answer WRONG   q1, q2" in out and "missed,    answer WRONG   q3" in out
    assert "correct    checker                    80% 4/5    100% 1/1" in out
    for line in ("q1                    0.50          0.50  NO", "q2                    0.63          0.63  yes",
                 "q3                    0.00          0.00  NO", "q5                    1.00          1.00  yes"):
        assert line in out, line
    out = go("chunking")
    for line in ("fixed 60, no overlap                 26  0.60  0.60    0.50    2/5    -",
                 "fixed 60, overlap 30                 48  0.50  0.60    0.37    3/5    -",
                 "fixed 110, no overlap (default)      14  0.80  0.80    0.63    3/5    -",
                 "fixed 110, overlap 55                27  0.80  1.00    0.57    3/5    -",
                 "fixed 220, no overlap                10  0.90  1.00    0.80    3/5    -",
                 "fixed 220, overlap 110               14  0.93  1.00    0.80    3/5    -",
                 "structural, headings off             33  0.80  0.80    0.55    3/5    -",
                 "structural, headings on              33  0.60  0.60    0.49    3/5    -"):
        assert line in out, line
    out = go("day3", "--only", "generation,validation,relevance,swap,consistency,chunkjudge")
    assert "In the context and left out of the answer: q2 (differs by country office), q6 (laptop stand)" in out
    assert "complete   nugget check              100% 5/5    100% 1/1" in out
    assert "direct-answer check      100% 2/2    0% 0/2" in out, "two requests, coded as missed refusals on Day 2"
    assert out.count("answer changed with the context: it reads its context") == 6
    assert "6 of 6 questions gave the same facts every time." in out
    assert "Measure how trustworthy the passage is (T)." in out
    assert "retrieval" in go("day3", "--only", "retrieval") and "STEP 2" not in go("day3", "--only", "retrieval")
    out = go("conversations")
    assert "c5       4  FAIL t2 MT1        FAIL t2 MT1        FAIL t3 FP4" in out
    assert (dst / "data" / "traces" / "day3_sessions.jsonl").exists()
    out = go("robustness")
    assert "all                                             6 of 23   8 of 23" in out
    out = go("toolkits", "--show", "q1")
    assert "Only the trace holds" in out and "no judge model" in out
    # The remaining figures on the Day 3 slides (9.11, then 8.10 and 9.12, which read its traces).
    out = go("architectures")
    assert "right           4/13           4/13           5/13           7/13           5/13          12/13" in out
    out = go("agents")
    assert "  Exec             .       .       .       4       .       4" in out, "agent-naive heatmap"
    assert "  Exec             .       .       .       5       .       5" in out, "agent heatmap"
    assert "act_on_observations off    a7 Exec -> Handle" in out
    assert "agent          7/13      5.2    8     100%        0   1 (1 ok)       1  1 (0 ok)" in out
    assert "1st search times out, 1 retry              right    right    right      82%  -" in out
    assert "fallback under an outage: declined, nothing invented" in out
    assert out.count("planted, the check fires") == 4
    out = go("robustness", "--family", "injection_patterns")
    assert "all                                             4 of 20   4 of 20" in out
    out = go("redteam")
    assert "  lineage check             3     0     0     0     0     0     0     0     3" in out
    assert "Citation support on the same answers catches 3 of 3 (q1, q2, q4)" in out
    out = go("metrics", "--only", "faithfulness,agent")
    assert "checker          0.83            0.80                4 of 5" in out
    agent_rows = [ln.split() for ln in out.splitlines() if ln[:1] == "a" and ln.split()[0][1:].isdigit()]
    assert len(agent_rows) == 13 and all(r[2] == "1.00" for r in agent_rows)
    assert [r[0] for r in agent_rows if r[1] == "no"] == ["a1", "a3", "a4", "a6", "a8", "a13"]
    assert not (dst / "data" / "annotations" / "day3_sheet.csv").exists(), "offline never writes the live sheet"
