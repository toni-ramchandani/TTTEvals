"""Day 2 on the reference run: synthetic questions, sheets, agreement, golden set.

Everything here is offline and writes only inside pytest's temporary folders.
The numbers pinned here are the ones the Day 2 slides show.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import evals.day2_evaluators as E
from evals import day2_agreement as G
from evals import day2_sheets as S
from evals.day2_error_analysis import Annotation, saturation
from evals.day2_examples import EXAMPLE_A, EXAMPLE_B, MODES
from evals.day2_synthetic import ASKERS, SCENARIOS, TOPICS, enumerate_tuples, generate
from helix import DEFAULT, answer, build_index
from toolkits.cases import cases_from_traces

ROOT = Path(__file__).resolve().parents[1]
DAY1 = json.loads((ROOT / "data" / "queries" / "day1_seed.json").read_text())["queries"]


@pytest.fixture(scope="module")
def reference():
    """The 30 reference traces: Day 1's six, then the 24 synthetic ones."""
    rows = generate(24, DEFAULT)
    index = build_index(DEFAULT)
    suite = [*DAY1, *rows]
    traces = [answer(q["text"], index, DEFAULT, user_office=q["user_office"], user_role=q["user_role"])
              for q in suite]
    return suite, traces, cases_from_traces(traces, suite)


def test_tuples_cover_every_combination_and_know_what_to_expect():
    tuples = enumerate_tuples(seed=0)
    assert len(tuples) == len(TOPICS) * len(ASKERS) * len(SCENARIOS) == 96
    assert len({(t.topic, t.asker, t.scenario) for t in tuples}) == 96
    assert enumerate_tuples(seed=0) == tuples, "the same seed gives the same order"
    for t in tuples:
        assert t.should_refuse == (t.scenario == "out of scope" or t.topic == "salary bands")


def test_synthetic_rows_carry_the_asker_and_the_expected_behaviour(reference):
    suite, traces, _ = reference
    rows = suite[6:]
    assert [r["id"] for r in rows] == [f"s{i:02d}" for i in range(1, 25)]
    assert len({r["text"] for r in rows}) == 24, "repeated wording is skipped"
    for r, t in zip(rows, traces[6:]):
        assert (t.user_office, t.user_role) == (r["user_office"], r["user_role"])
        assert isinstance(r["should_refuse"], bool)


def test_the_example_labels_describe_exactly_the_reference_questions(reference):
    suite, _, _ = reference
    texts = [q["text"] for q in suite]
    assert list(EXAMPLE_A) == texts, "A's notes, in reading order"
    assert set(EXAMPLE_B) == set(texts)
    for who in (EXAMPLE_A, EXAMPLE_B):
        for ok, note, labels in who.values():
            assert note and labels <= set(MODES)
            assert ok == (not labels), "a passing trace carries no failure mode"


def test_code_checks_agree_with_annotator_a_on_every_reference_trace(reference):
    """The teaching claim of Module 06: where the trace records the fact, code agrees with people."""
    suite, _, cases = reference
    for check in E.CODE_CHECKS:
        rs = [check(c) for c in cases]
        human = [rs[0].mode not in EXAMPLE_A[q["text"]][2] for q in suite]
        r = E.rates([x.passed for x in rs], human)
        assert r["tpr"] == 1.0 and r["tnr"] == 1.0, rs[0].evaluator


def test_kappa_by_hand():
    k = G.kappa([True, True, False, False], [True, False, False, False])
    assert (k["p_o"], k["p_e"], k["kappa"]) == (0.75, 0.5, 0.5)
    assert G.kappa([True, True], [True, True])["kappa"] is None
    with pytest.raises(ValueError):
        G.kappa([True], [True, False])


@pytest.mark.parametrize("k, name", [(None, "not defined: one answer for every item"),
                                     (-0.1, "worse than chance"), (0.17, "none"), (0.21, "minimal"),
                                     (0.59, "weak"), (0.60, "moderate"), (0.80, "strong"),
                                     (0.90, "strong"), (0.91, "almost perfect")])
def test_mchugh_reading(k, name):
    assert G.reading(k) == name


def test_reference_agreement_matches_the_slides(reference):
    suite, _, _ = reference
    a = [G.Label(q["text"], *EXAMPLE_A[q["text"]]) for q in suite]
    b = [G.Label(q["text"], *EXAMPLE_B[q["text"]]) for q in suite]
    k = G.kappa([not x.acceptable for x in a], [not y.acceptable for y in b])
    assert (k["both"], k["only_a"], k["only_b"], k["neither"]) == (17, 11, 0, 2)
    assert round(k["p_o"], 2) == 0.63 and round(k["p_e"], 2) == 0.56 and round(k["kappa"], 2) == 0.17
    modes = G.per_mode(a, b, list(MODES))
    got = {m: round(v["kappa"], 2) for m, v in modes.items()}
    assert got == {"stale_policy": 1.0, "wrong_scope_row": 0.29, "false_refusal": 1.0,
                   "entitlement_leak": 1.0, "missed_refusal": 0.74, "no_direct_answer": 0.33,
                   "incomplete": 0.0, "off_topic": 1.0}
    assert len([d for d in G.disagreements(a, b) if d[2] == "verdict"]) == 11


def test_the_split_gives_a_judge_examples_and_a_fair_test(reference):
    suite, traces, _ = reference
    rows = [G.GoldenRow(q.get("id", ""), "", "", q["text"], t.answer, EXAMPLE_A[q["text"]][0],
                        set(EXAMPLE_A[q["text"]][2]), "", "agreed") for q, t in zip(suite, traces)]
    G.assign_splits(rows, seed=0)
    sizes = {s: sum(r.split == s for r in rows) for s in ("train", "dev", "test")}
    assert sizes == {"train": 4, "dev": 13, "test": 13}
    assert [sum(r.acceptable for r in rows if r.split == s) for s in ("dev", "test")] == [1, 1]
    for m in MODES:
        n = sum(m in r.labels for r in rows)
        where = {r.split for r in rows if m in r.labels}
        if n >= 2:
            assert {"dev", "test"} <= where, m
        if n >= 3:
            assert "train" in where, f"a judge for {m} needs a failing example in train"
    train = sorted(r.query for r in rows if r.split == "train")
    assert train == sorted([DAY1[1]["text"], *[q["text"] for q in suite if q.get("id") in ("s16", "s17", "s20")]])
    again = G.assign_splits([G.GoldenRow(**{**r.__dict__, "split": ""}) for r in rows], seed=0)
    assert [r.split for r in again] == [r.split for r in rows], "the split is reproducible"


def test_the_judge_examples_come_from_train_only(reference):
    suite, traces, cases = reference
    golden = [G.GoldenRow(q.get("id", ""), "", "", q["text"], t.answer, EXAMPLE_A[q["text"]][0],
                          set(EXAMPLE_A[q["text"]][2]), EXAMPLE_A[q["text"]][1], "agreed")
              for q, t in zip(suite, traces)]
    G.assign_splits(golden, seed=0)
    by_q = {c.question: c for c in cases}
    train = [(r.id, by_q[r.query], r.acceptable, r.labels, r.note) for r in golden if r.split == "train"]
    scope = E.pick_examples(E.SCOPE_MODE, train)
    assert [(e.id, e.passed) for e in scope] == [("q2", False), ("s20", True)], \
        "the company-wide 12 weeks fails; the 25 days that apply to everyone pass"
    direct = E.pick_examples(E.DIRECT_MODE, train)
    assert [e.passed for e in direct] == [False, True] and direct[0].id == "s20"
    prompt = E.judge_prompt(cases[0], E.SCOPE_MODE, scope)
    assert "Asker: employee, Amsterdam office" in prompt and '"answer": "Fail"' in prompt
    assert prompt.count("Input ") == 2 and "Judge this criterion only." in prompt
    held_out = {r.query for r in golden if r.split in ("dev", "test")}
    assert not {e.case.question for e in scope + direct} & held_out, "no dev or test trace in a prompt"


def test_golden_set_round_trips(tmp_path):
    rows = [G.GoldenRow("q1", "day1", "employee, Amsterdam", "question?", "answer.", False,
                        {"stale_policy", "wrong_scope_row"}, "note, with a comma", "A", "dev",
                        reason="the 2024 page has a row per office", round=2)]
    back = G.read_golden(G.write_golden(rows, tmp_path / "g.csv"))
    assert back == rows
    assert G.next_round([]) == 1 and G.next_round(rows) == 2


@pytest.mark.parametrize("k, name", [(None, "not defined"), (-0.1, "poor"), (0.17, "slight"),
                                     (0.21, "fair"), (0.40, "fair"), (0.41, "moderate"),
                                     (0.61, "substantial"), (0.81, "almost perfect")])
def test_landis_and_koch_bands(k, name):
    assert G.landis_koch(k) == name


def test_fleiss_kappa_by_hand():
    # Four items, three raters: yes counts 3, 0, 2, 1. Shares 0.5, so p_e = 0.5;
    # per-item agreement 1, 1, 1/3, 1/3, so p_bar = 2/3 and kappa = 1/3.
    k = G.fleiss_kappa([[True, True, True], [False, False, False], [True, True, False], [True, False, False]])
    assert round(k, 6) == round(1 / 3, 6)
    assert G.fleiss_kappa([[True, True], [True, True]]) is None
    with pytest.raises(ValueError):
        G.fleiss_kappa([[True, True, True], [True, False]])


def test_saturation_counts_new_modes_per_block():
    anns = [Annotation("t", f"q{i}", "n", not labs, set(labs))
            for i, labs in enumerate([["a"], ["b"], [], ["a", "c"], ["b"], []])]
    assert saturation(anns, block=2) == [(1, 2, ["a", "b"]), (3, 4, ["c"]), (5, 6, [])]


def test_a_sheet_never_drops_what_a_person_wrote(tmp_path, reference):
    suite, traces, _ = reference
    items = [S.Item(q.get("id", ""), "day1", t, q) for q, t in zip(suite[:2], traces[:2])]
    notes = {traces[0].query: {"acceptable": "n", "open_code": "stale", "labels": "stale_policy"},
             "an old question": {"acceptable": "y", "open_code": "fine", "labels": ""},
             "an untouched old question": {"acceptable": "", "open_code": "", "labels": ""}}
    path = S.write_sheet(tmp_path / "sheet.csv", items, notes)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [r["query"] for r in rows] == [traces[0].query, traces[1].query, "an old question"]
    assert rows[0]["labels"] == "stale_policy" and rows[2]["source"] == "no trace"
    assert S.read_sheet(path)[traces[0].query]["open_code"] == "stale"


def test_the_reference_run_keeps_its_own_files():
    assert DEFAULT.offline
    for p in (S.SYN_SUITE, S.SYN_TRACES, S.SHEET, S.GOLDEN, S.LABELS, S.sheet_path("sam")):
        assert "_reference" in p.name, p
    assert S.sheet_path("Sam Lee").name == "day2_sheet_reference_sam-lee.csv"


def test_clean_keeps_the_sheets_people_wrote(tmp_path, monkeypatch):
    import run

    for rel in ("data/traces/day1.jsonl", "data/annotations/day1_triage_worksheet.csv",
                "data/annotations/day2_labels.csv", "data/annotations/day2_sheet.csv",
                "data/annotations/day2_sheet_sam.csv", "data/annotations/day2_golden.csv",
                "data/queries/day2_synthetic.json", "data/queries/day1_seed.json"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x", encoding="utf-8")
    monkeypatch.setattr(run, "ROOT", tmp_path)
    assert run.main(["clean"]) == 0
    left = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert left == ["data/annotations/day1_triage_worksheet.csv", "data/annotations/day2_golden.csv",
                    "data/annotations/day2_sheet.csv", "data/annotations/day2_sheet_sam.csv",
                    "data/queries/day1_seed.json"]
    assert run.main(["clean", "--all"]) == 0
    assert [p.name for p in tmp_path.rglob("*") if p.is_file()] == ["day1_seed.json"]


def test_annotating_in_the_terminal_saves_each_answer_and_resumes(tmp_path, monkeypatch, reference):
    """What a participant types in `day2 --annotate` and `day2 --label`, scripted."""
    import scripts.day2 as D

    suite, traces, _ = reference
    items = [S.Item(q.get("id", ""), "day1", t, q) for q, t in zip(suite[:3], traces[:3])]
    path = tmp_path / "sheet.csv"
    typed = iter(["n", "gave the 2022 figure", "y", "", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(typed))
    notes: dict = {}
    D.annotate_live(items, notes, path, redo=False)
    saved = S.read_sheet(path)
    assert saved[traces[0].query]["acceptable"] == "n"
    assert saved[traces[0].query]["open_code"] == "gave the 2022 figure"
    assert saved[traces[1].query]["acceptable"] == "y"
    assert saved[traces[2].query]["acceptable"] == "", "q stopped before the third trace"

    # Resume: only the unjudged trace is asked about.
    asked = []
    typed = iter(["n", "said it could not find it"])
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or next(typed))
    notes = S.read_sheet(path)
    D.annotate_live(items, notes, path, redo=False)
    assert len(asked) == 2 and S.read_sheet(path)[traces[2].query]["acceptable"] == "n"

    # Labels by number, by key, and a new mode of your own.
    typed = iter(["1, wrong_scope_row", "my_new_mode"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(typed))
    notes = S.read_sheet(path)
    D.label_live(items, notes, path, dict(MODES))
    saved = S.read_sheet(path)
    assert saved[traces[0].query]["labels"] == "stale_policy, wrong_scope_row"
    assert saved[traces[2].query]["labels"] == "my_new_mode"
    assert saved[traces[1].query]["labels"] == "", "a passing trace is not asked for labels"


def test_a_label_on_a_changed_answer_is_kept_but_no_longer_counts(tmp_path, reference):
    suite, traces, _ = reference
    items = [S.Item("q1", "day1", traces[0], suite[0])]
    path = S.write_sheet(tmp_path / "sheet.csv", items, {})
    notes = S.read_sheet(path)
    notes[traces[0].query].update(acceptable="n", open_code="stale")
    S.write_sheet(path, items, notes)
    assert S.notes_for([traces[0]], path)[traces[0].query]["open_code"] == "stale"

    changed = type(traces[0])(**{**traces[0].__dict__, "answer": "A different answer."})
    S.write_sheet(path, [S.Item("q1", "day1", changed, suite[0])], S.read_sheet(path))
    assert S.notes_for([changed], path) == {}, "the old label does not judge the new answer"
    rows = S.read_sheet_all(path)
    assert [r["source"] for r in rows] == ["day1", "earlier answer"]
    assert rows[1]["open_code"] == "stale" and rows[0]["acceptable"] == ""


def test_counting_a_sheet_about_your_own_system(tmp_path, capsys):
    import scripts.day2 as D

    sheet = tmp_path / "mine.csv"
    sheet.write_text("question,answer,acceptable,open_code,labels\n"
                     "Q1,a,n,old price list,stale_price\nQ2,b,y,,\nQ3,c,,not read yet,\n"
                     "Q4,d,no,wrong region,\"wrong_region, stale_price\"\n", encoding="utf-8")
    assert D.main(["--count", str(sheet)]) == 0
    out = capsys.readouterr().out
    assert "3 answers judged, 2 unacceptable." in out
    assert "stale_price                     2     67%" in out
    assert "2 new   stale_price, wrong_region" in out
    assert not (tmp_path / "data").exists(), "counting writes nothing"


def test_the_whole_reference_run_from_a_clean_copy(tmp_path):
    """`python run.py all --offline` on a fresh copy: every Day 2 figure the slides quote."""
    import shutil
    import subprocess
    import sys

    for part in ("src", "scripts", "run.py"):
        src = ROOT / part
        (shutil.copytree if src.is_dir() else shutil.copy)(src, tmp_path / part)
    (tmp_path / "data" / "queries").mkdir(parents=True)
    for name in ("day1_seed.json", "architectures.json"):
        shutil.copy(ROOT / "data" / "queries" / name, tmp_path / "data" / "queries" / name)
    run = subprocess.run([sys.executable, "run.py", "all", "--offline"], cwd=tmp_path,
                         capture_output=True, text=True, timeout=300)
    out = run.stdout
    assert run.returncode == 0, run.stderr[-2000:]
    for line in ("30 traces: 6 from Day 1, 24 synthetic.",
                 "28 of 30 traces judged unacceptable.",
                 "missed_refusal             9     30%",
                 "no_direct_answer           8     27%",
                 "entitlement_leak           6     20%",
                 "wrong_scope_row            5     17%",
                 "off_topic                  5     17%",
                 "stale_policy               4     13%",
                 "incomplete                 2      7%",
                 "false_refusal              1      3%",
                 "traces   1 to 10  7 new",
                 "traces  11 to 20  1 new   off_topic",
                 "traces  21 to 30  0 new",
                 "11 verdicts split.",
                 "stale evidence (code)      fails 4 of 30 traces",
                 "entitlement (code)         fails 6 of 30 traces",
                 "false refusal (code)       fails 1 of 30 traces",
                 "missed refusal (code)      fails 9 of 30 traces",
                 "Cohen's kappa       (p_o - p_e) / (1 - p_e) = +0.17   Landis and Koch: slight · McHugh: none",
                 "wrong_scope_row         5    1    1    87%  +0.29   fair / minimal",
                 "no_direct_answer: 6 disagreements (6 on the verdict, 0 on the modes only)",
                 "wrong_scope_row: 4 disagreements (1 on the verdict, 3 on the modes only)",
                 "missed_refusal: 3 disagreements (2 on the verdict, 1 on the modes only)",
                 "30 traces in the golden set: 15 A, 15 agreed.",
                 "train         4     0     4",
                 "dev          13     1    12",
                 "test         13     1    12",
                 "golden set written to data/annotations/day2_golden_reference.csv",
                 "  q2   Fail  gave the 2022 company-wide 12 weeks; the 2024 page replaced it wit",
                 "  s20  Pass  The figure in the answer is the one that applies to this asker. It",
                 "The prompt the binary judge receives, here for q1:",
                 "stale evidence (code)      stale_policy          100% 1/1  100% 12/12",
                 "entitlement (code)         entitlement_leak      100% 3/3  100% 10/10",
                 "missed refusal (code)      missed_refusal        100% 4/4    100% 9/9",
                 "Raw rate on the 13 labelled traces outside the dev split.",
                 "stale evidence (code)      15%        15%     15% to 15%      -    13",
                 "entitlement (code)         15%        15%     15% to 15%      -    13",
                 "false refusal (code)        0%         0%       0% to 0%      -    13",
                 "missed refusal (code)      31%        31%     31% to 31%      -    13",
                 "the Day 2 golden set"):
        assert line in out, line
    import csv
    with open(tmp_path / "data" / "annotations" / "day2_golden_reference.csv", newline="") as f:
        golden = {r["id"]: r for r in csv.DictReader(f)}
    assert (golden["q1"]["split"], golden["q6"]["split"]) == ("dev", "test"), "q1 is the prompt shown; q6 waits for the final look"
    passes = {r["split"]: rid for rid, r in golden.items() if r["acceptable"] == "yes"}
    assert passes == {"dev": "q5", "test": "s18"}
    written = sorted(p.name for p in (tmp_path / "data" / "annotations").iterdir())
    assert written == ["day1_triage_worksheet.csv", "day2_golden_reference.csv",
                       "day2_labels_reference.csv", "day2_sheet_reference.csv"]


def test_more_questions_extend_the_set_and_keep_the_old_ones():
    first = generate(10, DEFAULT)
    more = generate(16, DEFAULT, existing=first)
    assert len(more) == 6 and [r["id"] for r in more] == [f"s{i:02d}" for i in range(11, 17)]
    assert not {r["text"] for r in more} & {r["text"] for r in first}
    assert [r["text"] for r in first + more] == [r["text"] for r in generate(16, DEFAULT)], \
        "growing a set in two steps gives the same questions as writing it at once"


def test_q6_had_the_answer_in_front_of_it(reference):
    """The Day 2 opening: the evidence reached the model; the answer never says yes or no."""
    _, traces, _ = reference
    q6 = traces[5]
    assert q6.query.startswith("Can I claim a laptop stand")
    reached = [c for c in q6.retrieved if not c.dropped_at]
    assert any("a laptop stand" in c.text for c in reached)
    assert "laptop stand" not in q6.answer and "600 EUR" in q6.answer
    assert all(check(c).passed for check in E.CODE_CHECKS for c in [cases_from_traces([q6], [DAY1[5]])[0]])


def test_every_mode_seen_twice_reaches_dev_and_test_for_any_seed():
    import random

    rnd = random.Random(7)
    for seed in range(40):
        rows = []
        for i in range(rnd.randint(8, 40)):
            labs = set(rnd.sample(["a", "b", "c", "d"], rnd.randint(0, 2)))
            rows.append(G.GoldenRow(str(i), "", "", f"question {seed} {i}", "", not labs, labs, "", ""))
        G.assign_splits(rows, seed=seed)
        for m in ["a", "b", "c", "d", "pass"]:
            mine = [r for r in rows if (m in r.labels) or (m == "pass" and not r.labels)]
            if len(mine) >= 2:
                assert {"dev", "test"} <= {r.split for r in mine}, (seed, m)


def _agreement_on(monkeypatch, tmp_path, items, sheets):
    """Run `python run.py agreement` on given labels, with every file inside tmp_path."""
    import scripts.agreement as A

    paths = {name: tmp_path / f"day2_sheet_{name}.csv" for name in sheets}
    by_path = {paths[name]: labels for name, labels in sheets.items()}
    monkeypatch.setattr(A, "ROOT", tmp_path)
    monkeypatch.setattr(A, "SHEET", paths["you"])
    monkeypatch.setattr(A, "GOLDEN", tmp_path / "day2_golden.csv")
    monkeypatch.setattr(A, "load_items", lambda: items)
    monkeypatch.setattr(A, "other_sheets", lambda: {n: p for n, p in paths.items() if n != "you"})
    monkeypatch.setattr(A, "labels_from_sheet",
                        lambda its, path: {k: v for k, v in by_path[path].items() if k in {i.query for i in its}})
    return A.main([])


def test_round_two_measures_the_revised_rubric_on_fresh_traces(tmp_path, monkeypatch, capsys, reference):
    from evals.day2_examples import EXAMPLES

    suite, traces, _ = reference
    items = [S.Item(q.get("id", ""), "day1" if i < 6 else "synthetic", t, q)
             for i, (q, t) in enumerate(zip(suite, traces))]
    lab = {who: {it.query: G.Label(it.query, *EXAMPLES[who][it.query]) for it in items} for who in "AB"}
    first = {it.query for it in items[:20]}
    # Round 1: the first 20 traces, labelled blind.
    assert _agreement_on(monkeypatch, tmp_path, items,
                         {"you": {q: v for q, v in lab["A"].items() if q in first},
                          "sam": {q: v for q, v in lab["B"].items() if q in first}}) == 0
    out = capsys.readouterr().out
    assert "ROUND 2" not in out
    round1 = {r.query: r.split for r in G.read_golden(tmp_path / "day2_golden.csv")}
    assert len(round1) == 20
    # Round 2: ten fresh traces, labelled with the revised rubric, and now the two agree.
    sam = {**{q: v for q, v in lab["B"].items() if q in first},
           **{q: v for q, v in lab["A"].items() if q not in first}}
    assert _agreement_on(monkeypatch, tmp_path, items, {"you": lab["A"], "sam": sam}) == 0
    out = capsys.readouterr().out
    assert "ROUND 2: the revised rubric on 10 fresh traces nobody discussed" in out
    assert "Cohen's kappa       +1.00   Landis and Koch: almost perfect" in out
    golden = G.read_golden(tmp_path / "day2_golden.csv")
    assert sorted(r.round for r in golden) == [1] * 20 + [2] * 10
    assert all(round1[r.query] == r.split for r in golden if r.query in round1), "old traces keep their split"


def test_three_annotators_get_fleiss_kappa(tmp_path, monkeypatch, capsys, reference):
    from evals.day2_examples import EXAMPLES

    suite, traces, _ = reference
    items = [S.Item(q.get("id", ""), "day1", t, q) for q, t in zip(suite, traces)]
    lab = {who: {it.query: G.Label(it.query, *EXAMPLES[who][it.query]) for it in items} for who in "AB"}
    assert _agreement_on(monkeypatch, tmp_path, items, {"you": lab["A"], "sam": lab["B"], "kim": lab["A"]}) == 0
    out = capsys.readouterr().out
    assert "3 annotators: you, sam, kim. Traces labelled by all: 30." in out
    assert "Fleiss' kappa on the verdict" in out and "python run.py agreement --with sam" in out
    assert not (tmp_path / "day2_golden.csv").exists(), "no golden set from three sheets at once"
