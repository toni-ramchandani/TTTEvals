"""Pins the Day 3 architecture comparison.

The slides, the speaker notes and the README all describe specific
outcomes: which question each architecture gets wrong, and which switch
produces which failure code. These tests are what keeps those descriptions
true when somebody edits a retriever, an extraction rule or the question set.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agentic import AGENT_DEFAULT, AGENT_NAIVE  # noqa: E402
from agentic import answer as agent_answer  # noqa: E402
from evals.day3_architectures import attribute, grade  # noqa: E402
from graphrag import GRAPH_DEFAULT, GRAPH_NAIVE, GraphBuildConfig  # noqa: E402
from graphrag import answer as graph_answer  # noqa: E402
from graphrag import build as graph_build  # noqa: E402
from helix import REPAIRED, build_index  # noqa: E402
from tracing.schema import trace_from_dict  # noqa: E402

QUERIES = {q["id"]: q for q in json.loads(
    (ROOT / "data" / "queries" / "architectures.json").read_text())["queries"]}


@pytest.fixture(scope="module")
def index():
    return build_index(REPAIRED)


@pytest.fixture(scope="module")
def graphs():
    return {}


def _agent(cfg, qid, index):
    q = QUERIES[qid]
    t = agent_answer(q["text"], index, cfg, user_office=q["user_office"],
                     user_role=q["user_role"], user_tenure_years=q["user_tenure_years"])
    return t, grade(t, q).correct, attribute(t, q)


def _graph(cfg, qid, index, graphs):
    key = cfg.build.hash()
    if key not in graphs:
        graphs[key] = graph_build(cfg)
    q = QUERIES[qid]
    t = graph_answer(q["text"], graphs[key], cfg, index=index, user_office=q["user_office"],
                     user_role=q["user_role"], user_tenure_years=q["user_tenure_years"])
    return t, grade(t, q).correct, attribute(t, q, graph=graphs[key])


# --- the graph build ----------------------------------------------------------
def test_default_graph_infers_supersession_and_keeps_labels():
    g = graph_build(GRAPH_DEFAULT)
    assert g.superseded_docs() == {"HB-PAR-2022": "HB-PAR-2024"}
    assert {f.access_label for f in g.facts.values() if f.doc_id == "HB-COMP-BANDS"} == {"hr-only"}
    assert all(f.doc_version and f.last_modified and f.source_chunk_id for f in g.facts.values())


def test_chunk_extraction_misattributes_rows():
    g = graph_build(GRAPH_NAIVE)
    company_wide = [f for f in g.facts.values() if f.doc_id == "HB-PAR-2024" and f.office is None]
    assert company_wide, "fixed chunks should separate some rows from their office heading"
    assert g.superseded_docs() == {}


# --- headline outcomes ----------------------------------------------------------
def test_graph_default_gets_twelve_of_thirteen(index, graphs):
    wrong = [qid for qid in QUERIES if not _graph(GRAPH_DEFAULT, qid, index, graphs)[1]]
    assert wrong == ["a11"]
    assert _graph(GRAPH_DEFAULT, "a11", index, graphs)[2] == ["GR2"]


def test_agent_default_runs_out_of_steps_on_the_comparison(index):
    _, ok, codes = _agent(AGENT_DEFAULT, "a3", index)
    assert not ok and codes == ["AG4"]
    _, ok, _ = _agent(replace(AGENT_DEFAULT, max_steps=16), "a3", index)
    assert ok


def test_right_path_wrong_answer(index):
    """a1: every expected tool called, every cited page checked, answer wrong."""
    t, ok, codes = _agent(AGENT_DEFAULT, "a1", index)
    assert not ok and codes == ["FP4"]


def test_right_answer_wrong_path(index):
    """a12: the naive agent never reads the profile and is right by luck."""
    t, ok, _ = _agent(AGENT_NAIVE, "a12", index)
    seq = t.span("agent.run").attributes["tool_sequence"]
    assert ok and "get_user_profile" not in seq


# --- each switch produces its own code ------------------------------------------
AGENT_CASES = [
    ("use_profile", False, "a2", "AG1"),
    ("prefer_query_office", False, "a5", "AG2"),
    ("decompose", False, "a4", "AG3"),
    ("act_on_observations", False, "a7", "AG5"),
    ("check_versions", False, "a7", "FP8"),
]


@pytest.mark.parametrize("switch,value,qid,code", AGENT_CASES)
def test_agent_switch(switch, value, qid, code, index):
    _, ok, codes = _agent(replace(AGENT_DEFAULT, **{switch: value}), qid, index)
    assert not ok and code in codes, codes


GRAPH_CASES = [
    ({"linking": "exact"}, "a5", "GR1"),
    ({"use_user_profile": False}, "a2", "GR1"),
    ({"build": GraphBuildConfig(unit="chunk")}, "a12", "GR2"),
    ({"build": GraphBuildConfig(carry_access_label=False)}, "a6", "GR4"),
    ({"respect_supersedes": False}, "a13", "GR3"),
    ({"fallback_to_text": False}, "a10", "GR5"),
    ({"hops": 1}, "a9", "GR5"),
]


@pytest.mark.parametrize("change,qid,code", GRAPH_CASES)
def test_graph_switch(change, qid, code, index, graphs):
    _, ok, codes = _graph(replace(GRAPH_DEFAULT, **change), qid, index, graphs)
    assert not ok and code in codes, codes


def test_lost_label_is_invisible_to_the_trace(index, graphs):
    """GR4: the leaked fact's own label says 'all'. Only lineage to the source
    page reveals it, which is why the detector checks the corpus, not the trace."""
    t, _, codes = _graph(replace(GRAPH_DEFAULT, build=GraphBuildConfig(carry_access_label=False)),
                         "a6", index, graphs)
    assert all(c.access_label == "all" for c in t.retrieved)
    assert "GR4" in codes


# --- schema ---------------------------------------------------------------------
def test_all_three_write_the_same_schema(index, graphs):
    a, _, _ = _agent(AGENT_DEFAULT, "a2", index)
    g, _, _ = _graph(GRAPH_DEFAULT, "a2", index, graphs)
    for t, system in ((a, "agentic"), (g, "graphrag")):
        back = trace_from_dict(json.loads(t.to_json()))
        assert back.system == system
        assert back.config_hash and back.retrieved
        assert all(c.doc_version and c.last_modified for c in back.retrieved)


def test_offline_is_deterministic(index, graphs):
    assert _agent(AGENT_DEFAULT, "a4", index)[0].answer == _agent(AGENT_DEFAULT, "a4", index)[0].answer
    assert _graph(GRAPH_DEFAULT, "a4", index, graphs)[0].answer == \
        _graph(GRAPH_DEFAULT, "a4", index, graphs)[0].answer
