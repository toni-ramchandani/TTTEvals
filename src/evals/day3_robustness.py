"""Day 3, Module 09: behavioural tests against a baseline.

A pass rate on the suite says how Helix does on the questions someone wrote.
These tests ask how it does when the input changes in ways users and
attackers change it. Seven families, each from a published method:

  paraphrase, office_alias   CheckList invariance tests (Ribeiro et al. 2020):
                             a change that should not matter must not change
                             the answer's facts
  asker_office               a CheckList directional test: the same question
                             from another office must change the figure
  noise                      RGB noise robustness (Chen et al.): an irrelevant
                             chunk in the context must not change the answer
  negative_rejection         RGB negative rejection: a question the handbook
                             cannot answer must be declined
  direct_injection           OWASP LLM01 and LLM02: instructions in the message
                             must not unlock restricted content
  indirect_injection         OWASP LLM01, scenario 4: a planted page in the
                             store carries an instruction; the answer must not
                             follow it

Every check is code over the answer and the trace, like Day 2's. The planted
page and the noise chunk live only in memory while the command runs; the
handbook files are never touched.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from helix.config import Config
from helix.ingest import Index
from helix.llm import embed
from helix.retrieve import ENTITLEMENTS
from tracing.schema import Chunk, Trace

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "data" / "queries" / "day3_robustness.json"
FAMILIES = ("paraphrase", "office_alias", "asker_office", "noise", "negative_rejection",
            "direct_injection", "indirect_injection")
#: Run only when named (`--family injection_patterns`): twenty more direct
#: injection patterns, because three are too few to call a defence tested.
OPT_IN = ("injection_patterns",)


def load_suite(path: Path = SUITE) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def noise_chunk(suite: dict) -> Chunk:
    return Chunk(chunk_id="NOISE-CAFE-00", doc_id="NOISE-CAFE", text=suite["noise_chunk"],
                 doc_version="1", last_modified="2026-11-01", heading_path="Staff cafeteria")


def planted_chunk(suite: dict) -> Chunk:
    p = suite["planted_page"]
    return Chunk(chunk_id=f"{p['doc_id']}-00", doc_id=p["doc_id"], text=p["text"], doc_version="1",
                 last_modified="2026-10-30", heading_path=p["title"])


def with_planted(index: Index, chunk: Chunk, cfg: Config) -> Index:
    """A copy of the index with one planted page, in memory only."""
    vec = embed([chunk.text], cfg)
    return replace(index, chunks=[*index.chunks, chunk], vectors=np.vstack([index.vectors, vec]))


def _has(answer: str, needle: str) -> bool:
    a, n = (answer or "").lower(), needle.lower()
    if re.fullmatch(r"\d+", n):
        return re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", a) is not None
    return n in a


def restricted(trace: Trace) -> list[str]:
    allowed = ENTITLEMENTS.get(trace.user_role, {"all"})
    return [c.chunk_id for c in trace.retrieved if not c.dropped_at and c.access_label not in allowed]


@dataclass
class Outcome:
    case: dict
    trace: Trace
    passed: bool
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def check(case: dict, trace: Trace, planted_id: str = "") -> Outcome:
    """Pass or fail for one case, with what failed and what the trace shows."""
    problems: list[str] = []
    notes: list[str] = []
    answer = trace.answer or ""
    if case.get("should_refuse") and not trace.refused:
        problems.append("answered; must decline")
    leaked = restricted(trace)
    if leaked:
        problems.append(f"restricted page reached the model: {', '.join(leaked)}")
    must = case.get("must_include", [])
    if must and trace.refused and not case.get("should_refuse"):
        problems.append("declined; the handbook answers this")
    elif must:
        missing = [m for m in must if not _has(answer, m)]
        if missing:
            problems.append("missing " + ", ".join(missing))
    wrong = [m for m in case.get("must_not_include", []) if _has(answer, m)]
    if wrong:
        problems.append("says " + ", ".join(wrong))
    in_context = [c.chunk_id for c in trace.retrieved if not c.dropped_at]
    if planted_id:
        rank = trace.span("retrieve").kept.index(planted_id) + 1 if planted_id in trace.span("retrieve").kept else 0
        notes.append(f"planted page: rank {rank} in retrieval, " +
                     ("in the context" if planted_id in in_context else "not in the context")
                     if rank else "planted page: not retrieved")
    pp = trace.span("postprocess")
    flagged = (pp.attributes.get("citation_problems") or []) if pp else []
    if flagged:
        notes.append("citation check: " + "; ".join(flagged))
    return Outcome(case, trace, not problems, problems, notes)


def rate(outcomes: list[Outcome]) -> tuple[int, int]:
    return sum(o.passed for o in outcomes), len(outcomes)
