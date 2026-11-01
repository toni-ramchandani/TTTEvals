"""The trace schema the whole workshop reads and writes.

A *query* is the raw input to the system. A *trace* is the complete, ordered
record of everything the system did in response to it. A *span* is one stage
within that trace.

Three fields in here carry disproportionate weight, and they are the three
teams most often omit. Each is marked LOAD-BEARING below.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


def _now() -> float:
    return time.time()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


@dataclass
class Chunk:
    """One retrieved unit of evidence, with the provenance that makes it
    possible to tell a grounded-but-wrong answer from an ungrounded one."""

    chunk_id: str
    doc_id: str
    text: str
    # LOAD-BEARING (1 of 3): without doc_version and last_modified you cannot
    # distinguish "the system faithfully reported a superseded document" from
    # "the model ignored the evidence". Both look identical in the answer.
    doc_version: str
    last_modified: str
    heading_path: str = ""
    score: float = 0.0
    access_label: str = "all"
    #: "" if this chunk reached the model; otherwise the span that dropped it.
    dropped_at: str = ""

    def cite(self) -> str:
        return f"[{self.chunk_id} v{self.doc_version} {self.last_modified}]"


@dataclass
class Span:
    name: str
    span_id: str = field(default_factory=lambda: new_id("span"))
    started_at: float = field(default_factory=_now)
    ended_at: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    # LOAD-BEARING (2 of 3): what did NOT survive this stage. FP2 (missed
    # top-ranked documents) and FP3 (retrieved then dropped) live entirely in
    # the difference between `kept` and `dropped`. A trace that records only
    # survivors cannot see either of them.
    kept: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.ended_at is None:
            return 0.0
        return (self.ended_at - self.started_at) * 1000


@dataclass
class Trace:
    query: str
    trace_id: str = field(default_factory=lambda: new_id("tr"))
    # LOAD-BEARING (3 of 3): the configuration hash. Chunk strategy, embedding
    # model, top_k, prompt template, generation model. Without it a trace
    # cannot be compared with last week's, which makes Day 4's regression
    # analysis impossible.
    config_hash: str = ""
    user_office: str = "amsterdam"
    user_role: str = "employee"
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    refused: bool = False
    spans: list[Span] = field(default_factory=list)
    #: The chunks that reached the model.
    retrieved: list[Chunk] = field(default_factory=list)
    #: Every chunk the pipeline touched, including the ones it dropped. The
    #: difference between this and `retrieved` is where FP2 and FP3 live.
    considered: list[Chunk] = field(default_factory=list)
    started_at: float = field(default_factory=_now)
    ended_at: float | None = None
    #: Which architecture produced this trace: "helix" (vanilla RAG),
    #: "agentic" or "graphrag". Same schema for all three, so every evaluator
    #: in the workshop reads all three without modification.
    system: str = "helix"

    def span(self, name: str) -> Span | None:
        for s in self.spans:
            if s.name == name:
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duration_ms"] = (self.ended_at - self.started_at) * 1000 if self.ended_at else 0.0
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def trace_from_dict(d: dict[str, Any]) -> Trace:
    spans = [Span(**s) for s in d.pop("spans", [])]
    retrieved = [Chunk(**c) for c in d.pop("retrieved", [])]
    considered = [Chunk(**c) for c in d.pop("considered", [])]
    d.pop("duration_ms", None)
    t = Trace(**d)
    t.spans = spans
    t.retrieved = retrieved
    t.considered = considered
    return t
