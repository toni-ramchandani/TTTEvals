"""The tools the Helix agent can call.

Five tools, each a thin, deterministic wrapper over something the workshop
already has. The agent's power comes from choosing among them and choosing
their arguments; its failures come from the same place.

=========================  ===================================================
``get_user_profile``       who is asking: office, role, years of service.
                           Vanilla Helix has this and never uses it.
``list_offices``           the offices the handbook covers.
``search_handbook``        hybrid retrieval and reranking over the Helix index,
                           with an optional office filter.
``check_document_status``  is this page current, or superseded by a newer one?
                           In production this is the document management
                           system. Here it applies the same rule the GraphRAG
                           builder uses: same title, newer last_modified wins.
``finish``                 end the loop with an answer and its citations.
=========================  ===================================================

``TOOL_SPECS`` is the same set in the JSON-schema form a model's tool-calling
API expects. The live agent loop sends it, minus whatever the switches take
away (see ``agent.live_tools``); the reference-run planner calls the Python
methods directly. Both write identical spans.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from helix.config import Config
from helix.ingest import HANDBOOK, Index, document_status
from helix.retrieve import rerank, retrieve
from helix.structure import OFFICES, offices_in
from tracing.schema import Chunk


class Toolbox:
    def __init__(self, index: Index, cfg: Config, profile: dict[str, Any]) -> None:
        self.index = index
        self.cfg = cfg
        self.profile = dict(profile)
        self._status: dict[str, dict[str, Any]] = document_status(HANDBOOK)
        #: Every chunk any tool has returned, keyed by id, so the trace can
        #: record the full set the agent saw.
        self.seen: dict[str, Chunk] = {}

    # -- tools --------------------------------------------------------------
    def get_user_profile(self) -> dict[str, Any]:
        return dict(self.profile)

    def list_offices(self) -> list[str]:
        return list(OFFICES)

    def search_handbook(self, query: str, office: str | None = None) -> tuple[list[Chunk], list[Chunk]]:
        """Return (results, filtered_out).

        With an office, the query is extended with the office name and any
        chunk that names only *other* offices is filtered out. Chunks that name
        no office (company-wide pages) are kept, which is correct, and is also
        how a superseded company-wide page gets back in.
        """
        q = query
        wide = replace(self.cfg, top_k=max(self.cfg.top_k, 8))
        cands, blocked = retrieve(self.index, q, wide, user_role=self.profile.get("role", "employee"))
        filtered: list[Chunk] = []
        if office:
            keep = []
            for c in cands:
                named = offices_in(c.text)
                if named and office not in named:
                    filtered.append(c)
                else:
                    keep.append(c)
            cands = keep
        ranked, demoted = rerank(cands, q, self.cfg)
        for c in ranked + demoted + filtered + blocked:
            self.seen.setdefault(c.chunk_id, c)
        return ranked, filtered + demoted + blocked

    def check_document_status(self, doc_id: str) -> dict[str, Any]:
        return dict(self._status.get(doc_id, {"doc_id": doc_id, "status": "unknown"}))


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_user_profile",
            "description": "Return the asking employee's contracting office, role and years of service.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_offices",
            "description": "List the offices the handbook covers.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_handbook",
            "description": (
                "Search the employee handbook. Returns up to three extracts, each with its "
                "chunk id, document id, version and last-modified date. Pass `office` to "
                "exclude extracts about other offices."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "office": {"type": "string", "enum": list(OFFICES)},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_document_status",
            "description": "Report whether a handbook document is current or superseded, and by what.",
            "parameters": {
                "type": "object",
                "properties": {"doc_id": {"type": "string"}},
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Give the final answer, with the bracketed reference of every extract used.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["answer"],
            },
        },
    },
]
