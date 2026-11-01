"""The knowledge graph: nodes, facts and edges, each fact carrying provenance.

Four kinds of node:

* ``office``    Dublin, Amsterdam, Berlin, London, Singapore
* ``company``   one node, for company-wide facts that apply to every office
* ``document``  one per handbook page, carrying version and last-modified date
* ``topic``     parental_leave, notice_period, salary_band, annual_leave
* ``fact``      one atomic statement: one table row, or one sentence

Edges:

* ``office -[PART_OF]-> company``
* ``office -[HAS_FACT]-> fact`` and ``company -[HAS_FACT]-> fact``
* ``topic -[ABOUT]-> fact``
* ``fact -[STATED_IN]-> document``
* ``document -[SUPERSEDES]-> document``

A fact is the graph's unit of evidence, the way a chunk is vanilla RAG's. It is
smaller (one row, with its office and role attached as properties rather than
left behind in a heading), and it is decided at build time, not at query time.
Both of those are the point, and both are where GraphRAG's own failures are
born.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from tracing.schema import Chunk


@dataclass
class Fact:
    fact_id: str
    topic: str
    statement: str
    #: Canonical office, or None for a company-wide fact.
    office: str | None
    attrs: dict[str, Any]
    # Provenance. Every one of these is copied from the source document at
    # extraction time. If extraction drops one, nothing downstream can recover it.
    doc_id: str
    doc_version: str
    last_modified: str
    access_label: str
    source_chunk_id: str
    section: str

    def as_chunk(self, score: float = 0.0) -> Chunk:
        """A fact presented in the trace schema every evaluator already reads."""
        return Chunk(
            chunk_id=self.fact_id,
            doc_id=self.doc_id,
            text=self.statement,
            doc_version=self.doc_version,
            last_modified=self.last_modified,
            heading_path=self.section,
            score=round(score, 4),
            access_label=self.access_label,
        )


@dataclass
class Edge:
    src: str
    rel: str
    dst: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:
    node_id: str
    kind: str
    label: str
    attrs: dict[str, Any] = field(default_factory=dict)


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.facts: dict[str, Fact] = {}
        self.edges: list[Edge] = []
        self._adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.build_info: dict[str, Any] = {}

    # -- construction -------------------------------------------------------
    def add_node(self, node_id: str, kind: str, label: str = "", **attrs: Any) -> Node:
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id, kind, label or node_id, dict(attrs))
        return self.nodes[node_id]

    def add_edge(self, src: str, rel: str, dst: str, **attrs: Any) -> None:
        self.edges.append(Edge(src, rel, dst, dict(attrs)))
        self._adj[src].append((rel, dst))
        self._adj[dst].append((f"~{rel}", src))

    def add_fact(self, fact: Fact) -> None:
        if fact.fact_id in self.facts:
            return
        self.facts[fact.fact_id] = fact
        self.add_node(fact.fact_id, "fact", fact.statement)
        owner = f"office:{fact.office}" if fact.office else "company"
        self.add_edge(owner, "HAS_FACT", fact.fact_id)
        self.add_edge(f"topic:{fact.topic}", "ABOUT", fact.fact_id)
        self.add_edge(fact.fact_id, "STATED_IN", f"doc:{fact.doc_id}")

    # -- reading ------------------------------------------------------------
    def neighbours(self, node_id: str) -> list[tuple[str, str]]:
        return self._adj.get(node_id, [])

    def superseded_docs(self) -> dict[str, str]:
        """doc_id -> the doc_id that supersedes it."""
        out: dict[str, str] = {}
        for e in self.edges:
            if e.rel == "SUPERSEDES":
                out[e.dst.removeprefix("doc:")] = e.src.removeprefix("doc:")
        return out

    def stats(self) -> dict[str, int]:
        kinds: dict[str, int] = defaultdict(int)
        for n in self.nodes.values():
            kinds[n.kind] += 1
        rels: dict[str, int] = defaultdict(int)
        for e in self.edges:
            rels[e.rel] += 1
        return {**{f"nodes.{k}": v for k, v in sorted(kinds.items())},
                **{f"edges.{k}": v for k, v in sorted(rels.items())}}

    def to_dict(self) -> dict[str, Any]:
        return {
            "build_info": self.build_info,
            "nodes": [asdict(n) for n in self.nodes.values() if n.kind != "fact"],
            "facts": [asdict(f) for f in self.facts.values()],
            "edges": [asdict(e) for e in self.edges],
        }
