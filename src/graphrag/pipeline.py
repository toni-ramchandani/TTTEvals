"""Request time for GraphRAG: link, traverse, filter, rank, generate.

The request path is different from vanilla Helix in kind, not just in degree.
Helix asks "which chunks look like this question?". GraphRAG asks "which
entities does this question name, and what does the graph say about them?".
That makes some questions easy that are hard for chunk retrieval (comparisons
across offices, questions that need two facts joined on the same entity) and
makes a new set of things go wrong (an entity the linker does not recognise,
a fact the extractor attributed to the wrong node).

Spans, in order:

  graph.link       question -> offices, topics, role, tenure, comparative?
  graph.traverse   breadth-first from the linked nodes, up to ``hops``
  graph.filter     drop superseded and unentitled facts, and record both
  graph.rank       order what is left; derive a ranking for comparisons
  fallback.retrieve  only if the graph has nothing on the topic at all
  generate         the same generator Helix uses, reading facts not chunks
  postprocess      citations and refusal, exactly as in Helix

Facts are written into the trace as ``Chunk`` objects, so every Day 1 to Day 4
evaluator reads a GraphRAG trace without modification. That is a design choice
worth copying: a new architecture should change what fills the trace, not the
shape of the trace.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass, field

from helix.config import REPAIRED, Config
from helix.ingest import Index
from helix.llm import _REFUSAL, complete, is_refusal
from helix.prompts import PROMPT_VERSION, build_messages
from helix.retrieve import ENTITLEMENTS, rerank, retrieve
from helix.structure import (
    OFFICES,
    is_comparative,
    level_in,
    offices_in,
    role_in,
    tenure_band,
    tenure_in,
    topics_in,
)
from tracing.schema import Chunk, Trace, _now
from tracing.tracer import span

from .extract import GraphBuildConfig, build_graph
from .graph import Fact, Graph

_WORDS = __import__("re").compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class GraphConfig:
    build: GraphBuildConfig = field(default_factory=GraphBuildConfig)
    linking: str = "alias"            # "alias" | "exact"            GR1
    use_user_profile: bool = True     # seed the user's office when the question names none
    hops: int = 2
    respect_supersedes: bool = True   # drop facts from superseded pages   GR3
    enforce_entitlements: bool = True # drop facts the user may not see    GR4
    fallback_to_text: bool = True     # chunk retrieval when the graph has no facts on the topic   GR5
    max_facts: int = 4
    base: Config = REPAIRED           # fallback retrieval and the generation model

    def hash(self) -> str:
        payload = asdict(self)
        payload["base"] = self.base.hash()
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def describe(self) -> str:
        return (
            f"graph unit={self.build.unit} acl={'on' if self.build.carry_access_label else 'off'} "
            f"supersedes={'on' if self.build.infer_supersession else 'off'} | "
            f"link={self.linking} profile={'on' if self.use_user_profile else 'off'} hops={self.hops} "
            f"filter-stale={'on' if self.respect_supersedes else 'off'} "
            f"filter-acl={'on' if self.enforce_entitlements else 'off'} "
            f"fallback={'on' if self.fallback_to_text else 'off'} [{self.hash()}]"
        )


#: Every switch on. The configuration the Day 3 comparison treats as GraphRAG.
GRAPH_DEFAULT = GraphConfig()

#: Every switch off. Each failure this produces is one switch's worth of lesson.
GRAPH_NAIVE = GraphConfig(
    build=GraphBuildConfig(unit="chunk", carry_access_label=False, infer_supersession=False),
    linking="exact",
    use_user_profile=False,
    respect_supersedes=False,
    enforce_entitlements=False,
    fallback_to_text=False,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _overlap(a: str, b: str) -> int:
    return len(set(_WORDS.findall(a.lower())) & set(_WORDS.findall(b.lower())))


def _bfs(graph: Graph, seeds: list[str], hops: int) -> dict[str, int]:
    """node_id -> distance from the nearest seed, up to ``hops``."""
    dist = {s: 0 for s in seeds if s in graph.nodes}
    queue = deque(dist)
    while queue:
        node = queue.popleft()
        if dist[node] >= hops:
            continue
        for _, nxt in graph.neighbours(node):
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    return dist


def _view(fact: Fact, tenure: float | None) -> str:
    """The fact as the question needs it. A notice fact holds three tenure
    bands; with a known tenure the graph can answer with the one that applies."""
    if fact.topic == "notice_period" and tenure is not None and fact.attrs.get("header_found"):
        band = tenure_band(tenure)
        value = fact.attrs["bands"].get(band)
        if value:
            years = int(tenure) if float(tenure).is_integer() else tenure
            return (
                f"{fact.office.title()} office notice period on resignation with {years} years' "
                f"service ({band.lower()}): {value}."
            )
    return fact.statement


def _rank_score(fact: Fact, question: str, seed_offices: list[str], role: str | None, level: str | None) -> float:
    score = 0.1 * _overlap(fact.statement, question)
    if fact.office and fact.office in seed_offices:
        score += 3
    elif fact.office is None:
        score += 1
    frole = fact.attrs.get("role")
    if role and frole == role:
        score += 2
    elif role and frole not in (None, "any"):
        score -= 3
    if level and fact.attrs.get("level") == level:
        score += 2
    return score


def _derive_ranking(facts: list[Fact], role: str) -> Chunk | None:
    """For "which office has the longest ...": rank office-specific facts.

    This is the question type graphs are built for. Vanilla retrieval returns
    the k chunks most like the question and the generator reports one of them;
    the graph holds every office's figure as a property and can sort.
    """
    rows = [f for f in facts if f.topic == "parental_leave" and f.office and f.attrs.get("role") == role]
    if not rows:
        return None
    rows.sort(key=lambda f: f.attrs["weeks"], reverse=True)
    top = rows[0]
    rest = "; ".join(f"{f.office.title()} office, {f.attrs['weeks']} weeks" for f in rows[1:])
    missing = [o.title() for o in OFFICES if o not in {f.office for f in rows}]
    text = (
        f"Longest {role} caregiver parental leave: {top.office.title()} office, "
        f"{top.attrs['weeks']} weeks ({top.attrs['working_days']} working days)."
    )
    if rest:
        text += f" Then {rest}."
    if missing:
        text += f" No office-specific entry for {', '.join(missing)}."
    return Chunk(
        chunk_id=f"{top.fact_id}+ranked",
        doc_id=top.doc_id,
        text=text,
        doc_version=top.doc_version,
        last_modified=top.last_modified,
        heading_path="derived: ranking over " + ", ".join(f.fact_id for f in rows),
        score=1.0,
        access_label=top.access_label,
    )


# ---------------------------------------------------------------------------
# The request path
# ---------------------------------------------------------------------------
def answer(
    question: str,
    graph: Graph,
    gcfg: GraphConfig = GRAPH_DEFAULT,
    index: Index | None = None,
    user_office: str = "amsterdam",
    user_role: str = "employee",
    user_tenure_years: float | None = None,
    on_token=None,
) -> Trace:
    """Answer one question from the graph and return its trace.

    `on_token`, if given, receives the answer text as a live model streams it.
    """
    trace = Trace(query=question, config_hash=gcfg.hash(), user_office=user_office, user_role=user_role)
    trace.system = "graphrag"

    # 1. link ----------------------------------------------------------------
    with span(trace, "graph.link", linking=gcfg.linking, build_hash=graph.build_info.get("build_hash")) as s:
        offices = offices_in(question, aliases=gcfg.linking == "alias")
        topics = topics_in(question)
        role = role_in(question)
        level = level_in(question)
        comparative = is_comparative(question)
        tenure = tenure_in(question)
        profile_used = []
        if not offices and not comparative and gcfg.use_user_profile:
            offices = [user_office]
            profile_used.append("office")
        if tenure is None and gcfg.use_user_profile and user_tenure_years is not None:
            tenure = user_tenure_years
            profile_used.append("tenure")
        if comparative:
            seeds = [f"office:{o}" for o in OFFICES]
        elif offices:
            seeds = [f"office:{o}" for o in offices]
        else:
            seeds = [f"topic:{t}" for t in topics]
        s.kept = seeds
        s.attributes.update(
            offices_linked=offices, topics_linked=topics, role=role, level=level,
            tenure_years=tenure, comparative=comparative, profile_used=profile_used,
            offices_in_question_any_alias=offices_in(question, aliases=True),
        )

    # 2. traverse ------------------------------------------------------------
    with span(trace, "graph.traverse", hops=gcfg.hops) as s:
        dist = _bfs(graph, seeds, gcfg.hops)
        reached = [graph.facts[n] for n in dist if n in graph.facts]
        on_topic = [f for f in reached if not topics or f.topic in topics]
        s.kept = [f.fact_id for f in on_topic]
        s.dropped = [f.fact_id for f in reached if f not in on_topic]
        s.attributes["nodes_visited"] = len(dist)
        s.attributes["facts_reached"] = len(reached)
        s.attributes["graph_has_topic"] = {
            t: any(f.topic == t for f in graph.facts.values()) for t in topics
        }

    # 3. filter --------------------------------------------------------------
    superseded = graph.superseded_docs()
    allowed = ENTITLEMENTS.get(user_role, {"all"})
    with span(
        trace, "graph.filter",
        respect_supersedes=gcfg.respect_supersedes,
        enforce_entitlements=gcfg.enforce_entitlements,
        supersession_edges=superseded,
    ) as s:
        kept: list[Fact] = []
        stale, blocked = [], []
        for f in on_topic:
            if gcfg.respect_supersedes and f.doc_id in superseded:
                stale.append(f)
            elif gcfg.enforce_entitlements and f.access_label not in allowed:
                blocked.append(f)
            else:
                kept.append(f)
        s.kept = [f.fact_id for f in kept]
        s.dropped = [f.fact_id for f in stale + blocked]
        s.attributes["dropped_superseded"] = [f.fact_id for f in stale]
        s.attributes["blocked_by_entitlement"] = [f.fact_id for f in blocked]
    for f, why in [(f, "graph.filter") for f in stale + blocked]:
        c = f.as_chunk()
        c.dropped_at = why
        trace.considered.append(c)

    # 4. rank ----------------------------------------------------------------
    seed_offices = [sd.removeprefix("office:") for sd in seeds if sd.startswith("office:")]
    with span(trace, "graph.rank", max_facts=gcfg.max_facts) as s:
        scored = sorted(
            kept, key=lambda f: _rank_score(f, question, [] if comparative else seed_offices, role, level),
            reverse=True,
        )
        per_topic: dict[str, list[Chunk]] = {}
        context: list[Chunk] = []
        if comparative:
            derived = _derive_ranking(kept, role or "primary")
            if derived:
                context.append(derived)
                per_topic[topics[0] if topics else "parental_leave"] = [derived]
        for f in scored[: gcfg.max_facts]:
            c = f.as_chunk(_rank_score(f, question, seed_offices, role, level))
            c.text = _view(f, tenure)
            context.append(c)
            if not comparative:
                per_topic.setdefault(f.topic, []).append(c)
        s.kept = [c.chunk_id for c in context]
        s.dropped = [f.fact_id for f in scored[gcfg.max_facts:]]
        s.attributes["derived"] = bool(comparative and context and context[0].chunk_id.endswith("+ranked"))
        for f in scored[gcfg.max_facts:]:
            c = f.as_chunk()
            c.dropped_at = "graph.rank"
            trace.considered.append(c)

    # 5. fallback ------------------------------------------------------------
    graph_covers = any(f.topic in topics for f in graph.facts.values()) if topics else False
    used_fallback = False
    if not graph_covers and gcfg.fallback_to_text and index is not None:
        used_fallback = True
        with span(trace, "fallback.retrieve", reason="graph has no facts on this topic") as s:
            cands, filtered = retrieve(index, question, gcfg.base, user_role=user_role)
            ranked, demoted = rerank(cands, question, gcfg.base)
            context = ranked
            s.kept = [c.chunk_id for c in ranked]
            s.dropped = [c.chunk_id for c in demoted + filtered]
        per_topic = {"fallback": context}

    trace.retrieved = context
    trace.considered.extend(context)

    # 6. generate ------------------------------------------------------------
    with span(
        trace, "generate",
        model=gcfg.base.generation_model, prompt_version=PROMPT_VERSION, offline=gcfg.base.offline,
    ) as s:
        if not context:
            text, meta = _REFUSAL, {"finish_reason": "no_evidence"}
            if on_token:
                on_token(text)
        elif gcfg.base.offline:
            # The offline generator reports one statement per evidence set. The
            # graph hands it one set per topic the question asked about.
            parts, meta = [], {}
            order = [t for t in topics if t in per_topic] or list(per_topic)
            for t in order:
                part, meta = complete(build_messages(question, per_topic[t]), gcfg.base)
                if part != _REFUSAL:
                    parts.append(part)
            text = " ".join(parts) if parts else _REFUSAL
            if on_token:
                on_token(text)
        else:
            text, meta = complete(build_messages(question, context), gcfg.base, on_token=on_token)
        s.attributes.update(meta)
        s.attributes["evidence_sets"] = {k: [c.chunk_id for c in v] for k, v in per_topic.items()}

    # 7. postprocess ---------------------------------------------------------
    with span(trace, "postprocess") as s:
        citations = [c.cite() for c in context if c.cite() in text or c.chunk_id in text]
        refused = is_refusal(text)
        s.attributes.update(citations_emitted=citations, refused=refused, used_fallback=used_fallback)

    trace.answer, trace.citations, trace.refused = text, citations, refused
    trace.ended_at = _now()
    return trace


def build(gcfg: GraphConfig = GRAPH_DEFAULT) -> Graph:
    return build_graph(gcfg.build, model_cfg=gcfg.base)
