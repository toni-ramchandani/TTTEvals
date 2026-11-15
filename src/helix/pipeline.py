"""The full request path, instrumented.

Six spans, in the order a question travels through them. Every stage records
what it kept and what it dropped, because the difference between those two
sets is where most RAG failures are born.
"""

from __future__ import annotations

import re

from .config import Config
from .ingest import Index, build_index
from .llm import complete, count_tokens, is_refusal
from .prompts import PROMPT_VERSION, build_messages, render_context
from .retrieve import rerank, retrieve
from tracing.schema import Chunk, Trace
from tracing.tracer import span


def answer(
    question: str,
    index: Index,
    cfg: Config,
    user_office: str = "amsterdam",
    user_role: str = "employee",
    on_token=None,
    history: list[dict[str, str]] | None = None,
    standalone: str | None = None,
    extra_context: list[Chunk] | None = None,
) -> Trace:
    """Answer one question and return its full trace.

    `on_token`, if given, receives the answer text as the model streams it.

    Day 3 adds three optional inputs, all recorded in the trace:

    * `history`: the earlier turns of a conversation, as chat messages. The
      model sees them before the question (see `helix.conversation`).
    * `standalone`: the question retrieval should use instead of the raw
      message, when a conversation rewrote a follow-up into a standalone
      question in `query.transform`.
    * `extra_context`: chunks placed in front of the retrieved ones, which is
      how the robustness tests add an irrelevant chunk (`evals.day3_robustness`).
    """
    trace = Trace(
        query=question,
        config_hash=cfg.hash(),
        user_office=user_office,
        user_role=user_role,
    )

    # 1. query.transform -----------------------------------------------------
    with span(trace, "query.transform", original=question) as s:
        # A real system infers filters from user attributes here. Helix does
        # not, which is itself worth noticing: the office is known and unused.
        rewritten = (standalone or question).strip()
        s.attributes["rewritten"] = rewritten
        if history is not None:
            s.attributes["history_turns"] = sum(1 for m in history if m.get("role") == "user")
        s.attributes["inferred_filters"] = {}
        s.attributes["user_office_available_but_unused"] = user_office

    # 2. retrieve ------------------------------------------------------------
    with span(
        trace,
        "retrieve",
        retriever=cfg.retriever,
        top_k=cfg.top_k,
        embedding_model=cfg.embedding_model,
        enforce_entitlements=cfg.enforce_entitlements,
    ) as s:
        candidates, blocked = retrieve(index, rewritten, cfg, user_role=user_role)
        s.kept = [c.chunk_id for c in candidates]
        s.dropped = [c.chunk_id for c in blocked]
        s.attributes["blocked_by_entitlement"] = [c.chunk_id for c in blocked]
        s.attributes["candidates"] = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "doc_version": c.doc_version,
                "last_modified": c.last_modified,
                "access_label": c.access_label,
                "score": c.score,
            }
            for c in candidates
        ]

    for c in blocked:
        c.dropped_at = "retrieve"
    trace.considered.extend(blocked)

    # 3. rerank --------------------------------------------------------------
    with span(trace, "rerank", cutoff=cfg.rerank_cutoff) as s:
        ranked, demoted = rerank(candidates, rewritten, cfg)
        s.kept = [c.chunk_id for c in ranked]
        s.dropped = [c.chunk_id for c in demoted]
        for c in demoted:
            c.dropped_at = "rerank"
        trace.considered.extend(demoted)

    # 4. context.assemble ----------------------------------------------------
    with span(trace, "context.assemble", token_budget=cfg.context_token_budget) as s:
        assembled: list[Chunk] = []
        truncated: list[Chunk] = []
        used = 0
        for c in ranked:
            cost = count_tokens(c.text)
            if used + cost > cfg.context_token_budget:
                truncated.append(c)
                continue
            assembled.append(c)
            used += cost
        s.kept = [c.chunk_id for c in assembled]
        s.dropped = [c.chunk_id for c in truncated]
        s.attributes["tokens_used"] = used
        s.attributes["truncated"] = bool(truncated)
        for c in truncated:
            c.dropped_at = "context.assemble"
        trace.considered.extend(truncated)
        if extra_context:
            # Placed first, as a noisy retriever would place a confident miss.
            assembled = [*extra_context, *assembled]
            s.attributes["injected"] = [c.chunk_id for c in extra_context]

    trace.retrieved = assembled
    trace.considered.extend(assembled)

    # 5. generate ------------------------------------------------------------
    messages = build_messages(question.strip(), assembled, history=history)
    with span(
        trace,
        "generate",
        model=cfg.generation_model,
        temperature=cfg.temperature,
        prompt_version=PROMPT_VERSION,
        offline=cfg.offline,
    ) as s:
        text, meta = complete(messages, cfg, on_token=on_token)
        s.attributes.update(meta)
        s.attributes["context_chars"] = len(render_context(assembled))

    # 6. postprocess ---------------------------------------------------------
    with span(trace, "postprocess") as s:
        # A live model may shorten a reference, so a citation counts when the
        # chunk id appears in the answer, with or without version and date.
        citations = [c.cite() for c in assembled if c.cite() in text or c.chunk_id in text]
        refused = is_refusal(text)
        s.attributes["citations_emitted"] = citations
        s.attributes["refused"] = refused
        s.attributes["citation_problems"] = check_citations(text, assembled, user_role)
        s.attributes["citations_checked"] = True

    trace.answer = text
    trace.citations = citations
    trace.refused = refused
    from tracing.schema import _now

    trace.ended_at = _now()
    return trace


_CITED_ID = re.compile(r"\[([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*-\d{2})\b")


def check_citations(text: str, context: list[Chunk], user_role: str) -> list[str]:
    """Source-to-answer lineage, checked: what is wrong with the answer's citations.

    A citation should point to a chunk that was in the context, from a page
    that is current, that this asker may read. An empty list means no
    problem was found (or nothing was cited, which `refused` and Day 2's
    checks deal with).
    """
    from .ingest import document_status
    from .retrieve import ENTITLEMENTS

    status = document_status()
    allowed = ENTITLEMENTS.get(user_role, {"all"})
    in_context = {c.chunk_id: c for c in context}
    problems = []
    for cid in dict.fromkeys(_CITED_ID.findall(text or "")):
        c = in_context.get(cid)
        if c is None:
            problems.append(f"{cid} was not in the context")
            continue
        st = status.get(c.doc_id)
        if st is None:
            problems.append(f"{cid} is from {c.doc_id}, which is not a handbook page")
        elif st["status"] == "superseded":
            problems.append(f"{cid} is from {c.doc_id}, superseded by {st['superseded_by']}")
        if c.access_label not in allowed:
            problems.append(f"{cid} is {c.access_label}, not permitted for {user_role}")
    return problems


def build(cfg: Config) -> Index:
    """Convenience wrapper so a script reads as two lines."""
    return build_index(cfg)
