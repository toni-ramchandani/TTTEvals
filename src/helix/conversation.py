"""Helix in a conversation: one trace per turn, and three ways to carry history.

Day 1's Helix answers one message at a time. A real employee follows up:
"And the salary top-up?" only makes sense after the question before it. This
module wraps the same pipeline, so every turn is an ordinary trace with all
six spans, plus three fields that place it in a session (`session_id`,
`turn`, `history`). Every Day 1 to Day 3 evaluator reads it unchanged.

How much of the conversation the system carries is a switch, like the
agent's ablations, so its effect can be measured:

    last      each message alone: no history in the prompt, retrieval on the
              raw message. Day 1's Helix in a chat window.
    window    the last turns go into the prompt, but retrieval still uses the
              raw message. The common default in chat frameworks.
    condense  `query.transform` first rewrites the message as a standalone
              question from the conversation, retrieval uses that, and the
              prompt carries the turns too.

Identity always comes from login (`user_office`, `user_role`). Nothing a user
writes in the chat changes it.

Offline (the reference run) `condense` uses a fixed rule instead of a model:
it joins the user's earlier messages to the new one. Live, your model writes
the standalone question, and the trace records it.
"""

from __future__ import annotations

from dataclasses import replace

from .config import Config
from .ingest import Index
from .llm import complete
from .pipeline import answer
from tracing.schema import Trace, new_id

HISTORY_MODES = ("last", "window", "condense")
WINDOW = 3  # earlier exchanges kept in the prompt (user + assistant pairs)

CONDENSE = """\
Rewrite the user's last message as one standalone question that someone who
has not seen the conversation can understand. Keep every fact the user gave
about themselves (role, working pattern, years of service, dates) and the
topic. Do not answer it. Reply with the question only.

Conversation:
{transcript}

Last message: {message}

Standalone question:"""


def transcript(messages: list[dict[str, str]]) -> str:
    who = {"user": "User", "assistant": "Assistant"}
    return "\n".join(f"{who.get(m['role'], m['role'])}: {' '.join(m['content'].split())}"
                     for m in messages)


def condense(prior: list[dict[str, str]], message: str, cfg: Config) -> str:
    """The message as a standalone question, using the conversation so far."""
    if not prior:
        return message
    if cfg.offline:
        earlier = [m["content"].strip() for m in prior if m["role"] == "user"]
        return " ".join([*earlier, message.strip()])
    text = complete([{"role": "user", "content": CONDENSE.format(transcript=transcript(prior),
                                                                    message=message)}],
                    replace(cfg, temperature=0.0))[0]
    text = " ".join((text or "").split()).strip().strip('"')
    return text or message


def turn(message: str, prior: list[dict[str, str]], index: Index, cfg: Config, mode: str,
         user_office: str, user_role: str, session_id: str, number: int) -> Trace:
    """Answer one message in a conversation and return its trace."""
    if mode not in HISTORY_MODES:
        raise ValueError(f"history mode must be one of {HISTORY_MODES}, not {mode!r}")
    history = [] if mode == "last" else prior[-2 * WINDOW:]
    standalone = condense(prior, message, cfg) if mode == "condense" and prior else None
    t = answer(message, index, cfg, user_office=user_office, user_role=user_role,
               history=history if mode != "last" else None, standalone=standalone)
    t.session_id, t.turn, t.history = session_id, number, list(history)
    qt = t.span("query.transform")
    if qt is not None:
        qt.attributes["history_mode"] = mode
        qt.attributes["condensed"] = standalone is not None
    return t


def run(user_messages: list[str], index: Index, cfg: Config, mode: str, user_office: str,
        user_role: str, session_id: str | None = None,
        prefix: list[dict[str, str]] | None = None) -> list[Trace]:
    """A scripted conversation: the user's messages in order, one trace per turn.

    `prefix` starts the conversation from earlier messages (user and
    assistant) that are replayed as they were, not answered again. That is
    an N-1 test: the first N-1 turns fixed, the next one generated.
    """
    sid = session_id or new_id("ses")
    prior = list(prefix or [])
    start = sum(1 for m in prior if m["role"] == "user")
    out = []
    for i, msg in enumerate(user_messages, start + 1):
        t = turn(msg, prior, index, cfg, mode, user_office, user_role, sid, i)
        out.append(t)
        prior += [{"role": "user", "content": msg}, {"role": "assistant", "content": t.answer}]
    return out


def messages_of(traces: list[Trace]) -> list[dict[str, str]]:
    """The session as chat messages, user and assistant in turn."""
    out = []
    for t in traces:
        out += [{"role": "user", "content": t.query}, {"role": "assistant", "content": t.answer}]
    return out
