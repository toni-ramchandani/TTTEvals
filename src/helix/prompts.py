"""Versioned prompt templates.

A first-draft prompt has these parts: Role, Objective,
Instructions, Context, Reasoning, Output format, Examples, and Delimiters that
keep them apart. The template is kept here as data rather than inlined at the
call site so that a prompt change is a diffable, versioned event, and so that the
version can be written into the trace beside the model version and the input.

Changing the text below without bumping ``PROMPT_VERSION`` and the config
hash makes yesterday's traces quietly incomparable with today's.

v2 (live models): the refusal is one fixed sentence, so the pipeline can tell
a refusal from an answer, and the citation must be copied exactly, so the
pipeline can tell which extract was used.
"""

from __future__ import annotations

from tracing.schema import Chunk

from .llm import REFUSAL

PROMPT_VERSION = "v2"

SYSTEM = """\
### Role
You are Helix, an assistant for an internal employee handbook.

### Objective
Answer the employee's question using only the retrieved policy extracts
supplied in the context. Success is an answer the people team would sign.

### Instructions
- Use only the supplied context. Do not use general knowledge about employment law.
- If the context does not contain the answer, reply with exactly this sentence
  and nothing else: "{refusal}"
- If two extracts disagree, say that they disagree and name both.
- Cite the extract you used by copying its bracketed reference exactly as it
  appears above the extract, for example [HB-EXAMPLE-00 v1 2026-01-01].
- Keep the answer to three sentences or fewer. No preamble, no headings.

### Output format
A short answer, followed by the bracketed reference of the extract used.
"""

USER = """\
### Context
{context}

### Question
{question}
"""


def render_context(chunks: list[Chunk]) -> str:
    """One block per chunk, each carrying its provenance.

    The bracketed reference is what makes source-to-answer lineage possible:
    from a claim in the answer you can reach the chunk, the document, its
    version and its date.
    """
    blocks = []
    for c in chunks:
        blocks.append(f"{c.cite()}\n{c.text}")
    return "\n\n".join(blocks) if blocks else "(no extracts retrieved)"


def build_messages(question: str, chunks: list[Chunk]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM.format(refusal=REFUSAL)},
        {"role": "user", "content": USER.format(context=render_context(chunks), question=question)},
    ]
