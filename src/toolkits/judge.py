"""The judge both toolkits use: your judge model, called through helix.llm.

DeepEval and Ragas each ship model clients of their own (the OpenAI SDK,
Instructor, LangChain, LiteLLM). None of them is used here. Each toolkit
accepts a custom model object, and the two classes below are those objects:
they turn a toolkit's request into one `helix.llm.complete` call, and turn
the reply back into the structured object the toolkit expects.

Toolkit judges ask for JSON in a fixed shape. Small local models often get
that wrong, so every reply is validated against the toolkit's own schema,
one failed reply is answered with the validation error and a second chance,
and `JudgeStats` counts what happened. A judge that needs many repairs is
telling you something about the model, not about your RAG system.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any

from helix.config import Config
from helix.llm import complete, embed

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.M)


class JudgeParseError(ValueError):
    """The judge model did not return JSON the toolkit can use, twice."""


@dataclass
class JudgeStats:
    calls: int = 0
    repaired: int = 0
    failed: int = 0
    by_schema: dict[str, int] = field(default_factory=dict)

    def line(self) -> str:
        return (f"{self.calls} judge calls, {self.repaired} replies needed a second attempt, "
                f"{self.failed} could not be parsed")


def judge_model(cfg: Config) -> str:
    return cfg.judge_model or cfg.generation_model


def judge_config(cfg: Config) -> Config:
    """Judges run at temperature 0, whatever the answer model uses."""
    return replace(cfg, temperature=0.0)


def extract_json(text: str) -> str:
    """The JSON object inside a reply: fences, preamble and <think> removed."""
    body = _FENCE.sub("", (text or "").strip())
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end <= start:
        return body
    return body[start:end + 1]


def structured(prompt: str, schema_cls, cfg: Config, stats: JudgeStats | None = None):
    """Ask the judge for an instance of the Pydantic class `schema_cls`."""
    from pydantic import ValidationError

    stats = stats if stats is not None else JudgeStats()
    schema = schema_cls.model_json_schema()
    name = getattr(schema_cls, "__name__", "schema")
    stats.by_schema[name] = stats.by_schema.get(name, 0) + 1
    messages = [{"role": "user", "content": prompt}]
    jcfg = judge_config(cfg)
    last_error = ""
    for attempt in (1, 2):
        stats.calls += 1
        text = complete(messages, jcfg, model=judge_model(cfg), json_schema=schema)[0]
        try:
            return schema_cls.model_validate_json(extract_json(text))
        except (ValidationError, ValueError) as exc:
            last_error = str(exc).splitlines()[0][:300]
            if attempt == 1:
                stats.repaired += 1
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": (
                        "That reply is not valid JSON for the required schema "
                        f"({last_error}). Reply again with only the JSON object, matching this schema:\n"
                        + json.dumps(schema))},
                ]
    stats.failed += 1
    raise JudgeParseError(f"{judge_model(cfg)} returned unusable JSON for {name}: {last_error}")


def plain(prompt: str, cfg: Config, stats: JudgeStats | None = None) -> str:
    if stats is not None:
        stats.calls += 1
    return complete([{"role": "user", "content": prompt}], judge_config(cfg), model=judge_model(cfg))[0]


# ---------------------------------------------------------------------------
# DeepEval
# ---------------------------------------------------------------------------
def deepeval_judge(cfg: Config, stats: JudgeStats | None = None):
    """A DeepEval model object that answers through helix.llm."""
    from deepeval.models import DeepEvalBaseLLM

    stats = stats if stats is not None else JudgeStats()

    class HelixJudge(DeepEvalBaseLLM):
        def __init__(self) -> None:
            self.cfg, self.stats = cfg, stats
            super().__init__(model=judge_model(cfg))

        def load_model(self) -> "HelixJudge":
            return self

        def generate(self, prompt: str, schema=None, **_: Any):
            if schema is None:
                return plain(prompt, self.cfg, self.stats)
            return structured(prompt, schema, self.cfg, self.stats)

        async def a_generate(self, prompt: str, schema=None, **kwargs: Any):
            return await asyncio.to_thread(self.generate, prompt, schema, **kwargs)

        def get_model_name(self) -> str:
            return f"{judge_model(self.cfg)} (through helix.llm)"

    return HelixJudge()


# ---------------------------------------------------------------------------
# Ragas
# ---------------------------------------------------------------------------
def ragas_llm(cfg: Config, stats: JudgeStats | None = None):
    """A Ragas collections LLM that answers through helix.llm."""
    from ragas.llms.base import InstructorBaseRagasLLM

    stats = stats if stats is not None else JudgeStats()

    class HelixRagasLLM(InstructorBaseRagasLLM):
        def __init__(self) -> None:
            self.cfg, self.stats = cfg, stats

        def generate(self, prompt: str, response_model):
            return structured(prompt, response_model, self.cfg, self.stats)

        async def agenerate(self, prompt: str, response_model):
            return await asyncio.to_thread(self.generate, prompt, response_model)

        def __repr__(self) -> str:
            return f"HelixRagasLLM({judge_model(self.cfg)})"

    return HelixRagasLLM()


def ragas_embeddings(cfg: Config):
    """Ragas embeddings from the workshop's embedding model (Answer Relevancy needs them)."""
    from ragas.embeddings.base import BaseRagasEmbedding

    class HelixRagasEmbeddings(BaseRagasEmbedding):
        def __init__(self) -> None:
            super().__init__(cache=None)
            self.cfg = cfg

        def embed_text(self, text: str, **_: Any) -> list[float]:
            return [float(x) for x in embed([text], self.cfg)[0]]

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return await asyncio.to_thread(self.embed_text, text, **kwargs)

        def embed_texts(self, texts: list[str], **_: Any) -> list[list[float]]:
            return [[float(x) for x in row] for row in embed(list(texts), self.cfg)]

        async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return await asyncio.to_thread(self.embed_texts, texts, **kwargs)

    return HelixRagasEmbeddings()
