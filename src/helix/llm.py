"""The only place the project talks to a model.

Live mode (the default) calls a real model over plain HTTPS using only the
Python standard library: no model SDK, no gateway library, nothing extra to
install or trust. Ollama is called through its own API on your laptop; cloud
models through their OpenAI-compatible endpoints (OpenAI, Gemini, Anthropic).
The model is chosen in `.env`:

    HELIX_GENERATION_MODEL=ollama/llama3.1:8b          answers, tool calls
    HELIX_EMBEDDING_MODEL=ollama/nomic-embed-text      retrieval embeddings
    HELIX_JUDGE_MODEL=                                 Day 3 judge (empty = same as answers)

Offline mode (`HELIX_OFFLINE=1` or `python run.py <cmd> --offline`) swaps in a
deterministic stub and hash embeddings. It exists for one reason: the
reference numbers on the slides, which every laptop reproduces exactly.

Every call here returns plain Python values, and every failure is turned into
a ModelError that says what to fix (start Ollama, pull the model, set a key).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request

import numpy as np

from .config import Config

Messages = list[dict[str, str]]


# ---------------------------------------------------------------------------
# Live path: plain HTTPS, Python standard library only
# ---------------------------------------------------------------------------
class ModelError(RuntimeError):
    """A model call failed. The message says what to do about it."""


_THINK = re.compile(r"<think>.*?</think>", re.S)

#: Cloud providers, each reached through its OpenAI-compatible HTTP endpoint.
#: prefix -> (base URL, environment variable holding the key, docs)
PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY",
               "https://platform.openai.com/docs/api-reference/chat"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY",
               "https://ai.google.dev/gemini-api/docs/openai"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY",
                  "https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk"),
}
_OLLAMA_PREFIXES = ("ollama/", "ollama_chat/")  # ollama_chat/ kept for older .env files


def _timeout() -> float:
    # Generous: a local model's first answer includes loading it into memory.
    return float(os.getenv("HELIX_TIMEOUT", "300"))


def _route(model: str) -> tuple[str, str, str, dict]:
    """Model string -> (kind, model name, base URL, headers).

    "ollama/llama3.1:8b"       Ollama's own API on this machine
    "openai/gpt-5.4-mini"      OpenAI (a bare name like "gpt-5.4-mini" also works)
    "gemini/gemini-3.6-flash"  Google, through its OpenAI-compatible endpoint
    "anthropic/claude-..."     Anthropic, through its OpenAI-compatible endpoint
    """
    if model.startswith(_OLLAMA_PREFIXES):
        base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434").rstrip("/")
        return "ollama", model.split("/", 1)[1], base, {}
    prefix, _, name = model.partition("/")
    if not name:  # a bare name is OpenAI's
        prefix, name = "openai", model
    if prefix not in PROVIDERS:
        raise ModelError(f"Unknown model provider '{prefix}' in '{model}'. Use one of: ollama/, "
                         + ", ".join(f"{p}/" for p in PROVIDERS) + ".")
    base, key_var, _ = PROVIDERS[prefix]
    if prefix == "openai":
        base = os.getenv("OPENAI_BASE_URL", base)
    key = os.getenv(key_var, "")
    if not key:
        raise ModelError(f"{model} needs {key_var}. Put it in .env (copy .env.example to .env first).")
    return "openai", name, base.rstrip("/"), {"Authorization": f"Bearer {key}"}


class _HTTPFailure(Exception):
    def __init__(self, status: int | None, body: str) -> None:
        super().__init__(f"{status}: {body}")
        self.status, self.body = status, body


def _open(url: str, payload: dict, headers: dict):
    """POST JSON and return the open response. The only network call in the project."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "rag-evals-workshop", **headers},
    )
    try:
        return urllib.request.urlopen(req, timeout=_timeout())
    except urllib.error.HTTPError as exc:
        raise _HTTPFailure(exc.code, exc.read().decode("utf-8", "replace")[:500]) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise _HTTPFailure(None, str(getattr(exc, "reason", exc))) from exc


def _post(url: str, payload: dict, headers: dict) -> dict:
    with _open(url, payload, headers) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_lines(url: str, payload: dict, headers: dict):
    """POST JSON and yield the response line by line, for streaming."""
    with _open(url, payload, headers) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                yield line


def _hint(model: str, exc: _HTTPFailure) -> str:
    """Turn a failed call into one line that says what to do."""
    body = exc.body.lower()
    if model.startswith(_OLLAMA_PREFIXES):
        name = model.split("/", 1)[1]
        if exc.status is None:
            return ("Ollama is not running. Start the Ollama app (or run `ollama serve`) and try again. "
                    f"Server expected at {os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')}.")
        if exc.status == 404 or "not found" in body:
            return f"Ollama does not have this model yet. Run:  ollama pull {name}"
        return f"Ollama call failed for {name} ({exc.status}): {exc.body[:300]}"
    if exc.status is None:
        return f"Could not reach the provider for {model}: {exc.body[:200]}. Check your internet connection."
    if exc.status in (401, 403):
        return f"The API key for {model} was rejected ({exc.status}). Check the key in .env."
    if exc.status == 404:
        return f"The provider does not recognise the model '{model}' (404). Check the id in .env."
    if exc.status == 429:
        return f"Rate limit or quota reached for {model} (429). Wait a minute, or use a local Ollama model."
    return f"Model call failed for {model} ({exc.status}): {exc.body[:300]}"


def _clean(text: str) -> str:
    """Strip a reasoning model's <think> block; keep only the answer."""
    return _THINK.sub("", text or "").strip()


def _visible(partial: str) -> str:
    """The part of a still-streaming answer that is safe to show: closed
    <think> blocks removed, and nothing from an unclosed one."""
    out = _THINK.sub("", partial)
    return out.split("<think>", 1)[0].lstrip()


def _num_ctx() -> int:
    # Ollama's default window is small, and it truncates silently when the
    # agent's conversation (tool results included) outgrows it.
    return int(os.getenv("OLLAMA_NUM_CTX", "8192"))


def _ollama_messages(messages: list[dict]) -> list[dict]:
    """The agent speaks the OpenAI message format; Ollama's own API differs in
    two places: tool-call arguments are an object, and a tool result names
    its tool instead of a call id."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            m["tool_calls"] = [{"function": {
                "name": tc["function"]["name"],
                "arguments": (json.loads(tc["function"]["arguments"] or "{}")
                              if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]),
            }} for tc in m["tool_calls"]]
        if m.get("role") == "tool":
            m = {"role": "tool", "tool_name": m.get("name", ""), "content": m.get("content", "")}
        m["content"] = m.get("content") or ""
        out.append(m)
    return out


def _chat(model: str, messages: list[dict], cfg: Config, tools: list[dict] | None,
          on_token=None, json_schema: dict | None = None) -> tuple[str, list[dict], dict]:
    """One chat call to any provider. Returns (text, raw tool calls, metadata).

    `json_schema` asks for JSON output, which evaluation judges need. Ollama
    receives the schema itself (its structured outputs); OpenAI receives JSON
    mode. Other providers get no extra field: the judge prompt already asks
    for JSON, and the caller validates what comes back. A server that rejects
    the field is asked again without it.
    """
    kind, name, base, headers = _route(model)
    stream = on_token is not None and not tools
    parts: list[str] = []
    shown = 0
    usage: dict = {}
    finish = "stop"
    calls: list[dict] = []

    def emit(piece: str) -> None:
        nonlocal shown
        parts.append(piece)
        visible = _visible("".join(parts))
        if len(visible) > shown:
            on_token(visible[shown:])
            shown = len(visible)

    if kind == "ollama":
        payload: dict = {"model": name, "messages": _ollama_messages(messages), "stream": stream,
                         "options": {"temperature": cfg.temperature, "num_ctx": _num_ctx()}}
        if tools:
            payload["tools"] = tools
        if json_schema is not None:
            payload["format"] = json_schema
        url = f"{base}/api/chat"
        if not stream and json_schema is not None:
            try:
                data = _post(url, payload, headers)
            except _HTTPFailure as exc:
                # Older Ollama versions accept only "json", not a schema.
                if exc.status not in (400, 500):
                    raise
                payload["format"] = "json"
                data = _post(url, payload, headers)
            msg = data.get("message") or {}
            return msg.get("content") or "", [], {
                "finish_reason": data.get("done_reason", "stop"),
                "prompt_tokens": data.get("prompt_eval_count"),
                "completion_tokens": data.get("eval_count")}
        if stream:
            for line in _post_lines(url, payload, headers):
                event = json.loads(line)
                if event.get("error"):
                    raise _HTTPFailure(500, str(event["error"]))
                piece = (event.get("message") or {}).get("content") or ""
                if piece:
                    emit(piece)
                if event.get("done"):
                    finish = event.get("done_reason", "stop")
                    usage = {"prompt_tokens": event.get("prompt_eval_count"),
                             "completion_tokens": event.get("eval_count")}
        else:
            data = _post(url, payload, headers)
            msg = data.get("message") or {}
            parts.append(msg.get("content") or "")
            calls = msg.get("tool_calls") or []
            finish = data.get("done_reason", "stop")
            usage = {"prompt_tokens": data.get("prompt_eval_count"),
                     "completion_tokens": data.get("eval_count")}
        return "".join(parts), calls, {"finish_reason": finish, **usage}

    payload = {"model": name, "messages": messages, "temperature": cfg.temperature, "stream": stream}
    if tools:
        payload["tools"], payload["tool_choice"] = tools, "auto"
    if json_schema is not None and base.startswith("https://api.openai.com"):
        payload["response_format"] = {"type": "json_object"}
    url = f"{base}/chat/completions"
    for attempt in (1, 2):
        try:
            if stream:
                for line in _post_lines(url, payload, headers):
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    usage = event.get("usage") or usage
                    for choice in event.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content") or ""
                        if piece:
                            emit(piece)
                        finish = choice.get("finish_reason") or finish
            else:
                data = _post(url, payload, headers)
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                parts.append(msg.get("content") or "")
                calls = msg.get("tool_calls") or []
                finish = choice.get("finish_reason") or finish
                usage = data.get("usage") or {}
            break
        except _HTTPFailure as exc:
            # Some newer models accept only their default temperature. Retry
            # once without it rather than fail; the trace records the model.
            if attempt == 1 and exc.status == 400 and "temperature" in exc.body.lower() and not parts:
                payload.pop("temperature", None)
                continue
            if attempt == 1 and exc.status == 400 and "response_format" in payload and not parts:
                payload.pop("response_format")
                continue
            raise
    return "".join(parts), calls, {"finish_reason": finish,
                                   "prompt_tokens": usage.get("prompt_tokens"),
                                   "completion_tokens": usage.get("completion_tokens")}


def _live_complete(messages: Messages, cfg: Config, on_token=None, model: str | None = None,
                   json_schema: dict | None = None) -> tuple[str, dict]:
    model = model or cfg.generation_model
    t0 = time.perf_counter()
    try:
        text, _, meta = _chat(model, messages, cfg, None, on_token=on_token, json_schema=json_schema)
    except _HTTPFailure as exc:
        raise ModelError(_hint(model, exc)) from exc
    return _clean(text), {**meta, "latency_s": round(time.perf_counter() - t0, 2), "model": model}


#: Embeddings live in memory for the length of one command and are never
#: written to disk. Within one command, a text is embedded once: building six
#: indexes over the same handbook in `architectures` asks the model once per
#: chunk, not six times. The next command asks the model again.
_MEMO: dict[str, dict[str, list[float]]] = {}


def _embed_batch(model: str, texts: list[str]) -> list[list[float]]:
    if model.startswith("anthropic/"):
        raise ModelError("Anthropic has no embeddings API. Set HELIX_EMBEDDING_MODEL to an Ollama, "
                         "OpenAI or Gemini embedding model.")
    kind, name, base, headers = _route(model)
    if kind == "ollama":
        return _post(f"{base}/api/embed", {"model": name, "input": texts}, headers)["embeddings"]
    data = _post(f"{base}/embeddings", {"model": name, "input": texts}, headers)["data"]
    return [d["embedding"] for d in sorted(data, key=lambda d: d.get("index", 0))]


def _live_embed(texts: list[str], cfg: Config) -> np.ndarray:
    model = cfg.embedding_model
    cache = _MEMO.setdefault(model, {})
    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    missing = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in cache]
    if missing:
        try:
            for i in range(0, len(missing), 64):
                batch = missing[i:i + 64]
                for (k, _), vec in zip(batch, _embed_batch(model, [t for _, t in batch])):
                    cache[k] = [float(x) for x in vec]
        except _HTTPFailure as exc:
            raise ModelError(_hint(model, exc)) from exc
    return np.array([cache[k] for k in keys], dtype=np.float32)


def chat_with_tools(messages: list[dict], tools: list[dict], cfg: Config) -> dict:
    """One turn of a tool-calling conversation, used by the agent.

    Messages and tools use the OpenAI format whatever the provider. Returns
    {"content": str, "tool_calls": [{"id", "name", "arguments": dict}],
    "latency_s": float}. Arguments arrive as a JSON string from OpenAI-style
    APIs and as an object from Ollama; both come back as a dict.
    """
    model = cfg.generation_model
    t0 = time.perf_counter()
    try:
        text, raw_calls, _ = _chat(model, messages, cfg, tools)
    except _HTTPFailure as exc:
        raise ModelError(_hint(model, exc)) from exc
    calls = []
    for i, tc in enumerate(raw_calls):
        fn = tc.get("function") or {}
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"_unparsed": args}
        calls.append({"id": tc.get("id") or f"call_{i}", "name": fn.get("name"), "arguments": args})
    return {"content": _clean(text), "tool_calls": calls, "latency_s": round(time.perf_counter() - t0, 2)}


# ---------------------------------------------------------------------------
# Offline path
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"[a-z0-9]+")


def _hash_embed(texts: list[str], dim: int = 256) -> np.ndarray:
    """A bag-of-words hashing embedding.

    It is not a good embedding. It is a *deterministic* one, which is what a
    teaching fixture needs: every participant sees the same retrieval ranking
    and therefore the same three seeded failures.
    """
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for tok in _TOKEN.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            out[i, h % dim] += 1.0
        norm = np.linalg.norm(out[i])
        if norm:
            out[i] /= norm
    return out


_CITE = re.compile(r"^\[([^\]]+)\]\s*$", re.M)
_UNIT_SPLIT = re.compile(r"(?:\|\s*\|)|(?<=[.!?])\s+")
_REFUSAL = (
    "I don't have enough information in the retrieved policy documents "
    "to answer that. Please check with HR."
)


_BREADCRUMB = re.compile(r"^[^\n|]{1,80} > [^\n|]{1,80}$", re.M)
_HEADING = re.compile(r"#{1,6}\s*")


def _facts(block: str) -> list[str]:
    """Candidate statements of fact in a chunk, in the order they appear.

    Handles both prose sentences and pipe-delimited table rows, because fixed
    chunking flattens newlines and a table row stops looking like a row.
    Heading markers and breadcrumbs are navigation, not facts, so they are
    stripped before a unit is considered.
    """
    block = _BREADCRUMB.sub("", block)
    out: list[str] = []
    for unit in _UNIT_SPLIT.split(block):
        unit = _HEADING.sub("", unit).strip().strip("|").strip()
        if not unit or not re.search(r"\d", unit):
            continue
        if set(unit) <= set("- |"):  # markdown table separator row
            continue
        out.append(re.sub(r"\s*\|\s*", ", ", unit).strip(" ,"))
    return out


def _stub_complete(messages: Messages) -> tuple[str, dict]:
    """A generator that answers strictly from the top-ranked retrieved chunk.

    It never invents a number. It reports the first statement of fact in the
    highest-ranked chunk that reached it, preferring one that shares wording
    with the question. That single rule reproduces every Day 1 failure
    honestly, and each failure is attributable to the pipeline rather than to
    the stub's imagination:

    * FP10: when fixed chunking put Ireland's figures and the Amsterdam
      heading in the same chunk, the first fact in that chunk is Ireland's,
      so the answer confidently reports the wrong country;
    * FP8 / FP9: when the superseded 2022 page outranks the 2024 one, the
      stub faithfully reports the stale global figure;
    * FP3: when truncation dropped the chunk holding the answer, there is no
      fact to report and the stub refuses;
    * FP11: when the entitlement filter is off, the stub happily reads a
      restricted compensation band, because nothing told it not to.
    """
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    body = user.split("### Context", 1)[-1].split("### Question", 1)[0]
    question = user.split("### Question", 1)[-1] if "### Question" in user else ""
    q_terms = set(_TOKEN.findall(question.lower()))

    # One block per retrieved chunk, highest-ranked first.
    blocks: list[tuple[str, str]] = []
    current_cite, buf = "", []
    for line in body.splitlines():
        m = _CITE.match(line.strip())
        if m:
            if buf:
                blocks.append((current_cite, "\n".join(buf)))
                buf = []
            current_cite = f"[{m.group(1)}]"
        else:
            buf.append(line)
    if buf:
        blocks.append((current_cite, "\n".join(buf)))

    for cite, block in blocks:
        facts = _facts(block)
        if not facts:
            continue
        scored = sorted(
            facts,
            key=lambda f: -len(q_terms & set(_TOKEN.findall(f.lower()))),
        )
        best = scored[0] if q_terms & set(_TOKEN.findall(scored[0].lower())) else facts[0]
        text = f"{best} {cite}".strip()
        break
    else:
        text = _REFUSAL

    return text, {
        "finish_reason": "stop",
        "prompt_tokens": sum(len(_TOKEN.findall(m["content"])) for m in messages),
        "completion_tokens": len(_TOKEN.findall(text)),
    }


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------
def complete(messages: Messages, cfg: Config, on_token=None, model: str | None = None,
             json_schema: dict | None = None) -> tuple[str, dict]:
    """Return (text, metadata). Metadata always carries finish_reason and token counts.

    `on_token`, if given, is called with each piece of text as the model
    streams it, so a script can print the answer as it is written.
    `model` overrides the generation model (the Day 3 judge uses this).
    `json_schema` asks the model for JSON in that shape (the toolkit judges
    use this); the caller still validates the reply.
    """
    if cfg.offline:
        text, meta = _stub_complete(messages)
        if on_token:
            on_token(text)
        return text, meta
    return _live_complete(messages, cfg, on_token=on_token, model=model, json_schema=json_schema)


def embed(texts: list[str], cfg: Config) -> np.ndarray:
    if cfg.offline or cfg.embedding_model == "hash":
        return _hash_embed(texts)
    return _live_embed(texts, cfg)


#: The one sentence the prompt asks a model to use when the context is not enough.
REFUSAL = _REFUSAL

_REFUSAL_PATTERNS = re.compile(
    r"don.t have enough information|do not have enough information|"
    r"not (?:enough|sufficient) information|cannot answer|can.t answer|unable to answer|"
    r"(?:extracts|context|documents|handbook) (?:do|does) not (?:contain|include|cover|say|mention)|"
    r"no information (?:about|on)",
    re.I,
)


def is_refusal(text: str) -> bool:
    """Did the answer decline? Live models phrase it many ways; the prompt asks
    for one fixed sentence, and this also catches the common variants."""
    return bool(_REFUSAL_PATTERNS.search(text or ""))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def count_tokens(text: str) -> int:
    """A word-ish token count, good enough for a context budget demonstration.

    Real token counting is model-specific. The point this project needs to
    make is that a budget exists and that things fall off the end of it.
    """
    return max(1, math.ceil(len(_TOKEN.findall(text)) * 1.3))
