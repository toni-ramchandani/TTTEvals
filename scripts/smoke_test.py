"""Pre-session check: is your model reachable, and does the whole path work?

Send this to participants 48 hours before Day 1 with one instruction:
run `python run.py check` and paste the last line into the chat.

It checks, in order, and stops at the first problem with a line saying what
to do about it:

  1. the model server or API key (Ollama running and the models pulled, or
     the key for your cloud provider present)
  2. one live answer, one live embedding, one tool call, and one live call
     to the judge when it is a different model (local answers with a cloud
     judge, or the other way round)
  3. one question through the full Helix pipeline, streamed as it is written

Run:  python run.py check
      python run.py check --model openai/gpt-5.4-mini --embed openai/text-embedding-3-small
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evals.day1_triage import triage  # noqa: E402
from helix import DEFAULT, answer, build_index  # noqa: E402
from helix.console import Stream, banner  # noqa: E402
from helix.llm import PROVIDERS, ModelError, chat_with_tools, complete, embed  # noqa: E402


def fail(msg: str) -> int:
    print(f"\nCHECK FAILED  {msg}")
    return 1


def key_for(model: str) -> str | None:
    if model.startswith(("ollama/", "ollama_chat/")) or model == "hash":
        return None
    prefix = model.split("/", 1)[0] if "/" in model else "openai"  # bare names are OpenAI's
    return PROVIDERS[prefix][1] if prefix in PROVIDERS else None


def check_ollama(models: list[str]) -> str | None:
    """Is the Ollama server up, and has every model we need been pulled?"""
    base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=5) as r:
            have = {m["name"] for m in json.load(r).get("models", [])}
    except Exception:  # noqa: BLE001
        return (f"Ollama is not answering at {base}. Install it from https://ollama.com, "
                "start the app (or run `ollama serve`), then run this check again.")
    print(f"ollama   running at {base}, {len(have)} models pulled")
    for m in models:
        name = m.split("/", 1)[1]
        tagged = name if ":" in name else f"{name}:latest"
        if name not in have and tagged not in have:
            return f"Ollama does not have {name} yet. Run:  ollama pull {name}"
        print(f"         {name} is pulled")
    return None


def main() -> int:
    cfg = DEFAULT
    print(f"python   {sys.version.split()[0]}")
    banner(cfg)

    if cfg.offline:
        print("\nOffline reference run: skipping the model checks.")
    else:
        # 1. server or keys -------------------------------------------------------
        models = {cfg.generation_model, cfg.embedding_model, cfg.judge_model or cfg.generation_model}
        local = sorted(m for m in models if m.startswith(("ollama/", "ollama_chat/")))
        if local:
            problem = check_ollama(local)
            if problem:
                return fail(problem)
        for m in sorted(models):
            var = key_for(m)
            if var and not os.getenv(var):
                return fail(f"{m} needs {var}. Add it to .env (copy .env.example to .env first).")
            if var:
                print(f"key      {var} is set (for {m})")

        # 2. one call of each kind ----------------------------------------------
        try:
            t0 = time.perf_counter()
            text, _ = complete([{"role": "user", "content": "Reply with the single word: ready"}], cfg)
            print(f"answer   {text[:40]!r} in {time.perf_counter() - t0:.1f}s")
            t0 = time.perf_counter()
            v = embed(["parental leave in Amsterdam"], cfg)
            print(f"embed    {v.shape[1]} dimensions in {time.perf_counter() - t0:.1f}s")
            tool = [{"type": "function", "function": {
                "name": "get_weather", "description": "Weather for a city.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                               "required": ["city"]}}}]
            turn = chat_with_tools([{"role": "user", "content": "What is the weather in Dublin? Use the tool."}],
                                   tool, cfg)
            if turn["tool_calls"]:
                print(f"tools    called {turn['tool_calls'][0]['name']}({turn['tool_calls'][0]['arguments']})")
            else:
                print("tools    WARNING: the model answered without calling the tool. The agent")
                print("         on Day 3 needs tool calling; try ollama/qwen3:8b or a cloud model.")
            if cfg.judge_model and cfg.judge_model != cfg.generation_model:
                # A separate judge (often a cloud model beside local answers): call it live too.
                t0 = time.perf_counter()
                text, _ = complete([{"role": "user", "content": "Reply with the single word: ready"}], cfg,
                                   model=cfg.judge_model)
                print(f"judge    {text[:40]!r} from {cfg.judge_model} in {time.perf_counter() - t0:.1f}s")
        except ModelError as exc:
            return fail(str(exc))

    # 3. the full path ----------------------------------------------------------
    index = build_index(cfg)
    assert len(index) > 10, "index looks empty; is the handbook corpus present?"
    print(f"index    {len(index)} chunks built")
    q = "How much parental leave am I entitled to at the Amsterdam office?"
    print(f"question {q}")
    out = Stream("answer   ")
    try:
        t = answer(q, index, cfg, on_token=out)
    except ModelError as exc:
        return fail(str(exc))
    out.done()
    assert t.spans, "no spans recorded; tracing is not wired up"
    assert t.answer, "no answer produced"
    print(f"spans    {', '.join(s.name for s in t.spans)}")
    print(f"cited    {t.citations or 'nothing'}")
    print(f"triage   {[f.code for f in triage(t)] or 'no failure detected'}")
    print(f"\nCHECK PASSED  {cfg.describe_models()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
