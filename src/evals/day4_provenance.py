"""Day 4, Module 10: provenance, the fingerprint a score must carry.

A score is comparable with last week's only if you can say what produced
both. The config hash from Day 1 says part of it. This module writes down the
rest, and shows the gap Day 1 planted.

The gap. `Config.hash()` covers every Config field, including
`prompt_version`, a label. It does not read the prompt template, which lives
as text in helix/prompts.py with its own PROMPT_VERSION. The trace schema's
comment says the hash covers "prompt template". It covers the label. Edit the
template without changing the label and every trace keeps the same hash:
yesterday's traces and today's look comparable and are not. The fingerprint
below hashes the template text itself, so the edit shows.

Three more things the config hash cannot see, because they are not Config
fields: a change to what the index holds (Day 4's `current-only`), a scope
guard in front of the model, and the model weights behind a name. An Ollama
tag such as llama3.1:8b can be pulled again to new weights; Ollama's
/api/tags reports a digest of the weights, and the fingerprint records it. A
cloud model name can change behind the same name with no digest at all (Chen,
Zaharia and Zou 2023 measured the same API name answer a task 84 percent
right in March and 51 percent in June), so the fingerprint records the name
you asked for and the run date, and the anchor set re-measures the judge
(`python run.py production`).

Instruments are fingerprinted too, by their source code: a check edited
"to cut false alarms" is a new instrument, and its old error rates no
longer apply.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import urllib.request
from datetime import date
from pathlib import Path

from evals.day2_evaluators import check_leak, check_missed_refusal, check_stale
from evals.day3_generation import citation_support
from evals.day4_suite import GUARDS, Build
from helix import prompts
from helix.ingest import HANDBOOK

INSTRUMENTS = {
    "access": check_leak,
    "current": check_stale,
    "grounded": citation_support,
    "decline": check_missed_refusal,
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def template_sha(system: str | None = None, user: str | None = None) -> str:
    """A hash of the prompt template text, which the config hash never reads."""
    return sha((prompts.SYSTEM if system is None else system) + "\x00" + (prompts.USER if user is None else user))


def corpus_sha(handbook: Path = HANDBOOK) -> str:
    """A hash of every handbook page, name and bytes."""
    h = hashlib.sha256()
    for p in sorted(handbook.glob("*.md")):
        h.update(p.name.encode())
        # Line endings normalised, so a Windows checkout hashes the same pages the same way.
        h.update(p.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8"))
    return h.hexdigest()[:12]


def instrument_sha(fn) -> str:
    return sha(inspect.getsource(fn))


def ollama_digest(model: str) -> str:
    """The weights digest Ollama reports for an ollama/ model, or '' when unavailable."""
    if not model.startswith(("ollama/", "ollama_chat/")):
        return ""
    name = model.split("/", 1)[1]
    base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=3) as r:
            tags = json.loads(r.read().decode("utf-8")).get("models", [])
    except (OSError, ValueError):
        return ""
    want = {name, name if ":" in name else f"{name}:latest"}
    for m in tags:
        if m.get("name") in want or m.get("model") in want:
            return (m.get("digest") or "")[:12]
    return ""


def fingerprint(build: Build, suite_sha: str, run_date: str | None = None) -> dict[str, str]:
    """Everything that can change a score, for one build on one suite."""
    cfg = build.cfg
    fp = {
        "config hash": cfg.hash(),
        "prompt version (Config)": cfg.prompt_version,
        "prompt version (prompts.py)": prompts.PROMPT_VERSION,
        "prompt template sha": template_sha(),
        "corpus sha": corpus_sha(),
        "suite sha": suite_sha,
        "index transforms": ", ".join(build.transforms) or "none",
        "scope guard": (f"{build.guard} [{sha(json.dumps(GUARDS[build.guard], sort_keys=True))}]"
                        if build.guard else "none"),
    }
    if cfg.offline:
        fp["answer model"] = "offline stub"
        fp["embedding model"] = "hash"
        fp["judge"] = "none (code instruments only)"
    else:
        judge = cfg.judge_model or cfg.generation_model
        for key, model in (("answer model", cfg.generation_model), ("embedding model", cfg.embedding_model),
                           ("judge", judge)):
            digest = ollama_digest(model)
            fp[key] = f"{model} [{digest}]" if digest else f"{model} (no digest: a name, not a version)"
    for name, fn in INSTRUMENTS.items():
        fp[f"instrument {name}"] = instrument_sha(fn)
    fp["run date"] = run_date or date.today().isoformat()
    return fp


def diff(a: dict[str, str], b: dict[str, str], ignore: tuple[str, ...] = ("run date",)) -> list[tuple[str, str, str]]:
    return [(k, a.get(k, ""), b.get(k, "")) for k in dict.fromkeys([*a, *b])
            if k not in ignore and a.get(k) != b.get(k)]


def planted_gap(cfg) -> dict[str, str]:
    """The Day 1 gap, shown without touching a file.

    An edited copy of the template, in memory: one instruction changed. The
    config hash is computed from Config alone, so it cannot change.
    """
    edited = prompts.SYSTEM.replace("three sentences or fewer", "five sentences or fewer")
    assert edited != prompts.SYSTEM, "the template no longer holds the sentence this demonstration edits"
    return {
        "edit": 'Instructions: "Keep the answer to three sentences or fewer" -> "five sentences or fewer"',
        "config hash before": cfg.hash(),
        "config hash after": cfg.hash(),
        "prompt version before": prompts.PROMPT_VERSION,
        "prompt version after": prompts.PROMPT_VERSION,
        "template sha before": template_sha(),
        "template sha after": template_sha(system=edited),
    }
