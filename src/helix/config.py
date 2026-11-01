"""Every knob that changes answer quality, in one place, with a hash.

The hash is what makes two traces comparable. Change the chunk size and the
hash changes, so a Day 4 regression report can say *which* configuration
produced which numbers instead of guessing.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict, field

from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from .env into the environment.

    Twelve lines instead of a dependency. Variables already set in the real
    environment (or by run.py's --model and similar options) win.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


_load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    # --- ingestion ---------------------------------------------------------
    chunk_strategy: str = "fixed"          # "fixed" | "structural"
    chunk_tokens: int = 110
    chunk_overlap: int = 0
    prepend_heading: bool = False          # the FP10 switch, see README

    # --- retrieval ---------------------------------------------------------
    retriever: str = "hybrid"              # "bm25" | "dense" | "hybrid"
    top_k: int = 5
    rrf_k: int = 60                        # reciprocal rank fusion constant
    rerank_cutoff: int = 3

    # --- context assembly --------------------------------------------------
    context_token_budget: int = 340        # fits 2 of the 3 reranked chunks

    # --- generation --------------------------------------------------------
    # "provider/model". Ollama on your laptop: "ollama/llama3.1:8b".
    # A cloud model: "openai/gpt-5.4-mini", "gemini/gemini-3.6-flash", "anthropic/claude-haiku-5-5".
    generation_model: str = field(
        default_factory=lambda: os.getenv("HELIX_GENERATION_MODEL", "ollama/llama3.1:8b")
    )
    # "hash" = the built-in deterministic embedding, no model needed.
    embedding_model: str = field(
        default_factory=lambda: os.getenv("HELIX_EMBEDDING_MODEL", "ollama/nomic-embed-text")
    )
    temperature: float = 0.0               # reproducibility, not creativity
    prompt_version: str = "v2"

    # --- runtime -----------------------------------------------------------
    #: Live by default. HELIX_OFFLINE=1 (or `python run.py ... --offline`) swaps
    #: in the deterministic stub: no model, no network, the reference numbers
    #: quoted on the slides.
    offline: bool = field(default_factory=lambda: _env_bool("HELIX_OFFLINE", False))
    #: The model that grades answers on Day 3. Empty = the generation model.
    judge_model: str = field(default_factory=lambda: os.getenv("HELIX_JUDGE_MODEL", ""))

    # --- access control ----------------------------------------------------
    enforce_entitlements: bool = False     # the FP11 switch, see README

    def hash(self) -> str:
        """Stable short hash over everything that can change an answer.

        `offline` and `judge_model` are excluded: they change who produces or
        grades the text, not how the system is configured.
        """
        payload = {k: v for k, v in asdict(self).items() if k not in ("offline", "judge_model")}
        if self.offline:
            # The stub ignores the model settings, so the reference run hashes
            # the same on every laptop whatever .env says.
            payload["generation_model"], payload["embedding_model"] = "offline-stub", "hash"
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def describe(self) -> str:
        return (
            f"{self.chunk_strategy}/{self.chunk_tokens}tok "
            f"heading={'on' if self.prepend_heading else 'off'} "
            f"{self.retriever} k={self.top_k} "
            f"budget={self.context_token_budget} "
            f"entitlements={'on' if self.enforce_entitlements else 'off'} "
            f"[{self.hash()}]"
        )

    def describe_models(self) -> str:
        if self.offline:
            return "offline reference run (deterministic stub, no model)"
        judge = self.judge_model or self.generation_model
        return f"answers: {self.generation_model} | embeddings: {self.embedding_model} | judge: {judge}"


#: The configuration Day 1 ships with. It is deliberately imperfect: three of
#: its settings each plant one failure that participants localise in H1.
DEFAULT = Config()

#: The same system with the three Day 1 defects repaired. Used on Day 4 as the
#: "after" arm of a before/after comparison.
REPAIRED = Config(
    chunk_strategy="structural",
    prepend_heading=True,            # fixes FP10 (severed context)
    context_token_budget=900,        # fixes FP3 (retrieved then truncated)
    enforce_entitlements=True,       # fixes FP11 (entitlement leakage)
)
