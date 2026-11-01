"""Small helpers for printing a live run as it happens.

Nothing here changes what the systems do. It only makes a live run visible:
which model is answering, the answer appearing word by word, and each agent
step as the model takes it.
"""

from __future__ import annotations

import sys
import time

from .config import Config


def banner(cfg: Config) -> None:
    """Say which models this run uses, so nobody misreads a reference run as live."""
    if cfg.offline:
        print("models   offline reference run: deterministic stub, hash embeddings")
        print("         (the fixed numbers on the slides; drop --offline to use your models)")
    else:
        print(f"models   answers    {cfg.generation_model}")
        print(f"         embeddings {cfg.embedding_model}")
        print(f"         judge      {cfg.judge_model or cfg.generation_model}")



def traces_note(traces, cfg: Config, made_by: str = "python run.py day1") -> None:
    """Say which answers this command judges, and whether they are current.

    Every model call a command makes is live. The answers it judges are the
    ones `made_by` gave, because the human labels belong to those exact
    answers. When they came from other models or settings, say so plainly.
    """
    hashes = sorted({t.config_hash for t in traces})
    if hashes == [cfg.hash()]:
        print(f"answers  the {len(traces)} answers from your last `{made_by}`, same models and settings as now")
    else:
        print(f"answers  the {len(traces)} answers from your last `{made_by}`, made with other models or")
        print(f"         settings (config {', '.join(hashes)}; now {cfg.hash()}). Run `{made_by}`")
        print("         to answer again live with the models set now.")

class Stream:
    """Print a model's answer as it streams, on one indented line.

    Pass an instance as `on_token`. Newlines in the answer become spaces so
    a table of answers stays readable (keep_newlines=True keeps them, for
    longer output). Call `.done()` when the answer ends.
    """

    def __init__(self, prefix: str = "    -> ", keep_newlines: bool = False) -> None:
        self.prefix = prefix
        self.keep_newlines = keep_newlines
        self.started = False
        self.t0 = time.perf_counter()

    def __call__(self, piece: str) -> None:
        if not self.started:
            sys.stdout.write(self.prefix)
            self.started = True
        if self.keep_newlines:
            piece = piece.replace("\n", "\n" + " " * len(self.prefix))
        else:
            piece = piece.replace("\n", " ")
        sys.stdout.write(piece)
        sys.stdout.flush()

    def done(self, note: str = "") -> None:
        if not self.started:
            sys.stdout.write(self.prefix)
        took = time.perf_counter() - self.t0
        bits = ([f"{took:.1f}s"] if took >= 0.05 else []) + ([note] if note else [])
        sys.stdout.write(f"   ({', '.join(bits)})\n" if bits else "\n")
        sys.stdout.flush()


def step_printer(indent: str = "      "):
    """An `on_step` callback for the agent: one line per tool call, as it starts."""

    def on_step(n: int, tool: str, args: dict) -> None:
        shown = ", ".join(f"{k}={str(v)[:50]!r}" for k, v in args.items()) if args else ""
        print(f"{indent}step {n}  {tool}({shown})", flush=True)

    return on_step


def progress(label: str, i: int, n: int, note: str = "") -> None:
    """One line per finished item in a long live run."""
    print(f"  [{i:>2}/{n}] {label}{'  ' + note if note else ''}", flush=True)
