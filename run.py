"""One entry point for the whole workshop.

    python run.py                 list the commands
    python run.py check           check your model connection end to end
    python run.py day1            Day 1 hands-on: six questions, six traces, triage
    python run.py day2            Day 2 error analysis: open coding, axial coding, counts
    python run.py agreement       Day 2: two annotators, Cohen's kappa, the golden set
    python run.py evaluators      Day 2: an evaluator per failure mode, checked against people
    python run.py day3            retrieval metrics, grounded x correct, judge validation
    python run.py metrics         Day 3: our metrics beside Ragas and DeepEval, same traces
    python run.py architectures   vanilla vs agentic vs GraphRAG, with ablations
    python run.py day4            paired before/after comparison and the release gate
    python run.py ci              Day 4: the release gate as tests, run by DeepEval
    python run.py toolkits        check the DeepEval and Ragas environment
    python run.py ask "question"  ask the three systems one question and read the traces
    python run.py all             every day, end to end
    python run.py test            the test suite (needs: pip install pytest)
    python run.py clean           delete generated traces (keeps the sheets people wrote)

Every command runs live against the models named in .env: a model on your
own laptop through Ollama, or a cloud model through an API key. Options that
work with any command:

    --model NAME     answer with this model, e.g. ollama/qwen3:8b or openai/gpt-5.4-mini
    --embed NAME     embed with this model, e.g. ollama/nomic-embed-text
    --judge NAME     judge with this model (default: the answer model)
    --offline        the reference run: a deterministic stub, no model at all.
                     It reproduces the exact numbers on the slides.

In VS Code, press F5 and pick a command from the list instead of typing it.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

COMMANDS = {
    "check": ("scripts.smoke_test", "check your model connection end to end"),
    "day1": ("scripts.day1", "Day 1 hands-on: six questions, six traces, triage"),
    "day2": ("scripts.day2", "Day 2 error analysis: open coding, axial coding, counts"),
    "agreement": ("scripts.agreement", "Day 2: two annotators, Cohen's kappa, the golden set"),
    "evaluators": ("scripts.evaluators", "Day 2: an evaluator per failure mode, checked against people"),
    "day3": ("scripts.day3", "retrieval metrics, grounded x correct, judge validation"),
    "metrics": ("scripts.metrics", "Day 3: our metrics beside Ragas and DeepEval, same traces"),
    "architectures": ("scripts.architectures", "vanilla vs agentic vs GraphRAG, with ablations"),
    "day4": ("scripts.day4", "paired before/after comparison and the release gate"),
    "ci": ("scripts.ci", "Day 4: the release gate as tests, run by DeepEval"),
    "toolkits": ("scripts.toolkits", "check the DeepEval and Ragas environment"),
    "ask": ("scripts.ask", "ask the three systems one question and read the traces"),
    "verify": ("scripts.verify_fixture", "assert the Day 1 seeded failures still reproduce"),
}
ALL = ["day1", "day2", "agreement", "evaluators", "day3", "architectures", "metrics", "day4"]


#: Options any command accepts. Each one becomes an environment variable, set
#: before any workshop module is imported, so every script sees the same models.
GLOBAL_FLAGS = {
    "--model": "HELIX_GENERATION_MODEL",
    "--embed": "HELIX_EMBEDDING_MODEL",
    "--judge": "HELIX_JUDGE_MODEL",
}


def take_global_flags(argv: list[str]) -> list[str]:
    rest, i = [], 0
    while i < len(argv):
        arg = argv[i]
        name, eq, value = arg.partition("=")
        if arg == "--offline":
            os.environ["HELIX_OFFLINE"] = "1"
        elif name in GLOBAL_FLAGS:
            if not eq:
                i += 1
                value = argv[i] if i < len(argv) else ""
            os.environ[GLOBAL_FLAGS[name]] = value
            os.environ["HELIX_OFFLINE"] = "0"
        else:
            rest.append(arg)
        i += 1
    return rest


def run_module(module: str, argv: list[str]) -> int:
    from helix.llm import ModelError
    from toolkits.env import ToolkitMissing

    saved = sys.argv
    sys.argv = [module, *argv]
    try:
        runpy.run_module(module, run_name="__main__")
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0
    except ModelError as exc:
        print(f"\n\nMODEL ERROR  {exc}")
        print("Check your setup with:  python run.py check")
        return 2
    except ToolkitMissing as exc:
        print(f"\n{exc}")
        return 2
    except KeyboardInterrupt:
        print("\nstopped")
        return 130
    finally:
        sys.argv = saved
    return 0


def usage() -> int:
    print(__doc__)
    return 0


def main(argv: list[str]) -> int:
    # Before anything can import DeepEval or Ragas: telemetry off, nothing
    # written to disk, no cloud upload (see src/toolkits/env.py).
    from toolkits.env import prepare

    for note in prepare():
        print(f"note  {note}")
    if not argv or argv[0] in ("-h", "--help", "help"):
        return usage()
    cmd, rest = argv[0], take_global_flags(argv[1:])
    if cmd in ("test", "verify"):
        # The tests and the fixture check pin the reference run. They never call a model.
        os.environ["HELIX_OFFLINE"] = "1"

    if cmd in COMMANDS:
        return run_module(COMMANDS[cmd][0], rest)

    if cmd == "all":
        for name in ALL:
            print(f"\n{'#' * 78}\n# python run.py {name}\n{'#' * 78}")
            code = run_module(COMMANDS[name][0], [])
            if code:
                return code
        return 0

    if cmd == "test":
        try:
            import pytest
        except ImportError:
            print("pytest is not installed. Run:  pip install pytest")
            return 1
        return pytest.main(["-q", str(ROOT / "tests"), *rest])

    if cmd == "clean":
        # Traces and derived tables are made again by the next run. Sheets hold
        # what people wrote (triage, verdicts, notes, labels, the golden set)
        # and are kept unless you ask: `python run.py clean --all`.
        human = ("day1_triage_worksheet.csv", "day2_sheet*.csv", "day2_golden*.csv")
        patterns = ["data/traces/*.jsonl", "data/annotations/*.csv", "data/queries/day2_synthetic*.json"]
        removed, kept = 0, []
        for pattern in patterns:
            for p in ROOT.glob(pattern):
                if "--all" not in rest and any(p.match(h) for h in human):
                    kept.append(p.name)
                    continue
                p.unlink()
                removed += 1
        print(f"removed {removed} generated files")
        if kept:
            print(f"kept what people wrote: {', '.join(sorted(kept))} (--all removes them too)")
        return 0

    print(f"unknown command: {cmd}\n")
    return usage() or 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
