"""Make the toolkits safe to import, and say what is installed.

Call `prepare()` before importing deepeval or ragas. It is idempotent.

What it switches off, and why:

    DEEPEVAL_TELEMETRY_OPT_OUT=1   DeepEval sends usage events to PostHog by default
    DEEPEVAL_DISABLE_DOTENV=1      DeepEval reads .env files on import; ours holds API keys
    DEEPEVAL_EVAL_MODE=llm         judges run on your model, never a hosted one
    DEEPEVAL_FILE_SYSTEM=READ_ONLY DeepEval's evaluate() saves each test run and a
                                   result cache (.deepeval/) by default; read-only
                                   mode saves nothing, so no result is ever replayed
    RAGAS_DO_NOT_TRACK=true        Ragas posts usage events by default
    CONFIDENT_API_KEY (removed)    with this key DeepEval uploads results to Confident AI

Nothing is stored: every score is computed live, on the models you name,
and printed. A test (tests/test_toolkits.py) runs both toolkits with the
home folder and the working folder isolated and fails if any file appears.

A value you set yourself in the environment wins for the first four. The
Confident AI key is always removed for the workshop: the traces hold an
employee's questions, and they stay on this machine.
"""

from __future__ import annotations

import importlib.metadata as md
import importlib.util
import os

PINNED = {"deepeval": "4.2.8", "ragas": "0.4.3"}

SAFE_DEFAULTS = {
    "DEEPEVAL_TELEMETRY_OPT_OUT": "1",
    "DEEPEVAL_DISABLE_DOTENV": "1",
    "DEEPEVAL_EVAL_MODE": "llm",
    "DEEPEVAL_FILE_SYSTEM": "READ_ONLY",
    "RAGAS_DO_NOT_TRACK": "true",
}

INSTALL_HINT = (
    "The toolkits are optional and install into their own environment:\n"
    "  py -3.12 -m venv .venv-toolkits          (macOS/Linux: python3.12 -m venv .venv-toolkits)\n"
    "  .venv-toolkits\\Scripts\\activate          (macOS/Linux: source .venv-toolkits/bin/activate)\n"
    "  pip install -r requirements.txt -r requirements-toolkits.txt -c constraints-toolkits.txt\n"
    "  pip install pip-audit && pip-audit\n"
    "Then run this command again from that environment."
)

#: LiteLLM releases that carried malware in 2026 (PYSEC-2026-2 and the
#: 21 October .pth credential stealer). Neither toolkit needs LiteLLM.
COMPROMISED_LITELLM = {"1.82.7", "1.82.8", "1.107.0"}

_prepared = False


class ToolkitMissing(RuntimeError):
    """A toolkit is not installed in this environment."""


def prepare() -> list[str]:
    """Set the safe defaults. Returns notes worth printing once."""
    global _prepared
    notes: list[str] = []
    for key, value in SAFE_DEFAULTS.items():
        os.environ.setdefault(key, value)
    if os.environ.pop("CONFIDENT_API_KEY", None):
        notes.append("CONFIDENT_API_KEY was set; removed for this run so no results leave this machine.")
    _prepared = True
    return notes


def version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def require(name: str) -> None:
    """Raise ToolkitMissing with the install steps if `name` is absent."""
    if not installed(name):
        raise ToolkitMissing(f"{name} is not installed in this environment.\n{INSTALL_HINT}")
    if not _prepared:
        prepare()


def status() -> list[tuple[str, str, str]]:
    """(item, state, note) rows for `python run.py toolkits`."""
    rows: list[tuple[str, str, str]] = []
    for dist, pin in PINNED.items():
        v = version(dist) or ("source" if installed(dist) else None)
        if v is None:
            rows.append((dist, "missing", f"pinned {pin}"))
        elif v == "source":
            rows.append((dist, v, f"imported from a source tree, not installed; pinned {pin}"))
        elif v != pin:
            rows.append((dist, v, f"pinned {pin}: reinstall with the constraints file"))
        else:
            rows.append((dist, v, "pinned version"))
    lite = version("litellm") or ("present" if installed("litellm") else None)
    if lite:
        bad = " COMPROMISED RELEASE: remove this environment" if lite in COMPROMISED_LITELLM else ""
        rows.append(("litellm", lite, "installed, but nothing needs it; uninstall it" + bad))
    else:
        rows.append(("litellm", "absent", "as intended"))
    for key in SAFE_DEFAULTS:
        rows.append((key, os.environ.get(key, "(unset)"), "set by prepare()"))
    rows.append(("CONFIDENT_API_KEY", "set" if os.environ.get("CONFIDENT_API_KEY") else "absent",
                 "results stay on this machine"))
    return rows
