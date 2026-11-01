"""Make the workshop's src/ importable when DeepEval's runner or pytest loads the gate."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from toolkits.env import prepare  # noqa: E402

prepare()
