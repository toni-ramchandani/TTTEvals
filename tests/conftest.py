"""The tests pin the reference run, so they never call a model.

Setting this before any workshop module is imported makes every Config built
during the tests an offline one, whatever .env says.
"""

import os

os.environ["HELIX_OFFLINE"] = "1"

# Before any test imports DeepEval or Ragas: telemetry off, nothing written
# to disk, no cloud upload. The same settings every workshop command uses.
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toolkits.env import prepare  # noqa: E402

prepare()
