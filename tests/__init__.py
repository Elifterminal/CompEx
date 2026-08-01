"""Test package. Puts src/ on the path so the suite runs without installing."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EQUATIONS = ROOT / "equations"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
