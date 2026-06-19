"""3D field-config layout helper.

Thin re-export of ``data.to_NLLL`` so 3D analysis scripts can import a
short, local name without depending on the repo-root sys.path setup.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data import to_NLLL  # noqa: E402,F401
