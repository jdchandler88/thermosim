"""Regenerates the user-guide screenshots with the real runner (slow, so opt-in).

    THERMOSIM_SCREENSHOTS=docs/images QT_QPA_PLATFORM=offscreen uv run pytest \
        tests/gui_integration/test_screenshots.py
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import pytest

OUT = os.environ.get("THERMOSIM_SCREENSHOTS")
SCRIPT = Path(__file__).resolve().parents[2] / "packaging" / "make_screenshots.py"

EXPECTED = [
    "setup.png",
    "setup-heat-source.png",
    "main-window-fast-run.png",
    "measured-comparison.png",
    "realtime-run.png",
    "geometry-error.png",
]


@pytest.mark.skipif(not OUT, reason="set THERMOSIM_SCREENSHOTS=<dir> to regenerate screenshots")
def test_generate_screenshots() -> None:
    assert OUT is not None
    module = runpy.run_path(str(SCRIPT), run_name="make_screenshots")
    module["main"](Path(OUT))
    for name in EXPECTED:
        assert (Path(OUT) / name).stat().st_size > 10_000
