"""pytest configuration for the Smart Traffic India test suite.

The suite is **head-less**: it never opens a window and never creates an OpenGL context
(``STI_TEST_MODE=1``), because the grading machine may not have a display.  Anything that
needs a real GL context is covered by ``tools/render_smoke.py`` instead, which runs under
a virtual X server.

Two environment details matter:

* ``STI_TEST_MODE``           - tells modules that no window will be created
* ``STI_DATA_DIR``            - redirects the save file to a temporary directory so the
  tests never touch the player's real profile
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ["STI_TEST_MODE"] = "1"
os.environ.setdefault("STI_DATA_DIR", os.path.join(tempfile.gettempdir(), "sti_tests_data"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def pytest_configure(config) -> None:                      # pragma: no cover - pytest hook
    config.addinivalue_line("markers", "slow: heavier simulation tests")
