#!/usr/bin/env python3
"""Smart Traffic India - Ultimate 3D Driving Simulator.

Run the game::

    python main.py                    # normal start, main menu
    python main.py --free             # straight into Free Drive
    python main.py --mission m03      # straight into a mission
    python main.py --list-missions    # print every mission id
    python main.py --quality low      # weaker laptop
    python main.py --frames 300 --screenshot shot.png   # head-less smoke run

The launcher only adds ``src/`` to ``sys.path`` and hands over to
:func:`smart_traffic_india.app.main`, so the game can also be started with
``python -m smart_traffic_india`` once the package is installed.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
if os.path.isdir(SRC) and SRC not in sys.path:
    sys.path.insert(0, SRC)


def _missing_dependency_help(packages) -> int:
    print("Smart Traffic India needs these packages: " + ", ".join(packages))
    print("Install them with:\n    python -m pip install -r requirements.txt\n"
          "or, if you prefer a one-liner:\n    python -m pip install pygame PyOpenGL numpy")
    return 2


def main() -> int:
    try:
        import pygame  # noqa: F401
        import OpenGL  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        missing = []
        for module, package in (("pygame", "pygame"), ("OpenGL", "PyOpenGL"),
                                ("numpy", "numpy")):
            try:
                __import__(module)
            except ImportError:
                missing.append(package)
        return _missing_dependency_help(missing)

    from smart_traffic_india.app import main as app_main
    return app_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
