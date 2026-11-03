"""Smart Traffic India - Ultimate 3D Driving Simulator.

A 3D Indian driving game built with **Pygame** (window, input, 2D HUD) and **PyOpenGL**
(real-time 3D rendering with hand written GLSL shaders), written as a Computer Graphics
major project.

The package is intentionally flat: every module is a self-contained subsystem and the
public names below are the ones the game, the tools and the tests use.

Quick start
-----------
>>> from smart_traffic_india import Game, SaveManager        # doctest: +SKIP
>>> SaveManager().money                                      # doctest: +SKIP
25000

Head-less (no window) usage -- this is what the tests do::

    from smart_traffic_india import World, SaveManager, Settings

    world = World(Settings(), atlas, models)   # atlas/models come from signage/vehicle_models
    world.update(1 / 60, player)
"""

from __future__ import annotations

from . import settings as settings_module
from .settings import APP_TITLE, APP_VERSION, Settings, default_profile

# These imports are cheap and do not touch OpenGL, so importing the package is safe in a
# head-less test run.
from .camera import CLOSE, CHASE, HOOD, Camera
from .collision import CollisionSystem
from .environment import Environment
from .garage import Garage
from .hud import HUD, HUDSnapshot
from .input_manager import InputManager
from .menus import GarageScreen, MenuScreen
from .missions import (Mission, MissionResult, MissionRuntime, build_career_missions,
                       build_parking_challenges, build_school_lessons)
from .pedestrians import PedestrianManager
from .player import Player
from .road import RoadNetwork
from .save_manager import SaveManager
from .traffic_ai import TrafficManager
from .traffic_signals import SignalController
from .vehicle import PLAYER_SPECS, Vehicle, VehicleSpec
from .vehicle_models import ModelLibrary
from .world import World

__all__ = [
    "APP_TITLE", "APP_VERSION", "Settings", "default_profile", "settings_module",
    "Camera", "CHASE", "CLOSE", "HOOD",
    "CollisionSystem", "Environment", "Garage", "HUD", "HUDSnapshot", "InputManager",
    "GarageScreen", "MenuScreen", "Mission", "MissionResult", "MissionRuntime",
    "build_career_missions", "build_parking_challenges", "build_school_lessons",
    "PedestrianManager", "Player", "RoadNetwork", "SaveManager", "TrafficManager",
    "SignalController", "Vehicle", "VehicleSpec", "PLAYER_SPECS", "ModelLibrary", "World",
]

__version__ = APP_VERSION
__all__ += ["Game", "Renderer", "main"]


def __getattr__(name: str):
    """Lazy access to the heavy modules (they create OpenGL objects on construction)."""
    if name == "Game":
        from .game import Game
        return Game
    if name == "Renderer":
        from .renderer import Renderer
        return Renderer
    if name == "main":
        from .app import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
