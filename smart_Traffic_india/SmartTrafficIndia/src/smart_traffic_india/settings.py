"""Global configuration, tunable constants and runtime settings for Smart Traffic India.

Everything that a designer or a demonstrator might want to tweak lives here so that
gameplay, rendering and physics modules never hard-code magic numbers.

The module is pure Python (no pygame / OpenGL imports) so that it can be imported by
head-less unit tests.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Tuple

APP_NAME = "Smart Traffic India"
APP_TITLE = "Smart Traffic India: Ultimate 3D Driving Simulator"
VERSION = "1.0.0"
APP_VERSION = VERSION               # convenience alias used by app.py / __init__.py
AUTHOR = "Computer Graphics Major Project"
APP_TAGLINE = "Indian city traffic, real road rules, 3D driving"
WEATHER_PRESETS = ("clear", "cloudy", "rain", "random")

# --------------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------------
FIXED_DT = 1.0 / 60.0          # deterministic simulation step (seconds)
MAX_STEPS_PER_FRAME = 5        # avoid the "spiral of death" after a long stall
MAX_FRAME_TIME = 0.25          # clamp huge dt (window drag, alt-tab) to 0.25 s

# --------------------------------------------------------------------------------------
# Graphics quality presets.  Keys are consumed by Renderer / Game / TrafficManager.
# --------------------------------------------------------------------------------------
QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "low": {
        "shadows": False, "rain_particles": 0, "far_plane": 260.0,
        "traffic_scale": 0.45, "pedestrian_scale": 0.4, "fog_scale": 1.25,
        "glow_scale": 0.4, "max_sign_textures": 4, "multisample": 0,
    },
    "medium": {
        "shadows": True, "rain_particles": 420, "far_plane": 340.0,
        "traffic_scale": 0.7, "pedestrian_scale": 0.7, "fog_scale": 1.1,
        "glow_scale": 0.7, "max_sign_textures": 8, "multisample": 0,
    },
    "high": {
        "shadows": True, "rain_particles": 900, "far_plane": 420.0,
        "traffic_scale": 1.0, "pedestrian_scale": 1.0, "fog_scale": 1.0,
        "glow_scale": 1.0, "max_sign_textures": 14, "multisample": 2,
    },
    "ultra": {
        "shadows": True, "rain_particles": 1600, "far_plane": 520.0,
        "traffic_scale": 1.25, "pedestrian_scale": 1.25, "fog_scale": 0.85,
        "glow_scale": 1.2, "max_sign_textures": 20, "multisample": 4,
    },
}
QUALITY_ORDER = ["low", "medium", "high", "ultra"]

# --------------------------------------------------------------------------------------
# Traffic density presets (multiplier applied on top of the quality traffic scale)
# --------------------------------------------------------------------------------------
TRAFFIC_LEVELS: Dict[str, Dict[str, float]] = {
    "light":  {"density": 0.45, "speed": 1.10, "aggression": 0.7, "max_vehicles": 18},
    "medium": {"density": 0.75, "speed": 1.00, "aggression": 0.9, "max_vehicles": 30},
    "heavy":  {"density": 1.10, "speed": 0.90, "aggression": 1.05, "max_vehicles": 42},
    "busy":   {"density": 1.45, "speed": 0.80, "aggression": 1.15, "max_vehicles": 54},
}
TRAFFIC_LEVEL_ORDER = ["light", "medium", "heavy", "busy"]

# --------------------------------------------------------------------------------------
# Physics / vehicle constants
# --------------------------------------------------------------------------------------
GRAVITY = 9.81
AIR_DRAG = 0.42          # quadratic drag coefficient (already mass-normalised)
ROLLING_RESISTANCE = 0.55
STEER_SPEED_FALLOFF = 0.055   # how quickly the allowed steering angle shrinks with speed
BODY_ROLL_GAIN = 0.030
BODY_PITCH_GAIN = 0.020
SUSPENSION_STIFFNESS = 42.0
SUSPENSION_DAMPING = 7.5
FUEL_IDLE_BURN = 0.09    # litres / second at idle
BUMP_SPEED_LOSS = 0.90   # fraction of speed kept for a small bump
POTHOLE_SPEED_LOSS = 0.72
RAIN_GRIP = 0.74
WET_ROAD_GRIP = 0.88
OFFROAD_GRIP = 0.68
OFFROAD_DRAG = 2.4

# --------------------------------------------------------------------------------------
# Day / night cycle
# --------------------------------------------------------------------------------------
DAY_LENGTH_SECONDS = 300.0     # one full 24h cycle of free drive
TIME_PRESETS = {
    "day": 11.5,
    "evening": 18.1,
    "night": 22.4,
    "morning": 7.4,
}

# --------------------------------------------------------------------------------------
# Player economy
# --------------------------------------------------------------------------------------
START_MONEY = 18000
MISSION_REWARD_PERFECT_BONUS = 0.35
COLLISION_DAMAGE_SCALE = 1.2
PAINT_PRICE = 1500                 # cost of one respray in the garage
REPAIR_RATE = 14                   # rupees per damage point

# --------------------------------------------------------------------------------------
# Colour palette (0..1 floats, used by procedural mesh builders)
# --------------------------------------------------------------------------------------
class P:
    """Colour palette shared by the mesh builders and the UI."""

    ASPHALT = (0.155, 0.155, 0.168)
    ASPHALT_WORN = (0.205, 0.200, 0.196)
    CONCRETE = (0.62, 0.61, 0.58)
    FOOTPATH = (0.70, 0.68, 0.62)
    KERB = (0.80, 0.79, 0.74)
    LINE_WHITE = (0.92, 0.92, 0.88)
    LINE_YELLOW = (0.93, 0.78, 0.21)
    ZEBRA = (0.90, 0.90, 0.86)
    DIRT = (0.44, 0.37, 0.27)
    GRASS = (0.30, 0.42, 0.20)
    SAND = (0.72, 0.63, 0.44)
    GLASS = (0.075, 0.115, 0.145)
    GLASS_LIGHT = (0.16, 0.21, 0.25)
    CHROME = (0.72, 0.73, 0.76)
    TYRE = (0.075, 0.075, 0.080)
    HUB = (0.68, 0.68, 0.71)
    RED_LIGHT = (0.95, 0.13, 0.10)
    AMBER_LIGHT = (1.00, 0.66, 0.10)
    GREEN_LIGHT = (0.18, 0.95, 0.35)
    HEADLIGHT = (1.00, 0.96, 0.80)
    BARRICADE = (0.88, 0.30, 0.12)
    PLASTIC = (0.85, 0.85, 0.86)
    WATER = (0.24, 0.30, 0.26)

    # Paint colours offered in the garage (name -> RGB)
    PAINTS: Dict[str, Tuple[float, float, float]] = {
        "Sunset Orange": (0.86, 0.34, 0.11),
        "Pearl White": (0.90, 0.90, 0.88),
        "Metro Silver": (0.66, 0.68, 0.70),
        "Midnight Blue": (0.10, 0.16, 0.34),
        "Racing Red": (0.72, 0.10, 0.10),
        "Banana Yellow": (0.90, 0.74, 0.14),
        "Leaf Green": (0.16, 0.44, 0.22),
        "Regal Purple": (0.31, 0.16, 0.42),
        "Gunmetal Grey": (0.26, 0.28, 0.30),
        "Sky Blue": (0.26, 0.55, 0.78),
    }

    # Body colours used by the AI traffic (weighted, Indian-market biased)
    TRAFFIC_PAINT: List[Tuple[float, float, float]] = [
        (0.88, 0.88, 0.86),   # white
        (0.82, 0.82, 0.80),   # off-white
        (0.66, 0.67, 0.69),   # silver
        (0.14, 0.15, 0.17),   # black
        (0.55, 0.14, 0.12),   # maroon
        (0.13, 0.28, 0.53),   # blue
        (0.72, 0.72, 0.70),   # grey
        (0.80, 0.40, 0.12),   # orange
        (0.16, 0.40, 0.30),   # dark green
    ]


# --------------------------------------------------------------------------------------
# Default key bindings.  Values are pygame key names resolved by InputManager.
# --------------------------------------------------------------------------------------
DEFAULT_BINDINGS: Dict[str, List[str]] = {
    "throttle":      ["K_w", "K_UP"],
    "brake":         ["K_s", "K_DOWN"],
    "steer_left":    ["K_a", "K_LEFT"],
    "steer_right":   ["K_d", "K_RIGHT"],
    "handbrake":     ["K_SPACE"],
    "horn":          ["K_h"],
    "lights":        ["K_l"],
    "camera":        ["K_c"],
    "look_back":     ["K_b"],
    "indicator_left":  ["K_q"],
    "indicator_right": ["K_e"],
    "hazards":       ["K_x"],
    "navigate":      ["K_n"],          # compass / GPS toggle
    "reset":         ["K_r"],          # un-stick the car
    "pause":         ["K_ESCAPE", "K_p"],
    "accept":        ["K_RETURN", "K_KP_ENTER"],
    "refuel":        ["K_f"],
    "map":           ["K_m"],
}

# --------------------------------------------------------------------------------------
# Save / config locations
# --------------------------------------------------------------------------------------
SAVE_DIR_NAME = ".smart_traffic_india"


def user_data_dir() -> str:
    """Return a writable per-user directory (created on demand)."""
    override = os.environ.get("STI_DATA_DIR")
    if override:
        path = override
    elif os.name == "nt":
        path = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "SmartTrafficIndia")
    else:
        path = os.path.join(os.path.expanduser("~"), SAVE_DIR_NAME)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:                                   # pragma: no cover - exotic FS
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runtime_data")
        os.makedirs(path, exist_ok=True)
    return path


def save_file_path() -> str:
    return os.path.join(user_data_dir(), "profile.json")


def config_file_path() -> str:
    return os.path.join(user_data_dir(), "config.json")


# --------------------------------------------------------------------------------------
# Runtime settings object (persisted through SaveManager)
# --------------------------------------------------------------------------------------
@dataclass
class Settings:
    """User-facing options.  Every field is validated on load."""

    width: int = 1280
    height: int = 720
    fullscreen: bool = False
    vsync: bool = True
    quality: str = "high"
    traffic_level: str = "medium"
    time_of_day: str = "day"          # day | evening | night | morning | cycle
    weather: str = "clear"            # clear | cloudy | rain | random
    camera_mode: str = "chase"        # chase | close | hood
    master_volume: float = 0.6
    show_minimap: bool = True
    show_hud: bool = True
    invert_steering: bool = False
    show_fps: bool = False
    unit_kmh: bool = True
    headlights_auto: bool = True

    # ---- validation helpers -------------------------------------------------------
    def option_list(self, name: str) -> List[Any]:
        return {
            "quality": QUALITY_ORDER,
            "traffic_level": TRAFFIC_LEVEL_ORDER,
            "time_of_day": ["day", "evening", "night", "morning", "cycle"],
            "weather": ["clear", "cloudy", "rain", "random"],
            "camera_mode": ["chase", "close", "hood"],
        }[name]

    def cycle_option(self, name: str, step: int = 1) -> None:
        values = self.option_list(name)
        current = getattr(self, name)
        index = (values.index(current) + step) % len(values) if current in values else 0
        setattr(self, name, values[index])

    def sanitise(self) -> "Settings":
        self.width = int(min(max(self.width, 800), 3840))
        self.height = int(min(max(self.height, 480), 2160))
        if self.quality not in QUALITY_PRESETS:
            self.quality = "high"
        if self.traffic_level not in TRAFFIC_LEVELS:
            self.traffic_level = "medium"
        if self.time_of_day not in self.option_list("time_of_day"):
            self.time_of_day = "day"
        if self.weather not in self.option_list("weather"):
            self.weather = "clear"
        if self.camera_mode not in self.option_list("camera_mode"):
            self.camera_mode = "chase"
        self.master_volume = float(min(max(self.master_volume, 0.0), 1.0))
        for flag in ("fullscreen", "vsync", "show_minimap", "show_hud", "invert_steering",
                     "show_fps", "unit_kmh", "headlights_auto"):
            setattr(self, flag, bool(getattr(self, flag)))
        return self

    @property
    def quality_flags(self) -> Dict[str, Any]:
        return dict(QUALITY_PRESETS[self.quality])

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in (data or {}).items() if k in known}
        try:
            return cls(**kwargs).sanitise()
        except (TypeError, ValueError):               # pragma: no cover - defensive
            return cls()

    def apply_resolution(self, width: int, height: int) -> None:
        self.width, self.height = int(width), int(height)


def default_profile() -> Dict[str, Any]:
    """A brand-new player profile (also the shape used by SaveManager defaults)."""
    return {
        "version": 1,
        "money": START_MONEY,
        "unlocked": ["hatchback"],
        "owned": ["hatchback"],
        "upgrades": {},                 # vehicle_key -> {system: level}
        "paints": {},                   # vehicle_key -> paint name
        "damage": {},                   # vehicle_key -> 0..100 body damage
        "last_car": "hatchback",
        "career_index": 0,
        "completed_missions": [],
        "best_parking": {},
        "school_lessons": [],
        "stats": {
            "distance_km": 0.0, "collisions": 0, "violations": 0, "mission_runs": 0,
            "perfect_runs": 0, "playtime_s": 0.0, "best_parking_score": 0,
        },
        "settings": Settings().to_dict(),
    }


def python_version_ok() -> bool:
    return sys.version_info >= (3, 9)
