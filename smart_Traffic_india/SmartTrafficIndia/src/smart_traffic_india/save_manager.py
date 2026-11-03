"""Persistence layer: player profile, unlocked vehicles, upgrades and settings.

Design goals
------------
* **Never crash the game because of a bad save file.**  Every load path is defensive:
  missing files, truncated JSON, wrong types, unknown vehicle ids and negative money are
  all repaired instead of raising.
* **Atomic writes.**  We write to ``profile.json.tmp`` and ``os.replace`` it, so a crash
  during save cannot destroy the previous profile.
* **Testable without a display.**  The storage directory can be redirected with the
  ``STI_DATA_DIR`` environment variable, which the unit tests use.

This module is deliberately free of pygame / OpenGL imports.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

from . import settings as S

PROFILE_VERSION = 1

#: Upgrade systems available in the garage.  Each level costs
#: ``base_cost * level`` and grants a documented, measurable bonus.
UPGRADE_SYSTEMS: Dict[str, Dict[str, Any]] = {
    "engine":     {"label": "Engine",     "max": 5, "cost": 3200,
                   "effect": "+6% top speed, +9% acceleration per level"},
    "brakes":     {"label": "Brakes",     "max": 5, "cost": 2400,
                   "effect": "+10% braking force per level"},
    "tyres":      {"label": "Tyres",      "max": 5, "cost": 2800,
                   "effect": "+9% grip per level"},
    "suspension": {"label": "Suspension", "max": 5, "cost": 2600,
                   "effect": "-18% pothole impact per level"},
    "steering":   {"label": "Steering",   "max": 5, "cost": 2200,
                   "effect": "+8% steering response per level"},
}


def empty_profile() -> Dict[str, Any]:
    return copy.deepcopy(S.default_profile())


class SaveManager:
    """Loads, validates, repairs and stores the player profile."""

    def __init__(self, path: Optional[str] = None, auto_load: bool = True) -> None:
        self.path = path or S.save_file_path()
        self.data: Dict[str, Any] = empty_profile()
        self.last_error: Optional[str] = None
        self.loaded_from_disk = False
        if auto_load:
            self.load()

    # ------------------------------------------------------------------ helpers
    def _defaults(self) -> Dict[str, Any]:
        return empty_profile()

    def _repair(self, raw: Any) -> Dict[str, Any]:
        """Merge an arbitrary decoded object onto the defaults, dropping junk."""
        data = self._defaults()
        if not isinstance(raw, dict):
            raise ValueError("profile root is not an object")

        known_keys = set(data.keys())
        for key in known_keys:
            if key not in raw:
                continue
            value = raw[key]
            if key == "settings":
                if isinstance(value, dict):
                    from .settings import Settings
                    data["settings"] = Settings.from_dict(value).to_dict()
            elif key == "stats":
                if isinstance(value, dict):
                    for stat_key, stat_default in data["stats"].items():
                        if stat_key in value:
                            try:
                                data["stats"][stat_key] = type(stat_default)(value[stat_key])
                            except (TypeError, ValueError):
                                pass
            elif key == "money":
                data["money"] = max(0, int(value)) if isinstance(value, (int, float)) else data["money"]
            elif key in ("unlocked", "owned", "completed_missions", "school_lessons"):
                if isinstance(value, (list, tuple)):
                    data[key] = [str(v) for v in value if isinstance(v, str)]
            elif key == "upgrades":
                if isinstance(value, dict):
                    clean: Dict[str, Dict[str, int]] = {}
                    for car, systems in value.items():
                        if not isinstance(systems, dict):
                            continue
                        clean[str(car)] = {}
                        for system, level in systems.items():
                            if system in UPGRADE_SYSTEMS and isinstance(level, (int, float)):
                                clean[str(car)][system] = int(max(0, min(level, UPGRADE_SYSTEMS[system]["max"])))
                    data["upgrades"] = clean
            elif key == "damage":
                if isinstance(value, dict):
                    data["damage"] = {str(k): float(max(0.0, min(100.0, v)))
                                      for k, v in value.items()
                                      if isinstance(v, (int, float))}
            elif key == "paints":
                if isinstance(value, dict):
                    data["paints"] = {str(k): str(v) for k, v in value.items()
                                      if isinstance(v, str) and v in S.P.PAINTS}
            elif key == "best_parking":
                if isinstance(value, dict):
                    data["best_parking"] = {str(k): int(v) for k, v in value.items()
                                            if isinstance(v, (int, float))}
            elif key in ("last_car", "version") or key == "career_index":
                if key == "career_index":
                    data[key] = max(0, int(value)) if isinstance(value, (int, float)) else 0
                else:
                    data[key] = value

        # ---- cross-field consistency -------------------------------------------------
        if not data["owned"]:
            data["owned"] = ["hatchback"]
        data["unlocked"] = sorted(set(data["unlocked"]) | set(data["owned"]) | {"hatchback"})
        if data["last_car"] not in data["unlocked"]:
            data["last_car"] = data["unlocked"][0] if data["unlocked"] else "hatchback"
        for car in data["owned"]:
            data["upgrades"].setdefault(car, {})
            data["paints"].setdefault(car, "Sunset Orange")
        data["money"] = max(0, int(data["money"]))
        data["version"] = PROFILE_VERSION
        return data

    # --------------------------------------------------------------------- api
    def load(self) -> Dict[str, Any]:
        """Read the profile; on any failure fall back to a fresh profile."""
        self.loaded_from_disk = False
        try:
            if not os.path.exists(self.path):
                self.last_error = "no save file"
                self.data = self._defaults()
                return self.data
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self.data = self._repair(raw)
            self.loaded_from_disk = True
            self.last_error = None
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.data = self._defaults()
        return self.data

    def save(self) -> bool:
        """Atomically persist the profile.  Returns ``True`` on success."""
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        try:
            os.makedirs(directory, exist_ok=True)
            payload = dict(self.data)
            payload["version"] = PROFILE_VERSION
            payload["saved_at"] = time.time()
            fd, tmp_path = tempfile.mkstemp(prefix=".sti_", suffix=".json", dir=directory)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self.path)
            self.last_error = None
            return True
        except (OSError, TypeError, ValueError) as exc:
            self.last_error = f"save failed: {exc}"
            return False

    # ---------------------------------------------------------------- accessors
    @property
    def money(self) -> int:
        return int(self.data.get("money", 0))

    @money.setter
    def money(self, value: int) -> None:
        self.data["money"] = max(0, int(value))

    def add_money(self, amount: int) -> int:
        self.money = self.money + int(amount)
        return self.money

    def is_unlocked(self, car: str) -> bool:
        return car in self.data.get("unlocked", [])

    def unlock(self, car: str) -> None:
        if car not in self.data["unlocked"]:
            self.data["unlocked"].append(car)
            self.data["unlocked"].sort()

    def buy_car(self, car: str, price: int) -> bool:
        """Attempt a purchase.  Returns False when already owned or too expensive."""
        if self.is_unlocked(car) or self.money < price:
            return False
        self.money -= price
        self.unlock(car)
        self.data.setdefault("upgrades", {}).setdefault(car, {})
        self.data.setdefault("paints", {}).setdefault(car, "Sunset Orange")
        return True

    def upgrade_level(self, car: str, system: str) -> int:
        return int(self.data.get("upgrades", {}).get(car, {}).get(system, 0))

    def upgrade_cost(self, car: str, system: str) -> Optional[int]:
        """Cost of the *next* level, or None when already maxed."""
        spec = UPGRADE_SYSTEMS.get(system)
        if spec is None:
            return None
        level = self.upgrade_level(car, system)
        if level >= spec["max"]:
            return None
        return int(spec["cost"] * (level + 1))

    def buy_upgrade(self, car: str, system: str) -> bool:
        cost = self.upgrade_cost(car, system)
        if cost is None or self.money < cost:
            return False
        self.money -= cost
        systems = self.data.setdefault("upgrades", {}).setdefault(car, {})
        systems[system] = self.upgrade_level(car, system) + 1
        return True

    def set_paint(self, car: str, paint: str, price: int = 0) -> bool:
        if paint not in S.P.PAINTS:
            return False
        if price and self.money < price:
            return False
        if price:
            self.money -= price
        self.data.setdefault("paints", {})[car] = paint
        return True

    def paint_of(self, car: str) -> str:
        return self.data.get("paints", {}).get(car, "Sunset Orange")

    def damage_of(self, car: str) -> float:
        return float(self.data.get("damage", {}).get(car, 0.0))

    def set_damage(self, car: str, damage: float) -> None:
        self.data.setdefault("damage", {})[car] = float(max(0.0, min(100.0, damage)))

    def add_damage(self, car: str, damage: float) -> float:
        total = min(100.0, self.damage_of(car) + max(0.0, damage))
        self.set_damage(car, total)
        return total

    def repair_cost(self, car: str, damage: float) -> int:
        """Damage is 0..100; repair price scales with the damage left."""
        return int(max(0.0, damage) * S.REPAIR_RATE)

    def repair(self, car: str) -> bool:
        cost = self.repair_cost(car, self.damage_of(car))
        if cost <= 0:
            self.set_damage(car, 0.0)
            return True
        if self.money < cost:
            return False
        self.money -= cost
        self.set_damage(car, 0.0)
        return True

    # ------------------------------------------------------------------- stats
    def record_mission(self, mission_id: str, reward: int, perfect: bool,
                       collisions: int, violations: int) -> None:
        self.add_money(reward)
        stats = self.data.setdefault("stats", {})
        stats["mission_runs"] = int(stats.get("mission_runs", 0)) + 1
        stats["collisions"] = int(stats.get("collisions", 0)) + int(collisions)
        stats["violations"] = int(stats.get("violations", 0)) + int(violations)
        if perfect:
            stats["perfect_runs"] = int(stats.get("perfect_runs", 0)) + 1
        if mission_id and mission_id not in self.data["completed_missions"]:
            self.data["completed_missions"].append(mission_id)

    def record_parking(self, key: str, score: int) -> bool:
        """Store a parking score; returns True when it is a new personal best."""
        table = self.data.setdefault("best_parking", {})
        best = int(table.get(key, 0))
        if score > best:
            table[key] = int(score)
            stats = self.data.setdefault("stats", {})
            stats["best_parking_score"] = max(int(stats.get("best_parking_score", 0)), int(score))
            return True
        return False

    def add_distance(self, km: float) -> None:
        stats = self.data.setdefault("stats", {})
        stats["distance_km"] = round(float(stats.get("distance_km", 0.0)) + max(0.0, km), 4)

    def add_playtime(self, seconds: float) -> None:
        stats = self.data.setdefault("stats", {})
        stats["playtime_s"] = round(float(stats.get("playtime_s", 0.0)) + max(0.0, seconds), 2)

    def mark_lesson(self, lesson_id: str) -> None:
        lessons: List[str] = self.data.setdefault("school_lessons", [])
        if lesson_id not in lessons:
            lessons.append(lesson_id)

    # ---------------------------------------------------------------- settings
    def load_settings(self) -> S.Settings:
        try:
            return S.Settings.from_dict(self.data.get("settings", {}))
        except Exception:                              # pragma: no cover - defensive
            return S.Settings()

    def store_settings(self, settings: S.Settings) -> None:
        self.data["settings"] = settings.sanitise().to_dict()

    def reset_progress(self) -> None:
        settings_data = self.data.get("settings", {})
        self.data = self._defaults()
        self.data["settings"] = settings_data
