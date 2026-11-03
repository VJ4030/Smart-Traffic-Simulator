"""The garage: vehicle ownership, selection, upgrades, paint and repairs.

The garage is a thin, *testable* layer over :class:`SaveManager`: it never touches pygame
or OpenGL, it only reads and writes the profile.  Every purchase is checked against the
player's balance and every upgrade maps to a measurable change in the effective
:class:`~smart_traffic_india.vehicle.VehicleSpec`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .save_manager import UPGRADE_SYSTEMS, SaveManager
from .vehicle import PLAYER_SPECS, VEHICLE_ORDER, VehicleSpec, effective_spec, suspension_absorption
from . import settings as S


class Garage:
    """Vehicle roster + economy operations."""

    def __init__(self, save: SaveManager) -> None:
        self.save = save

    # ------------------------------------------------------------------ roster
    @staticmethod
    def car_order() -> List[str]:
        return list(VEHICLE_ORDER)

    @staticmethod
    def display_name(key: str) -> str:
        return PLAYER_SPECS[key].name if key in PLAYER_SPECS else key.title()

    @staticmethod
    def price_of(key: str) -> int:
        return PLAYER_SPECS[key].price if key in PLAYER_SPECS else 0

    def spec_of(self, key: str) -> VehicleSpec:
        """The spec *with* the upgrades the player has bought for this car."""
        return effective_spec(PLAYER_SPECS.get(key, PLAYER_SPECS["hatchback"]),
                              self.save.data.get("upgrades", {}).get(key, {}))

    def selected_key(self) -> str:
        key = str(self.save.data.get("last_car", "hatchback"))
        if key not in PLAYER_SPECS or not self.is_unlocked(key):
            key = "hatchback"
        return key

    def selected_spec(self) -> VehicleSpec:
        return self.spec_of(self.selected_key())

    def selected_upgrades(self) -> Dict[str, int]:
        return dict(self.save.data.get("upgrades", {}).get(self.selected_key(), {}))

    def selected_paint(self) -> Tuple[float, float, float]:
        name = self.paint_of(self.selected_key())
        return S.P.PAINTS.get(name, (0.86, 0.34, 0.11))

    # ------------------------------------------------------------------ ownership
    def is_unlocked(self, key: str) -> bool:
        return self.save.is_unlocked(key)

    def owns(self, key: str) -> bool:
        return key in self.save.data.get("owned", [])

    def can_afford(self, key: str) -> bool:
        return self.save.money >= self.price_of(key)

    def buy(self, key: str) -> bool:
        """Purchase a car.  Returns False when it is unaffordable or already owned."""
        if key not in PLAYER_SPECS or self.is_unlocked(key):
            return False
        if not self.save.buy_car(key, self.price_of(key)):
            return False
        self.save.data.setdefault("owned", [])
        if key not in self.save.data["owned"]:
            self.save.data["owned"].append(key)
        self.save.save()
        return True

    def select(self, key: str) -> bool:
        if not self.is_unlocked(key):
            return False
        self.save.data["last_car"] = key
        self.save.save()
        return True

    # ------------------------------------------------------------------ upgrades
    @staticmethod
    def upgrade_systems() -> Dict[str, Dict[str, object]]:
        return UPGRADE_SYSTEMS

    def upgrade_level(self, key: str, system: str) -> int:
        return self.save.upgrade_level(key, system)

    def upgrade_cost(self, key: str, system: str) -> Optional[int]:
        return self.save.upgrade_cost(key, system)

    def buy_upgrade(self, key: str, system: str) -> bool:
        if not self.save.buy_upgrade(key, system):
            return False
        self.save.save()
        return True

    # ------------------------------------------------------------------ paint / repair
    def paint_of(self, key: str) -> str:
        return self.save.paint_of(key)

    def set_paint(self, key: str, paint: str) -> bool:
        if paint not in S.P.PAINTS:
            return False
        current = self.paint_of(key)
        price = 0 if current == paint or current is None else S.PAINT_PRICE
        if not self.save.set_paint(key, paint, price):
            return False
        self.save.save()
        return True

    def damage_of(self, key: str) -> float:
        return self.save.damage_of(key)

    def set_damage(self, key: str, damage: float) -> None:
        self.save.set_damage(key, damage)

    def add_damage(self, key: str, damage: float) -> float:
        return self.save.add_damage(key, damage)

    def repair_cost(self, key: Optional[str] = None) -> int:
        key = key or self.selected_key()
        return self.save.repair_cost(key, self.damage_of(key))

    def repair(self, key: Optional[str] = None) -> bool:
        key = key or self.selected_key()
        ok = self.save.repair(key)
        if ok:
            self.save.save()
        return ok

    # ------------------------------------------------------------------ info
    def money(self) -> int:
        return self.save.money

    def absorption(self, key: Optional[str] = None) -> float:
        key = key or self.selected_key()
        return suspension_absorption(self.save.data.get("upgrades", {}).get(key, {}))

    def summary(self) -> Dict[str, object]:
        return {
            "money": self.money(),
            "selected": self.selected_key(),
            "unlocked": list(self.save.data.get("unlocked", [])),
            "upgrades": dict(self.save.data.get("upgrades", {}).get(self.selected_key(), {})),
            "paint": self.paint_of(self.selected_key()),
            "damage": self.damage_of(self.selected_key()),
        }
