"""Keyboard and mouse input with rebindable actions.

The rest of the game never looks at raw key codes: it asks for *actions* (``"throttle"``,
``"pause"``, ...).  That keeps the controls rebindable from the settings screen and makes
the player controller testable with a plain dictionary of booleans.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pygame

from . import settings as S


def key_code(name: str) -> int:
    """Resolve ``"K_w"`` (or a bare integer) to a pygame key constant."""
    if isinstance(name, int):
        return name
    code = getattr(pygame, name, None)
    if code is None:
        raise KeyError(f"unknown key name: {name}")
    return int(code)


def key_name(code: int) -> str:
    try:
        return pygame.key.name(code).upper()
    except Exception:                                   # pragma: no cover
        return f"KEY{code}"


class InputManager:
    """Tracks held and freshly pressed actions, plus mouse state."""

    def __init__(self, bindings: Optional[Dict[str, List[str]]] = None,
                 invert_steering: bool = False) -> None:
        self.bindings: Dict[str, List[int]] = {}
        self.held: Dict[str, bool] = {}
        self._pressed: Dict[str, bool] = {}
        self.mouse_position: Tuple[int, int] = (0, 0)
        self.mouse_clicked = False
        self.invert_steering = invert_steering
        self.load_bindings(bindings or S.DEFAULT_BINDINGS)

    # ------------------------------------------------------------------ bindings
    def load_bindings(self, bindings: Dict[str, Sequence[str]]) -> None:
        for action, keys in bindings.items():
            codes: List[int] = []
            for entry in keys:
                try:
                    codes.append(key_code(entry))
                except KeyError:
                    continue
            if codes:
                self.bindings[action] = codes
        for action in self.bindings:
            self.held.setdefault(action, False)
            self._pressed.setdefault(action, False)

    def rebind(self, action: str, key: str) -> bool:
        try:
            self.bindings[action] = [key_code(key)]
            return True
        except KeyError:
            return False

    def binding_label(self, action: str) -> str:
        codes = self.bindings.get(action, [])
        return " / ".join(key_name(code) for code in codes) if codes else "-"

    # ------------------------------------------------------------------ events
    def process_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            for action, codes in self.bindings.items():
                if event.key in codes:
                    self.held[action] = True
                    if not event.dict.get("repeat", False):
                        self._pressed[action] = True
        elif event.type == pygame.KEYUP:
            for action, codes in self.bindings.items():
                if event.key in codes:
                    self.held[action] = False
        elif event.type == pygame.MOUSEMOTION:
            self.mouse_position = event.pos
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.mouse_clicked = True
            self.mouse_position = event.pos

    def end_frame(self) -> None:
        """Clear the one-shot presses - call once per frame after updating the game."""
        for action in self._pressed:
            self._pressed[action] = False
        self.mouse_clicked = False

    # ------------------------------------------------------------------ queries
    def action(self, name: str) -> bool:
        return bool(self.held.get(name, False))

    def pressed(self, name: str) -> bool:
        return bool(self._pressed.get(name, False))

    def clear(self) -> None:
        for action in self.held:
            self.held[action] = False
        self.end_frame()

    def driving_snapshot(self) -> Dict[str, bool]:
        """The dictionary the player controller consumes each step."""
        snapshot = {
            "throttle": self.action("throttle"),
            "brake": self.action("brake"),
            "steer_left": self.action("steer_left"),
            "steer_right": self.action("steer_right"),
            "handbrake": self.action("handbrake"),
            "horn": self.pressed("horn"),
            "lights": self.pressed("lights"),
            "indicator_left": self.pressed("indicator_left"),
            "indicator_right": self.pressed("indicator_right"),
            "hazards": self.pressed("hazards"),
        }
        if self.invert_steering:
            snapshot["steer_left"], snapshot["steer_right"] = (
                snapshot["steer_right"], snapshot["steer_left"])
        return snapshot

    # ------------------------------------------------------------------ summary
    def control_help(self) -> List[Tuple[str, str]]:
        return [
            ("Accelerate", self.binding_label("throttle")),
            ("Brake / Reverse", self.binding_label("brake")),
            ("Steer left / right", f"{self.binding_label('steer_left')} / "
                                   f"{self.binding_label('steer_right')}"),
            ("Handbrake", self.binding_label("handbrake")),
            ("Horn", self.binding_label("horn")),
            ("Headlights", self.binding_label("lights")),
            ("Change camera", self.binding_label("camera")),
            ("Look back", self.binding_label("look_back")),
            ("Indicators left / right", f"{self.binding_label('indicator_left')} / "
                                        f"{self.binding_label('indicator_right')}"),
            ("Hazard lights", self.binding_label("hazards")),
            ("Refuel", self.binding_label("refuel")),
            ("Reset car (un-stick)", self.binding_label("reset")),
            ("Pause menu", self.binding_label("pause")),
        ]
