"""The in-game HUD: speedometer, objective panel, signals, warnings and minimap.

The HUD is drawn with pygame on a transparent surface that the renderer blits over the 3D
frame as a textured quad.  It reads a single immutable :class:`HUDSnapshot`, which keeps
the drawing code completely decoupled from the simulation (and easy to unit test).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pygame

# --------------------------------------------------------------------------------------
# palette
# --------------------------------------------------------------------------------------
PANEL = (11, 14, 20, 200)
PANEL_LIGHT = (22, 27, 36, 210)
BORDER = (58, 66, 82, 220)
ACCENT = (255, 168, 40)
ACCENT_DIM = (168, 108, 26)
CYAN = (72, 200, 255)
TEXT = (238, 240, 245)
MUTED = (152, 160, 174)
GOOD = (92, 222, 122)
WARN = (255, 196, 64)
BAD = (236, 84, 72)
RED_LIGHT = (240, 60, 50)
AMBER_LIGHT = (255, 168, 40)
GREEN_LIGHT = (60, 225, 110)


@dataclass
class HUDSnapshot:
    """Everything the HUD needs for one frame."""

    speed_kmh: float = 0.0
    gear: str = "D"
    fuel_fraction: float = 1.0
    damage: float = 0.0              # 0..100
    max_speed: float = 180.0
    car_name: str = ""
    money: int = 0
    score: int = 0
    objective: str = ""
    objective_detail: str = ""
    destination_distance: float = -1.0
    time_left: float = -1.0
    signal_state: str = ""
    signal_distance: float = -1.0
    rule: str = ""
    warnings: List[Tuple[str, float]] = field(default_factory=list)
    toasts: List[Tuple[str, str, float]] = field(default_factory=list)
    collisions: int = 0
    violations: int = 0
    combo: float = 1.0
    zone: str = ""
    time_of_day: str = "day"
    weather: str = "clear"
    traffic_count: int = 0
    traffic_level: str = "medium"
    fps: float = 0.0
    show_fps: bool = False
    show_minimap: bool = True
    minimap: Optional[Dict[str, object]] = None
    emergency: bool = False
    direction_hint: str = ""
    parking_hint: str = ""
    mission_progress: float = -1.0     # 0..1 progress bar (e.g. parking accuracy)
    indicator: int = 0
    headlights: bool = False
    wrong_way: bool = False


# --------------------------------------------------------------------------------------
# drawing helpers
# --------------------------------------------------------------------------------------
def _panel(surface: pygame.Surface, rect: pygame.Rect, radius: int = 12,
           color=PANEL, border=BORDER, border_width: int = 1) -> None:
    layer = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(layer, color, layer.get_rect(), border_radius=radius)
    if border_width:
        pygame.draw.rect(layer, border, layer.get_rect(), width=border_width, border_radius=radius)
    surface.blit(layer, rect.topleft)


def _bar(surface: pygame.Surface, rect: pygame.Rect, fraction: float, color,
         background=(30, 34, 44, 220), radius: int = 4) -> None:
    fraction = max(0.0, min(1.0, fraction))
    pygame.draw.rect(surface, background, rect, border_radius=radius)
    if fraction > 0.0:
        width = max(2, int(rect.width * fraction))
        pygame.draw.rect(surface, color, pygame.Rect(rect.x, rect.y, width, rect.height),
                         border_radius=radius)
    pygame.draw.rect(surface, (0, 0, 0, 120), rect, width=1, border_radius=radius)


class HUD:
    """Draws the whole HUD onto a transparent surface."""

    def __init__(self, width: int, height: int) -> None:
        self.resize(width, height)

    def resize(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.scale = max(0.72, min(1.7, height / 720.0))
        s = self.scale
        self.font_big = pygame.font.Font(None, int(46 * s))
        self.font = pygame.font.Font(None, int(28 * s))
        self.font_small = pygame.font.Font(None, int(22 * s))
        self.font_tiny = pygame.font.Font(None, int(19 * s))
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)

    # ------------------------------------------------------------------ entry point
    def render(self, snapshot: HUDSnapshot) -> pygame.Surface:
        surface = self.surface
        surface.fill((0, 0, 0, 0))
        self._state = snapshot.signal_state
        s = self.scale
        if snapshot.show_minimap and snapshot.minimap:
            self._draw_minimap(surface, snapshot)
        self._draw_objective(surface, snapshot)
        self._draw_stats(surface, snapshot)
        self._draw_speed(surface, snapshot)
        self._draw_rules(surface, snapshot)
        self._draw_toasts(surface, snapshot)
        if snapshot.emergency:
            self._draw_emergency(surface)
        if snapshot.wrong_way:
            self._draw_wrong_way(surface)
        if snapshot.show_fps:
            text = self.font_tiny.render(
                f"{snapshot.fps:5.1f} FPS   traffic {snapshot.traffic_count}",
                True, MUTED)
            surface.blit(text, (self.width - text.get_width() - int(12 * s), int(8 * s)))
        del s
        return surface

    # ------------------------------------------------------------------ panels
    def _draw_objective(self, surface: pygame.Surface, snap: HUDSnapshot) -> None:
        s = self.scale
        width, height = int(342 * s), int(120 * s)
        rect = pygame.Rect(int(14 * s), int(14 * s), width, height)
        _panel(surface, rect)
        x = rect.x + int(14 * s)
        y = rect.y + int(10 * s)
        title = self.font_tiny.render("CURRENT OBJECTIVE", True, ACCENT_DIM)
        surface.blit(title, (x, y))
        y += int(20 * s)
        lines = self._wrap(snap.objective or "Free Drive", self.font, width - int(28 * s))
        for line in lines[:2]:
            surface.blit(self.font.render(line, True, TEXT), (x, y))
            y += int(26 * s)
        if snap.objective_detail:
            surface.blit(self.font_small.render(snap.objective_detail, True, MUTED), (x, y))
            y += int(22 * s)
        if snap.destination_distance >= 0.0:
            arrow = self._compass(snap.direction_hint)
            text = self.font_small.render(
                f"{arrow}  Destination  {snap.destination_distance:5.0f} m", True, CYAN)
            surface.blit(text, (x, y))
            y += int(20 * s)
        if snap.time_left >= 0.0:
            colour = BAD if snap.time_left < 12 else (WARN if snap.time_left < 30 else TEXT)
            surface.blit(self.font_small.render(f"Time left  {snap.time_left:5.1f} s",
                                                True, colour), (x, y))
        if snap.mission_progress >= 0.0:
            bar = pygame.Rect(x, rect.bottom - int(18 * s), width - int(28 * s), int(9 * s))
            _bar(surface, bar, snap.mission_progress, ACCENT)
        if snap.parking_hint:
            surface.blit(self.font_tiny.render(snap.parking_hint, True, WARN),
                         (x, rect.bottom - int(34 * s)))

    def _draw_stats(self, surface: pygame.Surface, snap: HUDSnapshot) -> None:
        s = self.scale
        width, height = int(240 * s), int(112 * s)
        rect = pygame.Rect(self.width - width - int(14 * s), int(14 * s), width, height)
        _panel(surface, rect)
        x = rect.x + int(14 * s)
        y = rect.y + int(10 * s)
        surface.blit(self.font.render(f"Rs {snap.money:,}", True, GOOD), (x, y))
        y += int(28 * s)
        surface.blit(self.font_small.render(f"Score {snap.score:,}", True, TEXT), (x, y))
        y += int(22 * s)
        surface.blit(self.font_tiny.render(
            f"Collisions {snap.collisions}   Violations {snap.violations}", True, MUTED), (x, y))
        y += int(20 * s)
        if snap.combo > 1.05:
            surface.blit(self.font_small.render(f"Smooth driving x{snap.combo:.1f}",
                                                True, CYAN), (x, y))
        # signal indicator on the right edge of the panel
        self._draw_signal_box(surface, rect)

    def _draw_signal_box(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        s = self.scale
        box = pygame.Rect(rect.right - int(60 * s), rect.y + int(12 * s),
                          int(46 * s), int(88 * s))
        _panel(surface, box, radius=8, color=(18, 20, 26, 220), border=(70, 74, 84, 230))
        lights = [(RED_LIGHT, "red", box.y + int(14 * s)),
                  (AMBER_LIGHT, "amber", box.y + int(45 * s)),
                  (GREEN_LIGHT, "green", box.y + int(76 * s))]
        state = (self._state or "").lower()
        for colour, name, cy in lights:
            active = state.startswith(name[:3])
            radius = int(11 * s) if active else int(9 * s)
            draw_colour = colour if active else tuple(int(c * 0.22) for c in colour)
            pygame.draw.circle(surface, draw_colour, (box.centerx, cy), radius)
            if active:
                pygame.draw.circle(surface, (255, 255, 255), (box.centerx, cy), radius, 1)

    _state: Optional[str] = None

    def _draw_speed(self, surface: pygame.Surface, snap: HUDSnapshot) -> None:
        s = self.scale
        width, height = int(268 * s), int(158 * s)
        rect = pygame.Rect(int(14 * s), self.height - height - int(14 * s), width, height)
        _panel(surface, rect)
        centre = (rect.centerx - int(38 * s), rect.centery + int(14 * s))
        radius = int(58 * s)
        # gauge arc
        pygame.draw.circle(surface, (28, 32, 40, 255), centre, radius, width=int(9 * s))
        fraction = max(0.0, min(1.0, snap.speed_kmh / max(snap.max_speed, 1.0)))
        end_angle = math.pi * 1.25 - fraction * math.pi * 1.5
        points = []
        steps = 26
        for i in range(steps + 1):
            t = i / steps
            angle = math.pi * 1.25 - t * fraction * math.pi * 1.5
            points.append((centre[0] + math.cos(angle) * radius,
                           centre[1] - math.sin(angle) * radius))
        if len(points) > 2:
            colour = GREEN_LIGHT if fraction < 0.7 else (WARN if fraction < 0.9 else BAD)
            pygame.draw.lines(surface, colour, False, points, int(9 * s))
        # needle
        tip = (centre[0] + math.cos(end_angle) * (radius - int(10 * s)),
               centre[1] - math.sin(end_angle) * (radius - int(10 * s)))
        pygame.draw.line(surface, ACCENT, centre, tip, int(3 * s))
        pygame.draw.circle(surface, ACCENT, centre, int(5 * s))
        big = self.font_big.render(f"{snap.speed_kmh:3.0f}", True, TEXT)
        surface.blit(big, (centre[0] - big.get_width() // 2, centre[1] + int(14 * s)))
        unit = self.font_tiny.render("km/h", True, MUTED)
        surface.blit(unit, (centre[0] - unit.get_width() // 2, centre[1] + int(46 * s)))
        # gear + bars
        x = rect.right - int(96 * s)
        y = rect.y + int(18 * s)
        gear_colour = ACCENT if snap.gear == "D" else CYAN
        surface.blit(self.font_big.render(snap.gear, True, gear_colour), (x + int(18 * s), y))
        y += int(52 * s)
        surface.blit(self.font_tiny.render("FUEL", True, MUTED), (x, y))
        _bar(surface, pygame.Rect(x, y + int(14 * s), int(80 * s), int(10 * s)),
             snap.fuel_fraction, GOOD if snap.fuel_fraction > 0.2 else BAD)
        y += int(36 * s)
        surface.blit(self.font_tiny.render("DAMAGE", True, MUTED), (x, y))
        _bar(surface, pygame.Rect(x, y + int(14 * s), int(80 * s), int(10 * s)),
             snap.damage / 100.0, BAD if snap.damage > 55 else WARN)
        # car name / status chips
        name = self.font_tiny.render(snap.car_name or "", True, MUTED)
        surface.blit(name, (rect.x + int(12 * s), rect.bottom - int(20 * s)))
        chips = []
        if snap.headlights:
            chips.append(("LIGHTS", CYAN))
        if snap.indicator < 0:
            chips.append(("<< LEFT", WARN))
        elif snap.indicator > 0:
            chips.append(("RIGHT >>", WARN))
        chip_x = rect.centerx + int(24 * s)
        for label, colour in chips:
            text = self.font_tiny.render(label, True, colour)
            surface.blit(text, (chip_x, rect.bottom - int(20 * s)))
            chip_x += text.get_width() + int(8 * s)

    def _draw_rules(self, surface: pygame.Surface, snap: HUDSnapshot) -> None:
        s = self.scale
        width = int(300 * s)
        rect = pygame.Rect(self.width - width - int(14 * s),
                           self.height - int(150 * s), width, int(136 * s))
        _panel(surface, rect, color=(11, 14, 20, 175))
        x = rect.x + int(12 * s)
        y = rect.y + int(8 * s)
        if snap.rule:
            text = self.font.render(snap.rule, True, ACCENT)
            surface.blit(text, (x, y))
            y += int(26 * s)
        for label, _timer in snap.warnings[:3]:
            surface.blit(self.font_small.render(f"! {label}", True, WARN), (x, y))
            y += int(20 * s)
        info = f"{snap.zone.title()}   {snap.time_of_day.upper()} / {snap.weather.upper()}"
        surface.blit(self.font_tiny.render(info, True, MUTED), (x, rect.bottom - int(46 * s)))
        info2 = f"Traffic: {snap.traffic_level.upper()} ({snap.traffic_count} vehicles)"
        surface.blit(self.font_tiny.render(info2, True, MUTED), (x, rect.bottom - int(26 * s)))

    def _draw_toasts(self, surface: pygame.Surface, snap: HUDSnapshot) -> None:
        s = self.scale
        if not snap.toasts:
            return
        y = int(self.height * 0.22)
        for kind, message, timer in snap.toasts[:4]:
            colour = {"violation": BAD, "pothole": WARN, "fuel": BAD, "explore": GOOD,
                      "reward": GOOD, "barrier": BAD, "water": CYAN}.get(kind, TEXT)
            alpha = 255 if timer > 1.0 else int(255 * max(0.0, timer))
            text = self.font.render(message, True, colour)
            text.set_alpha(alpha)
            box = pygame.Rect(0, 0, text.get_width() + int(26 * s), int(34 * s))
            box.center = (self.width // 2, y)
            layer = pygame.Surface(box.size, pygame.SRCALPHA)
            alpha_panel = int(180 * (alpha / 255.0))
            pygame.draw.rect(layer, (10, 12, 18, alpha_panel), layer.get_rect(),
                             border_radius=int(8 * s))
            pygame.draw.rect(layer, (*colour, alpha), layer.get_rect(), width=1,
                             border_radius=int(8 * s))
            surface.blit(layer, box.topleft)
            surface.blit(text, (box.x + int(13 * s), box.y + int(7 * s)))
            y += int(40 * s)

    def _draw_emergency(self, surface: pygame.Surface) -> None:
        s = self.scale
        pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() * 0.008)
        colour = (int(230 + 25 * pulse), int(60 * pulse) + 30, int(30 * pulse) + 20)
        text = self.font.render("EMERGENCY VEHICLE - GIVE WAY", True, colour)
        box = pygame.Rect(0, 0, text.get_width() + int(40 * s), int(46 * s))
        box.center = (self.width // 2, int(self.height * 0.10))
        layer = pygame.Surface(box.size, pygame.SRCALPHA)
        pygame.draw.rect(layer, (30, 6, 6, 210), layer.get_rect(), border_radius=int(10 * s))
        pygame.draw.rect(layer, colour, layer.get_rect(), width=2, border_radius=int(10 * s))
        surface.blit(layer, box.topleft)
        surface.blit(text, (box.x + int(20 * s), box.y + int(12 * s)))

    def _draw_wrong_way(self, surface: pygame.Surface) -> None:
        s = self.scale
        text = self.font.render("WRONG SIDE - KEEP LEFT", True, BAD)
        box = pygame.Rect(0, 0, text.get_width() + int(30 * s), int(40 * s))
        box.center = (self.width // 2, int(self.height * 0.16))
        layer = pygame.Surface(box.size, pygame.SRCALPHA)
        pygame.draw.rect(layer, (30, 6, 6, 200), layer.get_rect(), border_radius=int(8 * s))
        surface.blit(layer, box.topleft)
        surface.blit(text, (box.x + int(15 * s), box.y + int(10 * s)))

    def _draw_minimap(self, surface: pygame.Surface, snap: HUDSnapshot) -> None:
        s = self.scale
        size = int(196 * s)
        rect = pygame.Rect(self.width - size - int(14 * s), self.height - size - int(14 * s),
                           size, size)
        _panel(surface, rect, radius=10, color=(9, 12, 17, 205))
        clip = surface.get_clip()
        surface.set_clip(rect.inflate(-int(8 * s), -int(8 * s)))
        info = snap.minimap or {}
        cx, cz = float(info.get("x", 0.0)), float(info.get("z", 0.0))
        scale = size / max(float(info.get("range", 260.0)), 1.0)
        centre = rect.center

        def to_screen(x: float, z: float) -> Tuple[int, int]:
            # north (-z) is up on the minimap
            return (int(centre[0] + (x - cx) * scale), int(centre[1] + (z - cz) * scale))

        for road in info.get("roads", []):
            x0, z0, x1, z1, width = road
            start = to_screen(x0, z0)
            end = to_screen(x1, z1)
            pygame.draw.line(surface, (58, 66, 78), start, end, max(2, int(width * scale)))
        for lot in info.get("parking", []):
            px, pz = to_screen(lot[0], lot[1])
            pygame.draw.rect(surface, (70, 110, 150), pygame.Rect(px - 4, pz - 4, 8, 8))
        for tx, tz in info.get("traffic", []):
            tx_s, tz_s = to_screen(tx, tz)
            pygame.draw.circle(surface, (150, 156, 168), (tx_s, tz_s), max(1, int(2 * s)))
        destination = info.get("destination")
        if destination:
            dx, dz = to_screen(destination[0], destination[1])
            pygame.draw.circle(surface, ACCENT, (dx, dz), int(6 * s))
            pygame.draw.circle(surface, (0, 0, 0), (dx, dz), int(6 * s), 1)
        # player arrow
        px, pz = to_screen(cx, cz)
        yaw = float(info.get("yaw", 0.0))
        fx, fz = math.cos(yaw), -math.sin(yaw)
        rx, rz = -fz, fx
        arrow = [(px + fx * 9 * s, pz + fz * 9 * s),
                 (px - fx * 6 * s + rx * 5 * s, pz - fz * 6 * s + rz * 5 * s),
                 (px - fx * 6 * s - rx * 5 * s, pz - fz * 6 * s - rz * 5 * s)]
        pygame.draw.polygon(surface, CYAN, arrow)
        del destination
        # compass north marker
        north = self.font_tiny.render("N", True, MUTED)
        surface.blit(north, (rect.centerx - 5, rect.y + int(6 * s)))
        surface.set_clip(clip)

    # ------------------------------------------------------------------ helpers
    def _wrap(self, text: str, font: pygame.font.Font, max_width: int) -> List[str]:
        words = text.split()
        lines: List[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [""]

    @staticmethod
    def _compass(hint: str) -> str:
        return {"left": "<<", "right": ">>", "behind": "vv", "ahead": "^^"}.get(hint, "^^")
