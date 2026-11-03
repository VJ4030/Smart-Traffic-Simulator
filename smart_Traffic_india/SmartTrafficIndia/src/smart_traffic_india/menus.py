"""Menu framework and every screen of the game.

:class:`MenuScreen` is a small, self-contained widget: a list of rows that can be moved
through with the arrow keys / W-S / the mouse, activated with Enter or a click, adjusted
with the left-right arrows, and cancelled with Escape.  Each menu declares an ``on_select``
callback that returns an action string which :mod:`game` interprets - so a menu never
contains game logic and every visible button really does something.

Screens included: main menu, career mission list, parking challenge list, driving school
lessons, garage, settings, how-to-play (with the controls), pause, briefing, results and
the free-drive loadout picker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pygame

from .hud import ACCENT, ACCENT_DIM, CYAN, GOOD, MUTED, TEXT, WARN, _bar, _panel
from . import settings as S

ACTION = "action"
TOGGLE = "toggle"
CYCLE = "cycle"
SLIDER = "slider"
INFO = "info"


@dataclass
class MenuRow:
    """One interactive row."""

    label: str
    kind: str = ACTION
    hint: str = ""
    value: Optional[Callable[[], str]] = None
    adjust: Optional[Callable[[int], None]] = None
    tag: str = ""
    enabled: Callable[[], bool] = lambda: True
    colour: Optional[Tuple[int, int, int]] = None


class MenuScreen:
    """A vertical list menu with keyboard and mouse support."""

    def __init__(self, title: str, subtitle: str = "", footer: str = "") -> None:
        self.title = title
        self.subtitle = subtitle
        self.footer = footer
        self.rows: List[MenuRow] = []
        self.index = 0
        self.scroll = 0
        self.visible_rows = 8
        self.font_title = pygame.font.Font(None, 52)
        self.font_sub = pygame.font.Font(None, 26)
        self.font_row = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 22)
        self.last_mouse = (0, 0)
        self._surface_width, self._surface_height = 1280, 720

    # ------------------------------------------------------------------ building
    def add(self, label: str, kind: str = ACTION, hint: str = "", value=None,
            adjust=None, tag: str = "", enabled=None,
            colour: Optional[Tuple[int, int, int]] = None) -> "MenuScreen":
        self.rows.append(MenuRow(label, kind, hint, value, adjust, tag,
                                 enabled or (lambda: True), colour))
        return self

    def add_gap(self) -> "MenuScreen":
        self.rows.append(MenuRow("", INFO, enabled=lambda: False))
        return self

    # ------------------------------------------------------------------ input
    def move(self, delta: int) -> None:
        if not self.rows:
            return
        count = len(self.rows)
        for _ in range(count):
            self.index = (self.index + delta) % count
            if self.rows[self.index].enabled():
                break
        self._clamp_scroll()

    def _clamp_scroll(self) -> None:
        if self.index < self.scroll:
            self.scroll = self.index
        elif self.index >= self.scroll + self.visible_rows:
            self.scroll = self.index - self.visible_rows + 1

    def activate(self) -> Optional[str]:
        if not self.rows:
            return None
        row = self.rows[self.index]
        if not row.enabled():
            return None
        if row.kind == TOGGLE:
            if row.adjust:
                row.adjust(0)
            return f"toggle:{row.tag or row.label}"
        if row.kind == SLIDER:
            return f"slider:{row.tag or row.label}"
        if row.kind == INFO:
            return None
        return f"select:{row.tag or row.label}"

    def adjust_current(self, direction: int) -> Optional[str]:
        if not self.rows:
            return None
        row = self.rows[self.index]
        if row.kind in (CYCLE, SLIDER, TOGGLE) and row.adjust and row.enabled():
            row.adjust(direction)
            if row.kind == TOGGLE:
                return f"toggle:{row.tag or row.label}"
            return f"adjust:{row.tag or row.label}"
        return None

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_DOWN, pygame.K_s):
                self.move(1)
            elif event.key in (pygame.K_UP, pygame.K_w):
                self.move(-1)
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                return self.activate()
            elif event.key == pygame.K_LEFT:
                return self.adjust_current(-1)
            elif event.key == pygame.K_RIGHT:
                return self.adjust_current(1)
        elif event.type == pygame.MOUSEMOTION:
            self.last_mouse = event.pos
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            hit = self._row_at(event.pos)
            if hit is not None:
                if hit == self.index:
                    return self.activate()
                self.index = hit
        return None

    def _row_at(self, pos: Tuple[int, int]) -> Optional[int]:
        for visible, row_index in enumerate(range(self.scroll,
                                                  min(len(self.rows), self.scroll + self.visible_rows))):
            rect = self.row_rect(visible, row_index)
            if rect.collidepoint(pos):
                return row_index
        return None

    # ------------------------------------------------------------------ drawing
    #: vertical metrics of a menu panel (all scaled from the window height)
    HEADER = 108
    ROW_PITCH = 48
    ROW_HEIGHT = 42
    FOOTER = 62

    def _metrics(self, size: Tuple[int, int]) -> Tuple[pygame.Rect, int, int]:
        """Return ``(panel, row_pitch, visible_rows)`` for a window of ``size``.

        The panel is as tall as the rows need (capped to the window), so a 4 row pause menu
        looks tidy and a 12 row mission list still fits - and if it does not, the list
        scrolls instead of running off the panel.
        """
        width, height = int(size[0]), int(size[1])
        panel_width = min(int(width * 0.62), 780)
        scale = max(0.62, min(1.0, height / 720.0))
        pitch = max(30, int(self.ROW_PITCH * scale))
        header = int(self.HEADER * scale)
        footer = int(self.FOOTER * scale)
        wanted = header + max(1, len(self.rows)) * pitch + footer
        panel_height = int(min(wanted, height * 0.94))
        panel = pygame.Rect((width - panel_width) // 2, (height - panel_height) // 2,
                            panel_width, panel_height)
        visible = max(1, (panel_height - header - footer) // pitch)
        return panel, pitch, visible

    def apply_metrics(self, size: Tuple[int, int]) -> None:
        """Update :attr:`visible_rows` for the current window size (call before scrolling)."""
        _, _, visible = self._metrics(size)
        self.visible_rows = max(1, visible)
        self._clamp_scroll()

    def layout(self, surface: pygame.Surface) -> Dict[str, pygame.Rect]:
        panel, _pitch, _visible = self._metrics(surface.get_size())
        return {"panel": panel}

    def row_rect(self, visible_index: int, row_index: int) -> pygame.Rect:
        size = (self._surface_width, self._surface_height)
        panel, pitch, _visible = self._metrics(size)
        top = panel.y + int(self.HEADER * max(0.62, min(1.0, size[1] / 720.0)))
        height = max(24, pitch - 6)
        return pygame.Rect(panel.x + 26, top + visible_index * pitch, panel.width - 52,
                           height)

    def render(self, surface: pygame.Surface) -> pygame.Surface:
        self._surface_width, self._surface_height = surface.get_size()
        width, height = surface.get_size()
        self.apply_metrics((width, height))
        panel = self.layout(surface)["panel"]
        _panel(surface, panel, radius=16, color=(10, 13, 19, 226), border=(66, 74, 92, 240))
        title = self.font_title.render(self.title, True, ACCENT)
        surface.blit(title, (panel.x + 28, panel.y + 22))
        if self.subtitle:
            surface.blit(self.font_sub.render(self.subtitle, True, MUTED),
                         (panel.x + 30, panel.y + 70))
        pygame.draw.line(surface, (52, 58, 72), (panel.x + 24, panel.y + 98),
                         (panel.right - 24, panel.y + 98), 1)

        last_visible = min(len(self.rows), self.scroll + self.visible_rows)
        for visible, row_index in enumerate(range(self.scroll, last_visible)):
            row = self.rows[row_index]
            rect = self.row_rect(visible, row_index)
            selected = row_index == self.index
            enabled = row.enabled()
            if row.kind == INFO and not row.label:
                continue
            if selected:
                highlight = pygame.Surface(rect.size, pygame.SRCALPHA)
                pygame.draw.rect(highlight, (40, 48, 62, 220), highlight.get_rect(),
                                 border_radius=9)
                pygame.draw.rect(highlight, ACCENT, highlight.get_rect(), width=1,
                                 border_radius=9)
                surface.blit(highlight, rect.topleft)
            if not enabled:
                colour = (96, 100, 110)
            elif row.colour:
                colour = row.colour
            else:
                colour = TEXT if selected else (206, 212, 222)
            label = self.font_row.render(row.label, True, colour)
            surface.blit(label, (rect.x + 16, rect.y + 8))
            if row.value:
                value_text = self.font_row.render(str(row.value()), True,
                                                  ACCENT if selected else CYAN)
                surface.blit(value_text, (rect.right - value_text.get_width() - 16, rect.y + 8))
            elif row.hint:
                hint = self.font_small.render(row.hint, True, MUTED)
                surface.blit(hint, (rect.right - hint.get_width() - 16, rect.y + 13))
        if self.scroll > 0:
            surface.blit(self.font_small.render("^ more", True, MUTED),
                         (panel.right - 90, panel.y + 100))
        if last_visible < len(self.rows):
            surface.blit(self.font_small.render("v more", True, MUTED),
                         (panel.right - 90, panel.bottom - 40))
        if len(self.rows) > self.visible_rows:
            arrow_up = self.font_small.render("^", True, ACCENT if self.scroll > 0 else MUTED)
            arrow_down = self.font_small.render(
                "v", True,
                ACCENT if self.scroll + self.visible_rows < len(self.rows) else MUTED)
            surface.blit(arrow_up, (panel.right - 22, panel.y + 104))
            surface.blit(arrow_down, (panel.right - 22, panel.bottom - 64))
        footer = self.footer or "Up/Down or mouse to choose   |   Enter to select   |   Esc to go back"
        text = self.font_small.render(footer, True, MUTED)
        surface.blit(text, (panel.x + 28, panel.bottom - 34))
        return surface


class GarageScreen(MenuScreen):
    """A richer garage layout: car list on the left, stats and actions on the right."""

    def __init__(self, garage) -> None:
        super().__init__("Garage", "Buy, select and upgrade your vehicles")
        self.garage = garage
        self.view = "cars"          # cars | upgrades | paint
        self.car_index = 0
        self.action_index = 0
        self.font_stat = pygame.font.Font(None, 24)

    # ------------------------------------------------------------------ helpers
    def car_keys(self) -> List[str]:
        return self.garage.car_order()

    def current_car(self) -> str:
        keys = self.car_keys()
        return keys[min(self.car_index, len(keys) - 1)]

    def actions(self) -> List[Tuple[str, str, Optional[str]]]:
        """(tag, label, hint) rows for the currently selected car."""
        car = self.current_car()
        rows: List[Tuple[str, str, Optional[str]]] = []
        if self.garage.is_unlocked(car):
            if self.garage.selected_key() == car:
                rows.append(("selected", "SELECTED", "in use"))
            else:
                rows.append(("select", "Select this car", "Enter"))
        else:
            price = self.garage.price_of(car)
            rows.append(("buy", f"Buy for Rs {price:,}",
                         "affordable" if self.garage.can_afford(car) else "not enough money"))
        rows.append(("upgrades", "Upgrades", "engine, brakes, tyres, suspension, steering"))
        rows.append(("paint", "Paint shop", "change the body colour"))
        rows.append(("repair", "Repair bodywork",
                     f"Rs {self.garage.repair_cost():,}" if self.garage.damage_of(car) > 1 else "no damage"))
        return rows

    def upgrade_rows(self) -> List[Tuple[str, str, str, bool]]:
        car = self.current_car()
        out = []
        for system, spec in self.garage.upgrade_systems().items():
            level = self.garage.upgrade_level(car, system)
            cost = self.garage.upgrade_cost(car, system)
            label = f"{spec['label']}  Lv {level}/{spec['max']}"
            hint = "MAX" if cost is None else f"Rs {cost:,}"
            out.append((system, label, hint, cost is not None))
        return out

    def paint_rows(self) -> List[Tuple[str, str, str]]:
        car = self.current_car()
        current = self.garage.paint_of(car)
        rows = []
        for name, rgb in S.P.PAINTS.items():
            rows.append((name, name, "current" if name == current else
                         ("free" if self.garage.owns(car) else "")))
        return rows

    # ------------------------------------------------------------------ input
    def _action_count(self) -> int:
        return {"cars": len(self.actions()), "upgrades": len(self.upgrade_rows()),
                "paint": len(self.paint_rows())}[self.view]

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_LEFT, pygame.K_a):
                if self.view == "cars":
                    self.car_index = (self.car_index - 1) % len(self.car_keys())
                return None
            if event.key in (pygame.K_RIGHT, pygame.K_d):
                if self.view == "cars":
                    self.car_index = (self.car_index + 1) % len(self.car_keys())
                return None
            if event.key in (pygame.K_UP, pygame.K_w):
                self.action_index = max(0, self.action_index - 1)
                return None
            if event.key in (pygame.K_DOWN, pygame.K_s):
                self.action_index = min(self._action_count() - 1, self.action_index + 1)
                return None
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                return self.activate()
            if event.key == pygame.K_ESCAPE:
                return self.back()
            return None
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            car_hit = self._car_at(event.pos)
            if car_hit is not None:
                self.car_index = car_hit
                self.action_index = 0
                return None
            action_hit = self._action_at(event.pos)
            if action_hit is not None:
                if action_hit == self.action_index:
                    return self.activate()
                self.action_index = action_hit
            return None
        if event.type == pygame.MOUSEMOTION:
            self.last_mouse = event.pos
        return None

    def _layout_rects(self, surface_size=None):
        width, height = surface_size or (self._surface_width, self._surface_height)
        panel = pygame.Rect(int(width * 0.06), int(height * 0.08),
                            int(width * 0.88), int(height * 0.84))
        column = pygame.Rect(panel.x + 24, panel.y + 78, int(panel.width * 0.34),
                             panel.height - 118)
        info = pygame.Rect(column.right + 18, column.y, panel.right - column.right - 42,
                           int(column.height * 0.46))
        actions = pygame.Rect(info.x, info.bottom + 14, info.width,
                              column.bottom - info.height - 14)
        return panel, column, info, actions

    def _car_at(self, pos) -> Optional[int]:
        _panel, column, _info, _actions = self._layout_rects()
        for i, _key in enumerate(self.car_keys()):
            rect = pygame.Rect(column.x + 10, column.y + 12 + i * 54, column.width - 20, 48)
            if rect.collidepoint(pos):
                return i
        return None

    def _action_at(self, pos) -> Optional[int]:
        _panel, _column, _info, actions = self._layout_rects()
        for i in range(min(self._action_count(), 6)):
            rect = pygame.Rect(actions.x + 10, actions.y + 32 + i * 40, actions.width - 20, 34)
            if rect.collidepoint(pos):
                return i
        return None

    def activate(self) -> Optional[str]:
        car = self.current_car()
        if self.view == "cars":
            rows = self.actions()
            if not rows:
                return None
            tag = rows[min(self.action_index, len(rows) - 1)][0]
            if tag == "upgrades":
                self.view = "upgrades"
                self.action_index = 0
                return "garage:view"
            if tag == "paint":
                self.view = "paint"
                self.action_index = 0
                return "garage:view"
            if tag == "select":
                return f"garage:select:{car}"
            if tag == "buy":
                return f"garage:buy:{car}"
            if tag == "repair":
                return f"garage:repair:{car}"
            return None
        if self.view == "upgrades":
            rows = self.upgrade_rows()
            if not rows:
                return None
            system = rows[min(self.action_index, len(rows) - 1)][0]
            return f"garage:upgrade:{car}:{system}"
        if self.view == "paint":
            rows = self.paint_rows()
            if not rows:
                return None
            paint = rows[min(self.action_index, len(rows) - 1)][0]
            return f"garage:paint:{car}:{paint}"
        return None

    def back(self) -> Optional[str]:
        if self.view != "cars":
            self.view = "cars"
            self.action_index = 0
            return "garage:view"
        return "garage:exit"

    # ------------------------------------------------------------------ drawing
    def render(self, surface: pygame.Surface) -> pygame.Surface:
        width, height = surface.get_size()
        backdrop = pygame.Surface((width, height), pygame.SRCALPHA)
        backdrop.fill((6, 8, 12, 150))
        surface.blit(backdrop, (0, 0))
        panel, column, info, actions_rect = self._layout_rects((width, height))
        _panel(surface, panel, radius=16, color=(10, 13, 19, 232), border=(66, 74, 92, 240))
        surface.blit(self.font_title.render("Garage", True, ACCENT), (panel.x + 26, panel.y + 18))
        surface.blit(self.font_sub.render(f"Rs {self.garage.money():,} available",
                                          True, GOOD), (panel.right - 260, panel.y + 28))

        # ---- car column ------------------------------------------------------------
        _panel(surface, column, radius=12, color=(14, 18, 26, 220))
        for i, key in enumerate(self.car_keys()):
            rect = pygame.Rect(column.x + 10, column.y + 12 + i * 54, column.width - 20, 48)
            selected = i == self.car_index
            unlocked = self.garage.is_unlocked(key)
            if selected:
                highlight = pygame.Surface(rect.size, pygame.SRCALPHA)
                pygame.draw.rect(highlight, (40, 48, 62, 230), highlight.get_rect(), border_radius=9)
                pygame.draw.rect(highlight, ACCENT, highlight.get_rect(), width=1, border_radius=9)
                surface.blit(highlight, rect.topleft)
            colour = TEXT if unlocked else (120, 126, 138)
            name = self.garage.display_name(key)
            surface.blit(self.font_row.render(name, True, colour), (rect.x + 12, rect.y + 4))
            sub = ("IN USE" if key == self.garage.selected_key() else
                   ("owned" if unlocked else f"Rs {self.garage.price_of(key):,}"))
            surface.blit(self.font_small.render(sub, True,
                                                CYAN if key == self.garage.selected_key() else MUTED),
                         (rect.x + 12, rect.y + 28))
            if not unlocked:
                lock = self.font_small.render("LOCKED", True, WARN)
                surface.blit(lock, (rect.right - lock.get_width() - 10, rect.y + 14))

        # ---- stats + actions --------------------------------------------------------
        _panel(surface, info, radius=12, color=(14, 18, 26, 220))
        car = self.current_car()
        spec = self.garage.spec_of(car)
        surface.blit(self.font_row.render(self.garage.display_name(car), True, TEXT),
                     (info.x + 16, info.y + 10))
        surface.blit(self.font_small.render(spec.description, True, MUTED),
                     (info.x + 16, info.y + 38))
        stats = spec.display_stats()
        y = info.y + 66
        for label, value in stats.items():
            surface.blit(self.font_stat.render(label, True, MUTED), (info.x + 16, y + 2))
            _bar(surface, pygame.Rect(info.x + 130, y + 4, info.width - 150, 12), value,
                 ACCENT if value > 0.6 else CYAN)
            y += 22
        if not self.garage.is_unlocked(car):
            price = self.font_row.render(f"Price  Rs {self.garage.price_of(car):,}",
                                         True, WARN if not self.garage.can_afford(car) else GOOD)
            surface.blit(price, (info.x + 16, info.bottom - 34))

        _panel(surface, actions_rect, radius=12, color=(14, 18, 26, 220))
        if self.view == "cars":
            rows = [(label, hint) for _tag, label, hint in self.actions()]
        elif self.view == "upgrades":
            rows = [(label, hint) for _tag, label, hint, _ok in self.upgrade_rows()]
        else:
            rows = [(label, hint) for _name, label, hint in self.paint_rows()]
        surface.blit(self.font_small.render(
            {"cars": "ACTIONS", "upgrades": "UPGRADES  (left/right to change)",
             "paint": "PAINT SHOP  (left/right to change)"}[self.view], True, ACCENT_DIM),
            (actions_rect.x + 16, actions_rect.y + 8))
        for i, (label, hint) in enumerate(rows[:6]):
            rect = pygame.Rect(actions_rect.x + 10, actions_rect.y + 32 + i * 40,
                               actions_rect.width - 20, 34)
            selected = i == self.action_index and self.view != "cars"
            if self.view == "cars":
                selected = i == self.action_index
            if selected:
                highlight = pygame.Surface(rect.size, pygame.SRCALPHA)
                pygame.draw.rect(highlight, (40, 48, 62, 220), highlight.get_rect(), border_radius=8)
                pygame.draw.rect(highlight, ACCENT, highlight.get_rect(), width=1, border_radius=8)
                surface.blit(highlight, rect.topleft)
            colour = TEXT if selected else (204, 210, 220)
            if label == "SELECTED":
                colour = GOOD
            surface.blit(self.font_row.render(label, True, colour), (rect.x + 12, rect.y + 4))
            if hint:
                hint_text = self.font_small.render(hint, True, MUTED)
                surface.blit(hint_text, (rect.right - hint_text.get_width() - 12, rect.y + 9))
        if self.view == "paint":
            for i, (_name, label, hint) in enumerate(rows[:6]):
                rect = pygame.Rect(actions_rect.x + 10, actions_rect.y + 32 + i * 40,
                                   actions_rect.width - 20, 34)
                rgb = S.P.PAINTS[label]
                pygame.draw.rect(surface, tuple(int(c * 255) for c in rgb),
                                 pygame.Rect(rect.right - 34, rect.y + 6, 22, 22),
                                 border_radius=5)
        footer = ("Left/Right: choose car   Up/Down: choose action   Enter: apply   "
                  "Esc: back")
        surface.blit(self.font_small.render(footer, True, MUTED),
                     (panel.x + 26, panel.bottom - 32))
        return surface


def build_main_menu() -> MenuScreen:
    menu = MenuScreen("Smart Traffic India", "Ultimate 3D Driving Simulator  |  v" + S.VERSION)
    menu.add("Career Mode", ACTION, "10 missions", tag="career")
    menu.add("Free Drive", ACTION, "drive anywhere, no timers", tag="free")
    menu.add("Parking Challenges", ACTION, "4 marked lots", tag="parking")
    menu.add("Driving School", ACTION, "learn the road rules", tag="school")
    menu.add("Garage", ACTION, "cars, upgrades and paint", tag="garage")
    menu.add("Settings", ACTION, "graphics, audio and controls", tag="settings")
    menu.add("How to Play", ACTION, "controls and rules", tag="help")
    menu.add_gap()
    menu.add("Exit", ACTION, "quit to desktop", tag="exit")
    return menu


def build_pause_menu() -> MenuScreen:
    menu = MenuScreen("Paused", "The city keeps moving - take your time")
    menu.add("Resume", ACTION, tag="resume")
    menu.add("Restart / Respawn", ACTION, tag="restart")
    menu.add("Camera", CYCLE, value=lambda: "see settings", tag="camera")
    menu.add("Settings", ACTION, tag="settings")
    menu.add("How to Play", ACTION, tag="help")
    menu.add("Quit to Main Menu", ACTION, tag="mainmenu")
    return menu


def build_results_menu(title: str, subtitle: str) -> MenuScreen:
    menu = MenuScreen(title, subtitle)
    menu.add("Continue in Free Drive", ACTION, tag="continue")
    menu.add("Next Mission", ACTION, tag="next")
    menu.add("Retry", ACTION, tag="retry")
    menu.add("Garage", ACTION, tag="garage")
    menu.add("Main Menu", ACTION, tag="mainmenu")
    return menu


def build_settings_menu(settings: S.Settings) -> MenuScreen:
    menu = MenuScreen("Settings", "Changes are saved automatically")

    def cycle(name: str):
        return lambda step: settings.cycle_option(name, step or 1)

    def toggle(name: str):
        return lambda step=0: setattr(settings, name, not getattr(settings, name))

    def volume(step: int) -> None:
        settings.master_volume = max(0.0, min(1.0, settings.master_volume + step * 0.1))

    menu.add("Graphics quality", CYCLE, value=lambda: settings.quality.upper(),
             adjust=cycle("quality"), tag="quality")
    menu.add("Traffic density", CYCLE, value=lambda: settings.traffic_level.upper(),
             adjust=cycle("traffic_level"), tag="traffic_level")
    menu.add("Time of day", CYCLE, value=lambda: settings.time_of_day.upper(),
             adjust=cycle("time_of_day"), tag="time_of_day")
    menu.add("Weather", CYCLE, value=lambda: settings.weather.upper(),
             adjust=cycle("weather"), tag="weather")
    menu.add("Camera", CYCLE, value=lambda: settings.camera_mode.upper(),
             adjust=cycle("camera_mode"), tag="camera_mode")
    menu.add("Volume", SLIDER, value=lambda: f"{int(settings.master_volume * 100)}%",
             adjust=volume, tag="volume")
    menu.add_gap()
    menu.add("Minimap", TOGGLE, value=lambda: "ON" if settings.show_minimap else "OFF",
             adjust=toggle("show_minimap"), tag="show_minimap")
    menu.add("HUD", TOGGLE, value=lambda: "ON" if settings.show_hud else "OFF",
             adjust=toggle("show_hud"), tag="show_hud")
    menu.add("Show FPS", TOGGLE, value=lambda: "ON" if settings.show_fps else "OFF",
             adjust=toggle("show_fps"), tag="show_fps")
    menu.add("Headlights on at night", TOGGLE,
             value=lambda: "ON" if settings.headlights_auto else "OFF",
             adjust=toggle("headlights_auto"), tag="headlights_auto")
    menu.add("Invert steering", TOGGLE, value=lambda: "ON" if settings.invert_steering else "OFF",
             adjust=toggle("invert_steering"), tag="invert_steering")
    menu.add_gap()
    menu.add("Fullscreen", TOGGLE, value=lambda: "ON" if settings.fullscreen else "OFF",
             adjust=toggle("fullscreen"), tag="fullscreen")
    menu.add("Back", ACTION, tag="back")
    return menu


def build_help_menu(controls: Sequence[Tuple[str, str]]) -> MenuScreen:
    menu = MenuScreen("How to Play", "Left-hand traffic, Indian road conditions")
    for label, keys in controls:
        menu.add(label, INFO, hint=keys)
    menu.add_gap()
    menu.add("Back", ACTION, tag="back")
    return menu
