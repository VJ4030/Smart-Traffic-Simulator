"""Procedural signage: shop boards, hoardings, street name plates and warning signs.

Real Indian streets read as "Indian" because of their dense, colourful lettering.  This
module renders every sign into a single **texture atlas** at start-up using the font that
ships with pygame, so the game gets readable board text without shipping any binary asset
and without one texture per sign (that would be a draw-call disaster).

If pygame's font module is unavailable the atlas simply reports itself as empty and the
environment falls back to plain coloured boards (the game still runs).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pygame

UVRect = Tuple[float, float, float, float]

# --------------------------------------------------------------------------------------
# sign content
# --------------------------------------------------------------------------------------
SHOP_NAMES: List[str] = [
    "SHARMA KIRANA STORE", "GUPTA MEDICAL", "ANNAPURNA BHOJANALAY", "JAI HIND TRAVELS",
    "SAGAR MOBILE RECHARGE", "NEW BOMBAY TAILORS", "KRISHNA SWEETS", "OM SAI DENTING",
    "LAXMI GENERAL STORES", "PUNJABI DHABA", "SUNRISE ELECTRONICS", "GANESH AUTO PARTS",
    "VISHAL SUPER MARKET", "A1 CHAI & SNACKS", "MAHARASHTRA BAKERY", "SHREE RAM TIMBER",
    "LAKSHMI HARDWARE", "MODERN HAIR SALON", "INDIA POST", "STATE BANK ATM",
    "PATEL PHOTO STUDIO", "BALAJI TRANSPORT", "APNA BAZAR", "SATKAR DHABA",
]
HOARDING_TEXTS: List[str] = [
    "NOIDA BUILDCON  |  BOOKINGS OPEN", "SAVE FUEL - CARPOOL TODAY",
    "GYM KHOLO - BODY BANAO", "LUCKY JEWELLERS - 75 YEARS OF TRUST",
    "MEGA SALE - 50% OFF ON EVERYTHING", "DENTAL CARE - SMILE CLINIC",
    "INCOME TAX RETURN FILING", "SHOP LOCAL - SUPPORT SMALL BUSINESS",
]
STREET_SIGNS: List[str] = [
    "MG ROAD", "GANDHI CHOWK", "NEHRU MARG", "MARKET STREET", "STATION ROAD",
    "OLD CITY LANE", "HOSPITAL ROAD", "SCHOOL MARG",
]
WARNING_SIGNS: List[str] = [
    "SPEED LIMIT 40", "SCHOOL AHEAD", "HOSPITAL - NO HORN", "PEDESTRIAN CROSSING",
    "NO PARKING", "NO ENTRY", "GO SLOW", "WORK IN PROGRESS",
]

STYLE_SHOP = dict(bg=(238, 240, 236, 255), fg=(176, 32, 30, 255), size=30, pad=6, font="bold")
STYLE_SHOP_DARK = dict(bg=(28, 30, 38, 255), fg=(250, 214, 74, 255), size=30, pad=6, font="bold")
STYLE_BOARD = dict(bg=(18, 86, 156, 255), fg=(255, 255, 255, 255), size=32, pad=6, font="bold")
STYLE_HOARDING = dict(bg=(247, 245, 238, 255), fg=(24, 26, 32, 255), size=46, pad=10, font="bold")
STYLE_STREET = dict(bg=(26, 60, 110, 255), fg=(255, 255, 255, 255), size=30, pad=8, font="normal")
STYLE_WARNING = dict(bg=(244, 196, 40, 255), fg=(24, 24, 24, 255), size=30, pad=8, font="bold")
STYLE_EMERGENCY = dict(bg=(214, 226, 240, 255), fg=(180, 24, 24, 255), size=34, pad=8, font="bold")


class SignAtlas:
    """A shelf-packed texture atlas of text signs."""

    def __init__(self, width: int = 1024, height: int = 1024) -> None:
        self.width, self.height = width, height
        self.entries: Dict[str, UVRect] = {}
        self.available = False
        self.surface: Optional[pygame.Surface] = None
        self._cursor = [4, 4]
        self._row_height = 0
        self._font_cache: Dict[Tuple[int, str], pygame.font.Font] = {}
        try:
            if not pygame.font.get_init():
                pygame.font.init()
            self.surface = pygame.Surface((width, height), pygame.SRCALPHA, 32)
            self.surface.fill((255, 255, 255, 255))
            self.available = True
        except Exception:                                     # pragma: no cover
            self.available = False

    # ------------------------------------------------------------------ fonts
    def _font(self, size: int, style: str) -> pygame.font.Font:
        key = (size, style)
        font = self._font_cache.get(key)
        if font is None:
            try:
                font = pygame.font.Font(None, size)
                font.set_bold(style == "bold")
            except Exception:                                 # pragma: no cover
                font = pygame.font.SysFont(None, size)
            self._font_cache[key] = font
        return font

    # ------------------------------------------------------------------ packing
    def add(self, key: str, text: str, style: Dict) -> Optional[UVRect]:
        """Render ``text`` into the atlas and return its UV rect (or None if full)."""
        if not self.available or self.surface is None:
            return None
        if key in self.entries:
            return self.entries[key]
        font = self._font(int(style["size"]), str(style.get("font", "bold")))
        rendered = font.render(text, True, style["fg"], style["bg"])
        pad = int(style["pad"])
        w = rendered.get_width() + pad * 2
        h = rendered.get_height() + pad * 2
        if w >= self.width - 4:
            w = self.width - 8
        x, y = self._cursor
        if x + w + 4 > self.width:
            x = 4
            y += self._row_height + 4
            self._row_height = 0
        if y + h + 4 > self.height:
            return None                                       # atlas full - fall back
        self.surface.fill((style["bg"][0], style["bg"][1], style["bg"][2], 255),
                          pygame.Rect(x, y, w, h))
        self.surface.blit(rendered, (x + pad, y + pad))
        # thin darker frame so the board reads as a physical object
        pygame.draw.rect(self.surface, tuple(max(0, c - 60) for c in style["bg"][:3]),
                         pygame.Rect(x, y, w, h), 2)
        self._cursor = [x + w + 4, y]
        self._row_height = max(self._row_height, h)
        rect = (x / self.width, y / self.height, (x + w) / self.width, (y + h) / self.height)
        self.entries[key] = rect
        return rect

    def uv(self, key: str) -> Optional[UVRect]:
        return self.entries.get(key)

    def count(self) -> int:
        return len(self.entries)

    def rgb_bytes(self) -> bytes:
        """Raw RGB payload for ``glTexImage2D``."""
        if self.surface is None:
            return b""
        return pygame.image.tostring(self.surface, "RGB", False)


def build_atlas(max_entries: int = 40) -> SignAtlas:
    """Create the standard atlas used by the environment and HUD props."""
    atlas = SignAtlas()
    if not atlas.available:
        return atlas
    for i, name in enumerate(SHOP_NAMES):
        if i >= max_entries:
            break
        atlas.add(f"shop{i}", name, STYLE_SHOP if i % 3 else STYLE_SHOP_DARK)
    for i, text in enumerate(HOARDING_TEXTS):
        atlas.add(f"hoarding{i}", text, STYLE_HOARDING)
    for i, text in enumerate(STREET_SIGNS):
        atlas.add(f"street{i}", text, STYLE_STREET)
    for i, text in enumerate(WARNING_SIGNS):
        atlas.add(f"warn{i}", text, STYLE_WARNING)
    atlas.add("emergency", "CIVIL HOSPITAL", STYLE_EMERGENCY)
    atlas.add("police", "TRAFFIC POLICE", STYLE_BOARD)
    atlas.add("bus_stop", "BEST BUS STOP", STYLE_BOARD)
    atlas.add("petrol", "INDANE PETROL PUMP", STYLE_BOARD)
    atlas.add("station", "CENTRAL RAILWAY STATION", STYLE_BOARD)
    return atlas
