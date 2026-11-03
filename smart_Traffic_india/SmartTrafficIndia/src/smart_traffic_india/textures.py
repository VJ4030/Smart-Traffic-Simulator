"""Procedural textures - no binary asset files are shipped with the project.

Everything the world shader samples is generated from numpy noise at start-up:

====================  ==========================================================
name                  used by
====================  ==========================================================
``asphalt``           road surfaces (cracks, patches, tar seams)
``kerb``              kerbs, footpath edges
``zebra``             pedestrian crossings and stop lines
``dirt``              building plots, verges, market ground
``grass``             park strips and dividers
``rust``              industrial roofs and shutters
``signs``             the sign / hoarding atlas painted by :mod:`signage`
``window``            building window grids (lit at night)
====================  ==========================================================

The functions are pure (surface in, surface out) so they can be unit tested without a
display or a GL context.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pygame

from . import settings as S


def _surface(size: int) -> pygame.Surface:
    return pygame.Surface((size, size), pygame.SRCALPHA)


def _noise(rng: np.random.Generator, size: int, scale: float) -> np.ndarray:
    """Smooth value noise in 0..1 built from a small random grid (bilinear upsample)."""
    grid = max(2, int(size / max(scale, 1.0)))
    small = rng.random((grid, grid)).astype(np.float32)
    # bilinear upscale with linear interpolation in both axes
    xs = np.linspace(0, grid - 1, size, dtype=np.float32)
    ys = np.linspace(0, grid - 1, size, dtype=np.float32)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, grid - 1)
    y1 = np.clip(y0 + 1, 0, grid - 1)
    fx = (xs - x0)[None, :]
    fy = (ys - y0)[:, None]
    top = small[y0][:, x0] * (1 - fx) + small[y0][:, x1] * fx
    bottom = small[y1][:, x0] * (1 - fx) + small[y1][:, x1] * fx
    return top * (1 - fy) + bottom * fy


def _blit_array(surface: pygame.Surface, array: np.ndarray, alpha: float = 1.0) -> None:
    """Multiply the surface with a 0..1 grey array (cheap tinting / wear)."""
    data = np.clip(array * 255.0 * alpha, 0, 255).astype(np.uint8)
    rgb = np.dstack([data, data, data]).copy()
    rgb[:, :, 0] = np.clip(rgb[:, :, 0] * 1.02, 0, 255)
    if rgb.shape[0] != surface.get_height() or rgb.shape[1] != surface.get_width():
        rgb = rgb[:surface.get_height(), :surface.get_width()]
    overlay = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    # NOTE: no convert_alpha() here - textures must be buildable without a display so the
    # unit tests (and the texture preview tool) can run head-less.
    surface.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)


def asphalt(size: int = 256, seed: int = 11) -> pygame.Surface:
    """Worn, cracked, patchy Indian asphalt."""
    rng = np.random.default_rng(seed)
    surface = _surface(size)
    surface.fill((46, 46, 49, 255))
    coarse = _noise(rng, size, 26.0)
    fine = _noise(rng, size, 5.0)
    grain = rng.random((size, size)).astype(np.float32)
    shade = 0.72 + 0.34 * coarse + 0.18 * fine + 0.10 * grain
    _blit_array(surface, np.clip(shade, 0, 1.3))
    # tar seams and cracks
    for _ in range(6):
        x = rng.integers(0, size)
        y = rng.integers(0, size)
        points = [(int(x), int(y))]
        for _step in range(rng.integers(4, 11)):
            x = (x + rng.integers(-28, 29)) % size
            y = (y + rng.integers(-28, 29)) % size
            points.append((int(x), int(y)))
        pygame.draw.lines(surface, (30, 30, 33, 190), False, points,
                          int(rng.integers(1, 4)))
    # patched repairs - slightly lighter, rounded rectangles
    for _ in range(int(size / 32)):
        w = int(rng.integers(size // 10, size // 4))
        h = int(rng.integers(size // 12, size // 5))
        rect = pygame.Rect(int(rng.integers(0, size - w)), int(rng.integers(0, size - h)), w, h)
        shade = int(rng.integers(-14, 16))
        colour = (max(0, 52 + shade), max(0, 52 + shade), max(0, 55 + shade), 210)
        pygame.draw.rect(surface, colour, rect, border_radius=int(rng.integers(4, 14)))
    # pothole speckle
    for _ in range(int(size * 0.9)):
        x = int(rng.integers(0, size))
        y = int(rng.integers(0, size))
        r = int(rng.integers(1, 3))
        dark = int(rng.integers(18, 40))
        pygame.draw.circle(surface, (dark, dark, dark + 2, 200), (x, y), r)
    return surface


def kerb(size: int = 128, seed: int = 21) -> pygame.Surface:
    rng = np.random.default_rng(seed)
    surface = _surface(size)
    surface.fill((176, 174, 164, 255))
    grain = _noise(rng, size, 8.0)
    _blit_array(surface, 0.82 + 0.30 * grain)
    for i in range(0, size, size // 4):                        # black/white kerb blocks
        if (i // (size // 4)) % 2 == 0:
            pygame.draw.rect(surface, (58, 58, 60, 255), pygame.Rect(i, 0, size // 8, size))
    for _ in range(size // 3):                                 # grime
        x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
        pygame.draw.circle(surface, (120, 116, 108, 60), (x, y), int(rng.integers(2, 7)))
    return surface


def zebra(size: int = 128, seed: int = 31) -> pygame.Surface:
    surface = _surface(size)
    surface.fill((38, 38, 40, 255))
    stripe = size // 8
    for i in range(0, size, stripe * 2):
        pygame.draw.rect(surface, (222, 222, 214, 255), pygame.Rect(i, 0, stripe, size))
    rng = np.random.default_rng(seed)
    for _ in range(size // 2):                                 # paint wear
        x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
        pygame.draw.circle(surface, (44, 44, 46, 90), (x, y), int(rng.integers(1, 4)))
    return surface


def dirt(size: int = 128, seed: int = 41) -> pygame.Surface:
    rng = np.random.default_rng(seed)
    surface = _surface(size)
    surface.fill((124, 104, 74, 255))
    _blit_array(surface, 0.72 + 0.44 * _noise(rng, size, 18.0))
    for _ in range(size * 2):
        x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
        shade = int(rng.integers(90, 150))
        pygame.draw.circle(surface, (shade, shade - 12, shade - 30, 120), (x, y),
                           int(rng.integers(1, 4)))
    return surface


def grass(size: int = 128, seed: int = 51) -> pygame.Surface:
    rng = np.random.default_rng(seed)
    surface = _surface(size)
    surface.fill((64, 96, 52, 255))
    _blit_array(surface, 0.70 + 0.50 * _noise(rng, size, 10.0))
    for _ in range(size * 3):
        x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
        shade = int(rng.integers(60, 120))
        pygame.draw.line(surface, (shade - 30, shade, shade - 40, 150), (x, y), (x, y - 3))
    return surface


def rust(size: int = 128, seed: int = 61) -> pygame.Surface:
    rng = np.random.default_rng(seed)
    surface = _surface(size)
    surface.fill((150, 96, 60, 255))
    _blit_array(surface, 0.65 + 0.55 * _noise(rng, size, 16.0))
    for _ in range(size // 2):
        x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
        pygame.draw.circle(surface, (96, 60, 40, 120), (x, y), int(rng.integers(3, 12)))
    return surface


def window(size: int = 128, seed: int = 71) -> pygame.Surface:
    """Concrete wall with a window grid; the lit variant is handled by the material."""
    rng = np.random.default_rng(seed)
    surface = _surface(size)
    surface.fill((196, 190, 178, 255))
    _blit_array(surface, 0.85 + 0.25 * _noise(rng, size, 12.0))
    cols = 4
    cell = size // cols
    for row in range(cols):
        for col in range(cols):
            rect = pygame.Rect(col * cell + 6, row * cell + 6, cell - 12, cell - 16)
            pygame.draw.rect(surface, (26, 30, 40, 255), rect, border_radius=3)
            pygame.draw.rect(surface, (150, 148, 140, 255), rect, width=2, border_radius=3)
            pygame.draw.line(surface, (70, 74, 84, 255),
                             (rect.centerx, rect.y + 2), (rect.centerx, rect.bottom - 2), 1)
    return surface


def build_textures(quality: Dict[str, object], size: int = 256) -> Dict[str, pygame.Surface]:
    """Build every texture the renderer needs (the sign atlas comes from ``signage``)."""
    if not pygame.font.get_init():
        pygame.font.init()
    # low quality presets also use smaller textures - less bandwidth on weak GPUs
    traffic_scale = float(quality.get("traffic_scale", 1.0) or 1.0)
    resolution = max(64, int(size * (0.5 if traffic_scale < 0.6 else 1.0)))
    return {
        "asphalt": asphalt(resolution, 11),
        "kerb": kerb(max(64, resolution // 2), 21),
        "zebra": zebra(max(64, resolution // 2), 31),
        "dirt": dirt(max(64, resolution // 2), 41),
        "grass": grass(max(64, resolution // 2), 51),
        "rust": rust(max(64, resolution // 2), 61),
        "window": window(max(64, resolution // 2), 71),
    }


def palette_swatch(colour, size: int = 48) -> pygame.Surface:
    """A small flat colour tile (garage paint swatches, HUD icons)."""
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    rgb = tuple(int(max(0.0, min(1.0, c)) * 255) for c in colour[:3])
    surface.fill(rgb + (255,))
    pygame.draw.rect(surface, (255, 255, 255, 60), surface.get_rect(), width=2,
                     border_radius=6)
    return surface


def default_asphalt_colour() -> tuple:
    return S.P.ASPHALT
