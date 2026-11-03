"""Recognisable vehicle models - all generated procedurally, no external assets.

Modelling technique
-------------------
Cars are built as **lofted bodies**: a sequence of cross sections (``x``, bottom height,
top height, half width at the bottom, half width at the top) is skinned into a closed
surface.  Varying the section heights gives a real silhouette - bonnet, waistline, boot,
roof - and raising the *bottom* edge at the axle positions cuts genuine wheel arches.  On
top of that come a glass "greenhouse", bumpers, grille, lamps, mirrors, door handles and
details.  Wheels are separate meshes so they can be rotated hierarchically by the
renderer (rotation animation + steering).

Every model is data driven through :data:`STYLES`, so the six player cars and the eleven
AI traffic vehicles stay consistent but visually distinct (the auto-rickshaw has three
wheels and a canopy, the bus is 10.5 m long with a window band, the motorcycle carries a
rider silhouette, and so on).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .mesh import Material, MeshBuilder, MeshData
from .vehicle import PLAYER_SPECS, VehicleSpec
from . import settings as S

Vec3 = Tuple[float, float, float]

# --------------------------------------------------------------------------------------
# materials (colour resolution happens at draw time so that paint can be changed)
# --------------------------------------------------------------------------------------
PAINT = "paint"
PAINT_ALT = "paint_alt"
GLASS = "glass"
TYRE = "tyre"
HUB = "hub"
TRIM = "trim"
CHROME = "chrome"
HEADLIGHT = "headlight"
TAILLIGHT = "taillight"
INDICATOR = "indicator"
PLATE = "plate"
CUSHION = "cushion"
METAL = "metal"
CANVAS = "canvas"
BEACON_RED = "beacon_red"
BEACON_BLUE = "beacon_blue"
BEACON_AMBER = "beacon_amber"
LAMP = "lamp"          # always-on bright surface (not affected by headlight state)


def resolve_material(name: str, paint: Vec3 = (0.85, 0.35, 0.12),
                     lights_on: bool = False, brake: bool = False,
                     indicator: int = 0, hazards: bool = False,
                     beacons: bool = False, albedo: Optional[Vec3] = None) -> Material:
    """Map a semantic material name to a concrete :class:`Material`."""
    body = albedo if albedo is not None else paint
    alt = tuple(min(1.0, c * 1.25 + 0.05) for c in body) if albedo is None else body
    table: Dict[str, Material] = {
        PAINT: Material("paint", body, specular=0.65),
        PAINT_ALT: Material("paint_alt", alt, specular=0.55),
        GLASS: Material("glass", S.P.GLASS, specular=0.95),
        TYRE: Material("tyre", S.P.TYRE, specular=0.12),
        HUB: Material("hub", S.P.HUB, specular=0.75),
        TRIM: Material("trim", (0.09, 0.09, 0.10), specular=0.25),
        CHROME: Material("chrome", S.P.CHROME, specular=0.95),
        HEADLIGHT: Material("headlight", (0.26, 0.26, 0.24), specular=0.85,
                            emissive=0.0 if not lights_on else 1.0,
                            emissive_color=(1.0, 0.96, 0.82)),
        TAILLIGHT: Material("taillight", (0.35, 0.06, 0.05), specular=0.7,
                            emissive=0.0 if not brake else 0.9,
                            emissive_color=(1.0, 0.12, 0.08)),
        INDICATOR: Material("indicator", (0.42, 0.24, 0.05), specular=0.6,
                            emissive=1.0 if indicator else 0.0,
                            emissive_color=(1.0, 0.62, 0.08)),
        PLATE: Material("plate", (0.90, 0.90, 0.86), specular=0.1),
        CUSHION: Material("cushion", (0.16, 0.17, 0.19), specular=0.1),
        METAL: Material("metal", (0.45, 0.46, 0.48), specular=0.55),
        CANVAS: Material("canvas", (0.34, 0.38, 0.44), specular=0.1),
        LAMP: Material("lamp", (1.0, 0.98, 0.9), specular=0.4, emissive=1.0,
                       emissive_color=(1.0, 0.97, 0.86)),
        BEACON_RED: Material("beacon_red", (0.6, 0.08, 0.06), specular=0.8,
                             emissive=1.0 if beacons else 0.15,
                             emissive_color=(1.0, 0.10, 0.08)),
        BEACON_BLUE: Material("beacon_blue", (0.08, 0.16, 0.62), specular=0.8,
                              emissive=1.0 if beacons else 0.15,
                              emissive_color=(0.20, 0.45, 1.0)),
        BEACON_AMBER: Material("beacon_amber", (0.7, 0.42, 0.05), specular=0.8,
                               emissive=1.0 if beacons else 0.1,
                               emissive_color=(1.0, 0.55, 0.05)),
    }
    if hazards:
        table[INDICATOR] = Material("indicator", (0.42, 0.24, 0.05), specular=0.6,
                                    emissive=1.0, emissive_color=(1.0, 0.62, 0.08))
    return table.get(name, Material(name, body))


# --------------------------------------------------------------------------------------
# lofting helper
# --------------------------------------------------------------------------------------
Section = Tuple[float, float, float, float, float]   # x, y_bottom, y_top, half_w_bottom, half_w_top


def add_loft(builder: MeshBuilder, sections: Sequence[Section],
             top: str, side: str, bottom: str, front: str, back: str,
             side_glass: Optional[Sequence[int]] = None,
             glass: str = GLASS,
             no_front_cap: bool = False, no_back_cap: bool = False) -> None:
    """Skin a list of cross sections into a closed surface.

    ``side_glass`` lists the section indices whose side quad is glass (the windows).
    """
    if len(sections) < 2:
        return
    glass_idx = set(side_glass or [])
    quads = []           # (mat_name, verts, ref_center)
    for i in range(len(sections) - 1):
        x0, y0a, y1a, hw0a, hw1a = sections[i]
        x1, y0b, y1b, hw0b, hw1b = sections[i + 1]
        ref = ((x0 + x1) * 0.5, (y0a + y0b + y1a + y1b) * 0.25, 0.0)
        # top face
        quads.append((top, [(x0, y1a, hw1a), (x1, y1b, hw1b), (x1, y1b, -hw1b), (x0, y1a, -hw1a)], ref))
        # bottom face
        quads.append((bottom, [(x0, y0a, hw0a), (x1, y0b, hw0b), (x1, y0b, -hw0b), (x0, y0a, -hw0a)], ref))
        side_mat = glass if i in glass_idx else side
        # right side (+Z) and left side (-Z)
        quads.append((side_mat, [(x0, y0a, hw0a), (x1, y0b, hw0b), (x1, y1b, hw1b), (x0, y1a, hw1a)], ref))
        quads.append((side_mat, [(x0, y0a, -hw0a), (x1, y0b, -hw0b), (x1, y1b, -hw1b), (x0, y1a, -hw1a)], ref))
    # end caps
    if not no_front_cap:
        x, y0, y1, hwb, hwt = sections[-1]
        quads.append((front, [(x, y0, hwb), (x, y1, hwt), (x, y1, -hwt), (x, y0, -hwb)],
                      (x - 0.01, (y0 + y1) * 0.5, 0.0)))
    if not no_back_cap:
        x, y0, y1, hwb, hwt = sections[0]
        quads.append((back, [(x, y0, hwb), (x, y1, hwt), (x, y1, -hwt), (x, y0, -hwb)],
                      (x + 0.01, (y0 + y1) * 0.5, 0.0)))
    for mat_name, verts, ref in quads:
        builder.add_quad_auto(resolve_material(mat_name), verts, ref=ref)


# --------------------------------------------------------------------------------------
# per style geometry data
# --------------------------------------------------------------------------------------
#: body sections: (x, y_bottom, y_top, half_width_bottom, half_width_top)
STYLES: Dict[str, Dict[str, object]] = {
    "hatchback": {
        "body": [
            (-1.80, 0.34, 0.86, 0.66, 0.70),
            (-1.58, 0.40, 0.94, 0.76, 0.79),
            (-1.18, 0.42, 0.97, 0.78, 0.80),
            (-0.78, 0.40, 0.98, 0.80, 0.81),
            (-0.20, 0.20, 0.99, 0.81, 0.82),
            (0.40, 0.20, 0.98, 0.81, 0.82),
            (0.78, 0.40, 0.96, 0.80, 0.81),
            (1.18, 0.42, 0.92, 0.79, 0.80),
            (1.58, 0.40, 0.85, 0.75, 0.77),
            (1.80, 0.36, 0.78, 0.64, 0.68),
        ],
        "cabin": [
            (0.42, 0.90, 1.00, 0.74, 0.72),
            (0.16, 0.90, 1.20, 0.76, 0.71),
            (-0.08, 0.90, 1.40, 0.77, 0.68),
            (-0.62, 0.90, 1.49, 0.77, 0.68),
            (-1.06, 0.90, 1.47, 0.76, 0.67),
            (-1.40, 0.88, 1.26, 0.74, 0.63),
            (-1.58, 0.86, 1.00, 0.70, 0.60),
        ],
        "cabin_glass_sides": [1, 2, 3, 4],
        "roof_rails": False, "spoiler": False, "roof_box": None,
        "mirror": (0.34, 1.02, 0.86),
        "tail_x": -1.82, "tail_y": 0.80, "tail_hw": 0.62,
        "head_x": 1.82, "head_y": 0.70, "head_hw": 0.58,
    },
    "sedan": {
        "body": [
            (-2.16, 0.34, 0.98, 0.72, 0.76),
            (-1.94, 0.38, 1.02, 0.82, 0.85),
            (-1.40, 0.40, 1.04, 0.85, 0.87),
            (-0.90, 0.40, 1.05, 0.86, 0.87),
            (-0.30, 0.20, 1.06, 0.87, 0.88),
            (0.50, 0.20, 1.04, 0.87, 0.88),
            (1.00, 0.40, 0.98, 0.85, 0.86),
            (1.50, 0.42, 0.90, 0.83, 0.84),
            (1.94, 0.40, 0.82, 0.78, 0.80),
            (2.16, 0.36, 0.76, 0.68, 0.71),
        ],
        "cabin": [
            (0.58, 0.98, 1.06, 0.78, 0.76),
            (0.30, 0.98, 1.26, 0.81, 0.76),
            (0.02, 0.98, 1.42, 0.82, 0.72),
            (-0.60, 0.98, 1.46, 0.82, 0.72),
            (-1.05, 0.98, 1.42, 0.81, 0.70),
            (-1.45, 0.98, 1.20, 0.79, 0.66),
            (-1.72, 0.98, 1.05, 0.75, 0.62),
        ],
        "cabin_glass_sides": [1, 2, 3, 4],
        "roof_rails": False, "spoiler": False, "roof_box": None,
        "mirror": (0.52, 1.08, 0.90),
        "tail_x": -2.18, "tail_y": 0.86, "tail_hw": 0.66,
        "head_x": 2.16, "head_y": 0.68, "head_hw": 0.62,
    },
    "suv": {
        "body": [
            (-2.26, 0.44, 1.24, 0.80, 0.86),
            (-2.06, 0.50, 1.28, 0.88, 0.92),
            (-1.48, 0.50, 1.30, 0.91, 0.94),
            (-0.92, 0.50, 1.31, 0.92, 0.94),
            (-0.30, 0.26, 1.32, 0.93, 0.94),
            (0.60, 0.26, 1.30, 0.93, 0.94),
            (1.14, 0.50, 1.24, 0.92, 0.93),
            (1.62, 0.52, 1.16, 0.90, 0.91),
            (2.06, 0.50, 1.06, 0.86, 0.88),
            (2.26, 0.46, 0.98, 0.78, 0.80),
        ],
        "cabin": [
            (0.80, 1.22, 1.34, 0.84, 0.82),
            (0.46, 1.22, 1.56, 0.86, 0.82),
            (0.16, 1.22, 1.74, 0.87, 0.80),
            (-0.80, 1.22, 1.78, 0.87, 0.81),
            (-1.50, 1.22, 1.76, 0.86, 0.80),
            (-1.94, 1.22, 1.54, 0.84, 0.76),
            (-2.16, 1.22, 1.34, 0.78, 0.68),
        ],
        "cabin_glass_sides": [1, 2, 3, 4],
        "roof_rails": True, "spoiler": False, "roof_box": None,
        "mirror": (0.80, 1.40, 0.98),
        "tail_x": -2.28, "tail_y": 1.06, "tail_hw": 0.72,
        "head_x": 2.26, "head_y": 0.88, "head_hw": 0.70,
    },
    "sports": {
        "body": [
            (-2.19, 0.30, 0.72, 0.80, 0.86),
            (-1.98, 0.34, 0.80, 0.90, 0.94),
            (-1.44, 0.36, 0.84, 0.94, 0.95),
            (-0.84, 0.36, 0.86, 0.95, 0.96),
            (-0.20, 0.18, 0.86, 0.95, 0.96),
            (0.60, 0.18, 0.84, 0.95, 0.96),
            (1.16, 0.36, 0.78, 0.94, 0.95),
            (1.70, 0.38, 0.70, 0.91, 0.92),
            (2.02, 0.36, 0.62, 0.86, 0.88),
            (2.19, 0.32, 0.56, 0.76, 0.80),
        ],
        "cabin": [
            (0.40, 0.80, 0.86, 0.80, 0.78),
            (0.10, 0.80, 1.02, 0.84, 0.80),
            (-0.20, 0.80, 1.18, 0.85, 0.78),
            (-0.78, 0.80, 1.22, 0.85, 0.80),
            (-1.20, 0.80, 1.18, 0.84, 0.76),
            (-1.62, 0.80, 1.00, 0.82, 0.68),
            (-1.92, 0.78, 0.84, 0.76, 0.62),
        ],
        "cabin_glass_sides": [1, 2, 3],
        "roof_rails": False, "spoiler": True, "roof_box": None,
        "mirror": (0.36, 0.92, 0.98),
        "tail_x": -2.21, "tail_y": 0.66, "tail_hw": 0.76,
        "head_x": 2.19, "head_y": 0.50, "head_hw": 0.72,
    },
    "utility": {
        "body": [
            (-2.34, 0.42, 1.44, 0.78, 0.84),
            (-2.10, 0.50, 1.52, 0.86, 0.90),
            (-1.40, 0.50, 1.56, 0.89, 0.90),
            (-0.80, 0.30, 1.60, 0.90, 0.91),
            (-0.20, 0.30, 1.62, 0.91, 0.91),
            (0.70, 0.30, 1.60, 0.91, 0.91),
            (1.30, 0.50, 1.52, 0.90, 0.90),
            (1.85, 0.52, 1.38, 0.88, 0.89),
            (2.20, 0.50, 1.18, 0.84, 0.86),
            (2.34, 0.46, 1.04, 0.76, 0.78),
        ],
        "cabin": [
            (0.72, 1.40, 1.58, 0.84, 0.86),
            (0.44, 1.40, 1.82, 0.87, 0.88),
            (0.18, 1.40, 1.96, 0.88, 0.88),
            (-0.90, 1.40, 1.98, 0.89, 0.89),
            (-1.70, 1.40, 1.96, 0.88, 0.88),
            (-2.06, 1.40, 1.80, 0.86, 0.84),
            (-2.30, 1.40, 1.52, 0.80, 0.74),
        ],
        "cabin_glass_sides": [0, 1, 6],
        "roof_rails": True, "spoiler": False, "roof_box": "carrier",
        "mirror": (0.70, 1.62, 0.98),
        "tail_x": -2.36, "tail_y": 1.30, "tail_hw": 0.70,
        "head_x": 2.34, "head_y": 0.92, "head_hw": 0.68,
    },
    "taxi": {
        "body": [
            (-2.10, 0.34, 0.98, 0.70, 0.74),
            (-1.88, 0.38, 1.02, 0.79, 0.83),
            (-1.36, 0.40, 1.04, 0.83, 0.85),
            (-0.86, 0.40, 1.05, 0.84, 0.85),
            (-0.30, 0.20, 1.06, 0.84, 0.86),
            (0.46, 0.20, 1.05, 0.84, 0.86),
            (0.98, 0.40, 0.99, 0.83, 0.84),
            (1.46, 0.42, 0.92, 0.81, 0.82),
            (1.90, 0.40, 0.84, 0.76, 0.78),
            (2.10, 0.36, 0.78, 0.66, 0.70),
        ],
        "cabin": [
            (0.54, 0.98, 1.08, 0.76, 0.74),
            (0.26, 0.98, 1.28, 0.79, 0.74),
            (0.00, 0.98, 1.46, 0.80, 0.71),
            (-0.58, 0.98, 1.52, 0.80, 0.71),
            (-1.02, 0.98, 1.48, 0.79, 0.69),
            (-1.40, 0.98, 1.26, 0.77, 0.65),
            (-1.66, 0.98, 1.08, 0.73, 0.60),
        ],
        "cabin_glass_sides": [1, 2, 3, 4],
        "roof_rails": False, "spoiler": False, "roof_box": "taxi_sign",
        "mirror": (0.50, 1.10, 0.88),
        "tail_x": -2.12, "tail_y": 0.88, "tail_hw": 0.64,
        "head_x": 2.10, "head_y": 0.70, "head_hw": 0.60,
    },
}

#: traffic vehicles that reuse a car silhouette from :data:`STYLES`
CAR_TRAFFIC_KINDS = {
    "hatchback": "hatchback",
    "sedan": "sedan",
    "suv": "suv",
    "taxi": "taxi",
}


# --------------------------------------------------------------------------------------
# car model
# --------------------------------------------------------------------------------------
@dataclass
class VehicleModel:
    """A renderable vehicle: body mesh + one wheel mesh + wheel positions."""

    key: str
    body: MeshData
    wheel: MeshData
    wheels: List[Tuple[float, float, float]]
    length: float
    width: float
    height: float
    wheel_radius: float
    wheel_width: float
    light_anchors: Dict[str, List[Vec3]] = field(default_factory=dict)

    @property
    def half_length(self) -> float:
        return self.length * 0.5

    @property
    def half_width(self) -> float:
        return self.width * 0.5

    @property
    def radius(self) -> float:
        return 0.5 * math.sqrt(self.length ** 2 + self.height ** 2)

    def triangles(self) -> int:
        return self.body.triangle_count + 4 * self.wheel.triangle_count


def build_wheel(radius: float, width: float, segments: int = 14) -> MeshData:
    """A single wheel: tyre cylinder + two hub caps (drawn 4x per vehicle)."""
    b = MeshBuilder("wheel")
    b.add_cylinder(resolve_material(TYRE), radius, width * 0.5, (0, 0, 0), segments,
                   axis="z", caps=True, cap_material=resolve_material(HUB))
    for side in (-1, 1):
        b.add_cylinder(resolve_material(HUB), radius * 0.58, 0.02,
                       (0, 0, side * (width * 0.5 + 0.005)), 8, axis="z", caps=True)
    return b.build("wheel")


def _detail_lights_and_trim(b: MeshBuilder, style: Dict[str, object], hw: float) -> None:
    """Bumpers, grille, lamps, number plates and side trim shared by all cars."""
    tail_x = float(style["tail_x"])          # type: ignore[arg-type]
    tail_y = float(style["tail_y"])          # type: ignore[arg-type]
    tail_hw = float(style["tail_hw"])        # type: ignore[arg-type]
    head_x = float(style["head_x"])          # type: ignore[arg-type]
    head_y = float(style["head_y"])          # type: ignore[arg-type]
    head_hw = float(style["head_hw"])        # type: ignore[arg-type]

    # front bumper + grille
    b.add_box(resolve_material(TRIM), (0.22, 0.26, hw * 2 - 0.10),
              (head_x - 0.06, head_y - 0.22, 0.0))
    b.add_box(resolve_material(CHROME), (0.10, 0.16, hw * 1.15),
              (head_x + 0.02, head_y + 0.06, 0.0))
    b.add_box(resolve_material(TRIM), (0.08, 0.11, hw * 0.72),
              (head_x + 0.03, head_y + 0.24, 0.0))
    # head lamps + fog lamps + indicators
    for side in (-1, 1):
        b.add_box(resolve_material(HEADLIGHT), (0.14, 0.16, 0.34),
                  (head_x - 0.02, head_y + 0.02, side * (head_hw - 0.18)))
        b.add_box(resolve_material(HEADLIGHT), (0.10, 0.10, 0.16),
                  (head_x - 0.06, head_y - 0.20, side * (head_hw - 0.30)))
        b.add_box(resolve_material(INDICATOR), (0.10, 0.11, 0.18),
                  (head_x - 0.04, head_y + 0.02, side * (head_hw + 0.06)))
        # tail lamps
        b.add_box(resolve_material(TAILLIGHT), (0.12, 0.18, 0.30),
                  (tail_x + 0.04, tail_y - 0.02, side * (tail_hw - 0.14)))
        b.add_box(resolve_material(TAILLIGHT), (0.10, 0.10, 0.14),
                  (tail_x + 0.04, tail_y + 0.16, side * (tail_hw - 0.10)))
    # rear bumper
    b.add_box(resolve_material(TRIM), (0.20, 0.24, hw * 2 - 0.12),
              (tail_x + 0.08, tail_y - 0.26, 0.0))
    # number plates
    b.add_box(resolve_material(PLATE), (0.05, 0.13, 0.44), (head_x + 0.04, head_y - 0.26, 0.0))
    b.add_box(resolve_material(PLATE), (0.05, 0.13, 0.44), (tail_x - 0.02, tail_y - 0.32, 0.0))
    # side trim / door handles / pillars
    for side in (-1, 1):
        b.add_box(resolve_material(TRIM), (2.5, 0.05, 0.05), (0.0, 0.34, side * hw * 1.01))
        b.add_box(resolve_material(TRIM), (0.16, 0.05, 0.06), (0.55, 0.72, side * (hw + 0.02)))
        b.add_box(resolve_material(TRIM), (0.16, 0.05, 0.06), (-0.75, 0.72, side * (hw + 0.02)))
    # exhaust
    b.add_cylinder(resolve_material(METAL), 0.055, 0.10, (tail_x - 0.02, tail_y - 0.42, -hw * 0.45),
                   8, axis="x", caps=True)


def build_car_model(key: str, spec: VehicleSpec) -> VehicleModel:
    """Build a complete car model from the style table."""
    style = STYLES[key if key in STYLES else "sedan"]
    b = MeshBuilder(f"car_{key}")
    hw = spec.width * 0.5
    r = spec.wheel_radius

    # ---- lower body (bonnet / waistline / boot) with real wheel arches ---------------
    add_loft(b, style["body"], top=PAINT, side=PAINT, bottom=TRIM,  # type: ignore[arg-type]
             front=PAINT, back=PAINT)
    # ---- greenhouse: glass sides, painted roof --------------------------------------
    add_loft(b, style["cabin"], top=PAINT, side=PAINT, bottom=TRIM,  # type: ignore[arg-type]
             front=GLASS, back=GLASS,
             side_glass=style["cabin_glass_sides"])                  # type: ignore[arg-type]
    # ---- roof / bonnet detailing -----------------------------------------------------
    cabin = style["cabin"]                                           # type: ignore[assignment]
    roof_mid_x = (cabin[2][0] + cabin[3][0]) * 0.5
    roof_y = cabin[2][2]
    b.add_box(resolve_material(PAINT_ALT), (abs(cabin[3][0] - cabin[2][0]) * 0.9, 0.03, hw * 1.34),
              (roof_mid_x, roof_y + 0.02, 0.0))
    if style["roof_rails"]:
        for side in (-1, 1):
            b.add_box(resolve_material(TRIM), (abs(cabin[4][0] - cabin[1][0]), 0.06, 0.06),
                      ((cabin[1][0] + cabin[4][0]) * 0.5, roof_y + 0.07, side * hw * 0.55))
    if style["spoiler"]:
        b.add_box(resolve_material(PAINT_ALT), (0.34, 0.05, hw * 1.7), (-1.92, 1.02, 0.0), top_scale=0.8)
        for side in (-1, 1):
            b.add_box(resolve_material(TRIM), (0.10, 0.16, 0.08), (-1.82, 0.94, side * hw * 0.62))
    if style["roof_box"] == "taxi_sign":
        b.add_box(resolve_material(PLATE), (0.62, 0.20, 0.30), (roof_mid_x, roof_y + 0.12, 0.0))
        b.add_box(resolve_material(TRIM), (0.54, 0.05, 0.06), (roof_mid_x, roof_y + 0.23, 0.0))
    if style["roof_box"] == "carrier":
        for side in (-1, 1):
            b.add_box(resolve_material(METAL), (3.0, 0.08, 0.08), (-0.9, roof_y + 0.10, side * hw * 0.62))
        b.add_box(resolve_material(METAL), (2.8, 0.06, hw * 1.3), (-0.9, roof_y + 0.16, 0.0))

    # ---- mirrors, wipers, plates, lamps ----------------------------------------------
    mx, my, mz = style["mirror"]                                    # type: ignore[misc]
    for side in (-1, 1):
        b.add_box(resolve_material(TRIM), (0.14, 0.05, 0.14), (mx + 0.08, my + 0.02, side * mz * 0.98))
        b.add_box(resolve_material(PAINT_ALT), (0.10, 0.13, 0.06), (mx + 0.16, my + 0.03, side * (mz + 0.06)))
    # windscreen wipers
    b.add_box(resolve_material(TRIM), (0.02, 0.02, 0.36), (cabin[0][0] + 0.06, cabin[0][2] - 0.03, 0.18))
    _detail_lights_and_trim(b, style, hw)

    # ---- driver silhouette (visible through the glass) -------------------------------
    b.add_box(resolve_material(CUSHION), (0.36, 0.44, 0.44), (-0.35, 0.98, 0.28))
    b.add_sphere(resolve_material(CUSHION), 0.13, (-0.32, 1.28, 0.28), 8, 6)

    body = b.build(f"car_{key}")
    wheel = build_wheel(r, max(0.18, spec.width * 0.145))
    x_off = spec.wheelbase * 0.5
    z_off = spec.width * 0.5 - r * 0.45
    wheels = [(x_off, r, z_off), (x_off, r, -z_off), (-x_off, r, z_off), (-x_off, r, -z_off)]
    anchors = {
        "head": [(float(style["head_x"]) - 0.05, float(style["head_y"]) + 0.02,
                  s * (float(style["head_hw"]) - 0.18)) for s in (-1, 1)],
        "tail": [(float(style["tail_x"]) + 0.05, float(style["tail_y"]) - 0.02,
                  s * (float(style["tail_hw"]) - 0.14)) for s in (-1, 1)],
        "indicator_front": [(float(style["head_x"]) - 0.05, float(style["head_y"]) + 0.02,
                             s * (float(style["head_hw"]) + 0.06)) for s in (-1, 1)],
        "indicator_rear": [(float(style["tail_x"]) + 0.05, float(style["tail_y"]) + 0.10,
                            s * (float(style["tail_hw"]) + 0.02)) for s in (-1, 1)],
    }
    return VehicleModel(key, body, wheel, wheels, spec.length, spec.width, spec.height,
                        r, max(0.18, spec.width * 0.145), anchors)


# --------------------------------------------------------------------------------------
# traffic vehicles
# --------------------------------------------------------------------------------------
def _traffic_auto_rickshaw() -> VehicleModel:
    """Three wheeler: one front wheel, two rear, canopy + open passenger cabin."""
    b = MeshBuilder("auto")
    hw = 0.68
    # floor plate + chassis
    b.add_box(resolve_material(TRIM), (2.35, 0.10, hw * 2), (0.05, 0.40, 0.0))
    # front cowl (bonnet-less nose with the headlamp)
    add_loft(b, [
        (1.20, 0.30, 0.78, 0.32, 0.34),
        (1.05, 0.26, 0.86, 0.44, 0.46),
        (0.86, 0.24, 0.92, 0.52, 0.55),
    ], top=PAINT, side=PAINT, bottom=TRIM, front=PAINT, back=TRIM)
    # passenger body walls
    add_loft(b, [
        (0.80, 0.44, 1.18, 0.60, 0.62),
        (0.10, 0.44, 1.24, 0.64, 0.66),
        (-0.80, 0.44, 1.26, 0.66, 0.68),
        (-1.05, 0.46, 1.24, 0.62, 0.64),
        (-1.20, 0.48, 1.10, 0.52, 0.54),
    ], top=CUSHION, side=PAINT, bottom=TRIM, front=PAINT, back=PAINT)
    # canopy roof
    add_loft(b, [
        (0.86, 1.06, 1.20, 0.64, 0.60),
        (0.30, 1.12, 1.52, 0.66, 0.60),
        (-0.55, 1.12, 1.58, 0.66, 0.62),
        (-1.06, 1.08, 1.48, 0.62, 0.54),
        (-1.24, 1.04, 1.22, 0.52, 0.42),
    ], top=PAINT_ALT, side=PAINT, bottom=CUSHION, front=GLASS, back=GLASS)
    # window frames
    for side in (-1, 1):
        b.add_box(resolve_material(TRIM), (1.5, 0.06, 0.05), (-0.15, 1.30, side * hw * 0.99))
        b.add_box(resolve_material(TRIM), (0.06, 0.5, 0.05), (0.60, 1.18, side * hw * 0.99))
        b.add_box(resolve_material(TRIM), (0.06, 0.5, 0.05), (-0.95, 1.18, side * hw * 0.99))
    # windscreen
    b.add_box(resolve_material(GLASS), (0.06, 0.52, hw * 1.75), (0.78, 1.16, 0.0), top_scale=0.85)
    # bench seat + headrest
    b.add_box(resolve_material(CUSHION), (0.62, 0.16, hw * 1.7), (-0.55, 0.72, 0.0))
    b.add_box(resolve_material(CUSHION), (0.14, 0.42, hw * 1.7), (-0.92, 0.95, 0.0))
    # head lamp, indicator, number plate, mirror
    b.add_box(resolve_material(HEADLIGHT), (0.12, 0.18, 0.26), (1.18, 0.72, 0.0))
    for side in (-1, 1):
        b.add_box(resolve_material(INDICATOR), (0.10, 0.10, 0.12), (1.10, 0.60, side * 0.34))
        b.add_box(resolve_material(TAILLIGHT), (0.10, 0.14, 0.16), (-1.19, 0.74, side * 0.40))
        b.add_box(resolve_material(TRIM), (0.05, 0.30, 0.05), (0.92, 1.10, side * 0.40))
        b.add_box(resolve_material(CHROME), (0.05, 0.14, 0.10), (0.92, 1.26, side * 0.44))
    b.add_box(resolve_material(PLATE), (0.05, 0.12, 0.34), (1.20, 0.52, 0.0))
    b.add_box(resolve_material(PLATE), (0.05, 0.12, 0.34), (-1.23, 0.86, 0.0))
    body = b.build("auto")
    wheel = build_wheel(0.26, 0.15, 12)
    # a rickshaw has a single front wheel and a pair of rear wheels
    wheels = [(0.95, 0.26, 0.0), (-0.85, 0.26, 0.56), (-0.85, 0.26, -0.56)]
    anchors = {"head": [(1.24, 0.72, 0.0)], "tail": [(-1.25, 0.86, s * 0.4) for s in (-1, 1)]}
    return VehicleModel("auto", body, wheel, wheels, 2.6, 1.36, 1.75, 0.26, 0.15, anchors)


def _traffic_motorcycle() -> VehicleModel:
    b = MeshBuilder("bike")
    hw = 0.34
    b.add_box(resolve_material(TRIM), (0.90, 0.16, hw * 1.4), (-0.10, 0.62, 0.0))     # engine
    b.add_box(resolve_material(PAINT), (0.70, 0.22, hw * 1.3), (0.28, 0.78, 0.0))     # tank
    b.add_box(resolve_material(CUSHION), (0.66, 0.14, hw * 1.2), (-0.62, 0.84, 0.0))  # seat
    b.add_box(resolve_material(PAINT), (0.55, 0.10, hw * 1.1), (-1.02, 0.70, 0.0))    # tail
    b.add_box(resolve_material(TRIM), (0.10, 0.60, 0.10), (0.72, 0.98, 0.0))          # fork
    b.add_box(resolve_material(CHROME), (0.06, 0.06, 0.62), (0.78, 1.12, 0.0))        # handlebar
    b.add_box(resolve_material(HEADLIGHT), (0.12, 0.16, 0.20), (0.84, 1.00, 0.0))
    b.add_box(resolve_material(TAILLIGHT), (0.08, 0.08, 0.14), (-1.26, 0.78, 0.0))
    b.add_box(resolve_material(METAL), (0.60, 0.06, 0.08), (-0.60, 0.62, 0.0))
    b.add_box(resolve_material(PLATE), (0.05, 0.10, 0.24), (-1.28, 0.66, 0.0))
    for side in (-1, 1):
        b.add_box(resolve_material(TRIM), (0.16, 0.06, 0.16), (0.72, 0.62, side * 0.20))  # foot rest
        b.add_cylinder(resolve_material(CHROME), 0.03, 0.10, (-0.35, 0.42, side * 0.22), 6, axis="x")
    # ---- rider silhouette (helmet + torso + legs) ------------------------------------
    b.add_box(resolve_material(PAINT_ALT), (0.34, 0.52, 0.42), (-0.25, 1.16, 0.0))     # torso
    b.add_box(resolve_material(CUSHION), (0.20, 0.36, 0.16), (-0.05, 0.92, 0.24))     # leg right
    b.add_box(resolve_material(CUSHION), (0.20, 0.36, 0.16), (-0.05, 0.92, -0.24))    # leg left
    b.add_box(resolve_material(CUSHION), (0.14, 0.44, 0.14), (0.22, 1.10, 0.20))      # arm
    b.add_box(resolve_material(CUSHION), (0.14, 0.44, 0.14), (0.22, 1.10, -0.20))
    b.add_sphere(Material("helmet", (0.14, 0.15, 0.18), specular=0.8),
                 0.16, (-0.20, 1.52, 0.0), 10, 7)                                     # helmet
    b.add_box(resolve_material(GLASS), (0.10, 0.10, 0.24), (-0.06, 1.52, 0.0))        # visor
    body = b.build("bike")
    wheel = build_wheel(0.30, 0.11, 12)
    model = VehicleModel("bike", body, wheel, [(0.72, 0.30, 0.0), (-0.86, 0.30, 0.0)],
                         2.05, 0.72, 1.70, 0.30, 0.11,
                         {"head": [(0.88, 1.00, 0.0)], "tail": [(-1.30, 0.78, 0.0)]})
    return model


def _traffic_bus() -> VehicleModel:
    b = MeshBuilder("bus")
    hw = 1.25
    # main body (rounded front, flat sides, window band, rounded roof)
    body_sections = [
        (-5.25, 0.95, 3.10, 1.10, 1.06),
        (-4.60, 0.80, 3.20, 1.22, 1.18),
        (-2.00, 0.80, 3.25, 1.25, 1.20),
        (2.00, 0.80, 3.25, 1.25, 1.20),
        (4.20, 0.80, 3.22, 1.25, 1.20),
        (4.90, 0.85, 3.10, 1.24, 1.18),
        (5.25, 1.00, 2.80, 1.16, 1.05),
    ]
    add_loft(b, body_sections, top=PAINT, side=PAINT, bottom=TRIM, front=PAINT, back=PAINT)
    # livery stripe + window band
    for side in (-1, 1):
        b.add_box(resolve_material(PAINT_ALT), (10.1, 0.34, 0.06), (0.0, 1.35, side * hw * 1.005))
        b.add_box(resolve_material(GLASS), (9.4, 0.95, 0.06), (0.0, 2.45, side * hw * 1.005))
        for i in range(7):
            b.add_box(resolve_material(TRIM), (0.10, 1.05, 0.08), (-4.4 + i * 1.45, 2.45, side * hw * 1.01))
        # wheels arches strip
        b.add_box(resolve_material(TRIM), (10.1, 0.30, 0.05), (0.0, 0.95, side * hw * 1.005))
    # front face: windscreen, destination board, lamps
    b.add_box(resolve_material(GLASS), (0.10, 1.30, 2.30), (5.20, 2.35, 0.0))
    b.add_box(resolve_material(PLATE), (0.08, 0.34, 1.60), (5.24, 3.02, 0.0))
    b.add_box(resolve_material(TRIM), (0.22, 0.60, 2.40), (5.14, 1.20, 0.0))
    for side in (-1, 1):
        b.add_box(resolve_material(HEADLIGHT), (0.14, 0.26, 0.44), (5.22, 1.28, side * 0.86))
        b.add_box(resolve_material(INDICATOR), (0.12, 0.16, 0.24), (5.22, 1.02, side * 1.02))
        b.add_box(resolve_material(TAILLIGHT), (0.12, 0.30, 0.30), (-5.28, 1.40, side * 0.88))
        b.add_box(resolve_material(TRIM), (0.08, 0.90, 0.08), (5.10, 2.10, side * 1.40))
        b.add_box(resolve_material(CHROME), (0.06, 0.26, 0.16), (5.06, 2.55, side * 1.48))   # mirror
    # entry door
    b.add_box(resolve_material(GLASS), (0.90, 1.70, 0.10), (4.30, 1.95, -1.26))
    b.add_box(resolve_material(TRIM), (1.00, 0.10, 0.12), (4.30, 2.85, -1.26))
    b.add_box(resolve_material(PLATE), (0.06, 0.16, 0.60), (-5.28, 0.90, 0.0))
    body = b.build("bus")
    wheel = build_wheel(0.52, 0.30, 16)
    wheels = [(3.40, 0.52, 1.10), (3.40, 0.52, -1.10),
              (-3.10, 0.52, 1.10), (-3.10, 0.52, -1.10),
              (-4.35, 0.52, 1.10), (-4.35, 0.52, -1.10)]
    model = VehicleModel("bus", body, wheel, wheels, 10.5, 2.5, 3.3, 0.52, 0.30,
                         {"head": [(5.26, 1.28, s * 0.86) for s in (-1, 1)],
                          "tail": [(-5.30, 1.40, s * 0.88) for s in (-1, 1)]})
    return model


def _traffic_truck() -> VehicleModel:
    b = MeshBuilder("truck")
    hw = 1.12
    # chassis rails
    b.add_box(resolve_material(METAL), (8.0, 0.22, 1.9), (-0.3, 0.80, 0.0))
    # cab
    add_loft(b, [
        (1.90, 0.95, 2.75, 1.05, 1.10),
        (1.20, 0.95, 2.80, 1.10, 1.12),
        (0.80, 1.55, 2.80, 1.12, 1.12),
    ], top=PAINT_ALT, side=PAINT, bottom=TRIM, front=PAINT, back=TRIM)
    b.add_box(resolve_material(GLASS), (0.10, 0.90, 2.00), (1.86, 2.20, 0.0))
    for side in (-1, 1):
        b.add_box(resolve_material(GLASS), (0.70, 0.70, 0.08), (1.40, 2.20, side * hw * 1.005))
        b.add_box(resolve_material(HEADLIGHT), (0.12, 0.24, 0.40), (1.94, 1.30, side * 0.70))
        b.add_box(resolve_material(INDICATOR), (0.10, 0.14, 0.20), (1.94, 1.06, side * 0.95))
        b.add_box(resolve_material(TRIM), (0.08, 1.00, 0.08), (1.80, 2.30, side * 1.34))
        b.add_box(resolve_material(CHROME), (0.06, 0.28, 0.16), (1.74, 2.66, side * 1.42))
    b.add_box(resolve_material(TRIM), (0.24, 0.55, 2.20), (1.80, 1.32, 0.0))
    b.add_box(resolve_material(PLATE), (0.06, 0.16, 0.70), (1.96, 1.14, 0.0))
    # cargo box with ribs + tarpaulin look
    b.add_box(resolve_material(CANVAS), (5.60, 2.30, 2.20), (-2.60, 2.20, 0.0))
    b.add_box(resolve_material(METAL), (5.60, 0.14, 2.34), (-2.60, 3.38, 0.0))
    for side in (-1, 1):
        for i in range(6):
            b.add_box(resolve_material(METAL), (0.12, 2.20, 0.10), (-5.0 + i * 1.0, 2.20, side * 1.14))
        b.add_box(resolve_material(PAINT_ALT), (5.6, 0.40, 0.05), (-2.6, 1.35, side * 1.115))
        b.add_box(resolve_material(TAILLIGHT), (0.12, 0.28, 0.28), (-5.42, 1.10, side * 0.90))
        b.add_box(resolve_material(TAILLIGHT), (0.12, 0.28, 0.28), (-5.42, 1.10, side * 1.10))
    b.add_box(resolve_material(TRIM), (0.20, 0.60, 2.20), (-5.45, 0.95, 0.0))
    b.add_box(resolve_material(PLATE), (0.06, 0.16, 0.70), (-5.50, 0.86, 0.0))
    body = b.build("truck")
    wheel = build_wheel(0.50, 0.28, 14)
    wheels = [(1.35, 0.50, 1.05), (1.35, 0.50, -1.05),
              (-2.60, 0.50, 1.05), (-2.60, 0.50, -1.05),
              (-3.90, 0.50, 1.05), (-3.90, 0.50, -1.05)]
    return VehicleModel("truck", body, wheel, wheels, 8.5, 2.4, 3.5, 0.50, 0.28,
                        {"head": [(1.98, 1.30, s * 0.70) for s in (-1, 1)],
                         "tail": [(-5.52, 1.10, s * 1.0) for s in (-1, 1)]})


def _traffic_van(kind: str, paint_livery: str, beacon: bool = False) -> VehicleModel:
    """Delivery van / tempo, also the base for the ambulance."""
    b = MeshBuilder(kind)
    hw = 0.92
    add_loft(b, [
        (-2.35, 0.42, 1.30, 0.80, 0.86),
        (-2.10, 0.50, 1.95, 0.88, 0.92),
        (-1.30, 0.50, 2.05, 0.92, 0.94),
        (-0.20, 0.30, 2.08, 0.93, 0.94),
        (1.05, 0.30, 2.05, 0.93, 0.94),
        (1.70, 0.46, 1.86, 0.92, 0.92),
        (2.10, 0.50, 1.55, 0.89, 0.90),
        (2.35, 0.45, 1.20, 0.82, 0.84),
    ], top=PAINT, side=PAINT, bottom=TRIM, front=PAINT, back=PAINT)
    for side in (-1, 1):
        b.add_box(resolve_material(GLASS), (1.00, 0.70, 0.07), (1.35, 1.70, side * hw * 1.005))
        b.add_box(resolve_material(PAINT_ALT), (4.4, 0.30, 0.06), (-0.2, 1.20, side * hw * 1.005))
        b.add_box(resolve_material(HEADLIGHT), (0.12, 0.22, 0.34), (2.30, 1.00, side * 0.60))
        b.add_box(resolve_material(INDICATOR), (0.10, 0.14, 0.18), (2.30, 0.82, side * 0.82))
        b.add_box(resolve_material(TAILLIGHT), (0.10, 0.34, 0.22), (-2.37, 1.10, side * 0.78))
        b.add_box(resolve_material(TRIM), (0.08, 0.90, 0.08), (2.20, 1.70, side * 1.10))
        b.add_box(resolve_material(CHROME), (0.06, 0.24, 0.14), (2.14, 2.00, side * 1.18))
    b.add_box(resolve_material(GLASS), (0.10, 0.80, 1.80), (2.32, 1.72, 0.0))
    b.add_box(resolve_material(PLATE), (0.06, 0.16, 0.60), (2.36, 1.06, 0.0))
    b.add_box(resolve_material(PLATE), (0.06, 0.16, 0.60), (-2.38, 1.10, 0.0))
    if paint_livery == "ambulance":
        for side in (-1, 1):
            b.add_box(resolve_material(TAILLIGHT), (3.6, 0.42, 0.07), (-0.4, 1.55, side * hw * 1.008))
        b.add_box(resolve_material(TAILLIGHT), (0.10, 0.42, 2.0), (0.25, 1.55, 0.0))
        b.add_box(resolve_material(TRIM), (1.0, 0.16, 1.5), (-0.35, 2.14, 0.0))
    elif paint_livery == "police":
        for side in (-1, 1):
            b.add_box(resolve_material(BEACON_BLUE), (3.4, 0.40, 0.07), (-0.4, 1.45, side * hw * 1.008))
    if beacon:
        b.add_box(resolve_material(BEACON_RED), (0.34, 0.14, 0.34), (0.85, 2.16, 0.0))
        b.add_box(resolve_material(BEACON_BLUE), (0.34, 0.14, 0.34), (0.45, 2.16, 0.0))
        b.add_box(resolve_material(BEACON_RED), (0.34, 0.14, 0.34), (-0.05, 2.16, 0.0))
        b.add_box(resolve_material(BEACON_BLUE), (0.34, 0.14, 0.34), (-0.45, 2.16, 0.0))
    body = b.build(kind)
    wheel = build_wheel(0.34, 0.20, 14)
    wheels = [(1.42, 0.34, 0.82), (1.42, 0.34, -0.82), (-1.40, 0.34, 0.82), (-1.40, 0.34, -0.82)]
    anchors = {"head": [(2.34, 1.00, s * 0.60) for s in (-1, 1)],
               "tail": [(-2.39, 1.10, s * 0.78) for s in (-1, 1)]}
    if beacon:
        anchors["beacon"] = [(0.85, 2.24, 0.0), (0.45, 2.24, 0.0), (-0.05, 2.24, 0.0), (-0.45, 2.24, 0.0)]
    return VehicleModel(kind, body, wheel, wheels, 4.7, 1.85, 2.2, 0.34, 0.20, anchors)


def build_traffic_model(kind: str) -> VehicleModel:
    """Return the model for an AI traffic category."""
    if kind in CAR_TRAFFIC_KINDS:
        return build_car_model(kind, PLAYER_SPECS[kind])
    if kind == "auto":
        return _traffic_auto_rickshaw()
    if kind == "bike":
        return _traffic_motorcycle()
    if kind == "bus":
        return _traffic_bus()
    if kind == "truck":
        return _traffic_truck()
    if kind == "tempo":
        return _traffic_van("tempo", "delivery")
    if kind == "ambulance":
        return _traffic_van("ambulance", "ambulance", beacon=True)
    if kind == "police":
        return _traffic_van("police", "police", beacon=True)
    return build_car_model("sedan", PLAYER_SPECS["sedan"])


# --------------------------------------------------------------------------------------
# model library (built once, shared by every vehicle instance)
# --------------------------------------------------------------------------------------
class ModelLibrary:
    """Builds and caches every :class:`VehicleModel` used by the game."""

    def __init__(self) -> None:
        self._models: Dict[str, VehicleModel] = {}

    def player(self, key: str) -> VehicleModel:
        return self.get(f"player:{key}")

    def traffic(self, kind: str) -> VehicleModel:
        return self.get(f"traffic:{kind}")

    def get(self, cache_key: str) -> VehicleModel:
        model = self._models.get(cache_key)
        if model is None:
            if cache_key.startswith("player:"):
                key = cache_key.split(":", 1)[1]
                model = build_car_model(key, PLAYER_SPECS.get(key, PLAYER_SPECS["hatchback"]))
            else:
                model = build_traffic_model(cache_key.split(":", 1)[1])
            self._models[cache_key] = model
        return model

    def preload(self) -> None:
        for key in PLAYER_SPECS:
            self.player(key)
        for kind in ("hatchback", "sedan", "suv", "taxi", "auto", "bike", "bus", "truck",
                     "tempo", "ambulance", "police"):
            self.traffic(kind)

    def stats(self) -> Dict[str, int]:
        return {key: model.triangles() for key, model in self._models.items()}

    def all_models(self) -> Iterable[Tuple[str, VehicleModel]]:
        return self._models.items()


#: convenience: dimensions of each traffic kind, used by the AI collision code
TRAFFIC_DIMENSIONS: Dict[str, Tuple[float, float, float]] = {
    "hatchback": (3.6, 1.62, 1.5),
    "sedan": (4.32, 1.74, 1.46),
    "suv": (4.52, 1.88, 1.78),
    "taxi": (4.2, 1.68, 1.52),
    "auto": (2.6, 1.36, 1.75),
    "bike": (2.05, 0.72, 1.7),
    "bus": (10.5, 2.5, 3.3),
    "truck": (8.5, 2.4, 3.5),
    "tempo": (4.7, 1.85, 2.2),
    "ambulance": (4.7, 1.85, 2.2),
    "police": (4.7, 1.85, 2.2),
}
