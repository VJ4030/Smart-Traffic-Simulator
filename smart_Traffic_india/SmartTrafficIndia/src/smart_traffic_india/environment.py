"""The Indian city: buildings, bazaars, street furniture, props and lighting.

Design notes
------------
* The static world is **baked into spatial chunks** (44 m tiles).  Each chunk is one
  :class:`MeshData`, so the renderer draws a handful of batches per chunk and culls whole
  chunks against the view frustum.  Nothing is re-created per frame.
* Everything is procedurally generated from the road network: buildings are placed along
  the roads of each district, streetlights and electric poles march along the kerbs, and
  the props (tea stalls, vendor carts, bus shelters, hoardings, petrol pump, railway
  station) are scattered by district rules.  Nothing is ever placed on top of a road.
* :meth:`Environment.lighting` converts the time of day and the weather into the actual
  shader uniforms (sun direction/colour, hemisphere ambient, fog, sky colours), which is
  what makes sunrise/noon/sunset/night look different.

The module is pygame-only (no OpenGL) so it can be imported by head-less tests.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .mesh import Material, MeshBuilder, MeshData
from .road import MAT_ASPHALT, MAT_DIRT, MAT_FOOTPATH, MAT_KERB, MAT_LINE_WHITE, RoadNetwork, MAT_CONSTRUCTION
from .signage import SignAtlas
from .vehicle_models import ModelLibrary
from . import settings as S

Vec3 = Tuple[float, float, float]

# --------------------------------------------------------------------------------------
# materials
# --------------------------------------------------------------------------------------
B_WALLS = [
    Material("wall_a", (0.84, 0.79, 0.68), specular=0.08),
    Material("wall_b", (0.72, 0.66, 0.56), specular=0.08),
    Material("wall_c", (0.62, 0.60, 0.58), specular=0.08),
    Material("wall_d", (0.86, 0.72, 0.58), specular=0.08),
    Material("wall_e", (0.55, 0.60, 0.66), specular=0.08),
    Material("wall_f", (0.78, 0.58, 0.52), specular=0.08),
]
MAT_GLASS = Material("glass", (0.09, 0.13, 0.17), specular=0.9)
MAT_WINDOW = Material("window", (0.10, 0.14, 0.18), specular=0.9)
MAT_WINDOW_LIT = Material("window_lit", (0.10, 0.14, 0.18), specular=0.9)
MAT_SHUTTER = Material("shutter", (0.35, 0.37, 0.40), specular=0.35)
MAT_AWNING = Material("awning", (0.35, 0.45, 0.55), specular=0.2)
MAT_AWNING2 = Material("awning2", (0.62, 0.28, 0.22), specular=0.2)
MAT_ROOF = Material("roof_slab", (0.48, 0.47, 0.45), specular=0.06)
MAT_TANK = Material("water_tank", (0.30, 0.36, 0.42), specular=0.35)
MAT_POLE = Material("pole", (0.42, 0.43, 0.45), specular=0.45)
MAT_POLE_CONCRETE = Material("pole_concrete", (0.55, 0.54, 0.50), specular=0.10)
MAT_WIRE = Material("wire", (0.12, 0.12, 0.13), specular=0.30)
MAT_TREE_TRUNK = Material("trunk", (0.28, 0.21, 0.13), specular=0.05)
MAT_LEAF_A = Material("leaf_a", (0.20, 0.40, 0.16), specular=0.10)
MAT_LEAF_B = Material("leaf_b", (0.26, 0.47, 0.18), specular=0.10)
MAT_LEAF_C = Material("leaf_c", (0.16, 0.34, 0.14), specular=0.10)
MAT_SHOP_BOARD = Material("shop_board", (0.92, 0.92, 0.88), specular=0.15, texture="signs")
MAT_HOARDING = Material("hoarding", (0.95, 0.94, 0.90), specular=0.20, texture="signs")
MAT_HOARDING_FRAME = Material("hoarding_frame", (0.32, 0.33, 0.35), specular=0.4)
MAT_SIGN_POLE = Material("sign_pole", (0.55, 0.56, 0.58), specular=0.5)
MAT_SIGN_FACE = Material("sign_face", (0.90, 0.90, 0.88), specular=0.2, texture="signs")
MAT_STALL_ROOF = Material("stall_roof", (0.55, 0.48, 0.35), specular=0.05)
MAT_CANVAS = Material("tarp", (0.36, 0.40, 0.46), specular=0.08)
MAT_BENCH = Material("bench", (0.45, 0.32, 0.20), specular=0.10)
MAT_PLASTIC = Material("plastic", (0.86, 0.86, 0.84), specular=0.25)
MAT_LAMP_GLASS = Material("lamp_glass", (0.95, 0.92, 0.78), specular=0.4, emissive=0.0,
                          emissive_color=(1.0, 0.92, 0.72))
MAT_WATER_TANK_BLUE = Material("tank_blue", (0.15, 0.25, 0.55), specular=0.35)
MAT_TRAIN = Material("train", (0.32, 0.36, 0.44), specular=0.45)
MAT_TRAIN_STRIPE = Material("train_stripe", (0.75, 0.30, 0.18), specular=0.35)


@dataclass
class Building:
    """A city block building / shop / house."""

    x: float
    z: float
    width: float          # along the road
    depth: float          # away from the road
    height: float
    yaw: float
    style: str            # concrete | shop | house | warehouse
    material_index: int = 0
    shop_key: Optional[str] = None
    floors: int = 1

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        hw = max(self.width, self.depth) * 0.5 + 0.4
        return (self.x - hw, self.z - hw, self.x + hw, self.z + hw)

    def half_extents(self) -> Tuple[float, float]:
        if abs(math.cos(self.yaw)) > 0.7:
            return self.depth * 0.5, self.width * 0.5
        return self.width * 0.5, self.depth * 0.5


@dataclass
class Prop:
    """Generic street furniture / scenery item."""

    kind: str
    x: float
    z: float
    yaw: float = 0.0
    size: float = 1.0
    extra: Dict[str, object] = field(default_factory=dict)


@dataclass
class Chunk:
    """A baked piece of static world geometry used for frustum culling."""

    key: Tuple[int, int]
    center: Vec3
    half_extents: Vec3
    builder: MeshBuilder
    mesh: Optional[MeshData] = None

    def finalise(self) -> MeshData:
        self.mesh = self.builder.build(f"chunk_{self.key[0]}_{self.key[1]}")
        return self.mesh


@dataclass
class ParkingLot:
    """A marked parking area used by the parking challenges and free roam."""

    key: str
    label: str
    x: float
    z: float
    yaw: float
    rows: int = 1
    bays_per_row: int = 4
    bay_length: float = 5.2
    bay_width: float = 2.5

    def bay_transform(self, index: int) -> Tuple[float, float, float]:
        """World position + yaw of parking bay number ``index``."""
        per_row = self.bays_per_row
        row = index // per_row
        col = index % per_row
        lateral = (col - (self.bays_per_row - 1) * 0.5) * self.bay_width
        depth = (row - (self.rows - 1) * 0.5) * (self.bay_length + 0.6)
        cos_a, sin_a = math.cos(self.yaw), math.sin(self.yaw)
        # bays are laid out sideways from the lot centre, facing outward along depth
        dx = -sin_a * lateral + cos_a * depth
        dz = cos_a * lateral + sin_a * depth
        return self.x + dx, self.z + dz, self.yaw

    def bays(self) -> List[Tuple[float, float, float]]:
        return [self.bay_transform(i) for i in range(self.rows * self.bays_per_row)]

    def bay_rect(self, index: int) -> Tuple[float, float, float, float]:
        bx, bz, yaw = self.bay_transform(index)
        hw, hl = self.bay_width * 0.5, self.bay_length * 0.5
        cos_a, sin_a = math.cos(yaw), math.sin(yaw)
        corners = []
        for sx in (-1, 1):
            for sz in (-1, 1):
                cx = bx + cos_a * hw * sx - sin_a * hl * sz
                cz = bz + sin_a * hw * sx + cos_a * hl * sz
                corners.append((cx, cz))
        xs = [c[0] for c in corners]
        zs = [c[1] for c in corners]
        return min(xs), min(zs), max(xs), max(zs)


PARKING_LOTS: Dict[str, ParkingLot] = {
    "hospital": ParkingLot("hospital", "Civil Hospital Parking", 46.0, -60.0, 0.0, 1, 5),
    "market": ParkingLot("market", "Bazaar Street Parking", -26.0, 83.0, math.pi / 2, 1, 5),
    "mall": ParkingLot("mall", "Metro Mall Parking", 54.0, 24.0, 0.0, 2, 5),
    "oldcity": ParkingLot("oldcity", "Purani Basti Parking", -108.0, 8.0, math.pi / 2, 1, 4),
}


class Environment:
    """Owns every static object of the city and the sky/lighting state."""

    CHUNK_SIZE = 44.0

    def __init__(self, network: RoadNetwork, atlas: SignAtlas, models: ModelLibrary,
                 quality: Dict[str, object], seed: int = 21, time_of_day: str = "day",
                 weather: str = "clear") -> None:
        self.network = network
        self.atlas = atlas
        self.models = models
        self.quality = quality
        self.rng = random.Random(seed)
        self.buildings: List[Building] = []
        self.props: List[Prop] = []
        self.street_lights: List[Vec3] = []
        self.lamp_heads: List[Vec3] = []
        self.chunks: Dict[Tuple[int, int], Chunk] = {}
        self.parking_lots = dict(PARKING_LOTS)
        self.available_signs = int(quality.get("max_sign_textures", 8))
        self.sign_lookup: Dict[str, Tuple[float, float, float, float]] = {}
        self.sign_inverted = False

        # time / weather state
        self.time_preset = time_of_day
        self.time_hours = float(S.TIME_PRESETS.get(time_of_day, 11.5))
        self.weather = weather if weather != "random" else self.rng.choice(["clear", "cloudy", "rain"])
        self.rain_intensity = 0.0
        self.wind = 0.3
        self.flash_timer = 0.0
        self.streetlights_on = False

        self._place_buildings()
        self._place_all_props()
        self._build_geometry()
        self._update_lighting_flags()

    # ------------------------------------------------------------------ chunking
    def chunk_key(self, x: float, z: float) -> Tuple[int, int]:
        return (int(math.floor(x / self.CHUNK_SIZE)), int(math.floor(z / self.CHUNK_SIZE)))

    def chunk_for(self, x: float, z: float) -> Chunk:
        key = self.chunk_key(x, z)
        chunk = self.chunks.get(key)
        if chunk is None:
            center = ((key[0] + 0.5) * self.CHUNK_SIZE, 4.0, (key[1] + 0.5) * self.CHUNK_SIZE)
            half = (self.CHUNK_SIZE * 0.5, 14.0, self.CHUNK_SIZE * 0.5)
            chunk = Chunk(key, center, half, MeshBuilder(f"chunk_{key}"))
            self.chunks[key] = chunk
        return chunk

    def _grow_chunk(self, chunk: Chunk, x: float, y: float, z: float, radius: float) -> None:
        """Expand a chunk's culling box so it always contains everything inside it."""
        cx, cy, cz = chunk.center
        hx, hy, hz = chunk.half_extents
        chunk.half_extents = (
            max(hx, abs(x - cx) + radius),
            max(hy, abs(y - cy) + radius + 6.0),
            max(hz, abs(z - cz) + radius),
        )

    # ------------------------------------------------------------------ placement
    def _occupied(self, x: float, z: float, radius: float) -> bool:
        if self.network.on_road(x, z, 2.0):
            return True
        if self.network.junction_near(x, z, 16.0):
            return True
        for b in self.buildings:
            bx0, bz0, bx1, bz1 = b.bounds
            if bx0 - radius < x < bx1 + radius and bz0 - radius < z < bz1 + radius:
                return True
        return False

    def _place_buildings(self) -> None:
        rng = self.rng
        for road in self.network.roads:
            info = _district_info(road.district)
            lo, hi = min(road.start, road.end), max(road.start, road.end)
            s = lo + 6.0
            while s < hi - 6.0:
                width = rng.uniform(7.0, 16.0)
                for side in (-1, 1):
                    if rng.random() > float(info["density"]) * 0.95:
                        continue
                    depth = rng.uniform(8.0, 17.0)
                    offset = road.half_width + 5.4 + depth * 0.5 + rng.uniform(0.4, 3.0)
                    x, z = road.position_at(s + width * 0.5, side * offset)
                    if self._occupied(x, z, max(width, depth) * 0.55):
                        continue
                    style = str(info["buildings"])
                    height = rng.uniform(*info["height"])         # type: ignore[arg-type]
                    yaw = 0.0 if road.axis == "z" else math.pi / 2
                    floors = max(1, int(height / 3.2))
                    shop_key = None
                    if style == "shop" and len(self.sign_lookup) < self.available_signs:
                        idx = len(self.sign_lookup)
                        shop_key = f"shop{idx % 24}"
                        rect = self.atlas.uv(shop_key)
                        if rect is not None:
                            self.sign_lookup[shop_key] = rect
                    self.buildings.append(Building(x, z, width if road.axis != "z" else depth,
                                                   depth if road.axis != "z" else width,
                                                   height, yaw, style,
                                                   rng.randrange(len(B_WALLS)), shop_key, floors))
                s += width + rng.uniform(2.0, 6.0)
        # a dense ring of low-rise housing filling the gaps between the districts
        for _ in range(70):
            x = rng.uniform(-165.0, 165.0)
            z = rng.uniform(-160.0, 160.0)
            if self._occupied(x, z, 10.0):
                continue
            w = rng.uniform(8.0, 14.0)
            d = rng.uniform(9.0, 15.0)
            if abs(x) < 26.0 or abs(z) < 26.0:
                continue
            self.buildings.append(Building(x, z, w, d, rng.uniform(6.0, 13.0),
                                           rng.choice([0.0, math.pi / 2]), "house",
                                           rng.randrange(len(B_WALLS)), None, 2))

    def _place_all_props(self) -> None:
        rng = self.rng
        # --- street lights + electric poles along every road ------------------------
        for road in self.network.roads:
            lo, hi = min(road.start, road.end), max(road.start, road.end)
            spacing = 34.0
            s = lo + 12.0
            i = 0
            while s < hi - 12.0:
                side = 1 if i % 2 == 0 else -1
                offset = road.half_width + 1.4
                x, z = road.position_at(s, side * offset)
                if not self.network.junction_near(x, z, 14.0):
                    to_road = ((-side, 0.0) if road.axis == "z" else (0.0, -side))
                    self.props.append(Prop("streetlight", x, z, 0.0, 1.0, {"dir": to_road}))
                s += spacing
                i += 1
            # electric poles with sagging wires, offset from the street lights
            s = lo + 24.0
            poles: List[Vec3] = []
            while s < hi - 12.0:
                x, z = road.position_at(s, (road.half_width + 3.2))
                if not self.network.junction_near(x, z, 12.0):
                    self.props.append(Prop("elec_pole", x, z, 1.0 if road.axis == "z" else math.pi / 2))
                    poles.append((x, 7.6, z))
                s += 42.0
            for a, b in zip(poles, poles[1:]):
                self.props.append(Prop("wire", a[0], a[2], 0.0, 1.0, {"to": (b[0], b[2])}))
        # --- trees ------------------------------------------------------------------
        for road in self.network.roads:
            lo, hi = min(road.start, road.end), max(road.start, road.end)
            info = _district_info(road.district)
            s = lo + 8.0
            while s < hi - 8.0:
                if rng.random() < float(info["trees"]):
                    for side in (-1, 1):
                        x, z = road.position_at(s + rng.uniform(-2, 2), side * (road.half_width + 4.2))
                        if not self._occupied(x, z, 1.6):
                            self.props.append(Prop("tree", x, z, rng.uniform(0, 6.28),
                                                   rng.uniform(0.8, 1.35)))
                s += 18.0
        # --- hoardings along the busy roads -----------------------------------------
        hoard_idx = 0
        for road in self.network.roads:
            if road.district not in ("downtown", "highway", "market", "railway"):
                continue
            lo, hi = min(road.start, road.end), max(road.start, road.end)
            s = lo + 20.0
            while s < hi - 20.0:
                if rng.random() < 0.5:
                    side = 1 if rng.random() < 0.5 else -1
                    x, z = road.position_at(s, side * (road.half_width + 6.5))
                    if not self._occupied(x, z, 3.0):
                        key = f"hoarding{hoard_idx % 8}"
                        rect = self.atlas.uv(key)
                        yaw = math.pi if side > 0 else 0.0
                        self.props.append(Prop("hoarding", x, z, yaw, 1.0,
                                               {"uv": rect, "key": key, "side": side,
                                                "axis": road.axis}))
                        hoard_idx += 1
                s += 30.0
        # --- district flavour: bazaar stalls, vendors, shelters ----------------------
        for road in self.network.roads:
            lo, hi = min(road.start, road.end), max(road.start, road.end)
            density = {"market": 0.75, "oldcity": 0.6, "downtown": 0.30,
                       "railway": 0.35, "residential": 0.22}.get(road.district, 0.18)
            s = lo + 10.0
            while s < hi - 10.0:
                if rng.random() < density:
                    for side in (-1, 1):
                        if rng.random() > 0.6:
                            continue
                        x, z = road.position_at(s + rng.uniform(-3, 3), side * (road.half_width + 3.6))
                        if self._occupied(x, z, 2.2):
                            continue
                        kind = rng.choice(["vendor_cart", "tea_stall", "bench", "dustbin",
                                           "bollard", "planter", "shelter", "kiosk"])
                        self.props.append(Prop(kind, x, z,
                                               (math.pi / 2 if road.axis == "z" else 0.0) +
                                               rng.uniform(-0.2, 0.2), rng.uniform(0.9, 1.15)))
                s += 14.0
        # --- parked vehicles along the kerb (they narrow the usable road) ------------
        vehicle_kinds = ["auto", "hatchback", "tempo", "bike", "suv", "truck"]
        for road in self.network.roads:
            lo, hi = min(road.start, road.end), max(road.start, road.end)
            s = lo + 16.0
            while s < hi - 16.0:
                if rng.random() < 0.22:
                    side = 1 if rng.random() < 0.5 else -1
                    x, z = road.position_at(s, side * (road.half_width - 1.1))
                    if not self.network.junction_near(x, z, 16.0):
                        yaw = road.parking_yaw(side)
                        kind = rng.choice(vehicle_kinds)
                        self.props.append(Prop("parked_vehicle", x, z, yaw,
                                               rng.uniform(0.95, 1.05), {"kind": kind}))
                s += 26.0
        # --- landmark props ---------------------------------------------------------
        self._place_landmarks()
        # --- parking lots -----------------------------------------------------------
        for lot in self.parking_lots.values():
            self.props.append(Prop("parking_lot", lot.x, lot.z, lot.yaw, 1.0,
                                   {"lot": lot}))

    def _place_landmarks(self) -> None:
        # petrol pump beside MG Road
        self.props.append(Prop("petrol_pump", -66.0, -26.0, math.pi / 2, 1.0, {"serves": "fuel"}))
        # bus terminal
        self.props.append(Prop("bus_terminal", -140.0, 46.0, 0.0, 1.0))
        # railway station: building, platform, a stationary local train
        self.props.append(Prop("railway_station", -140.0, -112.0, 0.0, 1.0))
        # a school building with a play area
        self.props.append(Prop("school", -52.0, 42.0, math.pi / 2, 1.0))
        # market gate arch
        self.props.append(Prop("market_gate", 0.0, 66.0, 0.0, 1.0))

    # ------------------------------------------------------------------ geometry
    def _build_geometry(self) -> None:
        for b in self.buildings:
            chunk = self.chunk_for(b.x, b.z)
            self._add_building(chunk.builder, b)
            self._grow_chunk(chunk, b.x, b.height * 0.5, b.z, max(b.width, b.depth) * 0.75)
        for prop in self.props:
            chunk = self.chunk_for(prop.x, prop.z)
            self._add_prop(chunk.builder, prop)
            radius = 9.0 if prop.kind in ("petrol_pump", "bus_terminal", "railway_station",
                                          "market_gate", "school") else 4.0
            self._grow_chunk(chunk, prop.x, 3.0, prop.z, radius)
        # bake every chunk builder into a MeshData (one VAO per chunk on the GPU) - this is
        # what the renderer frustum-culls, and it is the reason the city can be large without
        # costing anything per frame beyond the visible chunks
        for chunk in self.chunks.values():
            if chunk.builder.groups:
                chunk.finalise()
        # ground plane (grass / dirt) is a single large mesh drawn first
        self.ground = self._build_ground()

    def _build_ground(self) -> MeshData:
        b = MeshBuilder("ground")
        b.add_flat(Material("ground", (0.34, 0.30, 0.24), specular=0.02),
                   (520.0, 520.0), (0.0, -0.02, 0.0), 60.0)
        return b.build("ground")

    def _add_building(self, b: MeshBuilder, bld: Building) -> None:
        rng = random.Random(hash((round(bld.x, 2), round(bld.z, 2))) & 0xFFFFFFFF)
        walls = B_WALLS[bld.material_index % len(B_WALLS)]
        yaw = bld.yaw
        if bld.style == "shop":
            self._add_shop(b, bld, walls, rng)
            return
        h = bld.height
        b.add_box(walls, (bld.width, h, bld.depth), (bld.x, h * 0.5, bld.z), top_scale=0.98)
        # parapet + roof slab
        b.add_box(MAT_ROOF, (bld.width + 0.5, 0.35, bld.depth + 0.5), (bld.x, h + 0.1, bld.z))
        # windows in a grid on all four faces
        faces = self._face_axes(yaw)
        for axis, sign in faces:
            cols = max(1, int(bld.width / 3.2))
            rows = max(1, int(h / 3.4))
            for r in range(rows):
                for c in range(cols):
                    t = (c + 0.5) / cols - 0.5
                    y = 1.4 + r * (h - 2.0) / max(rows, 1)
                    if y > h - 0.8:
                        continue
                    lit = rng.random() < 0.35
                    self._add_window(b, bld, axis, sign, t, y, lit, rng)
        if bld.style == "warehouse":
            b.add_box(MAT_SHUTTER, (bld.width * 0.6, h * 0.6, 0.3),
                      (bld.x, h * 0.3, bld.z + (bld.depth * 0.5 + 0.2) * math.cos(yaw)))
        else:
            # rooftop water tank + stair room (very common on Indian roofs)
            tx = bld.x + rng.uniform(-bld.width * 0.3, bld.width * 0.3)
            tz = bld.z + rng.uniform(-bld.depth * 0.3, bld.depth * 0.3)
            b.add_box(MAT_ROOF, (2.2, 2.4, 2.4), (bld.x - bld.width * 0.2, h + 1.3, bld.z), top_scale=0.9)
            b.add_cylinder(MAT_WATER_TANK_BLUE, 0.75, 0.7, (tx, h + 1.5, tz), 12, axis="y")
            b.add_box(MAT_ROOF, (0.4, 0.4, 0.4), (tx, h + 0.35, tz))
            # a small AC unit or two
            if rng.random() < 0.7:
                b.add_box(MAT_POLE, (0.9, 0.7, 0.6), (bld.x + rng.uniform(-2, 2), h + 0.55,
                                                      bld.z + bld.depth * 0.3))

    def _face_axes(self, yaw: float) -> List[Tuple[str, int]]:
        """Return the four faces of a building as (axis, sign) pairs."""
        if abs(math.cos(yaw)) > 0.7:      # building faces along Z
            return [("z", 1), ("z", -1)]
        return [("x", 1), ("x", -1)]

    def _add_window(self, b: MeshBuilder, bld: Building, axis: str, sign: int,
                    t: float, y: float, lit: bool, rng: random.Random) -> None:
        mat = MAT_WINDOW_LIT if lit else MAT_WINDOW
        w, h = 1.3, 1.4
        if axis == "z":
            x = bld.x + t * bld.width
            z = bld.z + sign * (bld.depth * 0.5 + 0.06)
            b.add_box(mat, (w, h, 0.08), (x, y, z))
            b.add_box(MAT_ROOF, (w + 0.3, 0.12, 0.2), (x, y + h * 0.5, z + sign * 0.06))
        else:
            x = bld.x + sign * (bld.depth * 0.5 + 0.06)
            z = bld.z + t * bld.width
            b.add_box(mat, (0.08, h, w), (x, y, z))
            b.add_box(MAT_ROOF, (0.2, 0.12, w + 0.3), (x + sign * 0.06, y + h * 0.5, z))

    def _add_shop(self, b: MeshBuilder, bld: Building, walls: Material, rng: random.Random) -> None:
        """A street-level shop: awning, shutter, colourful board with real text."""
        h = max(4.4, bld.height)
        b.add_box(walls, (bld.width, h, bld.depth), (bld.x, h * 0.5, bld.z), top_scale=0.99)
        b.add_box(MAT_ROOF, (bld.width + 0.4, 0.3, bld.depth + 0.4), (bld.x, h + 0.05, bld.z))
        facing_z = abs(math.cos(bld.yaw)) > 0.7
        sign = 1.0 if rng.random() < 0.5 else -1.0
        board_w = bld.width * 0.86
        board_y = 3.35
        board_h = 0.95
        if facing_z:
            z = bld.z + sign * (bld.depth * 0.5 + 0.14)
            uv = self.sign_lookup.get(bld.shop_key or "")
            b.add_quad_auto(MAT_SHOP_BOARD if uv else MAT_AWNING2,
                            [(bld.x - board_w * 0.5, board_y - board_h * 0.5, z),
                             (bld.x + board_w * 0.5, board_y - board_h * 0.5, z),
                             (bld.x + board_w * 0.5, board_y + board_h * 0.5, z),
                             (bld.x - board_w * 0.5, board_y + board_h * 0.5, z)],
                            uv=[(uv[0], uv[1]), (uv[2], uv[1]), (uv[2], uv[3]), (uv[0], uv[3])] if uv else None,
                            ref=(bld.x, 0.0, bld.z))
            # awning
            b.add_box(MAT_AWNING, (board_w, 0.10, 1.5),
                      (bld.x, 2.75, z + sign * 0.75), top_scale=0.9)
            # shutter / door
            b.add_box(MAT_SHUTTER, (bld.width * 0.7, 2.2, 0.12), (bld.x, 1.2, z))
            b.add_box(MAT_ROOF, (bld.width * 0.75, 0.16, 0.5), (bld.x, 0.10, z + sign * 0.25))
            # upstairs windows
            for c in range(max(1, int(bld.width / 3.4))):
                t = (c + 0.5) / max(1, int(bld.width / 3.4)) - 0.5
                for r in range(max(1, int((h - 4.4) / 3.2))):
                    y = 5.2 + r * 3.0
                    if y > h - 0.6:
                        continue
                    self._add_window(b, bld, "z", 1 if sign > 0 else -1, t, y,
                                     rng.random() < 0.4, rng)
        else:
            x = bld.x + sign * (bld.depth * 0.5 + 0.14)
            uv = self.sign_lookup.get(bld.shop_key or "")
            b.add_quad_auto(MAT_SHOP_BOARD if uv else MAT_AWNING2,
                            [(x, board_y - board_h * 0.5, bld.z - board_w * 0.5),
                             (x, board_y - board_h * 0.5, bld.z + board_w * 0.5),
                             (x, board_y + board_h * 0.5, bld.z + board_w * 0.5),
                             (x, board_y + board_h * 0.5, bld.z - board_w * 0.5)],
                            uv=[(uv[0], uv[1]), (uv[2], uv[1]), (uv[2], uv[3]), (uv[0], uv[3])] if uv else None,
                            ref=(bld.x, 0.0, bld.z))
            b.add_box(MAT_AWNING, (1.5, 0.10, board_w), (x + sign * 0.75, 2.75, bld.z),
                      top_scale=0.9)
            b.add_box(MAT_SHUTTER, (0.12, 2.2, bld.width * 0.7), (x, 1.2, bld.z))
            b.add_box(MAT_ROOF, (0.5, 0.16, bld.width * 0.75), (x + sign * 0.25, 0.10, bld.z))
            for c in range(max(1, int(bld.width / 3.4))):
                t = (c + 0.5) / max(1, int(bld.width / 3.4)) - 0.5
                for r in range(max(1, int((h - 4.4) / 3.2))):
                    y = 5.2 + r * 3.0
                    if y > h - 0.6:
                        continue
                    self._add_window(b, bld, "x", 1 if sign > 0 else -1, t, y,
                                     rng.random() < 0.4, rng)

    # ------------------------------------------------------------------ props
    def _add_prop(self, b: MeshBuilder, prop: Prop) -> None:
        kind = prop.kind
        handler = getattr(self, f"_prop_{kind}", None)
        if handler is not None:
            handler(b, prop)
        else:                                          # pragma: no cover - safety net
            b.add_box(MAT_POLE, (1.0, 1.0, 1.0), (prop.x, 0.5, prop.z))

    def _prop_streetlight(self, b: MeshBuilder, prop: Prop) -> None:
        x, z = prop.x, prop.z
        dx, dz = prop.extra.get("dir", (1.0, 0.0))          # direction towards the road
        b.add_cylinder(MAT_POLE, 0.11, 4.1, (x, 4.1, z), 10, axis="y")
        # arm reaching over the road, built from two thin boxes so it reads as a bracket
        arm_length = 2.4
        mid = (x + dx * arm_length * 0.5, 8.15, z + dz * arm_length * 0.5)
        size = (arm_length, 0.10, 0.10) if abs(dx) > abs(dz) else (0.10, 0.10, arm_length)
        b.add_box(MAT_POLE, size, mid)
        b.add_box(MAT_POLE, (0.12, 0.9, 0.12), (x + dx * 0.3, 7.7, z + dz * 0.3))
        lamp_x = x + dx * (arm_length + 0.2)
        lamp_z = z + dz * (arm_length + 0.2)
        b.add_box(MAT_LAMP_GLASS, (0.62, 0.16, 0.34), (lamp_x, 8.02, lamp_z))
        b.add_box(MAT_POLE, (0.46, 0.20, 0.46), (lamp_x, 8.20, lamp_z), top_scale=0.7)
        self.street_lights.append((lamp_x, 8.0, lamp_z))
        self.lamp_heads.append((lamp_x, 8.0, lamp_z))

    def _prop_elec_pole(self, b: MeshBuilder, prop: Prop) -> None:
        x, z = prop.x, prop.z
        b.add_cylinder(MAT_POLE_CONCRETE, 0.17, 3.8, (x, 3.8, z), 8, axis="y",
                       radius_bottom=0.24)
        b.add_box(MAT_POLE_CONCRETE, (0.10, 0.10, 2.2), (x, 7.4, z))
        b.add_box(MAT_POLE_CONCRETE, (0.10, 0.10, 1.6), (x, 6.9, z))
        # transformer box on some poles
        if (abs(x) + abs(z)) % 3 < 1.0:
            b.add_box(MAT_POLE, (0.8, 1.1, 0.6), (x + 0.5, 6.2, z))

    def _prop_wire(self, b: MeshBuilder, prop: Prop) -> None:
        to = prop.extra.get("to")
        if not to:
            return
        x0, z0 = prop.x, prop.z
        x1, z1 = float(to[0]), float(to[1])          # type: ignore[index]
        length = math.hypot(x1 - x0, z1 - z0)
        if length < 1.0 or length > 90.0:
            return
        segments = max(3, int(length / 4.0))
        along_x = abs(x1 - x0) > abs(z1 - z0)
        seg_size = (length / segments + 0.05, 0.045, 0.045) if along_x else (0.045, 0.045,
                                                                             length / segments + 0.05)
        for dy in (0.0, 0.25, 0.5):
            for i in range(segments):
                t0 = i / segments
                t1 = (i + 1) / segments
                sag0 = math.sin(t0 * math.pi) * 0.7
                sag1 = math.sin(t1 * math.pi) * 0.7
                px0 = x0 + (x1 - x0) * t0
                pz0 = z0 + (z1 - z0) * t0
                px1 = x0 + (x1 - x0) * t1
                pz1 = z0 + (z1 - z0) * t1
                y0 = 7.4 + dy - sag0
                y1 = 7.4 + dy - sag1
                mid = ((px0 + px1) * 0.5, (y0 + y1) * 0.5, (pz0 + pz1) * 0.5)
                b.add_box(MAT_WIRE, seg_size, mid, uv_scale=0.2)

    def _prop_tree(self, b: MeshBuilder, prop: Prop) -> None:
        scale = prop.size
        x, z = prop.x, prop.z
        trunk_h = 2.4 * scale
        b.add_cylinder(MAT_TREE_TRUNK, 0.16 * scale, trunk_h * 0.5, (x, trunk_h * 0.5, z), 8,
                       axis="y", radius_bottom=0.26 * scale)
        rng = random.Random(int(abs(x * 13 + z * 7)) & 0xFFFF)
        for i in range(3):
            leaf = (MAT_LEAF_A, MAT_LEAF_B, MAT_LEAF_C)[i % 3]
            ox = rng.uniform(-0.8, 0.8) * scale
            oz = rng.uniform(-0.8, 0.8) * scale
            r = rng.uniform(1.1, 1.7) * scale
            b.add_sphere(leaf, r, (x + ox, trunk_h + 0.6 * scale + rng.uniform(0, 0.8), z + oz),
                         9, 6, squash=0.85)

    def _prop_hoarding(self, b: MeshBuilder, prop: Prop) -> None:
        uv = prop.extra.get("uv")
        axis = prop.extra.get("axis", "z")
        x, z = prop.x, prop.z
        yaw = prop.yaw
        w, h = 7.0, 3.0
        top = 2.6 + h * 0.5
        # support poles
        for t in (-0.35, 0.35):
            px = x + (t * w) * (math.cos(yaw + math.pi / 2))
            pz = z + (t * w) * (math.sin(yaw + math.pi / 2))
            b.add_cylinder(MAT_POLE, 0.16, top * 0.5, (px, top * 0.5, pz), 8, axis="y")
        if axis == "z":
            verts = [(x - w * 0.5, top - h * 0.5, z), (x + w * 0.5, top - h * 0.5, z),
                     (x + w * 0.5, top + h * 0.5, z), (x - w * 0.5, top + h * 0.5, z)]
        else:
            verts = [(x, top - h * 0.5, z - w * 0.5), (x, top - h * 0.5, z + w * 0.5),
                     (x, top + h * 0.5, z + w * 0.5), (x, top + h * 0.5, z - w * 0.5)]
        uv_coords = [(uv[0], uv[1]), (uv[2], uv[1]), (uv[2], uv[3]), (uv[0], uv[3])] if uv else None
        b.add_quad_auto(MAT_HOARDING if uv else MAT_AWNING2, verts, uv=uv_coords,
                        ref=(x - math.cos(yaw) * 2.0, 0.0, z - math.sin(yaw) * 2.0))
        b.add_box(MAT_HOARDING_FRAME, (w + 0.2, 0.16, 0.16), (x, top + h * 0.5, z))
        b.add_box(MAT_HOARDING_FRAME, (w + 0.2, 0.16, 0.16), (x, top - h * 0.5, z))

    def _prop_vendor_cart(self, b: MeshBuilder, prop: Prop) -> None:
        x, z, yaw = prop.x, prop.z, prop.yaw
        cos_a, sin_a = math.cos(yaw), math.sin(yaw)
        b.add_box(MAT_POLE, (1.9, 0.16, 1.05), (x, 0.95, z), top_scale=1.0)
        b.add_box(MAT_BENCH, (1.8, 0.5, 0.95), (x, 0.68, z))
        for sx in (-0.7, 0.7):
            for sz in (-0.42, 0.42):
                b.add_cylinder(MAT_POLE, 0.22, 0.05,
                               (x + cos_a * sx - sin_a * sz, 0.22, z + sin_a * sx + cos_a * sz),
                               10, axis="z")
        b.add_cylinder(MAT_POLE, 0.05, 1.1, (x - cos_a * 0.9, 1.55, z - sin_a * 0.9), 8, axis="y")
        b.add_cylinder(MAT_AWNING2, 1.35, 0.06, (x - cos_a * 0.9, 2.55, z - sin_a * 0.9), 12,
                       axis="y", radius_bottom=0.15)
        b.add_box(MAT_AWNING2, (0.5, 0.3, 0.9), (x + cos_a * 0.6, 1.32, z + sin_a * 0.6))

    def _prop_tea_stall(self, b: MeshBuilder, prop: Prop) -> None:
        x, z, yaw = prop.x, prop.z, prop.yaw
        cos_a, sin_a = math.cos(yaw), math.sin(yaw)
        b.add_box(MAT_BENCH, (2.2, 1.0, 1.4), (x, 0.5, z))
        b.add_box(MAT_POLE, (2.4, 0.14, 1.6), (x, 2.35, z))
        for sx in (-1.0, 1.0):
            for sz in (-0.6, 0.6):
                b.add_cylinder(MAT_POLE, 0.06, 0.75,
                               (x + cos_a * sx - sin_a * sz, 1.5, z + sin_a * sx + cos_a * sz),
                               8, axis="y")
        b.add_box(MAT_STALL_ROOF, (2.6, 0.10, 1.9), (x, 2.5, z), top_scale=0.9)
        b.add_box(MAT_PLASTIC, (0.28, 0.34, 0.28), (x + cos_a * 0.7, 2.55, z + sin_a * 0.7))
        b.add_box(MAT_SHUTTER, (1.4, 0.6, 0.1), (x, 1.5, z + 0.75))

    def _prop_bench(self, b: MeshBuilder, prop: Prop) -> None:
        x, z, yaw = prop.x, prop.z, prop.yaw
        cos_a, sin_a = math.cos(yaw), math.sin(yaw)
        along_x = abs(cos_a) >= abs(sin_a)
        seat = (2.0, 0.12, 0.52) if along_x else (0.52, 0.12, 2.0)
        back = (2.0, 0.46, 0.10) if along_x else (0.10, 0.46, 2.0)
        ox = 0.0 if along_x else 0.24
        oz = 0.24 if along_x else 0.0
        b.add_box(MAT_BENCH, seat, (x, 0.46, z))
        b.add_box(MAT_BENCH, back, (x + ox, 0.72, z + oz))
        for t in (-0.8, 0.8):
            lx = x + cos_a * t
            lz = z + sin_a * t
            b.add_box(MAT_POLE, (0.12, 0.44, 0.44) if along_x else (0.44, 0.44, 0.12),
                      (lx, 0.22, lz))

    def _prop_dustbin(self, b: MeshBuilder, prop: Prop) -> None:
        b.add_cylinder(MAT_PLASTIC, 0.32, 0.45, (prop.x, 0.45, prop.z), 10, axis="y")
        b.add_cylinder(MAT_POLE, 0.34, 0.05, (prop.x, 0.92, prop.z), 10, axis="y")

    def _prop_bollard(self, b: MeshBuilder, prop: Prop) -> None:
        b.add_cylinder(MAT_PLASTIC, 0.12, 0.45, (prop.x, 0.45, prop.z), 8, axis="y")
        b.add_cylinder(MAT_AWNING2, 0.13, 0.06, (prop.x, 0.72, prop.z), 8, axis="y")

    def _prop_planter(self, b: MeshBuilder, prop: Prop) -> None:
        b.add_cylinder(MAT_KERB, 0.5, 0.35, (prop.x, 0.35, prop.z), 10, axis="y",
                       radius_bottom=0.42)
        b.add_sphere(MAT_LEAF_B, 0.45, (prop.x, 0.85, prop.z), 8, 6, squash=0.7)

    def _prop_shelter(self, b: MeshBuilder, prop: Prop) -> None:
        x, z, yaw = prop.x, prop.z, prop.yaw
        cos_a, sin_a = math.cos(yaw), math.sin(yaw)
        for sx in (-1.5, 1.5):
            for sz in (-0.7, 0.7):
                b.add_cylinder(MAT_POLE, 0.08, 1.35,
                               (x + cos_a * sx - sin_a * sz, 1.35, z + sin_a * sx + cos_a * sz),
                               8, axis="y")
        b.add_box(MAT_CANVAS, (3.6, 0.12, 2.0), (x, 2.75, z))
        b.add_box(MAT_BENCH, (3.0, 0.1, 0.45), (x, 0.6, z - sin_a * 0.5))
        b.add_box(MAT_SHUTTER, (3.0, 1.5, 0.08), (x - sin_a * 0.85, 1.6, z + cos_a * 0.85))

    def _prop_kiosk(self, b: MeshBuilder, prop: Prop) -> None:
        x, z, yaw = prop.x, prop.z, prop.yaw
        b.add_box(MAT_POLE, (1.6, 2.3, 1.6), (x, 1.15, z), top_scale=0.96)
        b.add_box(MAT_AWNING, (1.9, 0.14, 1.9), (x, 2.42, z), top_scale=0.85)
        b.add_box(MAT_WINDOW, (1.2, 0.8, 0.08), (x, 1.7, z + math.cos(yaw) * 0.82))

    def _prop_parked_vehicle(self, b: MeshBuilder, prop: Prop) -> None:
        kind = str(prop.extra.get("kind", "hatchback"))
        model = self.models.traffic(kind)
        from .transform import model_matrix
        matrix = model_matrix((prop.x, 0.0, prop.z), prop.yaw, prop.size)
        merged = model.body.transformed(matrix)
        wheel_matrices = []
        for offset in model.wheels:
            local = model_matrix((offset[0], offset[1], offset[2]), 0.0, 1.0)
            wheel_matrices.append(matrix @ local)
        for wm in wheel_matrices:
            merged = merged.merge(model.wheel.transformed(wm))
        self._merge_into(b, merged)

    def _merge_into(self, b: MeshBuilder, mesh: MeshData) -> None:
        pass
        for surface in mesh.surfaces:
            grp = b.group(surface.material)
            base = len(grp.positions) // 3
            grp.positions.extend(surface.positions.reshape(-1).tolist())
            grp.normals.extend(surface.normals.reshape(-1).tolist())
            grp.uvs.extend(surface.uvs.reshape(-1).tolist())
            grp.indices.extend((surface.indices.astype("int64") + base).tolist())

    def _prop_petrol_pump(self, b: MeshBuilder, prop: Prop) -> None:
        x, z = prop.x, prop.z
        b.add_flat(MAT_CONSTRUCTION, (26.0, 20.0), (x + 4.0, 0.03, z), 8.0)
        for sx in (-5.0, 5.0):
            for sz in (-4.0, 4.0):
                b.add_box(MAT_POLE, (0.7, 6.0, 0.7), (x + sx + 3.0, 3.0, z + sz))
        b.add_box(MAT_LINE_WHITE, (13.0, 0.5, 12.0), (x + 3.0, 6.25, z))
        b.add_box(MAT_AWNING2, (13.4, 0.3, 12.4), (x + 3.0, 6.6, z), top_scale=0.94)
        for sx in (-2.0, 2.0):
            b.add_box(MAT_PLASTIC, (0.9, 1.5, 0.6), (x + 3.0 + sx, 0.85, z + 2.2))
            b.add_box(MAT_POLE, (0.6, 0.4, 0.5), (x + 3.0 + sx, 1.75, z + 2.2))
        # office + sign board
        b.add_box(MAT_KERB, (5.0, 3.4, 4.0), (x + 9.0, 1.7, z - 6.0))
        uv = self.atlas.uv("petrol")
        b.add_quad_auto(MAT_SIGN_FACE if uv else MAT_AWNING2,
                        [(x - 3.4, 5.2, z), (x - 3.4, 7.6, z), (x + 1.6, 7.6, z), (x + 1.6, 5.2, z)],
                        uv=[(uv[0], uv[3]), (uv[0], uv[1]), (uv[2], uv[1]), (uv[2], uv[3])] if uv else None,
                        ref=(x + 1.0, 0.0, z + 3.0))
        for sx in (-2.4, 0.6):
            b.add_cylinder(MAT_POLE, 0.2, 3.2, (x + sx, 3.2, z), 8, axis="y")

    def _prop_bus_terminal(self, b: MeshBuilder, prop: Prop) -> None:
        x, z = prop.x, prop.z
        b.add_flat(MAT_CONSTRUCTION, (44.0, 26.0), (x + 12.0, 0.03, z + 6.0), 6.0)
        b.add_box(MAT_KERB, (30.0, 5.0, 12.0), (x + 10.0, 2.5, z - 10.0), top_scale=0.97)
        b.add_box(MAT_AWNING, (30.6, 0.4, 12.6), (x + 10.0, 5.2, z - 10.0))
        for i in range(5):
            b.add_box(MAT_GLASS, (2.4, 1.6, 0.12), (x - 2.0 + i * 5.0, 3.4, z - 3.95))
        uv = self.atlas.uv("bus_stop")
        b.add_quad_auto(MAT_SIGN_FACE if uv else MAT_AWNING2,
                        [(x + 22.0, 5.4, z - 3.9), (x + 22.0, 6.6, z - 3.9),
                         (x + 30.0, 6.6, z - 3.9), (x + 30.0, 5.4, z - 3.9)],
                        uv=[(uv[0], uv[1]), (uv[0], uv[3]), (uv[2], uv[3]), (uv[2], uv[1])] if uv else None,
                        ref=(x + 10.0, 0.0, z + 4.0))
        # parked buses
        for i in range(3):
            model = self.models.traffic("bus")
            from .transform import model_matrix
            matrix = model_matrix((x + 6.0 + i * 12.0, 0.0, z + 8.0), math.pi / 2, 1.0)
            merged = model.body.transformed(matrix)
            for offset in model.wheels:
                wm = matrix @ model_matrix((offset[0], offset[1], offset[2]), 0.0, 1.0)
                merged = merged.merge(model.wheel.transformed(wm))
            self._merge_into(b, merged)

    def _prop_railway_station(self, b: MeshBuilder, prop: Prop) -> None:
        x, z = prop.x, prop.z
        # platform + tracks + a stationary local train
        b.add_box(MAT_FOOTPATH, (70.0, 0.9, 9.0), (x + 6.0, 0.45, z + 12.0))
        b.add_box(MAT_KERB, (70.0, 0.35, 9.4), (x + 6.0, 0.9, z + 12.0), top_scale=0.98)
        for i in range(3):
            b.add_box(MAT_POLE, (70.0, 0.12, 0.14), (x + 6.0, 0.06, z + 18.0 + i * 1.6))
        # train: three coaches
        for i in range(3):
            coach_x = x - 16.0 + i * 17.0
            b.add_box(MAT_TRAIN, (16.0, 3.2, 3.0), (coach_x, 2.6, z + 18.8), top_scale=0.94)
            b.add_box(MAT_TRAIN_STRIPE, (16.0, 0.5, 3.06), (coach_x, 1.6, z + 18.8))
            for k in range(6):
                b.add_box(MAT_WINDOW, (1.4, 0.9, 0.1), (coach_x - 6.0 + k * 2.4, 3.2, z + 17.28))
                b.add_box(MAT_WINDOW, (1.4, 0.9, 0.1), (coach_x - 6.0 + k * 2.4, 3.2, z + 20.32))
            b.add_box(MAT_POLE, (0.3, 0.9, 21.0), (coach_x + 8.5, 0.45, z + 18.8))
        # station building with a clock tower
        b.add_box(MAT_KERB, (34.0, 7.5, 12.0), (x + 2.0, 3.75, z + 2.0), top_scale=0.99)
        b.add_box(MAT_ROOF, (34.6, 0.5, 12.6), (x + 2.0, 7.6, z + 2.0))
        b.add_box(MAT_KERB, (5.0, 12.0, 5.0), (x - 10.0, 6.0, z - 1.0), top_scale=0.8)
        b.add_box(MAT_ROOF, (5.6, 1.2, 5.6), (x - 10.0, 12.4, z - 1.0), top_scale=0.4)
        uv = self.atlas.uv("station")
        b.add_quad_auto(MAT_SIGN_FACE if uv else MAT_AWNING2,
                        [(x - 8.0, 5.2, z + 8.06), (x + 12.0, 5.2, z + 8.06),
                         (x + 12.0, 6.4, z + 8.06), (x - 8.0, 6.4, z + 8.06)],
                        uv=[(uv[0], uv[1]), (uv[2], uv[1]), (uv[2], uv[3]), (uv[0], uv[3])] if uv else None,
                        ref=(x + 2.0, 1.0, z + 4.0))
        for i in range(6):
            b.add_box(MAT_WINDOW, (2.0, 1.6, 0.1), (x - 12.0 + i * 5.0, 3.6, z + 8.05))
        # foot over bridge
        b.add_box(MAT_POLE, (40.0, 1.0, 4.0), (x + 6.0, 6.5, z + 14.0))
        for sx in (-14.0, 14.0):
            for sz in (-1.6, 1.6):
                b.add_cylinder(MAT_POLE, 0.2, 3.0, (x + 6.0 + sx, 3.0, z + 14.0 + sz), 8, axis="y")

    def _prop_school(self, b: MeshBuilder, prop: Prop) -> None:
        x, z = prop.x, prop.z
        b.add_box(MAT_KERB, (30.0, 8.0, 14.0), (x, 4.0, z), top_scale=0.99)
        b.add_box(MAT_AWNING2, (30.6, 0.4, 14.6), (x, 8.2, z))
        for i in range(5):
            for side in (-1, 1):
                b.add_box(MAT_WINDOW, (2.0, 1.5, 0.1), (x - 12.0 + i * 5.0, 4.6, z + side * 7.05))
        b.add_box(MAT_SHUTTER, (3.0, 2.4, 0.2), (x, 1.2, z + 7.05))
        # play area with a swing frame
        b.add_flat(MAT_DIRT, (26.0, 16.0), (x + 4.0, 0.04, z + 16.0), 4.0)
        for sx in (-3.0, 3.0):
            b.add_cylinder(MAT_POLE, 0.12, 1.6, (x + 4.0 + sx, 1.6, z + 16.0), 8, axis="y")
        b.add_box(MAT_POLE, (6.6, 0.14, 0.14), (x + 4.0, 3.1, z + 16.0))
        b.add_box(MAT_AWNING, (0.5, 0.1, 0.5), (x + 2.0, 1.4, z + 16.0))
        b.add_box(MAT_AWNING, (0.5, 0.1, 0.5), (x + 6.0, 1.4, z + 16.0))

    def _prop_market_gate(self, b: MeshBuilder, prop: Prop) -> None:
        x, z = prop.x, prop.z
        for side in (-1, 1):
            b.add_box(MAT_AWNING2, (1.2, 7.0, 1.2), (x + side * 8.0, 3.5, z))
        b.add_box(MAT_AWNING2, (18.0, 1.6, 1.4), (x, 7.6, z), top_scale=0.9)
        uv = self.atlas.uv("market_gate") or self.atlas.uv("shop0")
        b.add_quad_auto(MAT_SIGN_FACE if uv else MAT_AWNING2,
                        [(x - 7.0, 6.9, z + 0.75), (x + 7.0, 6.9, z + 0.75),
                         (x + 7.0, 8.2, z + 0.75), (x - 7.0, 8.2, z + 0.75)],
                        uv=[(uv[0], uv[1]), (uv[2], uv[1]), (uv[2], uv[3]), (uv[0], uv[3])] if uv else None,
                        ref=(x, 1.0, z - 1.0))
        b.add_quad_auto(MAT_SIGN_FACE if uv else MAT_AWNING2,
                        [(x - 7.0, 6.9, z - 0.75), (x + 7.0, 6.9, z - 0.75),
                         (x + 7.0, 8.2, z - 0.75), (x - 7.0, 8.2, z - 0.75)],
                        uv=[(uv[0], uv[1]), (uv[2], uv[1]), (uv[2], uv[3]), (uv[0], uv[3])] if uv else None,
                        ref=(x, 1.0, z + 1.0))

    def _prop_parking_lot(self, b: MeshBuilder, prop: Prop) -> None:
        lot: ParkingLot = prop.extra["lot"]          # type: ignore[assignment]
        b.add_flat(MAT_ASPHALT, (lot.bays_per_row * lot.bay_width + 6.0,
                                 lot.rows * (lot.bay_length + 0.6) + 8.0),
                   (lot.x, 0.045, lot.z), 6.0)
        for i in range(lot.rows * lot.bays_per_row):
            bx, bz, yaw = lot.bay_transform(i)
            hw = lot.bay_width * 0.5
            # The bay's long axis is +X when yaw == 0, +Z when yaw == 90 degrees.
            if abs(math.cos(yaw)) > 0.7:
                for side in (-1, 1):
                    b.add_box(MAT_LINE_WHITE, (0.12, 0.02, lot.bay_length),
                              (bx, 0.055, bz + side * hw))
                b.add_box(MAT_LINE_WHITE, (lot.bay_width, 0.02, 0.12),
                          (bx - lot.bay_length * 0.5, 0.055, bz))
            else:
                for side in (-1, 1):
                    b.add_box(MAT_LINE_WHITE, (lot.bay_length, 0.02, 0.12),
                              (bx + side * hw, 0.055, bz))
                b.add_box(MAT_LINE_WHITE, (0.12, 0.02, lot.bay_width),
                          (bx, 0.055, bz - lot.bay_length * 0.5))

    # ------------------------------------------------------------------ lighting
    def update(self, dt: float, time_cycle: bool = True) -> None:
        if self.time_preset == "cycle" and time_cycle:
            self.time_hours = (self.time_hours + dt / S.DAY_LENGTH_SECONDS * 24.0) % 24.0
        target_rain = {"clear": 0.0, "cloudy": 0.0, "rain": 0.85}[self.weather]
        self.rain_intensity += (target_rain - self.rain_intensity) * min(1.0, dt * 0.4)
        self._update_lighting_flags()

    def _update_lighting_flags(self) -> None:
        h = self.time_hours
        self.streetlights_on = h < 6.4 or h > 18.0

    def set_time(self, preset: str) -> None:
        self.time_preset = preset
        if preset in S.TIME_PRESETS:
            self.time_hours = float(S.TIME_PRESETS[preset])
        self._update_lighting_flags()          # street lights react immediately

    def set_weather(self, weather: str) -> None:
        self.weather = weather if weather != "random" else self.rng.choice(
            ["clear", "cloudy", "rain"])

    @property
    def is_night(self) -> bool:
        return self.time_hours < 6.0 or self.time_hours > 19.0

    def sun_direction(self) -> Vec3:
        """Direction *from the surface towards the sun* (used by the shader)."""
        angle = (self.time_hours - 6.0) / 12.0 * math.pi     # 6h => sunrise, 18h => sunset
        elevation = math.sin(angle) * 1.05
        azimuth = angle * 0.6 - 0.4
        x = math.cos(azimuth) * 0.8
        y = max(elevation, -0.35)
        z = math.sin(azimuth) * 0.8
        length = math.sqrt(x * x + y * y + z * z) or 1.0
        return (x / length, y / length, z / length)

    def lighting(self) -> Dict[str, object]:
        """All shader lighting uniforms for the current time and weather."""
        h = self.time_hours
        rain = self.rain_intensity
        # dawn ramp 04:00 -> 06:30, dusk ramp 17:30 -> 20:00 (so 11:30 is full day and
        # 22:24 is full night, matching settings.TIME_PRESETS)
        dusk_centre = 18.9 if h >= 12.0 else 5.4
        dusk = max(0.0, 1.0 - abs(h - dusk_centre) / 2.2)
        if h >= 12.0:
            night = (h - 17.5) / 2.5           # 17:30 -> 0, 20:00 -> 1
        else:
            night = (6.5 - h) / 2.5            # 06:30 -> 0, 04:00 -> 1
        night = max(0.0, min(1.0, night))
        day = 1.0 - max(night, dusk * 0.55)

        # base palettes
        sky_day = (0.35, 0.55, 0.86)
        sky_night = (0.015, 0.025, 0.06)
        horizon_day = (0.72, 0.82, 0.94)
        horizon_night = (0.05, 0.06, 0.12)
        sun_day = (1.0, 0.96, 0.86)
        sun_dusk = (1.0, 0.55, 0.26)
        sun_night = (0.55, 0.62, 0.85)

        def mix(a, b, t):
            return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))

        zenith = mix(sky_night, sky_day, day)
        horizon = mix(horizon_night, horizon_day, day)
        horizon = mix(horizon, (0.98, 0.62, 0.35), dusk * 0.8)
        sun_color = mix(sun_night, mix(sun_day, sun_dusk, dusk), max(day, 0.25) if night < 0.9 else 0.2)
        light = mix((0.10, 0.12, 0.20), (1.0, 0.97, 0.90), day)
        light = mix(light, (0.55, 0.30, 0.18), dusk * 0.5)
        ambient_sky = mix((0.05, 0.07, 0.13), (0.42, 0.48, 0.58), day)
        ambient_ground = mix((0.03, 0.03, 0.05), (0.24, 0.22, 0.18), day)
        fog = mix((0.03, 0.04, 0.09), (0.62, 0.72, 0.86), day)
        fog = mix(fog, (0.75, 0.75, 0.78), rain * 0.6)
        density = 0.0055 * float(self.quality.get("fog_scale", 1.0))
        density *= (1.0 + rain * 0.9 + (1.0 - day) * 0.35)

        return {
            "sun_dir": self.sun_direction(),
            "sun_color": sun_color,
            "light": light,
            "ambient_sky": ambient_sky,
            "ambient_ground": ambient_ground,
            "fog_color": fog,
            "fog_density": density,
            "zenith": zenith,
            "horizon": horizon,
            "ground": mix((0.02, 0.02, 0.03), (0.28, 0.26, 0.22), day),
            "star_intensity": max(0.0, 1.0 - day * 2.2),
            "night_factor": 1.0 - day,
            "rain": rain,
        }

    # ------------------------------------------------------------------ accessors
    def colliders(self) -> List[Tuple[float, float, float, float, float]]:
        """(x, z, half_x, half_z, height) for every building, for collision queries."""
        out = []
        for b in self.buildings:
            hx, hz = b.half_extents()
            out.append((b.x, b.z, hx, hz, b.height))
        return out

    def chunk_list(self) -> List[Chunk]:
        return list(self.chunks.values())

    def statistics(self) -> Dict[str, int]:
        tris = 0
        for chunk in self.chunks.values():
            if chunk.mesh is not None:
                tris += chunk.mesh.triangle_count
        return {"buildings": len(self.buildings), "props": len(self.props),
                "chunks": len(self.chunks), "triangles": tris,
                "signs": len(self.sign_lookup)}


def _district_info(name: str) -> Dict[str, object]:
    from .road import DISTRICTS
    return DISTRICTS.get(name, DISTRICTS["residential"])
