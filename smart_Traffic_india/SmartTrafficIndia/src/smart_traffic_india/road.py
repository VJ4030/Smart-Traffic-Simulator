"""Road network, junctions, lane graph and road-surface mesh generation.

The city is a grid of axis-aligned roads.  Every road is described by a
:class:`Road` record; where two perpendicular roads cross we automatically create a
:class:`Junction`.  From this description the module builds:

* the asphalt / kerb / footpath / road-marking geometry (one merged mesh, a handful of
  draw calls),
* a **lane graph** used by the AI traffic (waypoints + successors + turn restrictions),
* hazard records (potholes, speed breakers, water logging, construction zones) which are
  both drawn *and* collide with the player's vehicle.

Coordinates: X is east, Z is north, the ground plane is Y = 0.  A road whose ``axis`` is
``"z"`` runs along the Z axis (its lateral coordinate is X).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .mesh import Material, MeshBuilder, MeshData
from . import settings as S

# --------------------------------------------------------------------------------------
# materials
# --------------------------------------------------------------------------------------
MAT_ASPHALT = Material("asphalt", S.P.ASPHALT, specular=0.06)
MAT_ASPHALT_WORN = Material("asphalt_worn", S.P.ASPHALT_WORN, specular=0.04)
MAT_PATCH = Material("asphalt_patch", (0.115, 0.115, 0.125), specular=0.05)
MAT_POTHOLE = Material("pothole", (0.055, 0.052, 0.048), specular=0.02)
MAT_WATER = Material("water", S.P.WATER, specular=0.75)
MAT_KERB = Material("kerb", S.P.KERB, specular=0.10)
MAT_FOOTPATH = Material("footpath", S.P.FOOTPATH, specular=0.08)
MAT_LINE_WHITE = Material("line_white", S.P.LINE_WHITE, specular=0.05)
MAT_LINE_YELLOW = Material("line_yellow", S.P.LINE_YELLOW, specular=0.05)
MAT_ZEBRA = Material("zebra", S.P.ZEBRA, specular=0.05)
MAT_BREAKER = Material("speed_breaker", (0.85, 0.72, 0.18), specular=0.12)
MAT_BREAKER_STRIPE = Material("breaker_stripe", (0.12, 0.12, 0.12), specular=0.10)
MAT_DIRT = Material("dirt", S.P.DIRT, specular=0.02)
MAT_BARRICADE = Material("barricade", S.P.BARRICADE, specular=0.15)
MAT_CONSTRUCTION = Material("construction", (0.55, 0.48, 0.35), specular=0.03)
MAT_DIVIDER = Material("divider", (0.80, 0.80, 0.76), specular=0.10)


# --------------------------------------------------------------------------------------
# description records
# --------------------------------------------------------------------------------------
@dataclass
class Road:
    """A straight road segment."""

    name: str
    axis: str                 # "z": runs along Z (fixed X) | "x": runs along X (fixed Z)
    center: float             # lateral coordinate of the road centre line
    start: float              # extent along the axis
    end: float
    half_width: float
    lanes_per_dir: int = 2
    lane_width: float = 3.4
    y_level: float = 0.015
    speed_limit: float = 50.0     # km/h
    district: str = "downtown"
    congestion: float = 1.0       # spawn weight multiplier
    condition: str = "smooth"     # smooth | cracked | patched | damaged
    median: bool = False          # raised central divider

    @property
    def length(self) -> float:
        return abs(self.end - self.start)

    def position_at(self, s: float, lateral: float) -> Tuple[float, float]:
        """World (x, z) at distance ``s`` along the road and ``lateral`` offset."""
        if self.axis == "z":
            return self.center + lateral, s
        return s, self.center + lateral

    def contains(self, s: float, margin: float = 0.0) -> bool:
        lo, hi = min(self.start, self.end), max(self.start, self.end)
        return lo - margin <= s <= hi + margin

    def lateral(self, x: float, z: float) -> float:
        return (x - self.center) if self.axis == "z" else (z - self.center)

    def along(self, x: float, z: float) -> float:
        return z if self.axis == "z" else x

    def nearest_point(self, x: float, z: float) -> Tuple[float, float]:
        s = min(max(self.along(x, z), min(self.start, self.end)), max(self.start, self.end))
        return self.position_at(s, 0.0)

    def on_surface(self, x: float, z: float, margin: float = 0.0) -> bool:
        return (self.contains(self.along(x, z)) and
                abs(self.lateral(x, z)) <= self.half_width + margin)

    def lane_offsets(self, direction: int) -> List[float]:
        """Lateral offsets (relative to the road centre) of each lane, left-hand traffic.

        A driver heading in the +direction keeps to the left, therefore the lanes for
        ``direction == +1`` sit on the ``+lateral`` side for a Z road and on the
        ``-lateral`` side for an X road (see docs/COMPUTER_GRAPHICS.md, section
        "coordinate systems").
        """
        offsets = []
        for i in range(self.lanes_per_dir):
            distance = self.lane_width * (i + 0.5)
            offsets.append(distance if self._left_is_positive(direction) else -distance)
        return offsets

    def _left_is_positive(self, direction: int) -> bool:
        # For a Z road heading +Z: left is +X.  For an X road heading +X: left is -Z.
        if self.axis == "z":
            return direction > 0
        return direction < 0

    def inbound_dir(self, s: float) -> int:
        return 1 if self.end >= self.start else -1

    def direction_yaw(self, direction: int) -> float:
        """Yaw (see transform.rotation_y) of a vehicle driving in ``direction``."""
        if self.axis == "z":
            return -math.pi / 2 if direction > 0 else math.pi / 2
        return 0.0 if direction > 0 else math.pi

    def parking_yaw(self, side: int) -> float:
        """Yaw of a car parked against the kerb on ``side`` of the road (left-hand
        traffic: a parked car faces the same way as the adjacent lane)."""
        forward = 1 if self.end >= self.start else -1
        if self.axis == "z":
            direction = forward if side > 0 else -forward
        else:
            direction = forward if side < 0 else -forward
        return self.direction_yaw(direction)


@dataclass
class Junction:
    """Crossing of two roads."""

    id: int
    x: float
    z: float
    road_a: Road
    road_b: Road
    name: str
    signalised: bool = False
    half_size: float = 9.0
    stop_offset: float = 7.0     # how far before the junction box vehicles stop

    @property
    def position(self) -> Tuple[float, float]:
        return self.x, self.z

    def approach_dir(self, road: Road) -> Tuple[int, int]:
        """(axis index, sign) describing how traffic enters this junction."""
        if road.axis == "z":
            return 0, (1 if self.z > road.center else -1)
        return 1, (1 if self.x > road.center else -1)


@dataclass
class Hazard:
    """A physical road hazard (pothole, speed breaker, water logging ...)."""

    kind: str                 # pothole | speed_breaker | water | construction | debris
    x: float
    z: float
    radius: float = 1.2
    strength: float = 1.0
    across: float = 0.0       # rotation of the hazard rectangle (radians)
    width: float = 2.0
    tag: str = ""

    def contains(self, x: float, z: float) -> bool:
        if self.kind == "speed_breaker":
            dx = x - self.x
            dz = z - self.z
            cos_a, sin_a = math.cos(-self.across), math.sin(-self.across)
            lx = dx * cos_a - dz * sin_a
            lz = dx * sin_a + dz * cos_a
            return abs(lx) <= 0.75 and abs(lz) <= self.width * 0.5
        return (x - self.x) ** 2 + (z - self.z) ** 2 <= self.radius ** 2


# --------------------------------------------------------------------------------------
# the city layout
# --------------------------------------------------------------------------------------
def build_city_roads() -> List[Road]:
    """Hard-coded, hand-tuned layout of the city (about 1.1 km x 0.8 km)."""
    return [
        # ---- the main arterial road (NS) -------------------------------------------------
        Road("MG Road", "z", 0.0, -200.0, 200.0, 9.0, lanes_per_dir=2, lane_width=3.6,
             y_level=0.020, speed_limit=60.0, district="downtown", congestion=1.25,
             condition="patched", median=True),
        # ---- east-west roads -------------------------------------------------------------
        Road("Station Road", "x", -140.0, -95.0, 95.0, 5.4, lanes_per_dir=1, lane_width=4.0,
             y_level=0.018, speed_limit=45.0, district="railway", congestion=1.1,
             condition="cracked"),
        Road("Nehru Marg", "x", -70.0, -100.0, 100.0, 5.4, lanes_per_dir=1, lane_width=4.0,
             y_level=0.016, speed_limit=45.0, district="industrial", congestion=0.9,
             condition="patched"),
        Road("Gandhi Chowk Road", "x", 30.0, -110.0, 130.0, 5.2, lanes_per_dir=1, lane_width=3.8,
             y_level=0.014, speed_limit=40.0, district="downtown", congestion=1.25,
             condition="cracked"),
        Road("Market Street", "x", 75.0, -90.0, 90.0, 4.9, lanes_per_dir=1, lane_width=3.6,
             y_level=0.012, speed_limit=30.0, district="market", congestion=1.6,
             condition="damaged"),
        Road("Hospital Avenue", "x", 140.0, -85.0, 130.0, 5.2, lanes_per_dir=1, lane_width=3.8,
             y_level=0.013, speed_limit=40.0, district="hospital", congestion=1.15,
             condition="smooth"),
        # NOTE: the bypass sits at z = -100, not on the same centre line as Station Road
        # (z = -140): two collinear roads would overlap and confuse every road lookup.
        Road("Highway Bypass", "x", -100.0, 60.0, 175.0, 6.4, lanes_per_dir=1, lane_width=4.4,
             y_level=0.019, speed_limit=70.0, district="highway", congestion=0.75,
             condition="smooth"),
        # ---- north-south cross roads -----------------------------------------------------
        Road("Old City Lane", "z", -95.0, -150.0, 150.0, 4.6, lanes_per_dir=1, lane_width=3.4,
             y_level=0.011, speed_limit=30.0, district="oldcity", congestion=1.5,
             condition="damaged"),
        Road("Hospital Road", "z", 42.0, -150.0, 150.0, 5.0, lanes_per_dir=1, lane_width=3.8,
             y_level=0.014, speed_limit=40.0, district="hospital", congestion=1.2,
             condition="cracked"),
        Road("Bypass Link Road", "z", 120.0, -140.0, 175.0, 5.4, lanes_per_dir=1, lane_width=4.0,
             y_level=0.017, speed_limit=55.0, district="highway", congestion=0.7,
             condition="smooth"),
        Road("School Marg", "z", -40.0, -60.0, 60.0, 4.4, lanes_per_dir=1, lane_width=3.4,
             y_level=0.010, speed_limit=30.0, district="school", congestion=1.3,
             condition="cracked"),
    ]


DISTRICTS: Dict[str, Dict[str, object]] = {
    "downtown":  {"label": "Downtown",        "buildings": "concrete", "height": (16, 40),
                  "density": 0.85, "trees": 0.25, "hoardings": 0.6},
    "market":    {"label": "Main Bazaar",     "buildings": "shop",     "height": (6, 12),
                  "density": 1.0, "trees": 0.15, "hoardings": 1.0},
    "hospital":  {"label": "Civil Hospital",  "buildings": "concrete", "height": (12, 22),
                  "density": 0.6, "trees": 0.5, "hoardings": 0.25},
    "school":    {"label": "School Zone",     "buildings": "concrete", "height": (8, 14),
                  "density": 0.5, "trees": 0.7, "hoardings": 0.1},
    "railway":   {"label": "Railway Station", "buildings": "concrete", "height": (10, 24),
                  "density": 0.75, "trees": 0.3, "hoardings": 0.5},
    "industrial":{"label": "Industrial Estate", "buildings": "warehouse", "height": (8, 16),
                  "density": 0.65, "trees": 0.1, "hoardings": 0.3},
    "highway":   {"label": "Bypass Highway",  "buildings": "warehouse", "height": (6, 14),
                  "density": 0.35, "trees": 0.35, "hoardings": 0.7},
    "oldcity":   {"label": "Purani Basti",    "buildings": "shop",     "height": (5, 10),
                  "density": 0.95, "trees": 0.2, "hoardings": 0.8},
    "residential": {"label": "Residential",   "buildings": "house",    "height": (6, 12),
                    "density": 0.6, "trees": 0.6, "hoardings": 0.1},
}

#: Landmarks (also used as mission destinations).
LANDMARKS: Dict[str, Tuple[float, float]] = {
    "Civil Hospital": (42.0, -70.0),
    "Central Railway Station": (-140.0, -95.0),
    "City Bus Terminal": (-140.0, 30.0),
    "Main Bazaar": (0.0, 75.0),
    "Sunrise Public School": (-40.0, 40.0),
    "Industrial Estate": (120.0, -70.0),
    "Metro Mall Downtown": (42.0, 30.0),
    "Bypass Toll Plaza": (120.0, 175.0),
    "Old City Parking": (-95.0, 0.0),
    "Airport Road Junction": (0.0, 140.0),
}


class RoadNetwork:
    """Owns the road layout, junctions, hazards, lane graph and road geometry."""

    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)
        self.roads: List[Road] = build_city_roads()
        self.junctions: List[Junction] = []
        self.hazards: List[Hazard] = []
        self.construction_zones: List[Tuple[float, float, float, str]] = []   # x, z, radius, label
        self.lanes: List["Lane"] = []
        self._lane_successors: Dict[int, List[Tuple[int, Tuple[float, float], str]]] = {}
        self._build_junctions()
        self._build_lane_graph()
        self._build_hazards()

    # ------------------------------------------------------------------ construction
    def _build_junctions(self) -> None:
        counter = 0
        for i, a in enumerate(self.roads):
            for b in self.roads[i + 1:]:
                if a.axis == b.axis:
                    continue
                z_road = a if a.axis == "z" else b
                x_road = b if a.axis == "z" else a
                if not (z_road.contains(x_road.center) and x_road.contains(z_road.center)):
                    continue
                counter += 1
                # ``z_road`` runs along Z so its *center* is an X coordinate, and
                # ``x_road`` runs along X so its center is a Z coordinate - mapping the two
                # onto the right axes is what places every junction (and therefore every
                # signal, stop line and turning lane) at the real crossing point.
                x = z_road.center
                z = x_road.center
                signalised = (z_road.name == "MG Road" or
                              (z_road.lanes_per_dir > 1) or
                              (x_road.district == "market" and z_road.district in ("market", "oldcity")) or
                              (x_road.district == "downtown" and z_road.district == "downtown"))
                name = f"{z_road.name} x {x_road.name}"
                junc = Junction(counter, x, z, z_road, x_road, name, signalised,
                                half_size=max(z_road.half_width, x_road.half_width) + 2.0,
                                stop_offset=z_road.half_width + 5.0)
                self.junctions.append(junc)

    def _build_lane_graph(self) -> None:
        """Create one lane *segment* per (direction, lane, stretch between junctions).

        Splitting lanes at every junction is what lets the AI turn at intersections: each
        segment knows where it ends, which junction it feeds, and which segments it may
        continue on (see :meth:`_connect_lanes`).
        """
        lane_id = 0
        for road in self.roads:
            for direction in (1, -1):
                lo, hi = min(road.start, road.end), max(road.start, road.end)
                stops = [lo] + [j.z if road.axis == "z" else j.x
                                for j in self._junctions_on(road)] + [hi]
                stops = sorted(set(stops))
                if direction < 0:
                    stops.reverse()
                for index, offset in enumerate(road.lane_offsets(direction)):
                    for segment in range(len(stops) - 1):
                        s0, s1 = stops[segment], stops[segment + 1]
                        junc_here = self._junction_at(road, s1)
                        junc_before = self._junction_at(road, s0)
                        points: List[Tuple[float, float]] = []
                        clearance = min(11.0, abs(s1 - s0) * 0.42)
                        s_b = s1 - math.copysign(clearance, s1 - s0)
                        steps = max(1, int(abs(s_b - s0) / 13.0))
                        for step in range(steps):
                            s_i = s0 + (s_b - s0) * step / steps
                            points.append(road.position_at(s_i, offset))
                        points.append(road.position_at(s1, offset))
                        self.lanes.append(Lane(
                            lane_id, road, direction, index, offset, points, segment, s1,
                            junc_here.id if junc_here else None, junc_here,
                            junc_before.id if junc_before else None))
                        lane_id += 1
        self._connect_lanes()

    def _junctions_on(self, road: Road) -> List[Junction]:
        out = [j for j in self.junctions if (j.road_a is road or j.road_b is road)]
        out.sort(key=lambda j: (j.z if road.axis == "z" else j.x))
        return out

    def _junction_at(self, road: Road, s: float) -> Optional[Junction]:
        for j in self._junctions_on(road):
            coord = j.z if road.axis == "z" else j.x
            if abs(coord - s) < 0.75:
                return j
        return None

    def junction_near(self, x: float, z: float, radius: float = 18.0) -> Optional[Junction]:
        best, best_d = None, radius * radius
        for j in self.junctions:
            d = (j.x - x) ** 2 + (j.z - z) ** 2
            if d < best_d:
                best, best_d = j, d
        return best

    def _connect_lanes(self) -> None:
        """Work out which lane segment a vehicle may continue on after each junction."""
        for lane in self.lanes:
            options: List[Tuple[int, Tuple[float, float], str]] = []
            junction = lane.end_junction
            if junction is None:
                self._lane_successors[lane.id] = options
                continue
            end_point = lane.points[-1]
            # --- straight on: the next segment of the same lane ------------------------
            for other in self.lanes:
                if (other.road is lane.road and other.direction == lane.direction
                        and other.index == lane.index and other.segment == lane.segment + 1):
                    options.append((other.id, other.points[0], "straight"))
            # --- turns: segments of the crossing road that leave this junction ---------
            cross_road = junction.road_b if lane.road is junction.road_a else junction.road_a
            for other in self.lanes:
                if other.road is not cross_road or other.start_junction_id != junction.id:
                    continue
                first = other.points[0]
                if math.dist(first, end_point) > cross_road.half_width + 14.0:
                    continue
                second = other.points[1] if len(other.points) > 1 else first
                if math.dist(second, (junction.x, junction.z)) < math.dist(
                        first, (junction.x, junction.z)) - 0.5:
                    continue                       # this segment leads *into* the junction
                turn = "left" if self._is_left_turn(lane.road, lane.direction, cross_road,
                                                    other.direction) else "right"
                if lane.road.lanes_per_dir > 1 and turn == "right" and lane.index == 0:
                    continue                       # inner lane cannot swing across traffic
                options.append((other.id, first, turn))
            self._lane_successors[lane.id] = options

    @staticmethod
    def _is_left_turn(road: Road, direction: int, cross: Road, cross_direction: int) -> bool:
        """Is the turn onto ``cross`` (travelling ``cross_direction``) a left turn?"""
        if road.axis == "z":
            heading = (0.0, float(direction))
            new_heading = (float(cross_direction), 0.0)
        else:
            heading = (float(direction), 0.0)
            new_heading = (0.0, float(cross_direction))
        cross_z = heading[0] * new_heading[1] - heading[1] * new_heading[0]
        # left = counter-clockwise when +Y is up (see transform.right_from_dir)
        return cross_z > 0

    def successors(self, lane_id: int) -> List[Tuple[int, Tuple[float, float], str]]:
        return self._lane_successors.get(lane_id, [])

    def lane_by_id(self, lane_id: int) -> "Lane":
        return self.lanes[lane_id]

    # ------------------------------------------------------------------- hazards
    def _build_hazards(self) -> None:
        rng = self.rng
        # potholes - more of them on damaged roads
        for road in self.roads:
            count = {"smooth": 2, "cracked": 6, "patched": 8, "damaged": 11}[road.condition]
            for _ in range(count):
                s = rng.uniform(min(road.start, road.end) + 8, max(road.start, road.end) - 8)
                lateral = rng.uniform(-road.half_width * 0.85, road.half_width * 0.85)
                x, z = road.position_at(s, lateral)
                if self.junction_near(x, z, 14.0):
                    continue
                radius = rng.uniform(0.55, 1.35)
                self.hazards.append(Hazard("pothole", x, z, radius,
                                           strength=0.6 + (radius - 0.55) * 0.75,
                                           tag=road.district))
        # speed breakers near schools, hospitals and markets
        for road in self.roads:
            zones = {"school": 4, "hospital": 3, "market": 3}.get(road.district, 1)
            for i in range(zones):
                s = rng.uniform(min(road.start, road.end) + 10, max(road.start, road.end) - 10)
                x, z = road.position_at(s, 0.0)
                if self.junction_near(x, z, 20.0):
                    continue
                across = 0.0 if road.axis == "x" else math.pi / 2.0
                self.hazards.append(Hazard("speed_breaker", x, z, 0.8, 1.0, across,
                                           width=road.half_width * 1.9, tag=road.district))
        # water logging (low lying spots, worse in rain)
        for road in self.roads:
            if road.district in ("market", "oldcity", "industrial"):
                s = rng.uniform(min(road.start, road.end) + 12, max(road.start, road.end) - 12)
                x, z = road.position_at(s, rng.uniform(-1.5, 1.5))
                if self.junction_near(x, z, 16.0):
                    continue
                self.hazards.append(Hazard("water", x, z, rng.uniform(3.0, 5.0), 0.45,
                                           tag=road.district))
        # construction zones with a marked lane closure
        for road in self.roads:
            if road.condition in ("damaged", "patched") and rng.random() < 0.55:
                s = rng.uniform(min(road.start, road.end) + 20, max(road.start, road.end) - 20)
                x, z = road.position_at(s, 0.0)
                if self.junction_near(x, z, 24.0):
                    continue
                self.construction_zones.append((x, z, 6.0, road.name))
                self.hazards.append(Hazard("construction", x, z, 6.0, 1.0, tag=road.name))

    def hazards_near(self, x: float, z: float, radius: float) -> List[Hazard]:
        r2 = radius * radius
        return [h for h in self.hazards if (h.x - x) ** 2 + (h.z - z) ** 2 <= r2]

    def hazard_at(self, x: float, z: float) -> Optional[Hazard]:
        for h in self.hazards:
            if h.contains(x, z):
                return h
        return None

    # -------------------------------------------------------------------- queries
    def road_at(self, x: float, z: float, margin: float = 0.0) -> Optional[Road]:
        best = None
        best_lat = 1e9
        for road in self.roads:
            if road.on_surface(x, z, margin):
                lat = abs(road.lateral(x, z))
                if lat < best_lat:
                    best, best_lat = road, lat
        return best

    def on_road(self, x: float, z: float, margin: float = 0.0) -> bool:
        return self.road_at(x, z, margin) is not None

    def in_junction(self, x: float, z: float, margin: float = 0.0) -> Optional[Junction]:
        for j in self.junctions:
            if abs(x - j.x) <= j.half_size + margin and abs(z - j.z) <= j.half_size + margin:
                return j
        return None

    def direction_of_travel(self, road: Road, x: float, z: float) -> int:
        """Which direction the *legal* traffic flows on the given side of the road."""
        lateral = road.lateral(x, z)
        if road.axis == "z":
            return 1 if (lateral > 0) == (road.end >= road.start) else -1
        return 1 if (lateral < 0) == (road.end >= road.start) else -1

    def speed_limit_at(self, x: float, z: float) -> float:
        road = self.road_at(x, z, 1.0)
        if road is None:
            return 25.0
        if self.in_junction(x, z, 2.0):
            return min(road.speed_limit, 30.0)
        return road.speed_limit

    def district_at(self, x: float, z: float) -> str:
        road = self.road_at(x, z, 6.0)
        if road is not None:
            return road.district
        best, best_d = "residential", 1e9
        for road in self.roads:
            px, pz = road.nearest_point(x, z)
            d = (px - x) ** 2 + (pz - z) ** 2
            if d < best_d:
                best, best_d = road.district, d
        return best

    # --------------------------------------------------------------------- mesh
    def build_mesh(self) -> MeshData:
        """Generate every static road surface in a single merged mesh."""
        b = MeshBuilder("roads")
        for road in self.roads:
            self._build_road_surface(b, road)
        for road in self.roads:
            self._build_markings(b, road)
        for junc in self.junctions:
            self._build_junction(b, junc)
        self._build_footpaths(b)
        self._build_hazards_mesh(b)
        return b.build("roads")

    # -- surfaces ------------------------------------------------------------------
    def _build_road_surface(self, b: MeshBuilder, road: Road) -> None:
        lo, hi = min(road.start, road.end), max(road.start, road.end)
        length = hi - lo
        mid = (lo + hi) * 0.5
        material = MAT_ASPHALT if road.condition in ("smooth", "cracked") else MAT_ASPHALT_WORN
        if road.axis == "z":
            b.add_flat(material, (road.half_width * 2, length), (road.center, road.y_level, mid))
            if road.condition in ("patched", "damaged"):
                for _ in range(6):
                    s = self.rng.uniform(lo + 6, hi - 6)
                    b.add_flat(MAT_PATCH, (road.half_width * 1.9, self.rng.uniform(6, 16)),
                               (road.center, road.y_level + 0.001, s))
        else:
            b.add_flat(material, (length, road.half_width * 2), (mid, road.y_level, road.center))
            if road.condition in ("patched", "damaged"):
                for _ in range(6):
                    s = self.rng.uniform(lo + 6, hi - 6)
                    b.add_flat(MAT_PATCH, (self.rng.uniform(6, 16), road.half_width * 1.9),
                               (s, road.y_level + 0.001, road.center))
        # kerbs along both sides
        kerb_w = 0.35
        kerb_h = 0.16
        if road.axis == "z":
            for side in (-1, 1):
                cx = road.center + side * (road.half_width + kerb_w * 0.5)
                b.add_box(MAT_KERB, (kerb_w, kerb_h, length), (cx, road.y_level + kerb_h * 0.5, mid))
        else:
            for side in (-1, 1):
                cz = road.center + side * (road.half_width + kerb_w * 0.5)
                b.add_box(MAT_KERB, (length, kerb_h, kerb_w), (mid, road.y_level + kerb_h * 0.5, cz))
        # raised median on the main arterial
        if road.median:
            if road.axis == "z":
                b.add_box(MAT_DIVIDER, (0.6, 0.30, length - 30.0),
                          (road.center, road.y_level + 0.15, mid))
            else:
                b.add_box(MAT_DIVIDER, (length - 30.0, 0.30, 0.6),
                          (mid, road.y_level + 0.15, road.center))

    def _build_markings(self, b: MeshBuilder, road: Road) -> None:
        lo, hi = min(road.start, road.end), max(road.start, road.end)
        y = road.y_level + 0.02
        # dashed lane separators
        for sign in (1, -1):
            for lane_index in range(1, road.lanes_per_dir + 1):
                if lane_index == road.lanes_per_dir:
                    continue
                lateral = sign * (road.lane_width * lane_index)
                s = lo + 4.0
                while s < hi - 4.0:
                    if not self.junction_near(*road.position_at(s, lateral), 11.0):
                        if road.axis == "z":
                            b.add_flat(MAT_LINE_WHITE, (0.14, 2.4),
                                       (road.center + lateral, y, s + 1.2))
                        else:
                            b.add_flat(MAT_LINE_WHITE, (2.4, 0.14),
                                       (s + 1.2, y, road.center + lateral))
                    s += 6.5
        # centre line: double yellow where there is no median, single white dash otherwise
        if road.median:
            return
        for sign in (-1, 1):
            lateral = sign * 0.18
            s = lo + 3.0
            while s < hi - 3.0:
                if not self.junction_near(*road.position_at(s, lateral), 10.0):
                    if road.axis == "z":
                        b.add_flat(MAT_LINE_YELLOW, (0.12, 2.8), (road.center + lateral, y, s + 1.4))
                    else:
                        b.add_flat(MAT_LINE_YELLOW, (2.8, 0.12), (s + 1.4, y, road.center + lateral))
                s += 4.2

    def _build_junction(self, b: MeshBuilder, junc: Junction) -> None:
        size_a = junc.road_a.half_width * 2
        size_b = junc.road_b.half_width * 2
        y = max(junc.road_a.y_level, junc.road_b.y_level) + 0.004
        b.add_flat(MAT_ASPHALT, (size_b, size_a), (junc.x, y, junc.z))
        # zebra crossings on all four arms
        for road, sign in ((junc.road_a, 1), (junc.road_a, -1),
                           (junc.road_b, 1), (junc.road_b, -1)):
            offset = road.half_width + 1.6
            if road.axis == "z":
                cx, cz = junc.x, junc.z + sign * offset
            else:
                cx, cz = junc.x + sign * offset, junc.z
            stripes = 7
            span = road.half_width * 1.7
            for i in range(stripes):
                t = (i + 0.5) / stripes - 0.5
                if road.axis == "z":
                    b.add_flat(MAT_ZEBRA, (0.55, span / stripes * 0.62),
                               (cx + t * span, y + 0.001, cz))
                else:
                    b.add_flat(MAT_ZEBRA, (span / stripes * 0.62, 0.55),
                               (cx, y + 0.001, cz + t * span))
            # stop line
            if road.axis == "z":
                b.add_flat(MAT_LINE_WHITE, (road.half_width, 0.4),
                           (cx - sign * road.half_width * 0.5, y + 0.002,
                            junc.z + sign * (road.half_width + 3.0)))
            else:
                b.add_flat(MAT_LINE_WHITE, (0.4, road.half_width),
                           (junc.x + sign * (road.half_width + 3.0), y + 0.002,
                            cz - sign * road.half_width * 0.5))

    def _build_footpaths(self, b: MeshBuilder) -> None:
        """Tall pavements behind the kerb, interrupted around junctions."""
        for road in self.roads:
            lo, hi = min(road.start, road.end), max(road.start, road.end)
            cuts = sorted((j.z if road.axis == "z" else j.x) for j in self._junctions_on(road))
            segments: List[Tuple[float, float]] = []
            cursor = lo
            for cut in cuts:
                if cut - 12.0 > cursor:
                    segments.append((cursor, cut - 12.0))
                cursor = max(cursor, cut + 12.0)
            if hi > cursor:
                segments.append((cursor, hi))
            for s0, s1 in segments:
                mid = (s0 + s1) * 0.5
                length = s1 - s0
                for side in (-1, 1):
                    lat = side * (road.half_width + 2.6)
                    if road.axis == "z":
                        b.add_box(MAT_FOOTPATH, (4.4, 0.22, length),
                                  (road.center + lat, 0.11, mid))
                        b.add_box(MAT_KERB, (0.3, 0.24, length),
                                  (road.center + side * (road.half_width + 0.5), 0.12, mid))
                    else:
                        b.add_box(MAT_FOOTPATH, (length, 0.22, 4.4),
                                  (mid, 0.11, road.center + lat))
                        b.add_box(MAT_KERB, (length, 0.24, 0.3),
                                  (mid, 0.12, road.center + side * (road.half_width + 0.5)))

    def _build_hazards_mesh(self, b: MeshBuilder) -> None:
        y = 0.05
        for hazard in self.hazards:
            if hazard.kind == "pothole":
                segments = 12
                for i in range(segments):
                    a0 = 2.0 * math.pi * i / segments
                    b.add_box(MAT_POTHOLE, (hazard.radius * 0.85, 0.02, hazard.radius * 0.85),
                              (hazard.x + math.cos(a0) * hazard.radius * 0.7, y,
                               hazard.z + math.sin(a0) * hazard.radius * 0.7),
                              top_scale=0.8)
            elif hazard.kind == "speed_breaker":
                # real 3D hump: a rounded box that the vehicle physically bounces over
                stripes = 6
                for i in range(stripes):
                    t = (i + 0.5) / stripes
                    half = hazard.width * 0.5 / stripes
                    along = (t - 0.5) * hazard.width
                    mat = MAT_BREAKER if i % 2 == 0 else MAT_BREAKER_STRIPE
                    if abs(hazard.across) > 0.1:      # breaker lies across a Z road
                        b.add_box(mat, (1.6, 0.10 + 0.05, half * 1.9),
                                  (hazard.x, y + 0.02, hazard.z + along))
                    else:
                        b.add_box(mat, (half * 1.9, 0.10 + 0.05, 1.6),
                                  (hazard.x + along, y + 0.02, hazard.z))
            elif hazard.kind == "water":
                b.add_flat(MAT_WATER, (hazard.radius * 1.8, hazard.radius * 1.8),
                           (hazard.x, 0.035, hazard.z))
            elif hazard.kind == "construction":
                self._build_construction(b, hazard)

    def _build_construction(self, b: MeshBuilder, hazard: Hazard) -> None:
        outer = hazard.radius * 1.35
        b.add_flat(MAT_CONSTRUCTION, (outer * 1.5, outer * 1.5), (hazard.x, 0.04, hazard.z))
        # barricades around the dig
        for i in range(6):
            a = 2.0 * math.pi * i / 6.0
            cx = hazard.x + math.cos(a) * outer
            cz = hazard.z + math.sin(a) * outer
            b.add_box(MAT_BARRICADE, (2.0, 0.9, 0.22), (cx, 0.55, cz), top_scale=0.9)
            b.add_box(MAT_LINE_WHITE, (1.9, 0.12, 0.26), (cx, 0.75, cz))
        b.add_box(MAT_DIRT, (outer * 0.8, 0.5, outer * 0.8), (hazard.x, 0.25, hazard.z))


# --------------------------------------------------------------------------------------
# lanes
# --------------------------------------------------------------------------------------
@dataclass
class Lane:
    """One lane *segment*: a polyline of waypoints that AI vehicles follow."""

    id: int
    road: Road
    direction: int
    index: int                 # lane index (0 = nearest the centre line)
    offset: float              # lateral offset from the road centre
    points: List[Tuple[float, float]]
    segment: int = 0           # which stretch of the road (0 = before the first junction)
    end_stop: float = 0.0      # distance along the road where this segment ends
    end_junction_id: Optional[int] = None
    end_junction: Optional[Junction] = None
    start_junction_id: Optional[int] = None

    @property
    def axis(self) -> str:
        return self.road.axis

    @property
    def start_point(self) -> Tuple[float, float]:
        return self.points[0]

    @property
    def end_point(self) -> Tuple[float, float]:
        return self.points[-1]

    def point_after(self, position: Tuple[float, float], lookahead: float,
                    extra: Sequence[Tuple[float, float]] = ()) -> Tuple[float, float]:
        """Return a point roughly ``lookahead`` metres ahead of ``position`` along the
        lane (and optionally into the connected lane's waypoints)."""
        points = self.points + list(extra)
        best_i, best_d = 0, 1e18
        for i in range(len(points) - 1):
            d = math.dist(position, points[i])
            if d < best_d:
                best_i, best_d = i, d
        travelled = 0.0
        prev = points[best_i]
        for i in range(best_i + 1, len(points)):
            seg = math.dist(prev, points[i])
            if travelled + seg >= lookahead:
                need = lookahead - travelled
                t = need / max(seg, 1e-6)
                return (prev[0] + (points[i][0] - prev[0]) * t,
                        prev[1] + (points[i][1] - prev[1]) * t)
            travelled += seg
            prev = points[i]
        return points[-1]

    def remaining_to_end(self, position: Tuple[float, float]) -> float:
        best_i, best_d = 0, 1e18
        for i, p in enumerate(self.points):
            d = math.dist(position, p)
            if d < best_d:
                best_i, best_d = i, d
        total = best_d
        for i in range(best_i, len(self.points) - 1):
            total += math.dist(self.points[i], self.points[i + 1])
        return total
