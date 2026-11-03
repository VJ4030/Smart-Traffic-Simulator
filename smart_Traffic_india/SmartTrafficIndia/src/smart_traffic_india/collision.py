"""Collision detection and response.

Techniques used
---------------
* 2D **oriented bounding boxes (OBB)** + the **separating axis theorem (SAT)** give an
  exact overlap test *and* the minimum translation vector, which is what we need to push
  the player out of whatever it hit (buildings, kerbs, other vehicles).
* A **uniform spatial hash** keeps the vehicle-vs-vehicle pass near-linear: only vehicles
  that share a grid cell are tested against each other.
* Per-vehicle **collision cooldowns** stop a single scrape from being reported (and
  penalised) on every frame.

Everything is pure Python so it is unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .road import Hazard
from .vehicle import Vehicle

Point = Tuple[float, float]

CELL_SIZE = 14.0


@dataclass
class CollisionEvent:
    """Something hit something."""

    kind: str                  # vehicle | building | barrier | pedestrian
    a_name: str
    b_name: str
    severity: float            # 0..2, relative impact speed
    position: Point
    normal: Point              # MTV normal, from a towards b
    depth: float
    is_player: bool = False

    @property
    def label(self) -> str:
        return f"{self.a_name} -> {self.b_name}"


def ground_height(x: float, z: float, hazards: Sequence[Hazard]) -> float:
    """Height of the drivable surface at *(x, z)*.

    The road itself is flat; only speed breakers (``Hazard.kind == "speed_breaker"``)
    raise the surface, which lets the suspension, the camera and the wheel animation
    react to real humps instead of pretending the road is a plane.
    """
    for hazard in hazards:
        if hazard.kind != "speed_breaker":
            continue
        if hazard.contains(x, z):
            # parabolic hump: the breaker is ~1.5 m along the road, wide across it
            dx = x - hazard.x
            dz = z - hazard.z
            t = (dz if abs(hazard.across) > 0.1 else dx) / 0.75
            return 0.16 * max(0.0, 1.0 - t * t)
    return 0.0


def obb_corners(x: float, z: float, yaw: float, half_length: float,
                half_width: float) -> List[Point]:
    """The four corners of a vehicle's footprint."""
    fx, fz = math.cos(yaw), -math.sin(yaw)
    rx, rz = -fz, fx
    return [
        (x + fx * half_length + rx * half_width, z + fz * half_length + rz * half_width),
        (x + fx * half_length - rx * half_width, z + fz * half_length - rz * half_width),
        (x - fx * half_length - rx * half_width, z - fz * half_length - rz * half_width),
        (x - fx * half_length + rx * half_width, z - fz * half_length + rz * half_width),
    ]


def vehicle_corners(vehicle: Vehicle) -> List[Point]:
    return obb_corners(vehicle.x, vehicle.z, vehicle.yaw,
                       vehicle.half_length, vehicle.half_width)


def _project(points: Sequence[Point], axis: Point) -> Tuple[float, float]:
    dots = [p[0] * axis[0] + p[1] * axis[1] for p in points]
    return min(dots), max(dots)


def sat_overlap(a: Sequence[Point], b: Sequence[Point]) -> Optional[Tuple[float, Point]]:
    """Separating axis test for two convex polygons.

    Returns ``(depth, normal)`` (minimum translation vector pushing ``a`` away from ``b``)
    or ``None`` when the boxes do not overlap.
    """
    best_depth = float("inf")
    best_axis: Point = (0.0, 0.0)
    for poly in (a, b):
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            axis = (-(p2[1] - p1[1]), p2[0] - p1[0])
            length = math.hypot(axis[0], axis[1])
            if length < 1e-9:
                continue
            axis = (axis[0] / length, axis[1] / length)
            min_a, max_a = _project(a, axis)
            min_b, max_b = _project(b, axis)
            overlap = min(max_a, max_b) - max(min_a, min_b)
            if overlap <= 0.0:
                return None
            if overlap < best_depth:
                best_depth = overlap
                best_axis = axis
    # orient the axis so it pushes a away from b
    centre_a = (sum(p[0] for p in a) / len(a), sum(p[1] for p in a) / len(a))
    centre_b = (sum(p[0] for p in b) / len(b), sum(p[1] for p in b) / len(b))
    if ((centre_a[0] - centre_b[0]) * best_axis[0] +
            (centre_a[1] - centre_b[1]) * best_axis[1]) < 0.0:
        best_axis = (-best_axis[0], -best_axis[1])
    return best_depth, best_axis


def point_in_obb(px: float, pz: float, x: float, z: float, yaw: float,
                 half_length: float, half_width: float) -> bool:
    fx, fz = math.cos(yaw), -math.sin(yaw)
    rx, rz = -fz, fx
    dx, dz = px - x, pz - z
    along = dx * fx + dz * fz
    across = dx * rx + dz * rz
    return abs(along) <= half_length and abs(across) <= half_width


class SpatialHash:
    """Uniform grid so only nearby vehicles are ever compared."""

    def __init__(self, cell: float = CELL_SIZE) -> None:
        self.cell = cell
        self.cells: Dict[Tuple[int, int], List[int]] = {}

    def clear(self) -> None:
        self.cells.clear()

    def insert(self, index: int, x: float, z: float) -> None:
        key = (int(math.floor(x / self.cell)), int(math.floor(z / self.cell)))
        self.cells.setdefault(key, []).append(index)

    def query(self, x: float, z: float) -> List[int]:
        cx = int(math.floor(x / self.cell))
        cz = int(math.floor(z / self.cell))
        out: List[int] = []
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                out.extend(self.cells.get((cx + dx, cz + dz), ()))
        return out


def _physics(agent):
    """Return the physical :class:`Vehicle` behind an agent (AI wrappers proxy one)."""
    return getattr(agent, "vehicle", agent)


class CollisionSystem:
    """Runs the per-frame collision pass and applies the responses."""

    def __init__(self, network, environment=None,
                 cooldown: float = 0.85, ai_cooldown: float = 1.2) -> None:
        self.network = network
        self.environment = environment
        self.cooldown = cooldown
        self.ai_cooldown = ai_cooldown
        self.hash = SpatialHash()
        self.stats = {"vehicle_pairs": 0, "obstacle_tests": 0, "hits": 0}
        self._building_grid: Optional[Dict[Tuple[int, int], List[int]]] = None
        self._buildings: List[Tuple[float, float, float, float, float]] = []
        if environment is not None:
            self.set_buildings(environment.colliders())

    # ------------------------------------------------------------------ buildings
    def set_buildings(self, buildings: Sequence[Tuple[float, float, float, float, float]]) -> None:
        self._buildings = list(buildings)
        self._building_grid = {}
        for index, (x, z, hx, hz, _h) in enumerate(self._buildings):
            keys = set()
            step = CELL_SIZE
            x0, x1 = x - hx, x + hx
            z0, z1 = z - hz, z + hz
            ix = int(math.floor(x0 / step))
            while ix * step <= x1:
                iz = int(math.floor(z0 / step))
                while iz * step <= z1:
                    keys.add((ix, iz))
                    iz += 1
                ix += 1
            for key in keys:
                self._building_grid.setdefault(key, []).append(index)

    def buildings_near(self, x: float, z: float) -> List[Tuple[float, float, float, float, float]]:
        if not self._building_grid:
            return []
        key = (int(math.floor(x / CELL_SIZE)), int(math.floor(z / CELL_SIZE)))
        out = []
        for kx in (key[0] - 1, key[0], key[0] + 1):
            for kz in (key[1] - 1, key[1], key[1] + 1):
                for index in self._building_grid.get((kx, kz), ()):
                    out.append(self._buildings[index])
        return out

    def building_overlap(self, vehicle: Vehicle) -> Optional[CollisionEvent]:
        """Test the vehicle against nearby buildings (axis aligned boxes)."""
        corners = vehicle_corners(vehicle)
        for bx, bz, hx, hz, _h in self.buildings_near(vehicle.x, vehicle.z):
            box = obb_corners(bx, bz, 0.0, hx, hz)
            hit = sat_overlap(corners, box)
            if hit is None:
                continue
            depth, normal = hit
            if depth <= 0.0:
                continue
            return CollisionEvent("building", vehicle.name, "building", 1.0,
                                  (vehicle.x, vehicle.z), (-normal[0], -normal[1]), depth,
                                  False)
        return None

    # ------------------------------------------------------------------- vehicles
    def check_vehicles(self, player: Optional[Vehicle],
                       traffic: Sequence[Vehicle]) -> List[CollisionEvent]:
        """Player-vs-traffic and traffic-vs-traffic overlap pass."""
        events: List[CollisionEvent] = []
        self.stats["vehicle_pairs"] = 0
        self.hash.clear()
        for i, v in enumerate(traffic):
            self.hash.insert(i, v.x, v.z)
        # --- traffic against traffic -------------------------------------------------
        tested = set()
        for i, v in enumerate(traffic):
            for j in self.hash.query(v.x, v.z):
                if j <= i or (i, j) in tested:
                    continue
                tested.add((i, j))
                other = traffic[j]
                if math.hypot(other.x - v.x, other.z - v.z) > (v.half_length + other.half_length +
                                                               v.half_width + other.half_width):
                    continue
                self.stats["vehicle_pairs"] += 1
                hit = sat_overlap(vehicle_corners(v), vehicle_corners(other))
                if hit is None:
                    continue
                depth, normal = hit
                self._separate(v, other, depth, normal)
                events.append(CollisionEvent("vehicle", v.name, other.name, 0.5,
                                             ((v.x + other.x) * 0.5, (v.z + other.z) * 0.5),
                                             normal, depth, False))
        # --- player against traffic ---------------------------------------------------
        # The player is *always* pushed out of an overlap (no clipping through traffic) but
        # a collision is only *reported* - damage, score, camera shake - once per cooldown,
        # so resting against a bumper cannot drain the score 60 times a second.
        if player is not None:
            hot = player.collision_cooldown <= 0.0
            for j in self.hash.query(player.x, player.z):
                other = traffic[j]
                distance = math.hypot(other.x - player.x, other.z - player.z)
                if distance > player.half_length + other.half_length + 6.0:
                    continue
                self.stats["vehicle_pairs"] += 1
                hit = sat_overlap(vehicle_corners(player), vehicle_corners(other))
                if hit is None:
                    continue
                depth, normal = hit
                if not hot:
                    self._separate(player, other, depth, normal)
                    continue
                relative = abs(player.speed - other.speed) + abs(player.speed) * 0.35
                severity = min(2.0, max(0.25, relative / 8.0))
                events.append(CollisionEvent("vehicle", player.name, other.name, severity,
                                             ((player.x + other.x) * 0.5,
                                              (player.z + other.z) * 0.5),
                                             normal, depth, True))
                self._bounce(player, other, depth, normal, severity)
                self.arm_player_cooldown(player)
        self.stats["obstacle_tests"] += len(tested)
        self.stats["hits"] += len(events)
        return events

    def _separate(self, a, b, depth: float, normal: Point) -> None:
        """Softly push two AI vehicles apart and slow the rear one down.

        ``a`` / ``b`` may be either raw :class:`~.vehicle.Vehicle` objects or the
        :class:`~.traffic_ai.TrafficVehicle` wrappers that AI traffic uses; the wrapper
        exposes read-only proxies, so the push is applied to the physical body.
        """
        push = min(depth * 0.5, 0.35) * 0.5
        body_a = _physics(a)
        body_b = _physics(b)
        body_a.x += normal[0] * push
        body_a.z += normal[1] * push
        body_b.x -= normal[0] * push
        body_b.z -= normal[1] * push
        # NOTE: no speed cut here.  Traffic-vs-traffic separation runs every frame while
        # two bodies touch, so cutting the desired speed here would turn a light bumper
        # touch into a permanent crawl.  The car-following model is responsible for
        # slowing down; only a *player* impact (``_bounce``) kills speed.

    def arm_player_cooldown(self, player: Vehicle) -> None:
        """Start the damage/score cooldown after a reported player impact.

        ``Vehicle.step`` ticks ``collision_cooldown`` down every frame, and
        :meth:`check_vehicles` only *reports* an impact while it is zero - so a car resting
        against a bus bumper cannot drain the score sixty times a second.
        """
        player.collision_cooldown = max(player.collision_cooldown, self.cooldown)

    def _bounce(self, player: Vehicle, other, depth: float, normal: Point,
                severity: float) -> None:
        """Push the player out and convert the impact into damage, shake and speed loss."""
        push = min(depth, 0.6)
        player.x += normal[0] * push
        player.z += normal[1] * push
        player.apply_impact(severity, "collision")
        body = _physics(other)
        body.x -= normal[0] * push * 0.5
        body.z -= normal[1] * push * 0.5
        if hasattr(other, "block_speed"):
            other.block_speed(0.5, 1.4)

    # ------------------------------------------------------------------- hazards
    def hazard_contact(self, vehicle: Vehicle, hazards: Sequence[Hazard]) -> List[Hazard]:
        """Which hazards the vehicle is currently physically touching."""
        out = []
        for hazard in hazards:
            if hazard.contains(vehicle.x, vehicle.z):
                out.append(hazard)
        return out

    def ground_height(self, x: float, z: float, hazards: Sequence[Hazard]) -> float:
        """Road height under a point - only speed breakers raise it (they are real humps)."""
        return ground_height(x, z, hazards)
