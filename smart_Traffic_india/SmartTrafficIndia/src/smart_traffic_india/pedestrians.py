"""Pedestrians: sidewalk wanderers and crossing users.

Pedestrians keep to the pavements, gather at shop fronts in the bazaar, and cross the
road only on marked zebra crossings.  Before stepping onto the road they *check for
traffic*: if a vehicle is close and fast the pedestrian waits, which is exactly the
behaviour the AI traffic mirrors by slowing down near crossings.  Nobody teleports or
walks through a building.

Like the rest of the simulation this module is pure Python: the state machine can be
tested without a display.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .road import Road, RoadNetwork
from .vehicle import Vehicle

Vec2 = Tuple[float, float]

WALK_SPEED = 1.25
WAIT_SPEED = 0.0
CROSS_SPEED = 1.45
AVOID_RADIUS = 26.0
SAFE_GAP = 24.0


@dataclass
class Pedestrian:
    """A single pedestrian walking a two-point route (pavement or crossing)."""

    x: float
    z: float
    yaw: float = 0.0
    speed: float = WALK_SPEED
    color_index: int = 0
    state: str = "walk"          # walk | wait | cross
    route: List[Vec2] = field(default_factory=list)
    waypoint: int = 0
    wait_timer: float = 0.0
    crossing: bool = False
    bob: float = 0.0
    alive_timer: float = 0.0
    check_timer: float = 0.0

    @property
    def position(self) -> Vec2:
        return (self.x, self.z)

    def target(self) -> Vec2:
        if not self.route:
            return (self.x, self.z)
        return self.route[self.waypoint % len(self.route)]


class PedestrianManager:
    """Spawns and updates the pedestrians of the city."""

    def __init__(self, network: RoadNetwork, quality: Dict[str, object],
                 seed: int = 5) -> None:
        self.network = network
        self.rng = random.Random(seed)
        self.quality = quality
        self.pedestrians: List[Pedestrian] = []
        self.max_pedestrians = int(26 * float(quality.get("pedestrian_scale", 1.0)))
        self.crossings = self._collect_crossings()
        self.routes = self._build_pavement_routes()
        self.spawn_timer = 0.0

    # ------------------------------------------------------------------ setup
    def _collect_crossings(self) -> List[Tuple[float, float, float, float]]:
        """Zebra crossings as ``(x0, z0, x1, z1)`` segments across the road."""
        out = []
        for junc in self.network.junctions:
            for road in (junc.road_a, junc.road_b):
                for sign in (1, -1):
                    offset = road.half_width + 1.6
                    if road.axis == "z":
                        cx, cz = junc.x, junc.z + sign * offset
                        out.append((cx - road.half_width, cz, cx + road.half_width, cz))
                    else:
                        cx, cz = junc.x + sign * offset, junc.z
                        out.append((cx, cz - road.half_width, cx, cz + road.half_width))
        return out

    def _build_pavement_routes(self) -> List[Tuple[Vec2, Vec2]]:
        """Straight pavement strolls along the kerb."""
        routes: List[Tuple[Vec2, Vec2]] = []
        for road in self.network.roads:
            lo, hi = min(road.start, road.end), max(road.start, road.end)
            for side in (-1, 1):
                s = lo + 10.0
                while s < hi - 24.0:
                    length = self.rng.uniform(16.0, 34.0)
                    a = road.position_at(s, side * (road.half_width + 3.2))
                    b = road.position_at(min(s + length, hi - 8.0), side * (road.half_width + 3.2))
                    if not self.network.junction_near(a[0], a[1], 12.0):
                        routes.append((a, b))
                    s += length + self.rng.uniform(6.0, 18.0)
        return routes

    # ------------------------------------------------------------------ spawning
    def spawn(self, near: Vec2, min_distance: float = 30.0,
              max_distance: float = 120.0) -> Optional[Pedestrian]:
        if len(self.pedestrians) >= self.max_pedestrians:
            return None
        crossing_mode = self.rng.random() < 0.35 and self.crossings
        if crossing_mode:
            x0, z0, x1, z1 = self.rng.choice(self.crossings)
            distance = math.hypot((x0 + x1) * 0.5 - near[0], (z0 + z1) * 0.5 - near[1])
            if distance < 22.0 or distance > 150.0:
                return None
            route = [(x0, z0), (x1, z1)]
            ped = Pedestrian(x0, z0, 0.0, CROSS_SPEED, self.rng.randrange(6),
                             state="walk", route=route, waypoint=1, crossing=True)
        else:
            if not self.routes:
                return None
            a, b = self.rng.choice(self.routes)
            distance = math.hypot(a[0] - near[0], a[1] - near[1])
            if distance < min_distance or distance > max_distance:
                return None
            route = [a, b]
            ped = Pedestrian(a[0], a[1], 0.0, WALK_SPEED, self.rng.randrange(6),
                             state="walk", route=route, waypoint=1)
        self.pedestrians.append(ped)
        return ped

    def seed(self, near: Vec2, count: int = 8) -> None:
        for _ in range(count):
            for _attempt in range(6):
                if self.spawn(near):
                    break

    # ------------------------------------------------------------------ update
    def update(self, dt: float, player: Optional[Vehicle] = None,
               traffic: Optional[Sequence[object]] = None) -> None:
        self.spawn_timer += dt
        if self.spawn_timer > 0.6:
            self.spawn_timer = 0.0
            if player is not None:
                self.spawn((player.x, player.z))
        for ped in self.pedestrians:
            self._update_one(ped, dt, player, traffic)
        self._cull(player)

    def _update_one(self, ped: Pedestrian, dt: float,
                    player: Optional[Vehicle], traffic: Optional[Sequence[object]]) -> None:
        ped.alive_timer += dt
        tx, tz = ped.target()
        dx, dz = tx - ped.x, tz - ped.z
        distance = math.hypot(dx, dz)
        if distance < 0.7:
            if ped.crossing:
                # reached the far side: turn around for the return crossing
                ped.route = list(reversed(ped.route))
                ped.waypoint = 1
                ped.state = "wait"
                ped.wait_timer = self.rng.uniform(3.0, 9.0)
            else:
                # turn around and stroll back
                ped.route = list(reversed(ped.route))
                ped.waypoint = 1
                ped.wait_timer = self.rng.uniform(0.0, 2.0)
                if ped.wait_timer > 0.0:
                    ped.state = "wait"
            return

        # pedestrians about to step onto the road must check for traffic
        road = self.network.road_at(ped.x, ped.z, 0.6)
        if road is not None and ped.state != "cross":
            ped.check_timer -= dt
            if ped.check_timer <= 0.0:
                ped.check_timer = 0.35
                if self._traffic_too_close(ped, player, traffic, road):
                    ped.state = "wait"
                    ped.wait_timer = 0.6
                else:
                    ped.state = "walk"
        if ped.state == "wait":
            ped.wait_timer -= dt
            if ped.wait_timer <= 0.0:
                ped.state = "cross" if ped.crossing else "walk"
            return

        desired_yaw = math.atan2(-dz, dx)
        ped.yaw = desired_yaw
        speed = CROSS_SPEED if ped.state == "cross" else WALK_SPEED
        ped.x += dx / distance * speed * dt
        ped.z += dz / distance * speed * dt
        ped.bob += dt * speed * 5.0
        ped.speed = speed

    def _traffic_too_close(self, ped: Pedestrian, player: Optional[Vehicle],
                           traffic: Optional[Sequence[object]], road: Road) -> bool:
        """True when a vehicle is approaching near this pedestrian."""
        if player is not None:
            d = math.hypot(player.x - ped.x, player.z - ped.z)
            if d < SAFE_GAP and abs(player.speed) > 3.0:
                closing = self._is_approaching(ped, player.x, player.z, player.yaw, player.speed)
                if closing or d < 9.0:
                    return True
        if traffic:
            for vehicle in traffic:
                dx = vehicle.x - ped.x
                dz = vehicle.z - ped.z
                d = math.hypot(dx, dz)
                if d > SAFE_GAP:
                    continue
                if abs(vehicle.speed) > 2.5 and self._is_approaching(
                        ped, vehicle.x, vehicle.z, vehicle.yaw, vehicle.speed):
                    return True
        return False

    @staticmethod
    def _is_approaching(ped: Pedestrian, x: float, z: float, yaw: float,
                        speed: float) -> bool:
        fx, fz = math.cos(yaw), -math.sin(yaw)
        dx, dz = ped.x - x, ped.z - z
        distance = math.hypot(dx, dz) or 1.0
        ahead = (dx * fx + dz * fz) / distance
        return ahead > 0.72 and speed > 0.0

    def _cull(self, player: Optional[Vehicle]) -> None:
        if player is None:
            return
        keep = []
        for ped in self.pedestrians:
            distance = math.hypot(ped.x - player.x, ped.z - player.z)
            if ped.alive_timer > 30.0 and distance > 160.0:
                continue
            keep.append(ped)
        self.pedestrians = keep

    # ------------------------------------------------------------------ queries
    def near(self, x: float, z: float, radius: float) -> bool:
        r2 = radius * radius
        for ped in self.pedestrians:
            if (ped.x - x) ** 2 + (ped.z - z) ** 2 <= r2:
                return True
        return False

    def count(self) -> int:
        return len(self.pedestrians)

    def statistics(self) -> Dict[str, int]:
        return {"pedestrians": len(self.pedestrians), "crossings": len(self.crossings),
                "routes": len(self.routes)}
