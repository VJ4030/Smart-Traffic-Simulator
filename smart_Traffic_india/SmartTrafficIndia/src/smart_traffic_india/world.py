"""The world: road network, signals, traffic, pedestrians, environment and collisions.

:class:`World` is the single place where the simulation lives.  It owns no window and no
OpenGL object, which means the whole city can be stepped head-less (that is exactly what
the unit tests and the performance benchmark do).

Gameplay modes only need to talk to this facade:
``world.update(dt, player)`` advances everything, ``world.apply_mission(mission)``
re-configures traffic density, time of day, weather and support vehicles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .collision import CollisionEvent, CollisionSystem
from .environment import Environment
from .missions import Mission
from .pedestrians import PedestrianManager
from .player import Player
from .road import Hazard, RoadNetwork
from .traffic_ai import TrafficManager, TrafficVehicle
from .traffic_signals import SignalController
from . import settings as S


@dataclass
class WorldFrame:
    """What happened during one simulated step."""

    collisions: List[CollisionEvent] = field(default_factory=list)
    emergency_near: bool = False
    traffic_count: int = 0
    pedestrian_count: int = 0
    signal_state: str = "green"
    zone: str = ""


class World:
    """Owns and steps the whole simulation."""

    def __init__(self, settings: S.Settings, atlas, models,
                 seed: int = 4242, build_geometry: bool = True) -> None:
        self.settings = settings
        self.quality = settings.quality_flags
        self.network = RoadNetwork(seed=seed)
        self.signals = SignalController(self.network)
        self.traffic = TrafficManager(self.network, self.signals, self.quality,
                                      level=settings.traffic_level, seed=seed + 7)
        self.pedestrians = PedestrianManager(self.network, self.quality, seed=seed + 11)
        self.environment = Environment(self.network, atlas, models, self.quality,
                                       seed=seed + 13,
                                       time_of_day=settings.time_of_day,
                                       weather=settings.weather)
        self.collision = CollisionSystem(self.network, self.environment)
        self.traffic.pedestrian_query = self.pedestrians.near
        self.zone = "downtown"
        self.static_meshes: Dict[str, object] = {}
        self.parking_lots = self.environment.parking_lots
        if build_geometry:
            self.build_geometry()

    # ------------------------------------------------------------------ setup
    def build_geometry(self) -> None:
        """Bake every static mesh (called once; the renderer uploads them lazily)."""
        self.static_meshes["roads"] = self.network.build_mesh()
        self.static_meshes["signals"] = self.signals.build_mesh()
        self.static_meshes["ground"] = self.environment.ground

    def total_static_triangles(self) -> int:
        total = 0
        for mesh in self.static_meshes.values():
            total += mesh.triangle_count                # type: ignore[attr-defined]
        for chunk in self.environment.chunk_list():
            if chunk.mesh is not None:
                total += chunk.mesh.triangle_count
        return total

    # ------------------------------------------------------------------ missions
    def apply_mission(self, mission: Mission, player: Player) -> None:
        """Configure the world for a mission without throwing the city away."""
        if mission.traffic_level:
            self.traffic.set_level(mission.traffic_level)
        else:
            self.traffic.set_level(self.settings.traffic_level)
        if mission.time_of_day:
            self.environment.set_time(mission.time_of_day)
        else:
            self.environment.set_time(self.settings.time_of_day)
        if mission.weather:
            self.environment.set_weather(mission.weather)
        else:
            self.environment.set_weather(self.settings.weather)
        self.traffic.despawn_emergency()
        if any(phase.kind == "give_way" for phase in mission.phases):
            self.traffic.spawn_emergency("ambulance", player=player.vehicle,
                                         player_pos=(player.x, player.z))
        self.traffic.spawn_accumulator = 0.0

    def release_mission(self) -> None:
        self.traffic.set_level(self.settings.traffic_level)

    # ------------------------------------------------------------------ stepping
    def update(self, dt: float, player: Player, time_cycle: bool = True) -> WorldFrame:
        frame = WorldFrame()
        # 1. signals (adaptive, queue driven)
        self.signals.update(dt)
        # 2. environment (time of day, weather)
        self.environment.update(dt, time_cycle=time_cycle)
        # 3. pedestrians (they read traffic and the player to decide when to cross)
        self.pedestrians.update(dt, player.vehicle, self.traffic.vehicles)
        # 4. traffic
        self.traffic.update(dt, player.vehicle, self.environment)
        # 5. collisions
        frame.collisions = self.collision.check_vehicles(
            player.vehicle, list(self.traffic.vehicles) + list(self.traffic.parked))
        building = self.collision.building_overlap(player.vehicle)
        if building is not None and player.vehicle.collision_cooldown <= 0.0:
            building.severity = min(1.6, 0.35 + abs(player.vehicle.speed) / 9.0)
            frame.collisions.append(building)
        for event in frame.collisions:
            if event.is_player:
                player.vehicle.collision_cooldown = self.collision.cooldown
                player.register_collision(event.severity, event.b_name)
            elif event.kind == "building" and event.a_name == player.vehicle.name:
                player.vehicle.collision_cooldown = self.collision.cooldown
                player.register_collision(event.severity, "building")
        frame.traffic_count = len(self.traffic.vehicles)
        frame.pedestrian_count = self.pedestrians.count()
        frame.emergency_near = self.traffic.emergency_near(player.x, player.z, 55.0)
        frame.zone = self.zone = self.network.district_at(player.x, player.z)
        if self.environment.is_night and self.settings.headlights_auto:
            player.vehicle.headlights = True
        return frame

    # ------------------------------------------------------------------ helpers
    @property
    def hazards(self) -> List[Hazard]:
        """Every road hazard in the city (potholes, breakers, water, construction)."""
        return self.network.hazards

    def seeds_pedestrians(self, near: Tuple[float, float], count: int = 10) -> None:
        self.pedestrians.seed(near, count)

    def prime_traffic(self, player: Player, target_fraction: float = 0.7) -> None:
        """Fill the city with traffic around a freshly spawned player."""
        target = int(self.traffic.max_vehicles * target_fraction)
        for _ in range(target):
            if len(self.traffic.vehicles) >= target:
                break
            self.traffic.spawn_vehicle(player_pos=(player.x, player.z))

    def nearby_traffic(self, x: float, z: float, radius: float) -> List[TrafficVehicle]:
        return self.traffic.nearby(x, z, radius)

    def statistics(self) -> Dict[str, object]:
        return {
            "roads": len(self.network.roads),
            "junctions": len(self.network.junctions),
            "lanes": len(self.network.lanes),
            "hazards": len(self.network.hazards),
            "traffic": self.traffic.statistics(),
            "pedestrians": self.pedestrians.statistics(),
            "signals": self.signals.statistics(),
            "environment": self.environment.statistics(),
            "static_triangles": self.total_static_triangles(),
        }
