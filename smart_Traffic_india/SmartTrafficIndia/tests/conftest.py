"""Shared fixtures: settings, save file, road network, world, player and AI helpers."""

from __future__ import annotations

import math
import os

import pytest

pygame = pytest.importorskip("pygame")                     # pygame is a hard requirement

from smart_traffic_india import settings as S
from smart_traffic_india.player import Player
from smart_traffic_india.road import RoadNetwork
from smart_traffic_india.save_manager import SaveManager
from smart_traffic_india.settings import Settings
from smart_traffic_india.signage import build_atlas
from smart_traffic_india.traffic_ai import TrafficContext, TrafficManager
from smart_traffic_india.traffic_signals import SignalController
from smart_traffic_india.vehicle import Vehicle, VehicleControl, get_spec
from smart_traffic_india.vehicle_models import ModelLibrary
from smart_traffic_india.world import World

#: every test uses the cheapest quality preset so the suite stays fast
TEST_QUALITY = dict(S.QUALITY_PRESETS["low"])
TEST_QUALITY["multisample"] = 0


@pytest.fixture(scope="session", autouse=True)
def _pygame_ready():
    pygame.init()
    yield
    pygame.quit()


@pytest.fixture()
def settings() -> Settings:
    value = Settings()
    value.width, value.height = 1280, 720
    value.quality = "low"
    value.traffic_level = "light"
    value.time_of_day = "day"
    value.weather = "clear"
    return value


@pytest.fixture()
def quality() -> dict:
    return dict(TEST_QUALITY)


@pytest.fixture(scope="session")
def atlas():
    return build_atlas(8)


@pytest.fixture(scope="session")
def models() -> ModelLibrary:
    library = ModelLibrary()
    library.preload()
    return library


@pytest.fixture()
def network() -> RoadNetwork:
    return RoadNetwork(seed=4242)


@pytest.fixture(scope="session")
def shared_world(atlas, models):
    """One fully built city shared by the slower integration tests (building is ~0.4 s)."""
    world = World(Settings(), atlas, models, seed=99)
    return world


@pytest.fixture()
def world(shared_world):
    """A private view of the shared city: traffic is reset per test, geometry is reused."""
    shared_world.traffic.vehicles.clear()
    shared_world.traffic.parked.clear()
    shared_world.traffic.emergency = None
    shared_world.traffic.set_level("light")
    return shared_world


@pytest.fixture()
def signals(network) -> SignalController:
    return SignalController(network)


@pytest.fixture()
def traffic(network, signals, quality) -> TrafficManager:
    return TrafficManager(network, signals, quality, level="light", seed=7)


@pytest.fixture()
def save(tmp_path) -> SaveManager:
    return SaveManager(path=str(tmp_path / "profile.json"))


@pytest.fixture()
def vehicle() -> Vehicle:
    return Vehicle(get_spec("hatchback"), 0.0, 0.0, 0.0, "test-car")


@pytest.fixture()
def player(network) -> Player:
    return Player("hatchback", {}, start=(1.8, -60.0, -math.pi / 2), network=network)


def drive(vehicle: Vehicle, seconds: float, throttle: float = 1.0, brake: float = 0.0,
          steer: float = 0.0, dt: float = S.FIXED_DT, steps: int = 0):
    """Step a bare vehicle (no world) for ``seconds`` - used by the physics tests."""
    control = VehicleControl()
    control.throttle = throttle
    control.brake = brake
    control.steer = steer
    count = steps or int(round(seconds / dt))
    for _ in range(count):
        vehicle.step(control, dt, None, "road", 1.0)
    return vehicle


def step_player(player: Player, signals: SignalController, seconds: float,
                inputs: dict | None = None, dt: float = S.FIXED_DT):
    """Step a :class:`Player` for ``seconds`` and return every feedback object produced."""
    base = {"throttle": False, "brake": False, "steer_left": False, "steer_right": False,
            "handbrake": False, "horn": False, "lights": False, "indicator_left": False,
            "indicator_right": False, "hazards": False, "look_back": False}
    if inputs:
        base.update(inputs)
    out = []
    for _ in range(int(round(seconds / dt))):
        out.append(player.update(dt, base, signals, []))
    return out


def lane_error(network, vehicle) -> float:
    """How far a vehicle is from the centre of its own lane (inf while in a junction)."""
    x, z = vehicle.x, vehicle.z
    if network.in_junction(x, z, 8.0):
        return 0.0
    return abs(vehicle.lane.road.lateral(x, z) - vehicle.lane.offset)


def make_traffic_context(manager: TrafficManager, vehicle=None, player_vehicle=None):
    grid = manager._build_grid()
    neighbours = manager._neighbours_from_grid(grid, vehicle) if vehicle else []
    return TrafficContext(neighbours=neighbours, player=player_vehicle,
                          pedestrian_near=manager.pedestrian_query,
                          congestion=1.0, rain=False, night=False)


def lane_with_signal(network: RoadNetwork, signals: SignalController):
    """The first lane that ends at a signalised junction it can actually leave."""
    for lane in network.lanes:
        junction = lane.end_junction
        if junction is None or not junction.signalised:
            continue
        if not network.successors(lane.id):
            continue
        return lane
    raise AssertionError("no signalised lane found")


def free_lane(network: RoadNetwork, min_points: int = 8):
    for lane in network.lanes:
        if len(lane.points) >= min_points:
            return lane
    raise AssertionError("no usable lane")


def run_until(predicate, seconds: float, step=None, dt: float = S.FIXED_DT,
              limit: int = 20000):
    """Advance the world until ``predicate`` returns truthy.

    ``step`` is an optional callable invoked once per simulated step (for example
    ``world.signals.update``); the predicate receives the elapsed simulated time and its
    return value is returned to the caller.
    """
    steps = int(seconds / dt)
    for index in range(min(steps, limit)):
        if step is not None:
            step(dt)
        time_now = index * dt
        result = predicate(time_now)
        if result:
            return result
    raise AssertionError(f"condition not met within {seconds:.1f} s of simulated time")


@pytest.fixture()
def save_dir(tmp_path, monkeypatch):
    """Point the save-file helpers at a temporary directory."""
    monkeypatch.setenv("STI_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"
