"""Collision geometry (SAT / OBB), the spatial hash and the damage cooldown."""

from __future__ import annotations

import math

import pytest

from smart_traffic_india import settings as S
from smart_traffic_india.collision import (CollisionEvent, CollisionSystem, SpatialHash,
                                           ground_height, obb_corners, point_in_obb,
                                           sat_overlap, vehicle_corners)
from smart_traffic_india.road import Hazard
from smart_traffic_india.vehicle import Vehicle, VehicleControl, get_spec


def test_obb_corners_are_the_four_rotated_corners():
    corners = obb_corners(0.0, 0.0, 0.0, 2.0, 1.0)
    assert len(corners) == 4
    xs = sorted(point[0] for point in corners)
    zs = sorted(point[1] for point in corners)
    assert xs[0] == pytest.approx(-2.0) and xs[-1] == pytest.approx(2.0)
    assert zs[0] == pytest.approx(-1.0) and zs[-1] == pytest.approx(1.0)
    rotated = obb_corners(0.0, 0.0, math.pi / 2, 2.0, 1.0)
    xs = sorted(point[0] for point in rotated)
    assert xs[0] == pytest.approx(-1.0, abs=1e-5)


def test_sat_separates_and_overlaps():
    a = obb_corners(0.0, 0.0, 0.0, 2.0, 1.0)
    far = obb_corners(20.0, 0.0, 0.0, 2.0, 1.0)
    assert sat_overlap(a, far) is None
    near = obb_corners(3.0, 0.0, 0.0, 2.0, 1.0)
    hit = sat_overlap(a, near)
    assert hit is not None
    depth, normal = hit
    assert depth > 0.0
    assert abs(math.hypot(normal[0], normal[1]) - 1.0) < 1e-5


def test_sat_handles_rotation():
    a = obb_corners(0.0, 0.0, 0.0, 2.0, 0.6)
    crossing = obb_corners(0.0, 0.0, math.pi / 2, 2.0, 0.6)
    assert sat_overlap(a, crossing) is not None
    corner = obb_corners(2.0, 2.0, math.radians(45), 1.0, 1.0)
    assert sat_overlap(a, corner) is not None


def test_point_in_obb():
    assert point_in_obb(0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0) is True
    assert point_in_obb(5.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0) is False
    assert point_in_obb(0.0, 1.0, 0.0, 0.0, math.pi / 2, 2.0, 1.0) is True


def test_spatial_hash_queries_neighbours_only():
    grid = SpatialHash(cell=10.0)
    for index, (x, z) in enumerate([(0.0, 0.0), (4.0, 4.0), (100.0, 100.0), (105.0, 105.0)]):
        grid.insert(index, x, z)
    near = set(grid.query(1.0, 1.0))
    assert 0 in near and 1 in near
    assert 2 not in near and 3 not in near
    grid.clear()
    assert grid.query(1.0, 1.0) == []


def test_vehicle_collision_reports_damage_once_per_cooldown():
    system = CollisionSystem(None, None, cooldown=1.0)
    player = Vehicle(get_spec("hatchback"), 0.0, 0.0, 0.0, "You")
    blocker = Vehicle(get_spec("bus"), 0.0, 0.0, 0.0, "bus-1")
    blocker.speed = 0.0
    overlapping = [player, blocker]

    events = system.check_vehicles(player, overlapping)
    assert any(event.is_player for event in events)
    assert player.collision_cooldown == pytest.approx(1.0)
    assert player.speed < 0.0 or abs(player.speed) < 1e-6      # pushed back / stopped

    # the very next frames must not report the same contact again
    for _ in range(30):
        again = system.check_vehicles(player, overlapping)
        assert not [event for event in again if event.is_player]

    # after the cooldown expires, a fresh contact is reported again
    player.collision_cooldown = 0.0
    player.x, player.z = 0.0, 0.0
    fresh = system.check_vehicles(player, overlapping)
    assert [event for event in fresh if event.is_player]


def test_collision_response_converts_speed_into_damage():
    system = CollisionSystem(None, None, cooldown=0.5)
    player = Vehicle(get_spec("hatchback"), 0.0, 0.0, 0.0, "You")
    player.speed = 14.0
    other = Vehicle(get_spec("sedan"), 1.0, 0.0, 0.0, "sedan-1")
    events = system.check_vehicles(player, [player, other])
    player_events = [event for event in events if event.is_player]
    assert player_events
    assert player.speed < 14.0
    assert player.damage > 0.0
    assert player.shake > 0.0


def test_ai_vehicles_are_separated_without_killing_their_speed():
    """Two AI cars in the same spot are pushed apart along the *least* penetration axis."""
    system = CollisionSystem(None, None)
    a = Vehicle(get_spec("sedan"), 0.0, 0.0, 0.0, "a")
    b = Vehicle(get_spec("sedan"), 1.2, 0.0, 0.0, "b")
    a.speed = b.speed = 6.0
    assert sat_overlap(vehicle_corners(a), vehicle_corners(b)) is not None
    for _ in range(60):
        system.check_vehicles(None, [a, b])
    remaining = sat_overlap(vehicle_corners(a), vehicle_corners(b))
    assert remaining is None or remaining[0] < 0.05      # pushed clear (within float noise)
    assert math.hypot(a.x - b.x, a.z - b.z) > 1.2
    assert abs(a.speed - 6.0) < 1e-6              # separation must not brake the cars
    assert abs(b.speed - 6.0) < 1e-6


def test_buildings_are_solid(network, world):
    system = world.collision
    assert system._buildings
    bx, bz, hx, hz, _height = system._buildings[0]
    vehicle = Vehicle(get_spec("hatchback"), bx, bz, 0.0, "You")
    event = system.building_overlap(vehicle)
    assert event is not None
    assert event.kind == "building"
    # a car far from every building is free
    vehicle.x, vehicle.z = 500.0, 500.0
    assert system.building_overlap(vehicle) is None


def test_ground_height_handles_speed_breakers_only():
    breaker = Hazard("speed_breaker", 0.0, 0.0, 3.0, 0.0, 1.0, 0.8, {"across": True})
    pothole = Hazard("pothole", 0.0, 0.0, 1.6, 0.0, 1.0, 0.8)
    assert ground_height(0.0, 0.0, [breaker]) > 0.05
    assert ground_height(0.0, 4.0, [breaker]) == 0.0
    assert ground_height(0.0, 0.0, [pothole]) == 0.0     # a hole is not a hump
    assert ground_height(0.0, 0.0, []) == 0.0


def test_collision_event_labels_and_severity():
    event = CollisionEvent("vehicle", "You", "bus-7", 1.2, (0.0, 0.0), (1.0, 0.0), 0.4, True)
    assert event.is_player is True
    assert "bus-7" in event.label                    # property, not a method
    assert event.severity > 0.0


def test_hazard_contact_and_system_statistics(world):
    system = world.collision
    vehicle = Vehicle(get_spec("hatchback"), 0.0, 0.0, 0.0, "You")
    hazard = world.network.hazards[0]
    vehicle.x, vehicle.z = hazard.x, hazard.z
    touching = system.hazard_contact(vehicle, world.network.hazards)
    assert hazard in touching
    for key in ("vehicle_pairs", "obstacle_tests", "hits"):
        assert key in system.stats


def test_high_speed_crash_into_a_building_is_bounded(network, world):
    system = world.collision
    bx, bz, _hx, _hz, _height = system._buildings[0]
    vehicle = Vehicle(get_spec("sports"), bx, bz, 0.0, "You")
    vehicle.speed = 40.0
    event = system.building_overlap(vehicle)
    assert event is not None
    assert event.severity <= 1.6
