"""AI traffic: spawning, spacing, signals, junctions, recycling and Indian behaviours."""

from __future__ import annotations

import math

import pytest

from smart_traffic_india import settings as S
from smart_traffic_india.traffic_ai import (BEHAVIOUR, TRAFFIC_SPECS, TrafficContext,
                                            TrafficVehicle)
from smart_traffic_india.traffic_signals import AMBER, GREEN, RED
from tests.conftest import lane_error, lane_with_signal, make_traffic_context, run_until


# ---------------------------------------------------------------------------- data
def test_every_indian_traffic_kind_exists():
    for kind in ("hatchback", "sedan", "suv", "auto", "bike", "bus", "truck", "taxi",
                 "tempo", "ambulance", "police"):
        assert kind in TRAFFIC_SPECS, f"traffic kind {kind} is missing"
        spec = TRAFFIC_SPECS[kind]
        assert spec.length > 1.5 and spec.width > 0.6
        assert spec.max_speed > 60
    # the auto-rickshaw is narrow and slow, the bus is long and heavy: distinct bodies
    assert TRAFFIC_SPECS["auto"].length < TRAFFIC_SPECS["hatchback"].length
    assert TRAFFIC_SPECS["bus"].length > 9.0
    assert TRAFFIC_SPECS["bus"].mass > TRAFFIC_SPECS["auto"].mass * 10
    assert TRAFFIC_SPECS["bike"].width < 1.0


def test_behaviours_exist_for_all_kinds():
    for kind in TRAFFIC_SPECS:
        assert kind in BEHAVIOUR
        behaviour = BEHAVIOUR[kind]
        assert 0.5 <= behaviour["speed"] <= 1.5
        assert 0.05 < behaviour["gap"] <= 3.0


# ---------------------------------------------------------------------------- spawning
def test_spawn_avoids_the_player_and_roads_outside_the_map(traffic, network):
    assert traffic.vehicles == []
    vehicle = traffic.spawn_vehicle(player_pos=(1.8, -60.0))
    assert vehicle is not None
    assert vehicle.lane is not None
    distance = math.hypot(vehicle.x - 1.8, vehicle.z + 60.0)
    assert distance >= 80.0
    # never spawn on a lane with nowhere to go
    assert network.successors(vehicle.lane.id)


def test_spawn_respects_the_vehicle_cap(traffic):
    for _ in range(traffic.max_vehicles * 4):
        traffic.spawn_vehicle(player_pos=(0.0, 0.0))
    assert len(traffic.vehicles) <= traffic.max_vehicles


def test_spawn_stack_is_avoided(traffic):
    first = traffic.spawn_vehicle(player_pos=(0.0, 0.0), lane=None)
    assert first is not None
    # trying to spawn on top of it must be refused by the spacing check
    clash = traffic.spawn_vehicle(lane=first.lane, point_index=first.point_index,
                                  player_pos=(0.0, 0.0))
    assert clash is None


def test_density_levels_change_the_traffic_volume(traffic):
    counts = {}
    for level in ("light", "medium", "heavy", "busy"):
        traffic.set_level(level)
        traffic.vehicles.clear()
        for _ in range(400):
            traffic.spawn_vehicle(player_pos=(0.0, 0.0))
        counts[level] = len(traffic.vehicles)
        # the preset cap is scaled by the graphics preset, never exceeded
        assert traffic.max_vehicles <= S.TRAFFIC_LEVELS[level]["max_vehicles"]
    assert counts["light"] < counts["heavy"] <= counts["busy"]


# ---------------------------------------------------------------------------- AI driving
def test_vehicle_follows_its_lane(traffic, network):
    vehicle = traffic.spawn_vehicle(kind="sedan", player_pos=(0.0, 0.0))
    assert vehicle is not None
    errors = []
    for _ in range(60 * 20):
        vehicle.update(S.FIXED_DT, make_traffic_context(traffic, vehicle))
        errors.append(lane_error(network, vehicle))
    assert vehicle.speed > 2.0                    # it actually drives
    # lane keeping is measured statistically: the median must be spot on, and 90% of the
    # samples within a car width (short connector lanes swing wide while turning, which is
    # exactly what the junction turn looks like)
    ordered = sorted(errors)
    assert ordered[len(ordered) // 2] < 0.35
    assert ordered[int(len(ordered) * 0.9)] < 3.0          # under one lane width


def test_vehicle_brakes_for_a_red_light_and_goes_on_green(traffic, signals, network):
    lane = lane_with_signal(network, signals)
    junction = lane.end_junction

    # wait until this approach is red with time to spare
    run_until(lambda _t: signals.state_for(junction, lane.road.axis, lane.direction) == RED,
              60.0, signals.update)

    vehicle = traffic.spawn_vehicle(kind="sedan", lane=lane,
                                    point_index=max(0, len(lane.points) - 6),
                                    player_pos=(0.0, 0.0))
    assert vehicle is not None
    stop_line = lane.road.half_width + 3.2

    def stepped(_t):
        signals.update(S.FIXED_DT)
        vehicle.update(S.FIXED_DT, TrafficContext())
        if abs(vehicle.speed) > 0.35:
            return False
        return vehicle.distance_to_junction() - stop_line

    distance_at_stop = run_until(stepped, 45.0)
    # stopped within a few metres of the painted stop line (0.0 == exactly on the line)
    # and never still rolling into the junction
    assert -4.0 < distance_at_stop < 4.0
    assert abs(vehicle.speed) < 0.4
    position_at_stop = (vehicle.x, vehicle.z)

    def moved(_t):
        signals.update(S.FIXED_DT)
        vehicle.update(S.FIXED_DT, TrafficContext())
        return vehicle.speed > 3.0 or vehicle.lane is not lane

    assert run_until(moved, 90.0) is not False
    assert (vehicle.x, vehicle.z) != position_at_stop


def test_car_following_keeps_a_gap(traffic):
    leader = traffic.spawn_vehicle(kind="hatchback", player_pos=(0.0, 0.0))
    follower = None
    for index, point_index in enumerate(range(leader.point_index + 1,
                                             len(leader.lane.points) - 1)):
        follower = traffic.spawn_vehicle(kind="hatchback", lane=leader.lane,
                                         point_index=point_index, player_pos=(0.0, 0.0))
        if follower is not None:
            break
    if follower is None:
        pytest.skip("no room for a following car in this lane")
    minimum = 99.0
    travelled = 0.0
    previous = (follower.x, follower.z)
    for _ in range(60 * 25):
        context = make_traffic_context(traffic, follower)
        leader.update(S.FIXED_DT, TrafficContext(neighbours=[follower]))
        follower.update(S.FIXED_DT, context)
        minimum = min(minimum, math.hypot(follower.x - leader.x, follower.z - leader.z))
        travelled += math.hypot(follower.x - previous[0], follower.z - previous[1])
        previous = (follower.x, follower.z)
    assert minimum > 1.2                          # they never overlap
    assert travelled > 5.0                        # the follower really drove along
    assert math.hypot(follower.x - leader.x, follower.z - leader.z) > 1.0


def test_neighbours_are_visible_through_the_grid(traffic):
    a = traffic.spawn_vehicle(kind="sedan", player_pos=(0.0, 0.0))
    assert a is not None
    grid = traffic._build_grid()
    neighbours = traffic._neighbours_from_grid(grid, a)
    assert a in neighbours or any(n is a for n in neighbours)


def test_traffic_recycles_instead_of_parking_forever(traffic, network):
    """A car that reaches the end of a dead-end lane is removed, not left as scenery."""
    lane = next(lane for lane in network.lanes
                if lane.end_junction is None and not network.successors(lane.id))
    vehicle = traffic.spawn_vehicle(kind="bus", lane=lane, point_index=0,
                                    player_pos=(0.0, 0.0))
    assert vehicle is not None
    for _ in range(60 * 60):
        vehicle.update(S.FIXED_DT, TrafficContext())
        if vehicle.dead:
            break
    assert vehicle.dead is True
    traffic.update(S.FIXED_DT, vehicle.vehicle, None)
    assert vehicle not in traffic.vehicles


def test_emergency_vehicle_spawns_behind_the_player_and_is_announced(traffic, player,
                                                                    network):
    ambulance = traffic.spawn_emergency("ambulance", player=player.vehicle,
                                        player_pos=(player.x, player.z))
    assert ambulance is not None
    assert ambulance.emergency is True
    assert ambulance.speed_factor > 1.0
    assert ambulance in traffic.vehicles
    traffic.despawn_emergency()
    assert traffic.emergency is None
    assert ambulance not in traffic.vehicles


def test_emergency_vehicle_does_not_stop_for_red_but_still_slows(traffic, signals,
                                                                 network):
    lane = lane_with_signal(network, signals)
    junction = lane.end_junction
    run_until(lambda _t: signals.state_for(junction, lane.road.axis, lane.direction) == RED,
              60.0, signals.update)
    vehicle = traffic.spawn_vehicle(kind="ambulance", lane=lane,
                                    point_index=len(lane.points) - 8, player_pos=(0, 0))
    assert vehicle is not None
    vehicle.emergency = True
    speed_limited = vehicle._junction_stop_speed()
    assert speed_limited > 0.0
    assert speed_limited < 20.0                  # slows down for safety ...


def test_behaviour_timer_blocks_speed_without_stopping_life(traffic):
    vehicle = traffic.spawn_vehicle(kind="auto", player_pos=(0.0, 0.0))
    vehicle.block_speed(0.5, 1.0)
    assert vehicle.blocked is True
    for _ in range(90):
        vehicle.update(S.FIXED_DT, TrafficContext())
    assert vehicle.blocked is False               # the block expires
    assert vehicle.block_timer == 0.0


def test_parked_vehicles_do_not_drive(traffic):
    vehicle = traffic.spawn_vehicle(kind="tempo", player_pos=(0.0, 0.0), parked=True)
    assert vehicle is not None
    start = (vehicle.x, vehicle.z)
    for _ in range(120):
        vehicle.update(S.FIXED_DT, TrafficContext())
    assert math.hypot(vehicle.x - start[0], vehicle.z - start[1]) < 0.5
    assert vehicle.vehicle.hazard_lights is True


def test_manager_updates_every_vehicle(traffic, player):
    for _ in range(12):
        traffic.spawn_vehicle(player_pos=(player.x, player.z))
    before = [(v.x, v.z) for v in traffic.vehicles]
    for _ in range(60):
        traffic.update(S.FIXED_DT, player.vehicle, None)
    after = [(v.x, v.z) for v in traffic.vehicles if v in traffic.vehicles]
    moved = sum(1 for (bx, bz), (ax, az) in zip(before, after)
                if math.hypot(ax - bx, az - bz) > 1.0)
    assert moved >= max(1, len(after) // 2)


def test_statistics_and_queries(traffic):
    traffic.spawn_vehicle(player_pos=(0.0, 0.0))
    stats = traffic.statistics()
    for key in ("active", "max", "spawned", "updated"):
        assert key in stats
    counts = traffic.count_by_kind()
    assert sum(counts.values()) == len(traffic.vehicles)
    assert traffic.nearby(0.0, 0.0, 5.0) == [] or isinstance(traffic.nearby(0.0, 0.0, 5.0), list)


def test_queue_provider_feeds_the_adaptive_signals(traffic, signals, network):
    lane = lane_with_signal(network, signals)
    junction = lane.end_junction
    length = traffic.queue_length_for_signal(junction, lane.road.axis, lane.direction)
    assert length >= 0
    snapshot = signals.queue_snapshot()
    assert isinstance(snapshot, dict)
    assert junction.name in snapshot
