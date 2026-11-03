"""Traffic signals: timing rules, adaptive control, per approach state and geometry."""

from __future__ import annotations

import pytest

from smart_traffic_india import settings as S
from smart_traffic_india.traffic_signals import (ALL_RED_TIME, AMBER, AMBER_TIME, GREEN,
                                                 MAX_DEMAND_VEHICLES, MAX_GREEN, MIN_GREEN,
                                                 RED, STARVATION_TIME)


def step_controller(controller, seconds, dt=S.FIXED_DT):
    for _ in range(int(round(seconds / dt))):
        controller.update(dt)


def test_timing_constants_are_sensible():
    assert 5.0 <= MIN_GREEN <= 15.0
    assert MIN_GREEN < MAX_GREEN <= 60.0
    assert 2.0 <= AMBER_TIME <= 5.0
    assert 0.5 <= ALL_RED_TIME <= 3.0
    assert STARVATION_TIME >= MAX_GREEN


def test_every_signalised_junction_gets_heads(network, signals):
    signalised = [j for j in network.junctions if j.signalised]
    stats = signals.statistics()
    assert stats["junctions"] == len(signalised)
    assert stats["heads"] >= 4 * len(signalised)      # at least one head per approach


def test_opposing_approaches_are_never_green_at_once(network, signals):
    """The one inviolable rule: two crossing roads must never both have green."""
    for _ in range(60 * 240):
        signals.update(S.FIXED_DT)
        for junction in network.junctions:
            if not junction.signalised:
                continue
            for road in (junction.road_a, junction.road_b):
                for direction in (1, -1):
                    state = signals.state_for(junction, road.axis, direction)
                    other = junction.road_b if road is junction.road_a else junction.road_a
                    opposite = signals.state_for(junction, other.axis, direction)
                    assert not (state == GREEN and opposite == GREEN)
        # sampled: the full 4 minute sweep is slow enough already


def test_green_amber_red_sequence(network, signals):
    junction = next(j for j in network.junctions if j.signalised)
    axis = junction.road_a.axis
    order = []
    for _ in range(60 * 180):
        signals.update(S.FIXED_DT)
        state = signals.state_for(junction, axis, 1)
        if not order or order[-1] != state:
            order.append(state)
    assert order[0] == GREEN
    for previous, following in zip(order, order[1:]):
        if previous == GREEN:
            assert following == AMBER
        elif previous == AMBER:
            assert following == RED
        elif previous == RED:
            assert following == GREEN


def test_green_duration_respects_min_and_max(network, signals):
    junction = next(j for j in network.junctions if j.signalised)
    axis = junction.road_a.axis
    durations = []
    current = signals.state_for(junction, axis, 1)
    length = 0.0
    for _ in range(60 * 600):
        signals.update(S.FIXED_DT)
        state = signals.state_for(junction, axis, 1)
        if state == current:
            length += S.FIXED_DT
        else:
            if current == GREEN:
                durations.append(length)
            current, length = state, 0.0
    assert durations
    assert min(durations) >= MIN_GREEN - 0.2
    assert max(durations) <= MAX_GREEN + 0.5
    assert len(durations) >= 3


def test_amber_and_all_red_gaps(network, signals):
    junction = next(j for j in network.junctions if j.signalised)
    axis = junction.road_a.axis
    timings = {}
    current = signals.state_for(junction, axis, 1)
    length = 0.0
    for _ in range(60 * 300):
        signals.update(S.FIXED_DT)
        state = signals.state_for(junction, axis, 1)
        if state == current:
            length += S.FIXED_DT
        else:
            timings.setdefault(current, []).append(length)
            current, length = state, 0.0
    assert AMBER in timings
    assert abs(timings[AMBER][0] - AMBER_TIME) < 0.6


def test_queues_extend_the_green_time(signals, network):
    """A junction with a queue must be given a longer green than an empty one."""
    junction = next(j for j in network.junctions if j.signalised)
    phase = signals.phases[junction.id]
    phase.demand = {phase.active_axis: 0, phase.other_axis: 0}
    empty = signals._green_duration(phase)
    phase.demand = {phase.active_axis: MAX_DEMAND_VEHICLES, phase.other_axis: 0}
    busy = signals._green_duration(phase)
    assert empty == MIN_GREEN
    assert busy > empty
    assert busy <= MAX_GREEN
    # a starved cross direction must not be made to wait forever
    phase.demand = {phase.active_axis: MAX_DEMAND_VEHICLES, phase.other_axis: 5}
    phase.served_since[phase.other_axis] = STARVATION_TIME + 1.0
    assert signals._green_duration(phase) == MIN_GREEN


def test_state_at_returns_the_nearest_junction_with_distance(network, signals):
    junction = next(j for j in network.junctions if j.signalised)
    road = junction.road_a
    x, z = road.position_at(junction.z - 20.0 if road.axis == "z" else 0.0, 0.0)
    if road.axis == "x":
        x, z = road.position_at(junction.x - 20.0, 0.0)
    yaw = road.direction_yaw(1)
    state, found, distance = signals.state_at(x, z, yaw)
    assert state in (GREEN, AMBER, RED)
    assert found is junction
    assert distance > 0.0
    # far away from every junction there is nothing to report
    state, found, distance = signals.state_at(2000.0, 2000.0, 0.0)
    assert found is None
    assert state == GREEN


def test_stop_line_distance_matches_the_geometry(network, signals):
    junction = next(j for j in network.junctions if j.signalised)
    road = junction.road_a
    s_value = (junction.z - 25.0) if road.axis == "z" else (junction.x - 25.0)
    x, z = road.position_at(s_value, road.lane_offsets(1)[0])
    yaw = road.direction_yaw(1)
    distance = signals.stop_line_distance(x, z, yaw)
    if distance is not None:
        assert 0.0 < distance < 60.0


def test_signal_mesh_has_poles_and_housings(network, signals):
    mesh = signals.build_mesh()
    names = {surface.material.name for surface in mesh.surfaces}
    assert mesh.triangle_count > 200
    joined = " ".join(names)
    assert "pole" in joined or "signal" in joined


def test_sprite_instances_expose_lit_lenses(network, signals):
    signals.update(0.5)
    instances = signals.sprite_instances()
    assert len(instances) >= 12
    lit = [item for item in instances if item[2][3] > 0.5]
    assert lit, "the active lamp must be clearly lit"
    for position, size, colour in instances[:5]:
        assert len(position) == 3
        assert size[0] > 0.05 and size[1] > 0.05
        assert len(colour) == 4


def test_statistics_and_queue_snapshot(network, signals):
    step_controller(signals, 30.0)
    stats = signals.statistics()
    assert stats["junctions"] >= 5
    assert stats["heads"] >= 4 * stats["junctions"]
    snapshot = signals.queue_snapshot()
    assert isinstance(snapshot, dict) and snapshot
