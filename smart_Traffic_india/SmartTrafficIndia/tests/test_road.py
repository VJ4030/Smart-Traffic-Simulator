"""The road network: topology, left-hand traffic rules, hazards and baking."""

from __future__ import annotations

import math

import pytest

from smart_traffic_india import road as R


def test_districts_and_landmarks_cover_the_brief():
    for name in ("residential", "market", "downtown", "hospital", "school", "railway",
                 "industrial", "highway", "oldcity"):
        assert name in R.DISTRICTS, f"district {name} missing"
        info = R.DISTRICTS[name]
        assert "buildings" in info and "height" in info
    for landmark in ("Civil Hospital", "Central Railway Station", "Main Bazaar",
                     "Sunrise Public School", "Industrial Estate", "Metro Mall Downtown",
                     "Airport Road Junction", "City Bus Terminal", "Bypass Toll Plaza"):
        assert landmark in R.LANDMARKS, f"landmark {landmark} missing"
        x, z = R.LANDMARKS[landmark]
        assert -400.0 < x < 400.0 and -400.0 < z < 400.0


def test_city_has_a_usable_road_network(network):
    assert len(network.roads) >= 8
    assert len(network.junctions) >= 10
    assert len(network.lanes) >= 24
    names = {road.name for road in network.roads}
    assert "MG Road" in names
    for road in network.roads:
        assert road.length > 100.0
        assert road.half_width > 3.0
        assert road.lane_width > 2.2
        assert 20.0 <= road.speed_limit <= 90.0
        assert 1 <= road.lanes_per_dir <= 3


def test_lane_graph_is_connected(network):
    """Cars must always have somewhere to go except on deliberate dead ends."""
    for lane in network.lanes:
        successors = network.successors(lane.id)
        if lane.end_junction is not None:
            assert successors, f"lane {lane.id} ends at a junction but leads nowhere"
        for lane_id, entry, turn in successors:
            assert turn in ("straight", "left", "right")
            target = network.lane_by_id(lane_id)
            assert target is not None
            assert math.hypot(entry[0] - target.points[0][0],
                              entry[1] - target.points[0][1]) < 12.0
            # a successor must start where the current lane ends (a real junction link)
            distance = math.hypot(entry[0] - lane.points[-1][0],
                                  entry[1] - lane.points[-1][1])
            assert distance < 40.0


def test_left_hand_traffic_lanes(network):
    """Lane offsets must sit on the left of the road for the direction of travel."""
    for road in network.roads:
        for direction in (1, -1):
            offsets = road.lane_offsets(direction)
            assert len(offsets) == road.lanes_per_dir
            assert all(abs(value) < road.half_width for value in offsets)
            for offset in offsets:
                lateral = offset
                # sample a quarter of the way along the road, away from the junctions
                # (inside a junction box either road is a correct answer)
                s = road.start + road.length * 0.25
                if road.axis == "z":
                    x, z = road.center + lateral, s
                else:
                    x, z = s, road.center + lateral
                if network.in_junction(x, z, road.half_width):
                    continue
                assert network.road_at(x, z, 2.0) is road
                assert network.direction_of_travel(road, x, z) == direction


def test_lane_waypoints_stay_on_the_road(network):
    for lane in network.lanes:
        for x, z in lane.points:
            assert network.road_at(x, z, lane.road.half_width) is not None
        # waypoints must be ordered along the lane (no zig-zag polyline)
        for index in range(len(lane.points) - 1):
            a, b = lane.points[index], lane.points[index + 1]
            assert math.hypot(b[0] - a[0], b[1] - a[1]) > 1.0


def test_hazards_are_physically_on_the_road(network):
    kinds = {}
    for hazard in network.hazards:
        kinds[hazard.kind] = kinds.get(hazard.kind, 0) + 1
        assert hazard.radius > 0.4
        assert network.road_at(hazard.x, hazard.z, 3.0) is not None, \
            f"{hazard.kind} at ({hazard.x}, {hazard.z}) is not on a road"
    for required in ("pothole", "speed_breaker", "water", "construction"):
        assert kinds.get(required, 0) >= 1, f"no {required} hazards in the city"


def test_hazard_contains_and_warning_range(network):
    hazard = network.hazards[0]
    assert hazard.contains(hazard.x, hazard.z) is True
    assert hazard.contains(hazard.x + hazard.radius * 4, hazard.z + hazard.radius * 4) is False
    near = network.hazards_near(hazard.x, hazard.z, 20.0)
    assert hazard in near
    assert network.hazard_at(hazard.x, hazard.z) is not None


def test_districts_and_speed_limits_resolve(network):
    for road in network.roads:
        s = road.start + road.length * 0.25          # mid-junction samples are ambiguous
        x, z = road.position_at(s, 0.0)
        if network.in_junction(x, z, road.half_width):
            continue
        assert network.road_at(x, z, 2.0) is road
        assert network.district_at(x, z) == road.district
        limit = network.speed_limit_at(x, z)
        assert 10.0 <= limit <= 90.0
    # far away from everything the player is "offroad"
    assert network.road_at(900.0, 900.0, 1.0) is None
    assert network.on_road(900.0, 900.0, 1.0) is False
    assert 10.0 <= network.speed_limit_at(900.0, 900.0) <= 30.0


def test_junctions_have_stop_lines_and_approaches(network):
    for junction in network.junctions:
        assert junction.half_size > 4.0
        assert junction.stop_offset >= junction.half_size
        x, z = junction.position
        assert network.road_at(x, z, 6.0) is not None
        lanes = [lane for lane in network.lanes
                 if lane.end_junction is junction or lane.start_junction_id == junction.id]
        assert lanes, f"{junction.name} has no lanes"


def test_mesh_is_baked_with_markings(network):
    mesh = network.build_mesh()
    assert mesh.triangle_count > 1500
    names = {material.name for surface in mesh.surfaces for material in [surface.material]}
    for expected in ("asphalt", "line_white", "line_yellow", "zebra"):
        assert any(expected in name for name in names), f"{expected} missing from the road mesh"


def test_lane_helpers(network):
    lane = next(lane for lane in network.lanes if len(lane.points) > 6)
    point = lane.point_after(lane.points[0], 12.0)
    assert math.hypot(point[0] - lane.points[0][0], point[1] - lane.points[0][1]) > 5.0
    assert lane.remaining_to_end(lane.points[0]) > 0.0
    assert lane.axis in ("x", "z")
    assert lane.start_point == lane.points[0]
    assert lane.end_point == lane.points[-1]


def test_road_geometry_helpers():
    road = R.Road("Test Road", "z", 0.0, -100.0, 100.0, 8.0, lanes_per_dir=2)
    assert road.length == pytest.approx(200.0)
    assert road.position_at(10.0, 3.0) == (3.0, 10.0)
    assert road.lateral(4.0, 0.0) == pytest.approx(4.0)
    assert road.along(4.0, 42.0) == pytest.approx(42.0)
    assert road.contains(0.0) is True
    assert road.contains(500.0) is False
    assert road.on_surface(1.0, 1.0, 1.0) is True
    assert road.on_surface(30.0, 1.0, 1.0) is False
    assert len(road.lane_offsets(1)) == 2
    assert road.direction_yaw(1) != road.direction_yaw(-1)
    assert abs(road.parking_yaw(1) - road.direction_yaw(1)) < 1e-9


def test_two_networks_from_the_same_seed_are_identical():
    a = R.RoadNetwork(seed=5)
    b = R.RoadNetwork(seed=5)
    c = R.RoadNetwork(seed=6)
    assert [road.name for road in a.roads] == [road.name for road in b.roads]
    assert [len(a.hazards), len(c.hazards)] != [0, 0]
    assert (len(a.lanes), len(a.hazards)) == (len(b.lanes), len(b.hazards))
    assert [h.kind for h in a.hazards] == [h.kind for h in c.hazards] or \
        len(a.hazards) != len(c.hazards)
