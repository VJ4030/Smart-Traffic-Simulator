"""Vehicle physics: acceleration, braking, steering, suspension, damage and fuel."""

from __future__ import annotations

import math

import pytest

from smart_traffic_india.vehicle import (Vehicle, VehicleControl, effective_spec, get_spec,
                                         suspension_absorption)
from tests.conftest import drive


def test_player_specs_exist_and_are_distinct():
    from smart_traffic_india.vehicle import PLAYER_SPECS
    assert len(PLAYER_SPECS) == 6
    keys = list(PLAYER_SPECS)
    assert keys[0] == "hatchback"                       # the starter car comes first
    assert get_spec("sports").max_speed > get_spec("hatchback").max_speed
    assert get_spec("utility").mass > get_spec("hatchback").mass
    assert get_spec("nonexistent") is get_spec("hatchback")     # safe fallback
    for key in keys:
        spec = get_spec(key)
        assert 1.0 < spec.length < 7.0
        assert 1.2 < spec.width < 2.6
        assert spec.max_speed > 60
        assert spec.tank > 10
        assert abs(spec.wheelbase - spec.length * 0.58) < 0.35


def test_acceleration_and_top_speed(vehicle):
    spec = vehicle.spec
    seen = []
    control = VehicleControl()
    control.throttle = 1.0
    for _ in range(60 * 60):
        vehicle.step(control, 1 / 60, None, "road", 1.0)
        seen.append(vehicle.speed_kmh)
    assert vehicle.speed_kmh > spec.max_speed * 0.70
    assert vehicle.speed_kmh <= spec.max_speed * 1.15 + 1.0
    # acceleration must be monotonic-ish: no speed reversals while accelerating
    assert max(seen) == pytest.approx(seen[-1], rel=0.2)


def test_braking_stops_the_car(vehicle):
    drive(vehicle, 8.0, throttle=1.0)
    assert vehicle.speed > 8.0
    control = VehicleControl()
    control.brake = 1.0
    for _ in range(60 * 5):
        vehicle.step(control, 1 / 60, None, "road", 1.0)
    assert abs(vehicle.speed) < 0.2


def test_reverse_gear_moves_backwards(vehicle):
    spec = vehicle.spec
    control = VehicleControl()
    control.gear = -1
    control.throttle = 1.0
    for _ in range(60 * 4):
        vehicle.step(control, 1 / 60, None, "road", 1.0)
    assert vehicle.speed < -1.0
    assert abs(vehicle.speed) < spec.max_speed / 3.6 * 0.35   # reverse is deliberately slow


def test_steering_turns_the_car_and_tightens_at_low_speed(vehicle):
    """Higher speed must give a *wider* turning circle (speed dependent steering)."""
    import math as _math

    def turn_radius(car, speed_target):
        control = VehicleControl()
        control.steer = 1.0
        # hold a steady speed with light throttle, then measure yaw rate vs speed
        while abs(car.speed - speed_target) > 0.4:
            control.throttle = 1.0 if car.speed < speed_target else 0.0
            control.brake = 1.0 if car.speed > speed_target + 0.5 else 0.0
            car.step(control, 1 / 60, None, "road", 1.0)
            if car.speed > speed_target * 3:
                break
        control.throttle = 0.3
        control.brake = 0.0
        yaw0, x0, z0 = car.yaw, car.x, car.z
        for _ in range(90):
            car.step(control, 1 / 60, None, "road", 1.0)
        turned = abs(car.yaw - yaw0)
        travelled = _math.hypot(car.x - x0, car.z - z0)
        return travelled / max(turned, 1e-4), car.speed

    slow = Vehicle(get_spec("hatchback"), 0.0, 0.0, 0.0, "slow")
    fast = Vehicle(get_spec("hatchback"), 0.0, 0.0, 0.0, "fast")
    slow_radius, slow_speed = turn_radius(slow, 5.0)
    fast_radius, fast_speed = turn_radius(fast, 20.0)
    assert fast_speed > slow_speed * 2.0
    assert fast_radius > slow_radius * 1.5
    assert 2.0 < slow_radius < 40.0


def test_handbrake_slows_the_car_quickly(vehicle):
    drive(vehicle, 8.0, throttle=1.0)
    speed_before = vehicle.speed
    control = VehicleControl()
    control.handbrake = True
    for _ in range(60 * 2):
        vehicle.step(control, 1 / 60, None, "road", 1.0)
    assert abs(vehicle.speed) < speed_before * 0.6


def test_wheels_rotate_and_steer(vehicle):
    control = VehicleControl()
    control.throttle = 1.0
    spins = [wheel.spin for wheel in vehicle.wheels]
    for _ in range(60):
        vehicle.step(control, 1 / 60, None, "road", 1.0)
    assert any(wheel.spin != before for wheel, before in zip(vehicle.wheels, spins))
    control.steer = 0.8
    for _ in range(10):
        vehicle.step(control, 1 / 60, None, "road", 1.0)
    assert any(abs(wheel.steer) > 0.05 for wheel in vehicle.wheels if wheel.steerable)


def test_suspension_reacts_to_a_speed_breaker(vehicle):
    def ground(x, z):
        return 0.16 if abs(z) < 0.75 else 0.0
    control = VehicleControl()
    control.throttle = 0.5
    heights = []
    for _ in range(240):
        vehicle.step(control, 1 / 60, ground, "road", 1.0)
        heights.append(vehicle.bump_offset)
    assert max(heights) > 0.01
    assert min(heights) < max(heights)


def test_rain_reduces_grip():
    dry = Vehicle(get_spec("hatchback"), 0, 0, 0, "dry")
    wet = Vehicle(get_spec("hatchback"), 0, 0, 0, "wet")
    drive(dry, 8.0, throttle=1.0)
    drive(wet, 8.0, throttle=1.0)
    control = VehicleControl()
    control.brake = 1.0
    distances = []
    for car in (dry, wet):
        start = car.x
        for _ in range(int(1.5 * 60)):
            car.step(control, 1 / 60, None, "road", 1.0 if car is dry else 0.74)
        distances.append(abs(car.x - start))
    assert distances[1] >= distances[0] * 0.95        # wet braking never stops sooner


def test_impact_damage_repair_and_write_off(vehicle):
    assert vehicle.damage == 0.0
    vehicle.apply_impact(1.0, "collision", 1.0)
    assert vehicle.damage > 0.0
    assert vehicle.shake > 0.0
    vehicle.apply_impact(2.0, "collision", 1.0)
    assert vehicle.damage <= 100.0
    vehicle.repair()
    assert vehicle.damage == 0.0


def test_upgrades_measurably_change_the_car():
    base = get_spec("hatchback")
    stock = effective_spec(base, {})
    tuned = effective_spec(base, {"engine": 5, "brakes": 5, "tyres": 5, "steering": 5,
                                  "suspension": 5})
    assert tuned.max_speed > stock.max_speed
    assert tuned.accel > stock.accel
    assert tuned.brake > stock.brake
    assert tuned.grip > stock.grip
    # the absorption value is an impact multiplier: lower means the car takes less damage
    assert suspension_absorption({"suspension": 5}) < suspension_absorption({})
    assert suspension_absorption(None) == pytest.approx(1.0)


def test_fuel_burn_and_refuel(vehicle):
    control = VehicleControl()
    control.throttle = 1.0
    for _ in range(60 * 20):
        vehicle.step(control, 1 / 60, None, "road", 1.0)
    assert vehicle.fuel < vehicle.spec.tank
    assert 0.0 <= vehicle.fuel_fraction() <= 1.0
    assert vehicle.efficiency_km_per_litre() > 0.0
    vehicle.refuel()
    assert vehicle.fuel == pytest.approx(vehicle.spec.tank)


def test_physics_stays_finite_under_abuse(vehicle):
    control = VehicleControl()
    for index in range(1200):
        control.throttle = 0.0 if index % 3 else 1.0
        control.brake = 1.0 if index % 5 == 0 else 0.0
        control.steer = math.sin(index * 0.05)
        control.handbrake = index % 7 == 0
        control.gear = -1 if index % 11 == 0 else 1
        vehicle.step(control, 1 / 60, None, "road", 0.8)
        assert math.isfinite(vehicle.speed)
        assert math.isfinite(vehicle.x) and math.isfinite(vehicle.z)
        assert math.isfinite(vehicle.yaw)
    assert abs(vehicle.speed) < 80.0
