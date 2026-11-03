"""The player controller: input mapping, road rules, violations, scoring and fuel.

:class:`Player` owns one :class:`~smart_traffic_india.vehicle.Vehicle`, converts the
current key state into a :class:`~smart_traffic_india.vehicle.VehicleControl`, applies the
world effects (surface grip, rain, potholes, speed breakers, water logging), tracks road
rule compliance and keeps the score.

The controller is written so that ``update()`` can be driven from a unit test with a fake
input dictionary - no pygame display is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .collision import ground_height
from .road import Hazard, Road, RoadNetwork
from .traffic_signals import AMBER, GREEN, RED, SignalController
from .transform import clamp
from .vehicle import Vehicle, VehicleControl, get_spec
from . import settings as S

#: how long a single violation is ignored before it can be reported again
VIOLATION_COOLDOWN = 6.0
#: how far ahead the player is warned about a hazard
HAZARD_WARNING_RANGE = 42.0


@dataclass
class PlayerFeedback:
    """Everything the HUD needs to know about what just happened."""

    events: List[Tuple[str, str]] = field(default_factory=list)     # (kind, message)
    warnings: List[Tuple[str, float]] = field(default_factory=list)  # (text, seconds)
    horn: bool = False
    pothole: bool = False
    speed_breaker: bool = False

    def add(self, kind: str, message: str) -> None:
        self.events.append((kind, message))

    def warn(self, message: str, seconds: float = 3.0) -> None:
        self.warnings.append((message, seconds))


class Player:
    """Drives one vehicle with keyboard input and scores the run."""

    def __init__(self, car_key: str = "hatchback", upgrades: Optional[Dict[str, int]] = None,
                 start: Tuple[float, float, float] = (4.0, -40.0, -math.pi / 2),
                 network: Optional[RoadNetwork] = None) -> None:
        self.car_key = car_key
        self.upgrades = dict(upgrades or {})
        self.vehicle = Vehicle(get_spec(car_key), start[0], start[1], start[2], name="You")
        self.control = VehicleControl()
        self.network = network

        # scoring / progression
        self.score = 0
        self.collisions = 0
        self.violations = 0
        self.potholes_hit = 0
        self.combo = 1.0
        self.trip_distance = 0.0
        self.trip_time = 0.0
        self.max_speed_seen = 0.0
        self.perfect = True
        self.weather_rain = False
        self._fuel_warned = False
        self.smooth_driving_time = 0.0

        # rule tracking
        self.violation_cooldown = 0.0
        self.hazard_cooldowns: Dict[int, float] = {}
        self.last_signal_state = GREEN
        self.signal_junction_watch: Optional[int] = None
        self.wrong_side_cooldown = 0.0
        self.overspeed_timer = 0.0
        self.hazard_warnings: Dict[int, float] = {}
        self.active_rule: str = ""
        self.rule_timer = 0.0
        self.zone: str = "downtown"

        # cosmetic state
        self.indicator_timer = 0.0
        self.horn_timer = 0.0
        self.explore_reward_timer = 0.0
        self.visited_districts: set = set()

    # ------------------------------------------------------------------ helpers
    @property
    def x(self) -> float:
        return self.vehicle.x

    @property
    def z(self) -> float:
        return self.vehicle.z

    @property
    def yaw(self) -> float:
        return self.vehicle.yaw

    @property
    def speed_kmh(self) -> float:
        return self.vehicle.speed_kmh

    def teleport(self, x: float, z: float, yaw: float) -> None:
        self.vehicle.reset(x, z, yaw)

    def set_rule(self, text: str, duration: float = 5.0) -> None:
        self.active_rule = text
        self.rule_timer = duration

    # ------------------------------------------------------------------ main step
    def update(self, dt: float, inputs: Dict[str, bool], signals: SignalController,
               hazards: Sequence[Hazard]) -> PlayerFeedback:
        fb = PlayerFeedback()
        vehicle = self.vehicle
        control = self.control

        # ---- gather input --------------------------------------------------------
        forward = bool(inputs.get("throttle"))
        backward = bool(inputs.get("brake"))
        gear = 1
        if backward and vehicle.speed < 0.6:
            gear = -1
        control.gear = gear
        control.throttle = 1.0 if (forward or (backward and gear == -1)) else 0.0
        control.brake = 1.0 if (backward and gear == 1) else 0.0
        steer_input = 0.0
        if inputs.get("steer_left"):
            steer_input += 1.0
        if inputs.get("steer_right"):
            steer_input -= 1.0
        control.steer = clamp(steer_input, -1.0, 1.0)
        control.handbrake = bool(inputs.get("handbrake"))

        # ---- indicators / lights --------------------------------------------------
        if inputs.get("indicator_left") and self.indicator_timer <= 0.0:
            vehicle.indicator = -1 if vehicle.indicator != -1 else 0
            self.indicator_timer = 0.25
        if inputs.get("indicator_right") and self.indicator_timer <= 0.0:
            vehicle.indicator = 1 if vehicle.indicator != 1 else 0
            self.indicator_timer = 0.25
        if inputs.get("hazards") and self.indicator_timer <= 0.0:
            vehicle.hazard_lights = not vehicle.hazard_lights
            self.indicator_timer = 0.25
        self.indicator_timer = max(0.0, self.indicator_timer - dt)
        if inputs.get("lights"):
            vehicle.headlights = not vehicle.headlights
        if inputs.get("horn") and self.horn_timer <= 0.0:
            self.horn_timer = 0.45
            fb.horn = True
        self.horn_timer = max(0.0, self.horn_timer - dt)

        # ---- world interaction ----------------------------------------------------
        network = self.network
        road: Optional[Road] = None
        surface = "road"
        grip = 1.0
        if network is not None:
            road = network.road_at(vehicle.x, vehicle.z, 1.2)
            if road is None:
                surface = "offroad"
                if network.hazard_at(vehicle.x, vehicle.z) is not None:
                    surface = "road"
            hazard_here = network.hazard_at(vehicle.x, vehicle.z)
            if hazard_here is not None and hazard_here.kind == "water":
                surface = "water"
            self.zone = network.district_at(vehicle.x, vehicle.z)
        grip *= S.RAIN_GRIP if self.weather_rain else 1.0
        if surface == "water":
            grip *= 0.85

        # ---- physics --------------------------------------------------------------
        vehicle.step(control, dt, ground_height_fn=lambda x, z: ground_height(x, z, hazards),
                     surface=surface, grip_modifier=grip)
        self.trip_distance += abs(vehicle.speed) * dt
        self.trip_time += dt
        self.max_speed_seen = max(self.max_speed_seen, vehicle.speed_kmh)

        # ---- hazards --------------------------------------------------------------
        if network is not None:
            fb.warnings.extend(self._hazard_warnings(network, dt))
            hit = self._hazard_collision(network, dt)
            if hit is not None:
                kind = hit.kind
                if kind == "pothole":
                    fb.pothole = True
                    fb.add("pothole", "POTHOLE! Suspension damaged")
                    self.potholes_hit += 1
                    self.combo = max(1.0, self.combo - 0.15)
                    self.smooth_driving_time = 0.0
                elif kind == "speed_breaker":
                    fb.speed_breaker = True
                elif kind == "water":
                    fb.add("water", "Water logged - slow down, grip reduced")
                elif kind == "construction":
                    fb.add("barrier", "BARRICADE - road under construction, keep clear")
        # ---- traffic signals / violations -----------------------------------------
        if signals is not None:
            state, junction, distance = signals.state_at(vehicle.x, vehicle.z, vehicle.yaw)
            self.last_signal_state = state
            self._check_signal_violation(state, junction, distance, fb)

        # ---- speed limit & other rules ---------------------------------------------
        if network is not None:
            limit = network.speed_limit_at(vehicle.x, vehicle.z)
            if vehicle.speed_kmh > limit + 12.0:
                self.overspeed_timer += dt
                if self.overspeed_timer > 2.5:
                    self._violation("OVERSPEEDING", -6, fb)
                    self.overspeed_timer = 0.0
            else:
                self.overspeed_timer = max(0.0, self.overspeed_timer - dt)
            self._check_wrong_side(road, dt, fb)
            self._check_road_rule_signs(road, fb)

        # ---- score & smoothing -----------------------------------------------------
        if self.violation_cooldown > 0.0:
            self.violation_cooldown = max(0.0, self.violation_cooldown - dt)
        if self.wrong_side_cooldown > 0.0:
            self.wrong_side_cooldown = max(0.0, self.wrong_side_cooldown - dt)
        if self.rule_timer > 0.0:
            self.rule_timer = max(0.0, self.rule_timer - dt)
            if self.rule_timer == 0.0:
                self.active_rule = ""
        speed_fraction = vehicle.speed_kmh / max(vehicle.spec.max_speed, 1.0)
        if 0.05 < speed_fraction < 0.75 and abs(vehicle.omega) < 0.5:
            self.smooth_driving_time += dt
        else:
            self.smooth_driving_time = max(0.0, self.smooth_driving_time - dt * 1.5)
        self.combo = clamp(1.0 + min(2.0, self.smooth_driving_time * 0.05), 1.0, 3.0)
        self.score += abs(vehicle.speed) * dt * 0.35 * self.combo

        # ---- fuel ------------------------------------------------------------------
        if vehicle.fuel <= 0.0 and vehicle.speed > 1.0:
            vehicle.speed *= 0.985
            if not self._fuel_warned:
                self._fuel_warned = True
                fb.add("fuel", "Out of fuel! Refuel at a petrol pump (press F)")
        district_here = self.zone
        if district_here and district_here not in self.visited_districts:
            self.visited_districts.add(district_here)
            self.score += 25
            fb.add("explore", f"New area explored: {district_here.title()}  (+25)")
        return fb

    # ------------------------------------------------------------------ internals
    def _hazard_warnings(self, network: RoadNetwork, dt: float) -> List[Tuple[str, float]]:
        """Short, relevant warnings for hazards the player is approaching."""
        out: List[Tuple[str, float]] = []
        vehicle = self.vehicle
        for key in list(self.hazard_warnings):
            self.hazard_warnings[key] = max(0.0, self.hazard_warnings[key] - dt)
        fx, fz = vehicle.forward()
        for hazard in network.hazards:
            dx, dz = hazard.x - vehicle.x, hazard.z - vehicle.z
            distance = math.hypot(dx, dz)
            if distance > HAZARD_WARNING_RANGE or distance < 1.0:
                continue
            ahead = (dx * fx + dz * fz) / distance
            if ahead < 0.75:
                continue
            key = id(hazard)
            if self.hazard_warnings.get(key, 0.0) > 0.0:
                continue
            label = {
                "pothole": "POTHOLE AHEAD", "speed_breaker": "SPEED BREAKER AHEAD",
                "water": "WATER LOGGING AHEAD", "construction": "ROAD WORK AHEAD",
            }.get(hazard.kind, "CAUTION")
            self.hazard_warnings[key] = 5.0
            out.append((label, 3.0))
        return out

    def _hazard_collision(self, network: RoadNetwork, dt: float) -> Optional[Hazard]:
        """Physical contact with a hazard, with a per-hazard cooldown."""
        vehicle = self.vehicle
        for hazard in network.hazards:
            key = id(hazard)
            cooldown = self.hazard_cooldowns.get(key, 0.0)
            if cooldown > 0.0:
                self.hazard_cooldowns[key] = cooldown - dt
                continue
            if not hazard.contains(vehicle.x, vehicle.z):
                continue
            if hazard.kind == "pothole":
                if abs(vehicle.speed) < 1.5:
                    continue
                severity = hazard.strength * min(1.6, abs(vehicle.speed) / 12.0)
                absorption = self.absorption
                vehicle.apply_impact(severity, "pothole", absorption)
                self.hazard_cooldowns[key] = 1.6
                self.score -= 12 * severity
                self.perfect = False
                return hazard
            if hazard.kind == "speed_breaker":
                if abs(vehicle.speed) < 2.0:
                    continue
                severity = min(1.4, abs(vehicle.speed) / 16.0)
                if abs(vehicle.speed) > 9.0:
                    vehicle.apply_impact(severity, "speed_breaker", self.absorption)
                    self.score -= 6 * severity
                    self.perfect = False
                self.hazard_cooldowns[key] = 1.2
                return hazard
            if hazard.kind == "water":
                if network.road_at(vehicle.x, vehicle.z, 0.5) is not None:
                    self.hazard_cooldowns[key] = 0.5
                    return hazard
            if hazard.kind == "construction":
                # barricades are solid: push the car back out of the dig
                dx = vehicle.x - hazard.x
                dz = vehicle.z - hazard.z
                distance = math.hypot(dx, dz) or 1.0
                push = hazard.radius * 1.35 - distance
                if push > 0.0:
                    vehicle.x += dx / distance * push
                    vehicle.z += dz / distance * push
                if abs(vehicle.speed) > 1.0:
                    vehicle.apply_impact(0.7, "collision", self.absorption)
                    self.score -= 10
                    self.perfect = False
                self.hazard_cooldowns[key] = 1.2
                return hazard
        return None

    def _check_signal_violation(self, state: str, junction, distance: float,
                                fb: PlayerFeedback) -> None:
        """A red light run is reported once: warning first, penalty only if ignored."""
        if junction is None:
            self.signal_junction_watch = None
            return
        if state in (RED, AMBER):
            if junction.id != self.signal_junction_watch:
                if state == RED:
                    self.set_rule("STOP ON RED", 4.0)
                    fb.warn("Signal: RED - stop before the line", 2.5)
                else:
                    self.set_rule("AMBER - PREPARE TO STOP", 3.0)
                self.signal_junction_watch = junction.id
            if distance < junction.half_size + 2.5:
                self._violation("JUMPED RED LIGHT", -35, fb)
                self.signal_junction_watch = None
        else:
            # green: rules about junctions still apply
            if distance < junction.half_size + 14.0:
                self.set_rule("DO NOT BLOCK JUNCTION", 3.0)
            self.signal_junction_watch = None

    def _check_wrong_side(self, road: Optional[Road], dt: float, fb: PlayerFeedback) -> None:
        if road is None or self.wrong_side_cooldown > 0.0 or self.vehicle.speed_kmh < 20.0:
            return
        legal = self.network.direction_of_travel(road, self.vehicle.x, self.vehicle.z)
        fz = math.sin(-self.vehicle.yaw)
        fx = math.cos(self.vehicle.yaw)
        along = fz if road.axis == "z" else fx
        if legal * along < -0.35 and abs(road.lateral(self.vehicle.x, self.vehicle.z)) > 1.5:
            self.set_rule("KEEP LEFT", 4.0)
            self._violation("WRONG SIDE DRIVING", -30, fb)
            self.wrong_side_cooldown = 8.0

    def _check_road_rule_signs(self, road: Optional[Road], fb: PlayerFeedback) -> None:
        """Contextual rule messages tied to the actual zone the player is in."""
        if road is None:
            return
        district = road.district
        if district == "school" and self.rule_timer <= 0.0:
            self.set_rule("SCHOOL ZONE - SLOW DOWN", 4.0)
            if self.vehicle.speed_kmh > 32.0:
                fb.warn("School zone: reduce speed", 2.0)
        elif district == "market" and self.rule_timer <= 0.0:
            self.set_rule("BAZAAR AREA - EXPECT PEDESTRIANS", 4.0)
        elif district == "hospital" and self.rule_timer <= 0.0:
            self.set_rule("HOSPITAL ZONE - NO HORN", 4.0)
        elif self.rule_timer <= 0.0 and self.last_signal_state == GREEN:
            self.set_rule("KEEP LEFT - MAINTAIN SAFE DISTANCE", 4.0)

    def _violation(self, label: str, penalty: float, fb: PlayerFeedback) -> None:
        if self.violation_cooldown > 0.0:
            return
        self.violation_cooldown = VIOLATION_COOLDOWN
        self.violations += 1
        self.score += penalty
        self.perfect = False
        self.combo = 1.0
        fb.add("violation", f"{label}   {int(penalty)} points")

    @property
    def absorption(self) -> float:
        from .vehicle import suspension_absorption
        return suspension_absorption(self.upgrades)

    # ------------------------------------------------------------------ reporting
    def summary(self) -> Dict[str, object]:
        return {
            "car": self.car_key,
            "score": int(self.score),
            "collisions": self.collisions,
            "violations": self.violations,
            "potholes": self.potholes_hit,
            "distance_km": self.trip_distance / 1000.0,
            "time_s": self.trip_time,
            "perfect": self.perfect and self.violations == 0,
            "top_speed": self.max_speed_seen,
        }

    def register_collision(self, severity: float, other: str = "traffic") -> None:
        self.vehicle.apply_impact(severity, "collision", self.absorption)
        self.collisions += 1
        self.perfect = False
        self.combo = 1.0
        self.score -= 25 * max(0.5, severity)
