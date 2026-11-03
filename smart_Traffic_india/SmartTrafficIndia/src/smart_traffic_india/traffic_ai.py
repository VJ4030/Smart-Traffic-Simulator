"""AI traffic: lane following, car following, signals, overtaking and Indian behaviours.

Every AI vehicle is a full :class:`~smart_traffic_india.vehicle.Vehicle`, so it uses the
same physics as the player (grip limit, suspension, brakes).  On top of the physics sits a
small driver model:

* **path following** - the lane graph is a set of waypoint polylines; the driver steers at
  a speed dependent look-ahead point (a classic pure-pursuit style controller),
* **car following** - an intelligent-driver-model style controller with a desired time
  gap, so queues form and dissolve smoothly,
* **signal obedience** - the driver computes the stop line of the junction it is feeding
  and brakes with a comfortable deceleration when the signal is red or amber,
* **overtaking** - a short lateral offset applied to the look-ahead point, only used when
  the neighbouring space is genuinely free,
* **Indian flavour** - auto-rickshaws weave between lanes, buses take wider turns and stop
  at bus stops, trucks accelerate slowly, delivery bikes pause at the kerb, taxis pull
  over, and emergency vehicles ignore signals while asking for a way through.

The module never imports pygame or OpenGL, so the driver model is unit-testable.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .road import Junction, Lane, RoadNetwork
from .traffic_signals import GREEN, SignalController
from .transform import clamp, damp
from .vehicle import Vehicle, VehicleControl, VehicleSpec
from . import settings as S

Vec2 = Tuple[float, float]


def _spec(kind: str, label: str, length: float, width: float, height: float,
          max_speed: float, accel: float, brake: float, grip: float, handling: float,
          mass: float, wheel_radius: float) -> VehicleSpec:
    return VehicleSpec(kind, label, "AI", 0, max_speed, accel, brake, grip, handling, mass,
                       14.0, length, width, height, length * 0.58, wheel_radius,
                       description="AI traffic")


#: Performance data of every AI vehicle category.
TRAFFIC_SPECS: Dict[str, VehicleSpec] = {
    "hatchback": _spec("hatchback", "Hatchback", 3.6, 1.62, 1.5, 110, 3.6, 7.6, 1.02, 28, 1040, 0.30),
    "sedan": _spec("sedan", "Sedan", 4.32, 1.74, 1.46, 122, 3.8, 8.0, 1.06, 27, 1245, 0.32),
    "suv": _spec("suv", "SUV", 4.52, 1.88, 1.78, 118, 3.0, 7.2, 1.0, 23, 1720, 0.36),
    "taxi": _spec("taxi", "Taxi", 4.2, 1.68, 1.52, 112, 3.4, 7.8, 1.0, 26, 1160, 0.31),
    "auto": _spec("auto", "Auto Rickshaw", 2.6, 1.36, 1.75, 74, 2.6, 6.4, 0.86, 30, 480, 0.26),
    "bike": _spec("bike", "Motorcycle", 2.05, 0.72, 1.7, 92, 3.6, 7.0, 0.92, 32, 180, 0.30),
    "bus": _spec("bus", "City Bus", 10.5, 2.5, 3.3, 82, 1.6, 5.6, 0.9, 16, 12500, 0.52),
    "truck": _spec("truck", "Truck", 8.5, 2.4, 3.5, 78, 1.4, 5.2, 0.86, 15, 14500, 0.50),
    "tempo": _spec("tempo", "Delivery Van", 4.7, 1.85, 2.2, 92, 2.4, 6.6, 0.94, 22, 2600, 0.34),
    "ambulance": _spec("ambulance", "Ambulance", 4.7, 1.85, 2.2, 126, 4.0, 8.4, 1.05, 26, 2800, 0.34),
    "police": _spec("police", "Police Jeep", 4.7, 1.85, 2.2, 120, 3.6, 8.0, 1.02, 25, 2700, 0.34),
}

#: Spawn weights per district - who you meet where.
DISTRICT_TRAFFIC: Dict[str, Dict[str, float]] = {
    "downtown":   {"hatchback": 3.0, "sedan": 3.0, "suv": 1.6, "taxi": 2.2, "auto": 2.0,
                   "bike": 2.4, "bus": 0.8, "tempo": 0.8, "truck": 0.4},
    "market":     {"auto": 4.0, "bike": 3.2, "hatchback": 2.2, "sedan": 1.4, "taxi": 1.6,
                   "tempo": 1.2, "suv": 0.8, "truck": 0.5},
    "oldcity":    {"auto": 4.5, "bike": 3.5, "hatchback": 2.0, "tempo": 1.4, "taxi": 1.0,
                   "sedan": 0.8, "truck": 0.6},
    "hospital":   {"sedan": 2.6, "hatchback": 2.4, "suv": 1.8, "taxi": 2.0, "ambulance": 0.5,
                   "bike": 1.4, "tempo": 0.8},
    "school":     {"auto": 2.6, "hatchback": 2.4, "bike": 1.6, "sedan": 1.6, "taxi": 1.2},
    "railway":    {"bus": 2.4, "auto": 3.0, "tempo": 1.6, "hatchback": 2.0, "sedan": 1.6,
                   "bike": 1.8, "taxi": 1.4},
    "industrial": {"truck": 3.4, "tempo": 2.4, "hatchback": 1.6, "suv": 1.2, "bike": 1.2,
                   "auto": 1.0},
    "highway":    {"truck": 3.0, "bus": 2.0, "sedan": 2.4, "suv": 1.8, "bike": 1.0,
                   "taxi": 1.0},
    "residential":{"hatchback": 3.0, "bike": 2.2, "auto": 1.6, "sedan": 1.2, "suv": 1.0},
}

#: per-kind driver behaviour
BEHAVIOUR: Dict[str, Dict[str, float]] = {
    "hatchback": {"speed": 0.95, "lane_change": 0.10, "turn_slow": 1.0, "gap": 1.0},
    "sedan":     {"speed": 1.0, "lane_change": 0.10, "turn_slow": 1.0, "gap": 1.0},
    "suv":       {"speed": 0.95, "lane_change": 0.08, "turn_slow": 1.1, "gap": 1.05},
    "taxi":      {"speed": 1.02, "lane_change": 0.18, "turn_slow": 1.0, "gap": 0.9},
    "auto":      {"speed": 0.78, "lane_change": 0.45, "turn_slow": 0.8, "gap": 0.75},
    "bike":      {"speed": 1.0, "lane_change": 0.55, "turn_slow": 0.9, "gap": 0.6},
    "bus":       {"speed": 0.82, "lane_change": 0.03, "turn_slow": 1.5, "gap": 1.25},
    "truck":     {"speed": 0.78, "lane_change": 0.04, "turn_slow": 1.6, "gap": 1.35},
    "tempo":     {"speed": 0.88, "lane_change": 0.14, "turn_slow": 1.1, "gap": 1.0},
    "ambulance": {"speed": 1.25, "lane_change": 0.35, "turn_slow": 1.1, "gap": 0.8},
    "police":    {"speed": 1.05, "lane_change": 0.25, "turn_slow": 1.0, "gap": 0.9},
}

EMERGENCY_KINDS = {"ambulance", "police"}


@dataclass
class TrafficContext:
    """Everything a driver needs to know about the current frame."""

    neighbours: List["TrafficVehicle"] = field(default_factory=list)
    player: Optional[Vehicle] = None
    pedestrian_near: Optional[Callable[[float, float, float], bool]] = None
    congestion: float = 1.0
    rain: bool = False
    night: bool = False


class TrafficVehicle:
    """One AI vehicle with its driver model."""

    def __init__(self, kind: str, lane: Lane, rng: random.Random,
                 speed_factor: float = 1.0, parked: bool = False,
                 point_index: int = 0) -> None:
        self.kind = kind
        self.spec = TRAFFIC_SPECS[kind]
        self.vehicle = Vehicle(self.spec, 0.0, 0.0, 0.0, name=f"{kind}-{id(self) % 9973}")
        self.rng = rng
        self.behaviour = BEHAVIOUR[kind]
        self.emergency = kind in EMERGENCY_KINDS
        self.parked = parked
        self.lane: Lane = lane
        self.point_index = point_index
        self.speed_factor = speed_factor
        self.lateral_offset = 0.0
        self.lateral_target = 0.0
        self.lane_change_timer = 0.0
        self.waypoint_reach = 5.5
        self.block_timer = 0.0
        self.blocked = False
        self.idle_timer = 0.0
        self.stuck_timer = 0.0
        self.lifetime = 0.0
        self.dead = False                 # set when the vehicle ran out of road
        self.stopped_for = 0.0
        self.last_signal_state = GREEN
        self.bus_stop_timer = 0.0
        self.delivery_pause = 0.0
        self.turning = 0.0
        self.crossing_junction = False
        self.finished_timer = 0.0

        # place the car on the lane
        start = self.lane.points[min(self.point_index, len(self.lane.points) - 1)]
        nxt = self.lane.points[min(self.point_index + 1, len(self.lane.points) - 1)]
        yaw = math.atan2(-(nxt[1] - start[1]), nxt[0] - start[0])
        self.vehicle.reset(start[0], start[1], yaw)
        if not parked:
            self.vehicle.speed = min(self.spec.max_speed_ms() * 0.6,
                                     self.road_limit() * 0.7)
            self.vehicle.headlights = True
        if parked:
            self.vehicle.hazard_lights = True
            self.lateral_target = self.lateral_offset = 0.0

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
    def speed(self) -> float:
        return self.vehicle.speed

    @property
    def name(self) -> str:
        return self.vehicle.name

    @property
    def half_length(self) -> float:
        return self.vehicle.half_length

    @property
    def half_width(self) -> float:
        return self.vehicle.half_width

    @property
    def position(self) -> Vec2:
        return (self.vehicle.x, self.vehicle.z)

    def road_limit(self) -> float:
        return self.lane.road.speed_limit / 3.6

    def block_speed(self, factor: float = 0.5, seconds: float = 1.2) -> None:
        """Temporarily cut the desired speed (used by collisions and road events)."""
        self.blocked = True
        self.block_timer = max(self.block_timer, seconds)
        self.vehicle.speed *= factor

    def distance_to(self, other: Vehicle) -> float:
        return math.hypot(other.x - self.vehicle.x, other.z - self.vehicle.z)

    # ------------------------------------------------------------------ driving
    def update(self, dt: float, ctx: TrafficContext) -> None:
        self.lifetime += dt
        if self.dead:
            return
        if self.block_timer > 0.0:
            self.block_timer = max(0.0, self.block_timer - dt)
        if not self.blocked or self.block_timer <= 0.0:
            self.blocked = False
        control = VehicleControl()

        if self.parked:
            control.brake = 0.0 if self.vehicle.speed > 0.05 else 0.0
            self.vehicle.step(control, dt, None, "road", 1.0)
            self.vehicle.hazard_lights = True
            return

        if self.finished_timer > 0.0:
            self.finished_timer -= dt
            if self.finished_timer <= 0.0:
                # ran out of lane at the edge of the map: let the manager recycle us
                # instead of parking a bus in the middle of the carriageway forever
                self.dead = True
                return
        self._update_progress(dt)
        desired = self._desired_speed(ctx)
        gap_speed = self._follow_speed(ctx)
        if self.blocked:
            gap_speed = min(gap_speed, 1.0)
        target = min(desired, gap_speed)

        # ---- longitudinal control: throttle / brake with smooth ramps --------------
        speed = self.vehicle.speed
        error = target - speed
        if error > 0.25:
            control.throttle = clamp(error * 0.45, 0.15, 1.0)
            control.brake = 0.0
        elif error < -0.3:
            # feed-forward plus proportional: without the constant term a car that needs to
            # lose 2 m/s touches the brake so gently that it coasts into the stop line
            control.brake = clamp(0.30 + (-error) * 0.30, 0.30, 1.0)
            control.throttle = 0.0
        else:
            control.throttle = 0.25 if target > 0.5 else 0.0
            control.brake = 0.0
        if target < 0.4 and speed < 1.2:
            control.throttle = 0.0
            control.brake = 1.0 if speed > 0.05 else 0.0

        # ---- steering: look ahead along the lane ----------------------------------
        look = clamp(4.0 + speed * 0.85, 5.0, 24.0)
        target_point = self._lookahead_point(look)
        control.steer = self._steer_towards(target_point, speed)
        self.vehicle.step(control, dt, None,
                          "road", S.RAIN_GRIP if ctx.rain else 1.0)

        # ---- lights ------------------------------------------------------------------
        self.vehicle.reverse_light = speed < -0.1
        if ctx.night or self.emergency:
            self.vehicle.headlights = True
        if self.emergency:
            self.vehicle.indicator = 0
        self._update_behaviours(dt, ctx, speed, target)

    # -- progress along the lane -----------------------------------------------------
    def _update_progress(self, dt: float) -> None:
        vehicle = self.vehicle
        while self.point_index < len(self.lane.points) - 1:
            px, pz = self.lane.points[self.point_index]
            if math.hypot(px - vehicle.x, pz - vehicle.z) > self.waypoint_reach:
                break
            self.point_index += 1
        if self.point_index >= len(self.lane.points) - 1:
            reached = self.lane.points[-1]
            if math.hypot(reached[0] - vehicle.x, reached[1] - vehicle.z) < self.waypoint_reach * 1.6:
                self._pick_next_lane()
        del dt

    def _pick_next_lane(self) -> bool:
        """Choose the next lane segment (straight / left / right) at the junction."""
        if self.network_ref is None:
            self.finished_timer = 2.0
            return False
        options = self.network_ref.successors(self.lane.id)
        if not options:
            self.finished_timer = 2.5        # road ends here - the manager recycles us
            return False
        # never route into a lane that itself leads nowhere unless there is no choice,
        # otherwise traffic would drift off the edge of the map instead of circulating
        if self.network_ref is not None and len(options) > 1:
            live = [o for o in options if self.network_ref.successors(o[0])]
            if live:
                options = live
        weights = []
        for _, _, turn in options:
            if turn == "straight":
                weights.append(0.66)
            elif turn == "left":
                weights.append(0.22)
            else:
                weights.append(0.12)
        total = sum(weights)
        pick = self.rng.random() * total
        chosen = options[-1]
        for option, weight in zip(options, weights):
            pick -= weight
            if pick <= 0.0:
                chosen = option
                break
        lane_id, _entry, turn = chosen
        self.lane = self.network_ref.lane_by_id(lane_id)
        self.point_index = 1
        self.lateral_target = 0.0
        self.turning = 0.9 if turn != "straight" else 0.0
        return True

    # -- speeds ----------------------------------------------------------------------
    def _desired_speed(self, ctx: TrafficContext) -> float:
        limit = self.road_limit()
        base = limit * self.speed_factor * self.behaviour["speed"]
        base /= max(0.55, (1.0 + (ctx.congestion - 1.0) * 0.35))
        if self.turning > 0.0:
            base = min(base, 4.2 * self.behaviour["turn_slow"])
        elif self.vehicle.omega != 0.0:
            base = min(base, limit)
        # slow down approaching a junction we must give way at / cannot clear
        junction = self.lane.end_junction
        if junction is not None:
            distance = self.distance_to_junction()
            if distance < 30.0:
                base = min(base, 6.0 + distance * 0.4)
                if not junction.signalised and distance < 12.0:
                    base = min(base, 4.0)        # unsignalised: ease off and look
        # red / amber signal ahead: brake comfortably for the stop line
        base = min(base, self._junction_stop_speed())
        # potholes ahead: slow down and steer around them
        base = min(base, self._hazard_speed())
        # pedestrians near a crossing
        if ctx.pedestrian_near is not None and ctx.pedestrian_near(self.vehicle.x, self.vehicle.z, 16.0):
            base = min(base, 3.0)
        if self.bus_stop_timer > 0.0 or self.delivery_pause > 0.0:
            base = 0.0
        return max(0.0, base)

    def _hazard_speed(self) -> float:
        network = self.network_ref
        vehicle = self.vehicle
        fx, fz = vehicle.forward()
        for hazard in network.hazards:
            if hazard.kind not in ("pothole", "water", "construction"):
                continue
            dx, dz = hazard.x - vehicle.x, hazard.z - vehicle.z
            distance = math.hypot(dx, dz)
            if distance > 22.0 or distance < 0.5:
                continue
            ahead = (dx * fx + dz * fz) / distance
            if ahead < 0.9:
                continue
            # is it in our lane?
            lateral = abs(dx * -fz + dz * fx)
            if lateral > self.half_width + 1.1:
                continue
            if hazard.kind == "pothole":
                return min(22.0 / 3.6, 2.0 + distance * 0.9)
            return min(18.0 / 3.6, 2.0 + distance * 0.9)
        return 99.0

    def _follow_speed(self, ctx: TrafficContext) -> float:
        """Intelligent-driver-model style gap control against vehicles ahead."""
        vehicle = self.vehicle
        fx, fz = vehicle.forward()
        rx, rz = -fz, fx
        gap_scale = self.behaviour["gap"]
        limit = 10e9
        obstacles: List[Vehicle] = [other.vehicle for other in ctx.neighbours
                                    if other is not self]
        if ctx.player is not None:
            obstacles.append(ctx.player)
        for other in obstacles:
            dx = other.x - vehicle.x
            dz = other.z - vehicle.z
            along = dx * fx + dz * fz
            lateral = abs(dx * rx + dz * rz)
            if along < 0.5 or along > 46.0:
                continue
            if lateral > other.half_width + self.half_width + 0.75:
                continue
            safe_gap = 2.6 + self.vehicle.speed * 0.9 * gap_scale + other.half_length
            closing = self.vehicle.speed - other.speed
            if along < safe_gap:
                limit = min(limit, max(0.0, other.speed - closing * 0.4))
            else:
                allowed = other.speed + (along - safe_gap) * 0.65
                limit = min(limit, max(0.0, allowed))
        return limit

    def _junction_stop_speed(self) -> float:
        """Speed limit imposed by a red / amber signal ahead."""
        junction = self.lane.end_junction
        if junction is None or not junction.signalised or self.signals_obj is None:
            return 10e9
        state = self.signals_obj.state_for(junction, self.lane.road.axis, self.lane.direction)
        self.last_signal_state = state
        if state == GREEN:
            return 10e9
        # aim ~3 m short of the painted line, so the car comes to rest *at* the line
        # instead of creeping over it (a real Indian stop, minus the optimism)
        stop_distance = (self.distance_to_junction()
                         - (self.lane.road.half_width + 3.2) + 2.2)
        if stop_distance <= 0.35:
            # At (or past) the stop line.  Hold if we are crawling - the classic Indian
            # "creep over the line" is what would let a car sail through on red - and only
            # clear the junction if we are already committed (fast, cannot stop).
            if abs(self.vehicle.speed) < 4.5:
                return 0.0
            return 10e9
        if self.emergency and stop_distance > 3.0:
            # emergency vehicles slow right down but do not stop
            return min(40.0 / 3.6, 3.0 + stop_distance * 1.4)
        required = (self.vehicle.speed ** 2) / max(2.0 * stop_distance, 0.5)
        if required > 6.5:
            return 10e9                     # genuinely too late: clear the junction safely
        # comfortable deceleration: v = sqrt(2 * a * d)
        return max(0.0, math.sqrt(max(0.0, 2.0 * 2.6 * stop_distance)))

    def distance_to_junction(self) -> float:
        vehicle = self.vehicle
        total = 0.0
        points = self.lane.points
        idx = min(self.point_index, len(points) - 1)
        px, pz = vehicle.x, vehicle.z
        for i in range(idx, len(points)):
            total += math.hypot(points[i][0] - px, points[i][1] - pz)
            px, pz = points[i]
        return total

    def _lookahead_point(self, lookahead: float) -> Vec2:
        vehicle = self.vehicle
        points = self.lane.points
        idx = min(self.point_index, len(points) - 1)
        prev = (vehicle.x, vehicle.z)
        travelled = 0.0
        for i in range(idx, len(points)):
            seg = math.hypot(points[i][0] - prev[0], points[i][1] - prev[1])
            if travelled + seg >= lookahead:
                t = (lookahead - travelled) / max(seg, 1e-6)
                point = (prev[0] + (points[i][0] - prev[0]) * t,
                         prev[1] + (points[i][1] - prev[1]) * t)
                break
            travelled += seg
            prev = points[i]
        else:
            point = points[-1]
        if abs(self.lateral_offset) > 0.01:
            # shift the target sideways for a lane change / overtake
            direction = math.atan2(-(point[1] - vehicle.z), point[0] - vehicle.x)
            side = (-math.sin(direction), math.cos(direction))
            point = (point[0] + side[0] * self.lateral_offset,
                     point[1] + side[1] * self.lateral_offset)
        return point

    def _steer_towards(self, point: Vec2, speed: float) -> float:
        vehicle = self.vehicle
        dx, dz = point[0] - vehicle.x, point[1] - vehicle.z
        if abs(dx) < 1e-6 and abs(dz) < 1e-6:
            return 0.0
        desired_yaw = math.atan2(-dz, dx)
        error = math.atan2(math.sin(desired_yaw - vehicle.yaw), math.cos(desired_yaw - vehicle.yaw))
        gain = 1.9 if speed < 8.0 else 1.25
        return clamp(error * gain, -1.0, 1.0)

    # -- behaviours ------------------------------------------------------------------
    def _update_behaviours(self, dt: float, ctx: TrafficContext,
                           speed: float, target: float) -> None:
        self.turning = max(0.0, self.turning - dt)
        if speed < 0.6:
            self.stopped_for += dt
        else:
            self.stopped_for = 0.0
        if self.vehicle.speed < 0.25 and not self.blocked:
            self.stuck_timer += dt
        else:
            self.stuck_timer = 0.0

        # bus stop / delivery pause
        if self.bus_stop_timer > 0.0:
            self.bus_stop_timer -= dt
        elif self.kind == "bus" and self.rng.random() < 0.0009:
            self.bus_stop_timer = self.rng.uniform(3.0, 6.0)
            self.lateral_target = self._kerb_side() * 1.1
        if self.delivery_pause > 0.0:
            self.delivery_pause -= dt
        elif self.kind in ("bike", "tempo") and self.rng.random() < 0.0012:
            self.delivery_pause = self.rng.uniform(1.5, 3.5)
            self.lateral_target = self._kerb_side() * 1.2

        # taxis pull over briefly
        if self.kind == "taxi" and self.rng.random() < 0.0006:
            self.delivery_pause = self.rng.uniform(2.0, 4.0)
            self.lateral_target = self._kerb_side() * 1.3

        # overtaking / lane changes (only when the neighbouring space is free)
        if self.lane_change_timer > 0.0:
            self.lane_change_timer -= dt
            if self.lane_change_timer <= 0.0:
                self.lateral_target = 0.0
        elif (self.rng.random() < self.behaviour["lane_change"] * 0.012 * self._traffic_pressure(ctx)
              and abs(self.lateral_offset) < 0.9):
            if self._space_beside_is_free(ctx):
                side = self.rng.choice([1.0, -1.0])
                self.lateral_target = side * min(1.5, self.lane.road.lane_width * 0.55)
                self.lane_change_timer = self.rng.uniform(2.0, 4.5)
        self.lateral_offset = damp(self.lateral_offset, self.lateral_target, 1.4, dt)

    def _traffic_pressure(self, ctx: TrafficContext) -> float:
        return clamp(0.4 + ctx.congestion, 0.4, 2.5)

    def _kerb_side(self) -> float:
        """Sign of the lateral offset that points towards the kerb."""
        # lane offsets are already measured from the road centre
        return 1.0 if self.lane.offset >= 0 else -1.0

    def _space_beside_is_free(self, ctx: TrafficContext) -> bool:
        vehicle = self.vehicle
        fx, fz = vehicle.forward()
        rx, rz = -fz, fx
        for other in ctx.neighbours:
            if other is self:
                continue
            dx = other.vehicle.x - vehicle.x
            dz = other.vehicle.z - vehicle.z
            along = dx * fx + dz * fz
            lateral = abs(dx * rx + dz * rz)
            if abs(along) < 16.0 and lateral < 3.6:
                return False
        if ctx.player is not None:
            dx = ctx.player.x - vehicle.x
            dz = ctx.player.z - vehicle.z
            along = dx * fx + dz * fz
            lateral = abs(dx * rx + dz * rz)
            if abs(along) < 16.0 and lateral < 3.6:
                return False
        return True

    def skip_to(self, lane: Lane, point_index: int = 0) -> None:
        self.lane = lane
        self.point_index = point_index
        point = lane.points[min(point_index, len(lane.points) - 1)]
        nxt = lane.points[min(point_index + 1, len(lane.points) - 1)]
        yaw = math.atan2(-(nxt[1] - point[1]), nxt[0] - point[0])
        self.vehicle.reset(point[0], point[1], yaw)

    # ------------------------------------------------------------------ network
    network_ref: RoadNetwork = None          # type: ignore[assignment]
    signals_obj: Optional[SignalController] = None

    def bind(self, network: RoadNetwork, signals: SignalController) -> None:
        self.network_ref = network
        self.signals_obj = signals


# --------------------------------------------------------------------------------------
# manager
# --------------------------------------------------------------------------------------
class TrafficManager:
    """Spawns, streams, updates and recycles the AI traffic around the player."""

    def __init__(self, network: RoadNetwork, signals: SignalController,
                 quality: Dict[str, object], level: str = "medium",
                 seed: int = 1337, event_sink: Optional[Callable[[str, str], None]] = None) -> None:
        self.network = network
        self.signals = signals
        self.quality = quality
        self.rng = random.Random(seed)
        self.level = level if level in S.TRAFFIC_LEVELS else "medium"
        self.vehicles: List[TrafficVehicle] = []
        self.parked: List[TrafficVehicle] = []
        self.pedestrian_query: Optional[Callable[[float, float, float], bool]] = None
        self.event_sink = event_sink
        self.emergency: Optional[TrafficVehicle] = None
        self.emergency_announced = False
        self.spawn_accumulator = 0.0
        self.stats = {"spawned": 0, "despawned": 0, "updated": 0}
        self.signal_queue: Dict[Tuple[int, str, int], int] = {}
        signals.queue_provider = self.queue_length_for_signal

    # ------------------------------------------------------------------ settings
    @property
    def level_config(self) -> Dict[str, float]:
        return S.TRAFFIC_LEVELS[self.level]

    def set_level(self, level: str) -> None:
        if level in S.TRAFFIC_LEVELS:
            self.level = level

    @property
    def max_vehicles(self) -> int:
        scale = float(self.quality.get("traffic_scale", 1.0))
        return int(max(6, self.level_config["max_vehicles"] * scale))

    # ------------------------------------------------------------------ spawning
    def _spawn_weights_for_district(self, district: str) -> Dict[str, float]:
        return DISTRICT_TRAFFIC.get(district, DISTRICT_TRAFFIC["residential"])

    def _pick_kind(self, district: str) -> str:
        weights = self._spawn_weights_for_district(district)
        total = sum(weights.values())
        pick = self.rng.random() * total
        for kind, weight in weights.items():
            pick -= weight
            if pick <= 0.0:
                return kind
        return "hatchback"

    def _lane_spawn_candidates(self, player_pos: Vec2, min_distance: float = 80.0,
                               max_distance: float = 250.0) -> List[Tuple[Lane, int]]:
        """Lane points that are far enough away to pop in unnoticed."""
        out: List[Tuple[Lane, int]] = []
        for lane in self.network.lanes:
            road = lane.road
            if road.speed_limit < 30.0:
                continue
            if not self.network.successors(lane.id):
                continue                    # map edge: cars would have nowhere to go
            for i, point in enumerate(lane.points):
                if i >= len(lane.points) - 1:
                    break
                distance = math.hypot(point[0] - player_pos[0], point[1] - player_pos[1])
                if distance < min_distance or distance > max_distance:
                    continue
                if self.network.in_junction(point[0], point[1], 2.0):
                    continue
                out.append((lane, i))
        return out

    def spawn_vehicle(self, kind: Optional[str] = None, lane: Optional[Lane] = None,
                      point_index: int = 0, player_pos: Vec2 = (0.0, 0.0),
                      parked: bool = False) -> Optional[TrafficVehicle]:
        if not parked and len(self.vehicles) >= self.max_vehicles:
            return None                     # the density limit is a hard cap
        if lane is None:
            candidates = self._lane_spawn_candidates(player_pos)
            if not candidates:
                return None
            # bias the spawn towards quiet stretches so new cars appear in traffic that
            # actually flows instead of right behind a queue (uses the live congestion field)
            weights = []
            for candidate_lane, candidate_index in candidates:
                point = candidate_lane.points[min(candidate_index,
                                                  len(candidate_lane.points) - 1)]
                weights.append(max(0.15, 1.0 - self._congestion_at(point[0], point[1])))
            total = sum(weights) or 1.0
            pick = self.rng.random() * total
            chosen = candidates[0]
            for candidate, weight in zip(candidates, weights):
                pick -= weight
                if pick <= 0.0:
                    chosen = candidate
                    break
            lane, point_index = chosen
        if kind is None:
            district = lane.road.district
            kind = self._pick_kind(district)
        point = lane.points[min(point_index, len(lane.points) - 1)]
        for other in self.vehicles:
            if math.hypot(other.vehicle.x - point[0], other.vehicle.z - point[1]) < 9.0:
                return None
        vehicle = TrafficVehicle(kind, lane, self.rng,
                                 speed_factor=self.rng.uniform(0.88, 1.08), parked=parked,
                                 point_index=point_index)
        vehicle.bind(self.network, self.signals)
        if parked:
            self.parked.append(vehicle)
        else:
            self.vehicles.append(vehicle)
        self.stats["spawned"] += 1
        return vehicle

    def spawn_emergency(self, kind: str = "ambulance", lane: Optional[Lane] = None,
                        point_index: int = 0, player_pos: Vec2 = (0.0, 0.0),
                        player: Optional[Vehicle] = None) -> Optional[TrafficVehicle]:
        """Spawn an emergency vehicle that asks for a way through.

        When a player vehicle is supplied we try to seed the ambulance *behind* the player
        on the same road travelling the same way, so that it actually catches up - which is
        what makes the "give way" challenge work in a live demo.
        """
        if self.emergency is not None and self.emergency in self.vehicles:
            return self.emergency
        vehicle: Optional[TrafficVehicle] = None
        if lane is None and player is not None:
            fx, fz = player.forward()
            road = self.network.road_at(player.x, player.z, 2.5)
            best: Optional[Tuple[float, Lane, int]] = None
            if road is not None:
                for candidate in self.network.lanes:
                    if candidate.road is not road:
                        continue
                    for i in range(1, len(candidate.points) - 1):
                        px, pz = candidate.points[i]
                        dx, dz = px - player.x, pz - player.z
                        distance = math.hypot(dx, dz)
                        if distance < 55.0 or distance > 150.0:
                            continue
                        behind = -(dx * fx + dz * fz) / max(distance, 1e-6)
                        if behind < 0.6:
                            continue
                        score = distance - behind * 20.0
                        if best is None or score < best[0]:
                            best = (score, candidate, i)
            if best is not None:
                vehicle = self.spawn_vehicle(kind, best[1], best[2], player_pos)
        if vehicle is None:
            vehicle = self.spawn_vehicle(kind, lane, point_index, player_pos)
        if vehicle is None and lane is None:
            for fallback in self.network.lanes:
                if len(fallback.points) > 3:
                    vehicle = self.spawn_vehicle(kind, fallback, 1, player_pos)
                    if vehicle is not None:
                        break
        if vehicle is not None:
            vehicle.speed_factor = 1.35
            self.emergency = vehicle
            self.emergency_announced = False
            if self.event_sink:
                self.event_sink("emergency", "Ambulance dispatched - give way!")
        return vehicle

    def despawn_emergency(self) -> None:
        if self.emergency is not None and self.emergency in self.vehicles:
            self.vehicles.remove(self.emergency)
        self.emergency = None

    # ------------------------------------------------------------------ update
    def update(self, dt: float, player_vehicle: Vehicle, environment=None) -> None:
        player_pos = (player_vehicle.x, player_vehicle.z)
        self._player_pos = player_pos
        self._spawn_loop(dt)
        self._despawn_far(player_pos)

        congestion = self._congestion_at(player_vehicle.x, player_vehicle.z)
        rain = bool(getattr(environment, "rain_intensity", 0.0) > 0.4) if environment else False
        night = bool(getattr(environment, "is_night", False)) if environment else False
        grid = self._build_grid()

        for vehicle in self.vehicles:
            neighbours = self._neighbours_from_grid(grid, vehicle)
            ctx = TrafficContext(neighbours=neighbours, player=player_vehicle,
                                 pedestrian_near=self.pedestrian_query,
                                 congestion=congestion, rain=rain, night=night)
            if environment is not None:
                road = self.network.road_at(vehicle.x, vehicle.z, 3.0)
                surface = "road" if road is not None else "offroad"
            else:
                surface = "road"
            vehicle.update(dt, ctx)
            if surface == "offroad" and abs(vehicle.speed) > 4.0:
                vehicle.vehicle.speed *= 0.985
        for vehicle in self.parked:
            vehicle.update(dt, TrafficContext())
        if any(v.dead for v in self.vehicles):
            self.vehicles = [v for v in self.vehicles if not v.dead]
            if self.emergency is not None and self.emergency.dead:
                self.emergency = None
            self.stats["recycled"] = self.stats.get("recycled", 0) + 1
        self._update_queue_table()
        self.stats["updated"] += 1

    def _spawn_loop(self, dt: float) -> None:
        self.spawn_accumulator += dt
        if self.spawn_accumulator < 0.35:
            return
        self.spawn_accumulator = 0.0
        if len(self.vehicles) >= self.max_vehicles:
            return
        player_pos = self._player_pos
        self.spawn_vehicle(player_pos=player_pos)

    _player_pos: Vec2 = (0.0, 0.0)

    def _despawn_far(self, player_pos: Vec2) -> None:
        keep: List[TrafficVehicle] = []
        for vehicle in self.vehicles:
            distance = math.hypot(vehicle.x - player_pos[0], vehicle.z - player_pos[1])
            if (distance > 330.0 or vehicle.stuck_timer > 40.0 or vehicle.lifetime > 600.0
                    or vehicle.finished_timer > 0.0 and distance > 70.0):
                self.stats["despawned"] += 1
                if vehicle is self.emergency:
                    self.emergency = None
                continue
            keep.append(vehicle)
        self.vehicles = keep

    # ------------------------------------------------------------------ queries
    def _build_grid(self) -> Dict[Tuple[int, int], List[TrafficVehicle]]:
        grid: Dict[Tuple[int, int], List[TrafficVehicle]] = {}
        for vehicle in self.vehicles:
            key = (int(vehicle.x // 16.0), int(vehicle.z // 16.0))
            grid.setdefault(key, []).append(vehicle)
        return grid

    def _neighbours_from_grid(self, grid: Dict[Tuple[int, int], List[TrafficVehicle]],
                              vehicle: TrafficVehicle) -> List[TrafficVehicle]:
        key = (int(vehicle.x // 16.0), int(vehicle.z // 16.0))
        out: List[TrafficVehicle] = []
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                out.extend(grid.get((key[0] + dx, key[1] + dz), ()))
        return out

    def _congestion_at(self, x: float, z: float) -> float:
        road = self.network.road_at(x, z, 6.0)
        base = road.congestion if road is not None else 1.0
        return base * (0.55 + 0.75 * float(self.level_config["density"]))

    def queue_length_for_signal(self, junction: Junction, axis: str, direction: int) -> int:
        """Number of vehicles queued on one approach - feeds the adaptive controller."""
        count = 0
        for vehicle in self.vehicles:
            lane = vehicle.lane
            if lane.road.axis != axis or lane.direction != direction:
                continue
            if lane.end_junction is not junction:
                continue
            distance = vehicle.distance_to_junction()
            if distance < 52.0 and abs(vehicle.speed) < 4.5:
                count += 1
        return count

    def _update_queue_table(self) -> None:
        self.signal_queue.clear()
        for phase in self.signals.phases.values():
            for axis in ("z", "x"):
                for direction in (1, -1):
                    self.signal_queue[(phase.junction.id, axis, direction)] = \
                        self.queue_length_for_signal(phase.junction, axis, direction)

    def nearby(self, x: float, z: float, radius: float) -> List[TrafficVehicle]:
        r2 = radius * radius
        return [v for v in self.vehicles if (v.x - x) ** 2 + (v.z - z) ** 2 <= r2]

    def emergency_near(self, x: float, z: float, radius: float = 55.0) -> bool:
        vehicle = self.emergency
        if vehicle is None or vehicle not in self.vehicles:
            return False
        return math.hypot(vehicle.x - x, vehicle.z - z) <= radius

    def count_by_kind(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for vehicle in self.vehicles:
            counts[vehicle.kind] = counts.get(vehicle.kind, 0) + 1
        return counts

    def statistics(self) -> Dict[str, int]:
        stats = dict(self.stats)
        stats.update({"active": len(self.vehicles), "parked": len(self.parked),
                      "max": self.max_vehicles})
        return stats
