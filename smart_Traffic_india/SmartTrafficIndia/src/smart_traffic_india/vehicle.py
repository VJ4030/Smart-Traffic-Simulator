"""Vehicle description, performance specs and the driving simulator.

The physics model is a **kinematic bicycle model with a grip limit** - the standard
approach for arcade-simulation racing games:

* the front wheel angle (``steer``) is rate-limited and shrinks with speed, which makes
  the car stable at high speed and nimble while parking,
* yaw rate ``omega = v * tan(steer) / wheelbase``,
* a lateral acceleration limit ``|v * omega| <= grip * g`` (reduced in rain / off-road)
  removes the "car turns like a top" arcade feel,
* longitudinal forces: engine curve, braking, rolling resistance and quadratic drag,
* four independent spring-damper suspension struts react to road height (speed breakers)
  and to impacts (potholes), which drives the visual pitch/roll of the body.

Everything here is pure Python + numpy so it can be unit tested without a display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

from . import settings as S
from .transform import clamp, damp, wrap_angle

# --------------------------------------------------------------------------------------
# specifications
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class VehicleSpec:
    """Static performance data of a vehicle."""

    key: str
    name: str
    category: str
    price: int
    max_speed: float          # km/h
    accel: float              # m/s^2 at low speed
    brake: float              # m/s^2
    grip: float               # lateral acceleration budget in g
    handling: float           # max steering angle in degrees
    mass: float               # kg
    fuel_efficiency: float    # km per litre
    length: float
    width: float
    height: float
    wheelbase: float
    wheel_radius: float
    tank: float = 42.0
    description: str = ""
    paint_default: str = "Sunset Orange"

    @property
    def half_length(self) -> float:
        return self.length * 0.5

    @property
    def half_width(self) -> float:
        return self.width * 0.5

    def max_speed_ms(self) -> float:
        return self.max_speed / 3.6

    def display_stats(self) -> Dict[str, float]:
        """Normalised 0..1 figures used by the garage bars."""
        return {
            "Speed": min(1.0, self.max_speed / 240.0),
            "Acceleration": min(1.0, self.accel / 7.0),
            "Braking": min(1.0, self.brake / 11.5),
            "Grip": min(1.0, self.grip / 1.3),
            "Handling": min(1.0, self.handling / 35.0),
            "Efficiency": min(1.0, self.fuel_efficiency / 20.0),
        }


#: The six player cars.  Proportions are deliberately different - a hatchback is short
#: and tall, the sports coupe is long, low and wide, the utility van is tall and boxy.
PLAYER_SPECS: Dict[str, VehicleSpec] = {
    "hatchback": VehicleSpec(
        "hatchback", "Trail Hatch 1.2", "Hatchback", price=0, max_speed=140.0, accel=4.4,
        brake=8.6, grip=1.05, handling=31.0, mass=1040.0, fuel_efficiency=18.5,
        length=3.60, width=1.62, height=1.50, wheelbase=2.36, wheel_radius=0.30,
        description="Light, cheap and easy to park. Great for learning the city.",
        paint_default="Sunset Orange"),
    "sedan": VehicleSpec(
        "sedan", "Metro Sedan 1.5V", "Sedan", price=26000, max_speed=168.0, accel=4.7,
        brake=9.2, grip=1.10, handling=29.0, mass=1245.0, fuel_efficiency=15.5,
        length=4.32, width=1.74, height=1.46, wheelbase=2.62, wheel_radius=0.32,
        description="Balanced family sedan with a comfortable ride and a big boot.",
        paint_default="Pearl White"),
    "suv": VehicleSpec(
        "suv", "Ridge SUV 4x4", "SUV", price=42000, max_speed=158.0, accel=3.5,
        brake=8.2, grip=1.02, handling=24.5, mass=1720.0, fuel_efficiency=12.0,
        length=4.52, width=1.88, height=1.78, wheelbase=2.72, wheel_radius=0.36,
        description="Heavy and sure-footed. Soaks up potholes better than anything else.",
        paint_default="Gunmetal Grey"),
    "sports": VehicleSpec(
        "sports", "Falcon GT Coupe", "Sports Car", price=98000, max_speed=238.0, accel=6.9,
        brake=11.4, grip=1.28, handling=34.0, mass=1180.0, fuel_efficiency=9.0,
        length=4.38, width=1.90, height=1.22, wheelbase=2.54, wheel_radius=0.33,
        description="Low, loud and fast. Needs a careful right foot in bazaar traffic.",
        paint_default="Racing Red"),
    "taxi": VehicleSpec(
        "taxi", "City Cab Compact", "Taxi", price=22000, max_speed=142.0, accel=4.1,
        brake=8.4, grip=1.02, handling=28.0, mass=1160.0, fuel_efficiency=16.5,
        length=4.20, width=1.68, height=1.52, wheelbase=2.54, wheel_radius=0.31,
        description="Metered city cab. Passengers love it, traffic police watch it.",
        paint_default="Banana Yellow"),
    "utility": VehicleSpec(
        "utility", "Cargo Utility Van", "Utility Vehicle", price=30000, max_speed=124.0,
        accel=3.2, brake=7.6, grip=0.94, handling=22.0, mass=1980.0, fuel_efficiency=10.5,
        length=4.68, width=1.82, height=1.98, wheelbase=2.86, wheel_radius=0.34,
        description="Delivery workhorse. Slow to accelerate, extremely stable.",
        paint_default="Sky Blue"),
}

VEHICLE_ORDER: List[str] = ["hatchback", "taxi", "sedan", "utility", "suv", "sports"]


def get_spec(key: str) -> VehicleSpec:
    return PLAYER_SPECS.get(key, PLAYER_SPECS["hatchback"])


def effective_spec(base: VehicleSpec, upgrades: Dict[str, int] | None) -> VehicleSpec:
    """Apply the garage upgrades to a spec, returning a modified copy.

    * engine:      +6% top speed, +9% acceleration per level
    * brakes:      +10% braking per level
    * tyres:       +9% grip per level
    * suspension:  (impact absorption - handled by the vehicle, not the spec)
    * steering:    +8% handling per level
    """
    if not upgrades:
        return base
    engine = int(upgrades.get("engine", 0))
    brakes = int(upgrades.get("brakes", 0))
    tyres = int(upgrades.get("tyres", 0))
    steering = int(upgrades.get("steering", 0))
    return replace(
        base,
        max_speed=base.max_speed * (1.0 + 0.06 * engine),
        accel=base.accel * (1.0 + 0.09 * engine),
        brake=base.brake * (1.0 + 0.10 * brakes),
        grip=base.grip * (1.0 + 0.09 * tyres),
        handling=base.handling * (1.0 + 0.08 * steering),
    )


def suspension_absorption(upgrades: Dict[str, int] | None) -> float:
    """Multiplier applied to impact severity (lower is better)."""
    level = int((upgrades or {}).get("suspension", 0))
    return max(0.10, 1.0 - 0.18 * level)


# --------------------------------------------------------------------------------------
# runtime state
# --------------------------------------------------------------------------------------
@dataclass
class VehicleControl:
    """Normalised control input for one simulation step."""

    throttle: float = 0.0        # 0 .. 1
    brake: float = 0.0           # 0 .. 1  (also drives reverse)
    steer: float = 0.0           # -1 .. 1  (+1 = left)
    handbrake: bool = False
    reverse: bool = False
    gear: int = 1                # 1 = drive, -1 = reverse


@dataclass
class Wheel:
    """One wheel: hierarchical child of the vehicle (position + spin + suspension)."""

    offset: Tuple[float, float, float]      # local (x forward, y up, z right)
    steerable: bool = False
    radius: float = 0.32
    spin: float = 0.0
    steer: float = 0.0                      # front wheels turn with the steering input
    compression: float = 0.0
    compression_velocity: float = 0.0
    contact: bool = True
    ground_height: float = 0.0

    def update_spin(self, speed: float, dt: float) -> None:
        self.spin = wrap_angle(self.spin + (speed / max(self.radius, 0.05)) * dt)


class Vehicle:
    """A simulated vehicle (used for the player *and* for the traffic AI)."""

    def __init__(self, spec: VehicleSpec, x: float = 0.0, z: float = 0.0,
                 yaw: float = 0.0, name: str = "vehicle") -> None:
        self.spec = spec
        self.name = name
        self.x = float(x)
        self.z = float(z)
        self.yaw = float(yaw)
        self.speed = 0.0                 # signed, m/s (+ forward)
        self.steer = 0.0                 # current front wheel angle (radians)
        self.omega = 0.0                 # yaw rate, rad/s
        self.accel_long = 0.0            # last longitudinal acceleration
        self.lateral_accel = 0.0

        # visual state (damped values used by the renderer)
        self.body_roll = 0.0
        self.body_pitch = 0.0
        self.bump_offset = 0.0
        self.shake = 0.0
        self.wheel_radius = spec.wheel_radius
        self.wheels: List[Wheel] = []
        self._build_wheels()

        # gameplay state
        self.fuel = spec.tank
        self.damage = 0.0                # 0..100
        self.headlights = False
        self.indicator = 0               # -1 left, 0 off, +1 right
        self.hazard_lights = False
        self.horn_timer = 0.0
        self.collision_cooldown = 0.0
        self.impact_flash = 0.0
        self.distance_travelled = 0.0
        self.grip_modifier = 1.0         # weather / surface
        self.surface = "road"            # road | offroad | water
        self.stuck_timer = 0.0
        self.reverse_light = False
        self.brake_light = 0.0
        self.last_impact_speed = 0.0

    # --------------------------------------------------------------------- setup
    def _build_wheels(self) -> None:
        spec = self.spec
        x_off = spec.wheelbase * 0.5
        z_off = max(spec.width * 0.5 - spec.wheel_radius * 0.65, 0.55)
        y = spec.wheel_radius
        self.wheels = [
            Wheel((x_off, y, z_off), True, spec.wheel_radius),
            Wheel((x_off, y, -z_off), True, spec.wheel_radius),
            Wheel((-x_off, y, z_off), False, spec.wheel_radius),
            Wheel((-x_off, y, -z_off), False, spec.wheel_radius),
        ]

    # ------------------------------------------------------------------- geometry
    @property
    def half_length(self) -> float:
        return self.spec.half_length

    @property
    def half_width(self) -> float:
        return self.spec.half_width

    def forward(self) -> Tuple[float, float]:
        return math.cos(self.yaw), -math.sin(self.yaw)

    def right(self) -> Tuple[float, float]:
        return math.sin(self.yaw), math.cos(self.yaw)

    def nose_position(self) -> Tuple[float, float]:
        fx, fz = self.forward()
        return self.x + fx * self.half_length, self.z + fz * self.half_length

    def tail_position(self) -> Tuple[float, float]:
        fx, fz = self.forward()
        return self.x - fx * self.half_length, self.z - fz * self.half_length

    def wheel_world_position(self, wheel: Wheel) -> Tuple[float, float]:
        fx, fz = self.forward()
        rx, rz = self.right()
        return (self.x + fx * wheel.offset[0] + rx * wheel.offset[2],
                self.z + fz * wheel.offset[0] + rz * wheel.offset[2])

    @property
    def corner_positions(self) -> List[Tuple[float, float]]:
        """Four body corners (used for OBB collision tests)."""
        fx, fz = self.forward()
        rx, rz = self.right()
        hl, hw = self.half_length, self.half_width
        out = []
        for sx in (1.0, -1.0):
            for sz in (1.0, -1.0):
                out.append((self.x + fx * hl * sx + rx * hw * sz,
                            self.z + fz * hl * sx + rz * hw * sz))
        return out

    @property
    def speed_kmh(self) -> float:
        return abs(self.speed) * 3.6

    # ---------------------------------------------------------------------- step
    def step(self, control: VehicleControl, dt: float, ground_height_fn=None,
             surface: str = "road", grip_modifier: float = 1.0) -> None:
        """Advance the simulation by ``dt`` seconds.

        ``ground_height_fn(x, z)`` returns the road height under a point (speed breakers
        raise it) and is optional so the physics can be tested in isolation.
        """
        spec = self.spec
        self.surface = surface
        self.grip_modifier = grip_modifier
        speed = self.speed

        # ---- steering: rate limited and speed dependent -------------------------
        max_angle = math.radians(spec.handling) / (1.0 + abs(speed) * 3.6 * S.STEER_SPEED_FALLOFF)
        max_angle = max(max_angle, math.radians(3.0))
        target_steer = clamp(control.steer, -1.0, 1.0) * max_angle
        steer_rate = 7.5 if abs(target_steer) > abs(self.steer) else 11.0
        self.steer = damp(self.steer, target_steer, steer_rate, dt)
        for wheel in self.wheels:
            if wheel.steerable:
                wheel.steer = self.steer

        # ---- longitudinal forces -------------------------------------------------
        prev_speed = speed
        accel = 0.0
        v_max = spec.max_speed_ms()
        if control.gear >= 0:
            throttle = clamp(control.throttle, 0.0, 1.0)
            brake = clamp(control.brake, 0.0, 1.0)
            if throttle > 0.0 and self.fuel > 0.0:
                # engine torque falls off near the top speed
                curve = max(0.0, 1.0 - (abs(speed) / max(v_max, 0.1)) ** 2.2)
                accel += spec.accel * throttle * curve * (0.65 + 0.35 * (1.0 - min(1.0, abs(speed) / 12.0)))
            if brake > 0.0:
                if speed > 0.35:
                    accel -= spec.brake * brake
                else:
                    # Holding the brake pedal at a standstill simply keeps the car still:
                    # the deceleration exactly cancels the leftover speed so the car never
                    # creeps, and never rolls backwards (reverse is the reverse gear's job,
                    # see ``control.gear < 0`` below and Player.update).
                    accel -= min(spec.brake * brake, speed / max(dt, 1e-5))
                    self.reverse_light = False
        if control.gear < 0:                            # explicit reverse gear
            demand = clamp(max(control.throttle, control.brake), 0.0, 1.0)
            if demand > 0.0:
                if speed > 0.2:
                    accel -= spec.brake * 0.6 * demand          # stop first
                elif speed > -v_max * 0.24:          # reverse is deliberately slow
                    accel -= spec.accel * 0.6 * demand          # then back up
                self.reverse_light = True
            else:
                self.reverse_light = False

        if control.handbrake:
            accel -= math.copysign(min(14.0, abs(speed) * 6.0 + 3.0), speed if speed else 1.0)

        # ---- resistance ---------------------------------------------------------------
        # rolling resistance is nearly constant (it dominates at low speed); aerodynamic
        # drag is quadratic (it dominates at high speed).  The two together set where the
        # engine's torque fall-off curve balances out, i.e. the real top speed of the car,
        # which lands within a few percent of spec.max_speed for every vehicle.
        accel -= S.ROLLING_RESISTANCE * 0.12
        accel -= (S.AIR_DRAG * 0.0012 / max(spec.mass / 1200.0, 0.6)) * speed * abs(speed)
        if surface == "offroad":
            accel -= S.OFFROAD_DRAG * math.copysign(min(1.0, abs(speed) / 6.0), speed if speed else 0.0)
            accel += S.OFFROAD_DRAG * 0.5 * control.throttle * 0.5
        if surface == "water":
            accel -= 3.0 * math.copysign(min(1.0, abs(speed) / 5.0), speed if speed else 0.0)

        self.speed = speed + accel * dt
        # clamp to the spec ceiling
        self.speed = clamp(self.speed, -v_max * 0.5, v_max)
        if abs(self.speed) < 0.06 and abs(accel) < 0.35:
            self.speed = 0.0
        self.accel_long = (self.speed - prev_speed) / max(dt, 1e-5)

        # ---- lateral / yaw (bicycle model + grip limit) ---------------------------
        if abs(self.speed) > 0.05:
            omega = self.speed * math.tan(self.steer) / max(spec.wheelbase, 1.0)
            grip = spec.grip * grip_modifier * (
                S.OFFROAD_GRIP if surface == "offroad" else 1.0)
            max_lat = grip * S.GRAVITY
            lateral = abs(self.speed * omega)
            if lateral > max_lat and abs(omega) > 1e-6:
                omega *= max_lat / lateral
            self.lateral_accel = self.speed * omega
        else:
            omega = 0.0
            self.lateral_accel = 0.0
        self.omega = omega
        self.yaw = wrap_angle(self.yaw + omega * dt)

        fx, fz = self.forward()
        dx = fx * self.speed * dt
        dz = fz * self.speed * dt
        self.x += dx
        self.z += dz
        self.distance_travelled += abs(self.speed) * dt

        # ---- fuel (efficiency aware: a 9 km/l sports car drinks twice a hatchback) ---
        efficiency_factor = 15.5 / max(spec.fuel_efficiency, 4.0)
        burn = (S.FUEL_IDLE_BURN + abs(self.speed) / 1000.0 * 3.0) * efficiency_factor * dt
        if self.fuel > 0.0:
            self.fuel = max(0.0, self.fuel - burn)

        # ---- suspension, wheels, body attitude ----------------------------------
        self._update_suspension(dt, ground_height_fn)
        for wheel in self.wheels:
            wheel.update_spin(self.speed, dt)
        target_roll = clamp(-self.lateral_accel * S.BODY_ROLL_GAIN, -0.16, 0.16)
        target_pitch = clamp(-self.accel_long * S.BODY_PITCH_GAIN, -0.09, 0.09)
        self.body_roll = damp(self.body_roll, target_roll, 7.0, dt)
        self.body_pitch = damp(self.body_pitch, target_pitch, 7.0, dt)
        self.bump_offset = damp(self.bump_offset, 0.0, 6.0, dt)
        self.shake = damp(self.shake, 0.0, 4.0, dt)
        self.impact_flash = damp(self.impact_flash, 0.0, 5.0, dt)
        self.brake_light = damp(self.brake_light, 1.0 if control.brake > 0.05 else 0.0, 12.0, dt)
        if self.horn_timer > 0.0:
            self.horn_timer = max(0.0, self.horn_timer - dt)
        if self.collision_cooldown > 0.0:
            self.collision_cooldown = max(0.0, self.collision_cooldown - dt)
        if abs(self.speed) < 0.35 and control.throttle > 0.5:
            self.stuck_timer += dt
        else:
            self.stuck_timer = 0.0

    def _update_suspension(self, dt: float, ground_height_fn) -> None:
        """Spring-damper per wheel - makes speed breakers and potholes *feel* real."""
        for wheel in self.wheels:
            wx, wz = self.wheel_world_position(wheel)
            target_height = 0.0
            if ground_height_fn is not None:
                target_height = float(ground_height_fn(wx, wz))
            wheel.ground_height = target_height
            # suspension compresses by the height of the ground under the wheel
            rest = 0.0
            displacement = rest - (target_height - 0.0)
            wheel.contact = True
            force = S.SUSPENSION_STIFFNESS * (displacement - wheel.compression) \
                - S.SUSPENSION_DAMPING * wheel.compression_velocity
            wheel.compression_velocity += force * dt
            wheel.compression += wheel.compression_velocity * dt
            wheel.compression = clamp(wheel.compression, -0.18, 0.06)
        if self.wheels:
            # the body rides on the average of the four contact heights, so the car visibly
            # climbs a speed breaker and dips into a pothole (the camera follows it too)
            average = sum(wheel.ground_height for wheel in self.wheels) / len(self.wheels)
            self.bump_offset = damp(self.bump_offset, average * 0.9, 9.0, dt)

    # ------------------------------------------------------------------ gameplay
    def apply_impact(self, severity: float, kind: str = "bump",
                     absorption: float = 1.0) -> float:
        """Apply a road impact (pothole / speed breaker / collision).

        Returns the speed loss in m/s.  ``severity`` is 0..1+.
        """
        severity = max(0.0, severity) * absorption
        if kind == "pothole":
            loss = abs(self.speed) * (1.0 - S.POTHOLE_SPEED_LOSS) * severity
            self.damage = min(100.0, self.damage + 1.6 * severity)
            self.shake = min(1.0, self.shake + 0.55 * severity)
            self.bump_offset = -0.075 * severity
        elif kind == "speed_breaker":
            loss = abs(self.speed) * 0.10 * severity
            self.shake = min(1.0, self.shake + 0.22 * severity)
            self.bump_offset = -0.05 * severity
        elif kind == "collision":
            loss = abs(self.speed) * 0.55 * min(severity, 2.0)
            self.damage = min(100.0, self.damage + 4.5 * severity)
            self.shake = min(1.0, self.shake + 0.7 * min(severity, 1.4))
            self.impact_flash = min(1.0, self.impact_flash + 0.8)
        else:
            loss = abs(self.speed) * 0.15 * severity
        self.speed = math.copysign(max(0.0, abs(self.speed) - loss), self.speed)
        for wheel in self.wheels:
            wheel.compression_velocity -= 2.2 * severity
        return loss

    def repair(self) -> None:
        self.damage = 0.0

    def refuel(self) -> None:
        self.fuel = self.spec.tank

    def fuel_fraction(self) -> float:
        return clamp(self.fuel / max(self.spec.tank, 1e-6), 0.0, 1.0)

    def efficiency_km_per_litre(self) -> float:
        eff = self.spec.fuel_efficiency
        if self.surface == "offroad":
            eff *= 0.7
        return eff

    def reset(self, x: float, z: float, yaw: float) -> None:
        self.x, self.z, self.yaw = float(x), float(z), float(yaw)
        self.speed = 0.0
        self.steer = 0.0
        self.omega = 0.0
        self.body_roll = self.body_pitch = 0.0
        self.stuck_timer = 0.0

    def unstick(self, network=None) -> None:
        """Nudge the car back onto the nearest road (used by the R key)."""
        if network is not None:
            best = None
            for r in network.roads:
                px, pz = r.nearest_point(self.x, self.z)
                d = (px - self.x) ** 2 + (pz - self.z) ** 2
                if best is None or d < best[0]:
                    best = (d, px, pz, r)
            if best is not None:
                _, px, pz, r = best
                self.x, self.z = px, pz
                self.yaw = 0.0 if r.axis == "z" else math.pi / 2
        self.speed = 0.0
        self.bump_offset = 0.0
        self.shake = 0.0
