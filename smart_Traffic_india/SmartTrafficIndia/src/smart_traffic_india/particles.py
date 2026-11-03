"""Particles: tyre smoke, water spray, dust, exhaust and impact debris.

The system is a fixed size pool of simple ballistic particles.  It never allocates during
play (the pool is preallocated and reused) and it hands its live particles to the
renderer's instanced sprite batch, so a thousand particles still cost one draw call.

Particles are *feedback only*: they never affect physics or gameplay fairness, which keeps
the "no unfair penalty" rule of the brief intact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .vehicle import Vehicle

Vec3 = Tuple[float, float, float]

#: particle flavours - the colour and behaviour differ per kind
KIND_TYRE_SMOKE = "tyre_smoke"
KIND_SPRAY = "spray"
KIND_DUST = "dust"
KIND_EXHAUST = "exhaust"
KIND_DEBRIS = "debris"
KIND_SIGNAL_GLOW = "glow"


@dataclass
class Particle:
    kind: str = KIND_DUST
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    life: float = 0.0
    max_life: float = 1.0
    size: float = 1.0
    grow: float = 0.6
    colour: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.4)
    drag: float = 1.6
    gravity: float = -1.2
    active: bool = False


class ParticleSystem:
    """Preallocated particle pool with the emitters the game needs."""

    def __init__(self, capacity: int = 320, quality: float = 1.0) -> None:
        self.capacity = max(32, int(capacity * max(0.25, quality)))
        self.pool: List[Particle] = [Particle() for _ in range(self.capacity)]
        self.cursor = 0
        self.emitted = 0
        self.live = 0
        self._exhaust_timer = 0.0
        self._seed = 0.137

    # ------------------------------------------------------------------ emitting
    def _rng(self) -> float:
        """Cheap deterministic noise in 0..1 (avoids a random call per particle)."""
        self._seed = (self._seed * 1103515245.0 + 12345.0) % 2147483648.0
        return self._seed / 2147483648.0

    def _take(self) -> Particle:
        particle = self.pool[self.cursor]
        self.cursor = (self.cursor + 1) % self.capacity
        if not particle.active:
            self.live += 1
        particle.active = True
        self.emitted += 1
        return particle

    def emit(self, kind: str, position: Vec3, velocity: Vec3, life: float, size: float,
             colour: Tuple[float, float, float, float], grow: float = 0.6,
             drag: float = 1.6, gravity: float = -1.2) -> None:
        particle = self._take()
        particle.kind = kind
        particle.x, particle.y, particle.z = position
        particle.vx, particle.vy, particle.vz = velocity
        particle.life = particle.max_life = max(0.05, life)
        particle.size = size
        particle.grow = grow
        particle.colour = colour
        particle.drag = drag
        particle.gravity = gravity

    # -- specific emitters ----------------------------------------------------------
    def tyre_smoke(self, vehicle: Vehicle, intensity: float = 1.0) -> None:
        """Smoke when the tyres are sliding (handbrake, hard braking, wheelspin)."""
        count = max(1, int(3 * intensity))
        fx, fz = vehicle.forward()
        rx, rz = vehicle.right()
        for i in range(count):
            side = 1.0 if i % 2 == 0 else -1.0
            jitter = self._rng() - 0.5
            self.emit(KIND_TYRE_SMOKE,
                      (vehicle.x - fx * vehicle.half_length * 0.7 + rx * side * 0.7,
                       0.35, vehicle.z - fz * vehicle.half_length * 0.7 + rz * side * 0.7),
                      (-fx * (1.5 + self._rng()) + rx * jitter * 1.4, 0.7 + self._rng() * 0.6,
                       -fz * (1.5 + self._rng()) + rz * jitter * 1.4),
                      0.9 + self._rng() * 0.7, 0.55 + self._rng() * 0.5,
                      (0.86, 0.86, 0.88, 0.34), grow=1.4, drag=1.2, gravity=0.15)

    def spray(self, vehicle: Vehicle, intensity: float = 1.0) -> None:
        """Water thrown up by the wheels when driving through rain or water logging."""
        count = max(1, int(4 * intensity))
        fx, fz = vehicle.forward()
        rx, rz = vehicle.right()
        for i in range(count):
            side = 1.0 if i % 2 == 0 else -1.0
            jitter = self._rng() - 0.5
            self.emit(KIND_SPRAY,
                      (vehicle.x - fx * vehicle.half_length * 0.6 + rx * side * 0.85,
                       0.25, vehicle.z - fz * vehicle.half_length * 0.6 + rz * side * 0.85),
                      (-fx * 2.2 + rx * jitter * 2.4, 1.5 + self._rng(), -fz * 2.2 + rz * jitter * 2.4),
                      0.45 + self._rng() * 0.3, 0.34, (0.72, 0.80, 0.92, 0.30),
                      grow=1.1, drag=2.4, gravity=-6.0)

    def dust(self, position: Vec3, strength: float = 1.0) -> None:
        for _ in range(2):
            self.emit(KIND_DUST,
                      (position[0] + (self._rng() - 0.5) * 0.8, position[1] + 0.15,
                       position[2] + (self._rng() - 0.5) * 0.8),
                      ((self._rng() - 0.5) * 1.6, 0.5 + self._rng(), (self._rng() - 0.5) * 1.6),
                      0.7 + self._rng() * 0.6, 0.4 * strength, (0.62, 0.54, 0.40, 0.28),
                      grow=1.5, drag=1.4, gravity=-0.5)

    def debris(self, position: Vec3, severity: float = 1.0) -> None:
        """Plastic / glass shards from an impact."""
        count = max(2, int(6 * severity))
        for _ in range(count):
            self.emit(KIND_DEBRIS,
                      position,
                      ((self._rng() - 0.5) * 6.0, 2.0 + self._rng() * 3.0,
                       (self._rng() - 0.5) * 6.0),
                      0.6 + self._rng() * 0.5, 0.12 + self._rng() * 0.09,
                      (0.85, 0.84, 0.80, 0.9), grow=0.0, drag=0.6, gravity=-9.0)

    def exhaust(self, vehicle: Vehicle, load: float) -> None:
        fx, fz = vehicle.forward()
        self.emit(KIND_EXHAUST,
                  (vehicle.x - fx * (vehicle.half_length + 0.25), 0.28,
                   vehicle.z - fz * (vehicle.half_length + 0.25)),
                  (-fx * 1.1, 0.35, -fz * 1.1),
                  0.5 + 0.4 * load, 0.22, (0.45, 0.45, 0.47, 0.22),
                  grow=1.3, drag=2.0, gravity=0.1)

    # ------------------------------------------------------------------ update
    def update(self, dt: float) -> None:
        live = 0
        for particle in self.pool:
            if not particle.active:
                continue
            particle.life -= dt
            if particle.life <= 0.0:
                particle.active = False
                continue
            damp = max(0.0, 1.0 - particle.drag * dt)
            particle.vx *= damp
            particle.vz *= damp
            particle.vy = particle.vy * damp + particle.gravity * dt
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            particle.z += particle.vz * dt
            if particle.y < 0.06:
                particle.y = 0.06
                particle.vy = 0.0
            particle.size += particle.grow * dt
            live += 1
        self.live = live

    def vehicle_feedback(self, vehicle: Vehicle, dt: float, wet: bool, offroad: bool,
                         locked_brake: bool) -> None:
        """Turn the vehicle's own state into particles (called once per frame)."""
        speed = abs(vehicle.speed)
        if wet and speed > 4.0:
            self.spray(vehicle, min(1.6, speed / 18.0))
        if offroad and speed > 3.0:
            self.dust((vehicle.x, 0.1, vehicle.z), min(1.4, speed / 12.0))
        slip = abs(vehicle.lateral_accel) * 0.12 + (0.5 if locked_brake else 0.0)
        if slip > 0.45 and speed > 3.0:
            self.tyre_smoke(vehicle, min(1.5, slip))
        self._exhaust_timer -= dt
        if self._exhaust_timer <= 0.0:
            self._exhaust_timer = 0.09
            if vehicle.speed > -0.5:
                self.exhaust(vehicle, min(1.0, abs(vehicle.accel_long) / 4.0))

    # ------------------------------------------------------------------ render
    def push_to(self, renderer, camera_position: Optional[Tuple[float, float, float]] = None,
                max_distance: float = 55.0) -> int:
        """Push every live particle into the renderer's instanced sprite batch."""
        pushed = 0
        for particle in self.pool:
            if not particle.active:
                continue
            if camera_position is not None:
                distance = math.hypot(particle.x - camera_position[0],
                                      particle.z - camera_position[2])
                if distance > max_distance:
                    continue
            fraction = max(0.0, min(1.0, particle.life / max(particle.max_life, 1e-4)))
            fade = fraction ** 0.7
            colour = (particle.colour[0], particle.colour[1], particle.colour[2],
                      particle.colour[3] * fade)
            size = max(0.05, particle.size) * (0.6 + 0.4 * (1.0 - fraction))
            renderer.push_sprite((particle.x, particle.y, particle.z), (size, size), colour)
            pushed += 1
        return pushed

    def statistics(self) -> dict:
        return {"capacity": self.capacity, "live": self.live, "emitted": self.emitted}

    def clear(self) -> None:
        for particle in self.pool:
            particle.active = False
        self.live = 0
