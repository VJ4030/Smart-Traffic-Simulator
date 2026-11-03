"""Cameras: third-person chase, close chase, hood, plus orbit and cinematic modes.

Concepts demonstrated: camera movement, view transformation (``look_at``), perspective
projection, smooth interpolation over time, and damping to avoid jitter.

Design rules that keep the driving readable:
* the camera always sits *behind* the vehicle and above it, so the road ahead is visible,
* all motion is critically damped - no snapping when the car turns or hits a pothole,
* the field of view widens a little with speed for a sense of velocity,
* the camera is pushed up when it would clip into a building or the road surface,
* a small amount of impulse shake is added on impacts (decaying quickly).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from .transform import (Frustum, Mat4, damp, look_at, perspective, wrap_angle)
from .vehicle import Vehicle

Vec3 = Tuple[float, float, float]

CHASE = "chase"
CLOSE = "close"
HOOD = "hood"
ORBIT = "orbit"

MODE_ORDER = [CHASE, CLOSE, HOOD]

PRESETS = {
    # a generous chase distance keeps the whole car and the road ahead in frame
    CHASE: dict(distance=10.8, height=4.6, look_height=1.45, fov=60.0, stiffness=5.4, lookahead=8.5),
    CLOSE: dict(distance=6.6, height=2.99, look_height=1.15, fov=66.0, stiffness=8.0, lookahead=5.5),
    HOOD:  dict(distance=-0.35, height=1.24, look_height=1.02, fov=74.0, stiffness=22.0, lookahead=14.0),
}


class Camera:
    """A damped follow camera with several modes."""

    def __init__(self, aspect: float = 16 / 9, mode: str = CHASE) -> None:
        self.mode = mode if mode in PRESETS else CHASE
        self.aspect = float(aspect)
        self.near = 0.6
        self.far = 420.0
        self.position = np.array([0.0, 4.0, -10.0], dtype=np.float32)
        self.target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._yaw = 0.0
        self._distance_scale = 1.0
        self._height_offset = 0.0
        self.fov = PRESETS[self.mode]["fov"]
        self.shake = 0.0
        self._shake_seed = 0.0
        self.view = np.identity(4, dtype=np.float32)
        self.projection = perspective(self.fov, self.aspect, self.near, self.far)
        self.view_projection = np.identity(4, dtype=np.float32)
        self.orbit_angle = 0.0
        self.cinematic = False
        self.collision_hint: Optional[Sequence[Tuple[float, float, float, float, float]]] = None

    # ------------------------------------------------------------------ helpers
    def set_aspect(self, aspect: float) -> None:
        self.aspect = max(aspect, 0.2)
        self.projection = perspective(self.fov, self.aspect, self.near, self.far)

    def set_far(self, far: float) -> None:
        self.far = float(far)
        self.projection = perspective(self.fov, self.aspect, self.near, self.far)

    def cycle_mode(self, forward: bool = True) -> str:
        index = MODE_ORDER.index(self.mode) if self.mode in MODE_ORDER else 0
        index = (index + (1 if forward else -1)) % len(MODE_ORDER)
        self.mode = MODE_ORDER[index]
        return self.mode

    def look_behind(self, vehicle: Vehicle, amount: float = 1.0) -> None:
        """Temporarily swing the camera to look backwards (the B key)."""
        self._look_behind = amount

    # ------------------------------------------------------------------ update
    def update(self, dt: float, vehicle: Vehicle, terrain_height: float = 0.0,
               reverse_view: bool = False) -> None:
        preset = PRESETS.get(self.mode if self.mode in PRESETS else CHASE)
        speed = abs(vehicle.speed)
        speed_frac = min(1.0, speed / max(vehicle.spec.max_speed_ms(), 1.0))

        if self.cinematic:
            self._update_cinematic(dt, vehicle)
            return

        heading = vehicle.yaw
        if reverse_view:
            heading = wrap_angle(heading + math.pi)
        distance = preset["distance"] + speed * 0.07 * (1.0 - 0.4 * speed_frac)
        height = preset["height"] + speed * 0.012 + self._height_offset
        if self.mode == HOOD:
            # hood camera: rigid, mounted on the bonnet
            fx, fz = vehicle.forward()
            desired = np.array([vehicle.x + fx * 0.55, 1.24 + vehicle.bump_offset * 0.5,
                                vehicle.z + fz * 0.55], dtype=np.float32)
            self.position = desired
            self.target = np.array([vehicle.x + fx * (0.55 + preset["lookahead"]),
                                    1.05 + vehicle.bump_offset * 0.4,
                                    vehicle.z + fz * (0.55 + preset["lookahead"])],
                                   dtype=np.float32)
        else:
            fx = math.cos(heading)
            fz = -math.sin(heading)
            # look slightly ahead of the car so junctions are visible early
            lead = preset["lookahead"] * (0.35 + 0.65 * speed_frac)
            anchor_x = vehicle.x + fx * lead
            anchor_z = vehicle.z + fz * lead
            desired = np.array([anchor_x - fx * distance, height, anchor_z - fz * distance],
                               dtype=np.float32)
            target = np.array([anchor_x, preset["look_height"] + vehicle.bump_offset,
                               anchor_z], dtype=np.float32)
            desired[1] = max(desired[1], terrain_height + 0.85)
            rate = preset["stiffness"]
            self.position = damp_vec_f32(self.position, desired, rate, dt)
            self.target = damp_vec_f32(self.target, target, rate * 1.35, dt)

        # ---- collision avoidance: keep the camera out of buildings and above ground ---
        if self.collision_hint:
            px, py, pz = float(self.position[0]), float(self.position[1]), float(self.position[2])
            for bx, bz, hx, hz, height_b in self.collision_hint:
                if (abs(px - bx) < hx + 0.6 and abs(pz - bz) < hz + 0.6 and py < height_b):
                    self.position[1] = float(height_b + 1.0)
                    break
        self.position[1] = max(self.position[1], terrain_height + 0.8)

        # ---- shake -------------------------------------------------------------------
        self.shake = max(vehicle.shake, self.shake * math.exp(-6.0 * dt))
        if self.shake > 0.001:
            self._shake_seed += dt * 40.0
            amp = 0.16 * self.shake
            self.position[0] += math.sin(self._shake_seed * 1.7) * amp
            self.position[1] += math.sin(self._shake_seed * 2.3) * amp * 0.7
            self.position[2] += math.cos(self._shake_seed * 1.3) * amp

        # ---- fov: widen with speed, spring back at rest --------------------------------
        target_fov = preset["fov"] + 10.0 * speed_frac
        self.fov = damp(self.fov, target_fov, 2.5, dt)
        self.projection = perspective(self.fov, self.aspect, self.near, self.far)
        self._update_matrices()

    def _update_cinematic(self, dt: float, vehicle: Vehicle) -> None:
        self.orbit_angle += dt * 0.16
        radius = 26.0
        self.position = np.array([vehicle.x + math.cos(self.orbit_angle) * radius, 11.0,
                                  vehicle.z + math.sin(self.orbit_angle) * radius],
                                 dtype=np.float32)
        self.target = np.array([vehicle.x, 1.5, vehicle.z], dtype=np.float32)
        self.fov = damp(self.fov, 46.0, 2.0, dt)
        self.projection = perspective(self.fov, self.aspect, self.near, self.far)
        self._update_matrices()

    def _update_matrices(self) -> None:
        self.view = look_at(self.position, self.target, self.up)
        self.view_projection = (self.projection @ self.view).astype(np.float32)

    # ------------------------------------------------------------------ outputs
    def frustum(self) -> Frustum:
        return Frustum(self.view_projection)

    def right_vector(self) -> Vec3:
        """Camera right/up vectors, used by the billboard shader."""
        forward = self.target - self.position
        forward = forward / max(float(np.linalg.norm(forward)), 1e-6)
        right = np.cross(forward, self.up)
        right = right / max(float(np.linalg.norm(right)), 1e-6)
        return (float(right[0]), float(right[1]), float(right[2]))

    def up_vector(self) -> Vec3:
        forward = self.target - self.position
        forward = forward / max(float(np.linalg.norm(forward)), 1e-6)
        right = np.cross(forward, self.up)
        right = right / max(float(np.linalg.norm(right)), 1e-6)
        true_up = np.cross(right, forward)
        return (float(true_up[0]), float(true_up[1]), float(true_up[2]))

    def inverse_view_projection(self) -> Mat4:
        try:
            return np.linalg.inv(self.view_projection.astype(np.float64)).astype(np.float32)
        except np.linalg.LinAlgError:                       # pragma: no cover
            return np.identity(4, dtype=np.float32)


def damp_vec_f32(current: np.ndarray, target: np.ndarray, rate: float,
                 dt: float) -> np.ndarray:
    factor = math.exp(-rate * dt)
    return (target + (current - target) * factor).astype(np.float32)
