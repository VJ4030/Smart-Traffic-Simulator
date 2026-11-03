"""3D maths for Smart Traffic India.

Concepts demonstrated here (see docs/COMPUTER_GRAPHICS.md):
  * 3D coordinate system (right-handed, +Y up, ground plane = XZ)
  * translation / rotation / scaling matrices and how they compose
  * model, view and projection transformations
  * perspective projection matrix construction
  * surface normals and the normal matrix
  * vector maths used by the vehicle simulator (forward/right basis vectors)

All matrices are 4x4 numpy arrays in **row-major** order, i.e. ``M @ v`` transforms the
column vector ``v``.  When uploading to OpenGL we pass ``transpose=GL_TRUE`` because GL
expects column-major storage.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

Mat4 = np.ndarray
Vec3 = np.ndarray
F32 = np.float32

__all__ = [
    "identity", "translation", "scaling", "rotation_x", "rotation_y", "rotation_z",
    "perspective", "ortho", "look_at", "transform_point", "transform_direction",
    "normal_matrix", "matmul", "model_matrix", "yaw_from_dir", "dir_from_yaw",
    "left_from_dir", "right_from_dir", "normalize", "length", "lerp", "clamp",
    "angle_lerp", "wrap_angle", "angle_delta", "damp", "vec", "Frustum",
]


# --------------------------------------------------------------------------------------
# basic builders
# --------------------------------------------------------------------------------------
def identity() -> Mat4:
    return np.identity(4, dtype=F32)


def translation(x: float, y: float, z: float) -> Mat4:
    m = np.identity(4, dtype=F32)
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m


def scaling(x: float, y: float | None = None, z: float | None = None) -> Mat4:
    if y is None:
        y = z = x
    elif z is None:
        z = x
    m = np.identity(4, dtype=F32)
    m[0, 0], m[1, 1], m[2, 2] = x, y, z
    return m


def rotation_x(a: float) -> Mat4:
    c, s = math.cos(a), math.sin(a)
    m = np.identity(4, dtype=F32)
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    return m


def rotation_y(a: float) -> Mat4:
    """Yaw rotation.  Maps local +X to ``(cos a, 0, -sin a)``."""
    c, s = math.cos(a), math.sin(a)
    m = np.identity(4, dtype=F32)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def rotation_z(a: float) -> Mat4:
    c, s = math.cos(a), math.sin(a)
    m = np.identity(4, dtype=F32)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def matmul(*mats: Mat4) -> Mat4:
    """Multiply matrices left-to-right (``matmul(A, B) == A @ B``)."""
    out = mats[0]
    for m in mats[1:]:
        out = out @ m
    return out.astype(F32, copy=False)


def model_matrix(position: Sequence[float], yaw: float = 0.0, scale: float | float = 1.0,
                 pitch: float = 0.0, roll: float = 0.0) -> Mat4:
    """Compose a model matrix: translate * yaw * pitch * roll * uniform scale."""
    if isinstance(scale, (int, float)):
        scale_m = scaling(float(scale))
    else:  # pragma: no cover - convenience only
        scale_m = scaling(*scale)
    m = translation(position[0], position[1], position[2])
    m = matmul(m, rotation_y(yaw))
    if pitch:
        m = matmul(m, rotation_z(pitch))
    if roll:
        m = matmul(m, rotation_x(roll))
    return matmul(m, scale_m)


# --------------------------------------------------------------------------------------
# projections
# --------------------------------------------------------------------------------------
def perspective(fov_y_degrees: float, aspect: float, near: float, far: float) -> Mat4:
    """Classic OpenGL perspective projection matrix (NDC z in [-1, 1])."""
    f = 1.0 / math.tan(math.radians(fov_y_degrees) * 0.5)
    m = np.zeros((4, 4), dtype=F32)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def ortho(left: float, right: float, bottom: float, top: float,
          near: float = -1.0, far: float = 1.0) -> Mat4:
    m = np.identity(4, dtype=F32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def look_at(eye: Sequence[float], target: Sequence[float],
            up: Sequence[float] = (0.0, 1.0, 0.0)) -> Mat4:
    """View matrix (world -> camera space)."""
    eye = np.asarray(eye, dtype=F32)
    target = np.asarray(target, dtype=F32)
    up = np.asarray(up, dtype=F32)
    forward = target - eye
    forward = forward / max(float(np.linalg.norm(forward)), 1e-8)
    side = np.cross(forward, up)
    side = side / max(float(np.linalg.norm(side)), 1e-8)
    true_up = np.cross(side, forward)
    m = np.identity(4, dtype=F32)
    m[0, :3] = side
    m[1, :3] = true_up
    m[2, :3] = -forward
    m[0, 3] = -float(np.dot(side, eye))
    m[1, 3] = -float(np.dot(true_up, eye))
    m[2, 3] = float(np.dot(forward, eye))
    return m


def transform_point(m: Mat4, point: Sequence[float]) -> Vec3:
    v = np.array([point[0], point[1], point[2], 1.0], dtype=F32)
    return (m @ v)[:3]


def transform_direction(m: Mat4, direction: Sequence[float]) -> Vec3:
    v = np.array([direction[0], direction[1], direction[2], 0.0], dtype=F32)
    return (m @ v)[:3]


def normal_matrix(m: Mat4) -> np.ndarray:
    """Inverse-transpose of the upper-left 3x3 block (keeps normals correct under
    non-uniform scaling / rotation)."""
    upper = np.asarray(m[:3, :3], dtype=np.float64)
    try:
        inv = np.linalg.inv(upper)
    except np.linalg.LinAlgError:              # pragma: no cover - degenerate matrix
        inv = np.identity(3)
    return inv.T.astype(F32)


# --------------------------------------------------------------------------------------
# vehicle / world helpers
# --------------------------------------------------------------------------------------
def yaw_from_dir(dx: float, dz: float) -> float:
    """Convert a ground direction to the yaw used by :func:`rotation_y`."""
    return math.atan2(-dz, dx)


def dir_from_yaw(yaw: float) -> Tuple[float, float]:
    """Ground-plane forward vector ``(dx, dz)`` for a yaw angle."""
    return (math.cos(yaw), -math.sin(yaw))


def left_from_dir(dx: float, dz: float) -> Tuple[float, float]:
    """Left-hand side vector of a ground direction (up x forward)."""
    return (dz, -dx)


def right_from_dir(dx: float, dz: float) -> Tuple[float, float]:
    return (-dz, dx)


def normalize(v: Sequence[float]) -> Vec3:
    arr = np.asarray(v, dtype=F32)
    n = float(np.linalg.norm(arr))
    return arr / n if n > 1e-8 else arr


def length(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=F32)))


def vec(x: float, y: float, z: float = 0.0) -> Vec3:
    return np.array([x, y, z], dtype=F32)


# --------------------------------------------------------------------------------------
# interpolation / angles
# --------------------------------------------------------------------------------------
def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


def wrap_angle(a: float) -> float:
    """Wrap an angle into (-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


def angle_delta(a: float, b: float) -> float:
    return wrap_angle(b - a)


def angle_lerp(a: float, b: float, t: float) -> float:
    return wrap_angle(a + angle_delta(a, b) * clamp(t, 0.0, 1.0))


def damp(current: float, target: float, rate: float, dt: float) -> float:
    """Frame-rate independent exponential approach (critically damped feel)."""
    return target + (current - target) * math.exp(-rate * dt)


def damp_vec(current: np.ndarray, target: np.ndarray, rate: float, dt: float) -> np.ndarray:
    return target + (current - target) * math.exp(-rate * dt)


# --------------------------------------------------------------------------------------
# frustum culling
# --------------------------------------------------------------------------------------
class Frustum:
    """Six frustum planes extracted from a view-projection matrix.

    Used for view-frustum culling so that off-screen buildings and vehicles are skipped
    before they reach the GPU (a core "efficient scene organisation" technique).
    """

    def __init__(self, view_projection: Mat4) -> None:
        m = np.asarray(view_projection, dtype=np.float64)
        planes = np.empty((6, 4), dtype=np.float64)
        # rows of the clip-space matrix: left, right, bottom, top, near, far
        planes[0] = m[3] + m[0]
        planes[1] = m[3] - m[0]
        planes[2] = m[3] + m[1]
        planes[3] = m[3] - m[1]
        planes[4] = m[3] + m[2]
        planes[5] = m[3] - m[2]
        norms = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        self.planes = (planes / norms).astype(F32)

    def sphere_visible(self, center: Sequence[float], radius: float) -> bool:
        c = np.array([center[0], center[1], center[2]], dtype=F32)
        d = self.planes[:, :3] @ c + self.planes[:, 3]
        return bool(np.all(d >= -radius))

    def box_visible(self, center: Sequence[float], half: Sequence[float]) -> bool:
        c = np.array([center[0], center[1], center[2]], dtype=F32)
        h = np.array([abs(half[0]), abs(half[1]), abs(half[2])], dtype=F32)
        d = self.planes[:, :3] @ c + self.planes[:, 3]
        r = np.abs(self.planes[:, :3]) @ h
        return bool(np.all(d >= -r))
