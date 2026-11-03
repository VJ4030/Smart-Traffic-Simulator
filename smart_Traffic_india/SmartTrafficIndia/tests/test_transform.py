"""Linear algebra and the frustum used for culling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from smart_traffic_india import transform as T


def test_matmul_and_identity():
    identity = T.identity()
    assert np.allclose(identity @ identity, identity)
    a = T.translation(1.0, 2.0, 3.0)
    b = T.scaling(2.0, 2.0, 2.0)
    assert np.allclose((a @ b)[:3, 3], [1.0, 2.0, 3.0])


def test_model_matrix_places_and_rotates():
    # yaw = +90 degrees rotates the local +X axis onto world -Z (right handed, Y up)
    matrix = T.model_matrix((10.0, 0.0, -5.0), math.pi / 2, 1.0)
    point = T.transform_point(matrix, (1.0, 0.0, 0.0))
    assert abs(point[0] - 10.0) < 1e-5
    assert abs(point[2] - (-6.0)) < 1e-5
    # ... and the vehicle helper agrees with it
    dx, dz = T.dir_from_yaw(math.pi / 2)
    assert abs(dx) < 1e-6 and abs(dz + 1.0) < 1e-6
    scaled = T.model_matrix((0, 0, 0), 0.0, 3.0)
    assert abs(T.transform_point(scaled, (1.0, 0.0, 0.0))[0] - 3.0) < 1e-5


def test_perspective_and_look_at_produce_a_valid_view():
    projection = T.perspective(60.0, 16 / 9, 0.5, 400.0)
    view = T.look_at((0.0, 2.0, -10.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    view_projection = projection @ view
    # the look-at target must land in front of the camera, inside the clip volume
    clip = T.transform_point(view_projection, (0.0, 1.0, 0.0))
    assert -1.0 < clip[0] < 1.0 and -1.0 < clip[1] < 1.0
    # something behind the camera must be outside
    behind = T.transform_point(view_projection, (0.0, 1.0, -60.0))
    assert behind[2] < 0.0 or behind[2] > 1.0


def test_direction_helpers_are_consistent():
    for yaw in (0.0, 0.5, -1.2, 2.9, -3.0):
        dx, dz = T.dir_from_yaw(yaw)
        assert abs(T.yaw_from_dir(dx, dz) - yaw) < 1e-5
        left = T.left_from_dir(dx, dz)
        right = T.right_from_dir(dx, dz)
        assert abs(left[0] + right[0]) < 1e-9
        assert abs(left[1] + right[1]) < 1e-9
        # left is a 90 degree rotation of forward
        assert abs(left[0] * dx + left[1] * dz) < 1e-6


def test_angle_wrapping_and_lerp():
    assert abs(T.wrap_angle(3 * math.pi) - math.pi) < 1e-6
    assert abs(T.angle_delta(0.1, 6.2)) < 0.3
    middle = T.angle_lerp(-0.1, 0.1, 0.5)
    assert abs(middle) < 1e-6
    assert T.clamp(5.0, 0.0, 1.0) == 1.0


def test_damping_converges_without_overshoot():
    value = 0.0
    for _ in range(240):
        value = T.damp(value, 10.0, 6.0, 1 / 60)
    assert 9.5 < value <= 10.0


def test_frustum_culls_behind_the_camera():
    projection = T.perspective(60.0, 1.0, 0.5, 300.0)
    view = T.look_at((0.0, 2.0, 0.0), (0.0, 2.0, 10.0), (0.0, 1.0, 0.0))
    frustum = T.Frustum(projection @ view)
    assert frustum.sphere_visible((0.0, 2.0, 20.0), 2.0) is True
    assert frustum.sphere_visible((0.0, 2.0, -80.0), 2.0) is False
    assert frustum.box_visible((0.0, 2.0, 20.0), (2.0, 2.0, 2.0)) is True
    assert frustum.box_visible((0.0, 2.0, -60.0), (2.0, 2.0, 2.0)) is False


def test_normal_matrix_survives_non_uniform_scale():
    matrix = T.model_matrix((1.0, 2.0, 3.0), 0.7, 1.0)
    matrix = matrix @ np.diag([2.0, 1.0, 3.0, 1.0])
    normals = T.normal_matrix(matrix)
    assert normals.shape == (3, 3)
    assert np.isfinite(normals).all()
