# Copyright (c) 2026 Robert Bosch GmbH
# Author: Roman Freiberg
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import jax
import jax.numpy as jnp


def theta_norm_to_denorm(
    theta_norm: jnp.ndarray, min_ranges: jnp.ndarray, max_ranges: jnp.ndarray
):
    theta_denorm = (theta_norm * 0.5 + 0.5) * (max_ranges - min_ranges) - min_ranges
    return theta_denorm


def theta_denorm_to_norm(
    theta_denorm: jnp.ndarray, min_ranges: jnp.ndarray, max_ranges: jnp.ndarray
):
    theta_norm = (theta_denorm + min_ranges) / (max_ranges - min_ranges) * 2 - 1
    return theta_norm


def quaternion_raw_multiply_jax(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    aw, ax, ay, az = jnp.unstack(a, axis=-1)
    bw, bx, by, bz = jnp.unstack(b, axis=-1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return jnp.stack((ow, ox, oy, oz), -1)


def quaternion_invert_jax(quaternion: jnp.ndarray) -> jnp.ndarray:
    scaling = jnp.array([1, -1, -1, -1])
    return quaternion * scaling


def axis_angle_to_quaternion_jax(axis_angle: jnp.ndarray) -> jnp.ndarray:
    angles = jnp.linalg.norm(axis_angle, axis=-1, keepdims=True)  # Shape (..., 1)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = jnp.abs(angles) < eps  # Shape (..., 1), bool

    angles_squeezed = angles.squeeze(-1)  # Shape (...)
    half_angles_squeezed = half_angles.squeeze(-1)  # Shape (...)
    small_angles_squeezed = small_angles.squeeze(-1)  # Shape (...)

    # Compute sin_half_angles_over_angles avoiding division by zero
    sin_half_angles_over_angles = jnp.where(
        small_angles_squeezed,
        0.5 - (angles_squeezed**2) / 48,
        jnp.sin(half_angles_squeezed) / angles_squeezed,
    )  # Shape (...)

    # Expand back to shape (..., 1)
    sin_half_angles_over_angles = sin_half_angles_over_angles[..., None]

    quaternions = jnp.concatenate(
        [jnp.cos(half_angles), axis_angle * sin_half_angles_over_angles], axis=-1
    )
    return quaternions


def quaternion_apply_jax(quaternion: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
    if point.shape[-1] != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = jnp.zeros(point.shape[:-1] + (1,))
    point_as_quaternion = jnp.concatenate((real_parts, point), -1)
    out = quaternion_raw_multiply_jax(
        quaternion_raw_multiply_jax(quaternion, point_as_quaternion),
        quaternion_invert_jax(quaternion),
    )
    return out[..., 1:]


def transform_points_jax(points: jnp.ndarray, Ts: jnp.ndarray) -> jnp.ndarray:
    q, t = Ts[..., :4], Ts[..., 4:]
    points = quaternion_apply_jax(q, points) + t
    return points


def se3_raw_mupltiply(Ts_one: jnp.ndarray, Ts_two: jnp.ndarray) -> jnp.ndarray:
    q_one, _ = Ts_one[..., :4], Ts_one[..., 4:]
    q_two, t_two = Ts_two[..., :4], Ts_two[..., 4:]
    q = quaternion_raw_multiply_jax(q_one, q_two)
    t = transform_points_jax(t_two, Ts_one)
    return jnp.concatenate([q, t], axis=-1)


def se3_invert(Ts: jnp.ndarray) -> jnp.ndarray:
    q, t = Ts[..., :4], Ts[..., 4:]
    q_inv = quaternion_invert_jax(q)
    t = -quaternion_apply_jax(q_inv, t)
    return jnp.concatenate([q_inv, t], axis=-1)


def similarity_transform(Ts_sim: jnp.ndarray, Ts: jnp.ndarray) -> jnp.ndarray:
    return se3_raw_mupltiply(se3_raw_mupltiply(Ts_sim, Ts), se3_invert(Ts_sim))


def quaternion_from_axis_angle(axis: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    axis = axis / jnp.linalg.norm(axis)
    half_angle = angle / 2.0
    s = jnp.sin(half_angle)
    q = jnp.array([jnp.cos(half_angle), axis[0] * s, axis[1] * s, axis[2] * s])
    return q


def rot_mat_from_axis_angle(axis: jnp.ndarray, angle: jnp.ndarray):
    axis = axis / jnp.linalg.norm(axis)
    s = jnp.sin(angle)
    c = jnp.cos(angle)
    R = jnp.diag(jnp.array([c, c, c]))
    R = R + jnp.outer(axis, axis) * (1.0 - c)
    axis = axis * s
    R = R + jnp.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return R


def quat_to_rot_mat(q: jnp.ndarray) -> jnp.ndarray:
    q = q / jnp.linalg.norm(q)
    w, x, y, z = q[0], q[1], q[2], q[3]
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    R = jnp.array(
        [
            [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz],
        ]
    )
    return R


def rot_pos_to_se3(rot: jnp.ndarray, pos: jnp.ndarray) -> jnp.ndarray:
    se3 = jnp.eye(4)
    se3 = se3.at[:3, :3].set(rot)
    se3 = se3.at[:3, 3].set(pos)
    return se3


def quat_to_se3(T: jnp.ndarray) -> jnp.ndarray:
    R = quat_to_rot_mat(T[:4])
    t = T[4:]
    se3 = rot_pos_to_se3(R, t)
    return se3


def rotate_vec_by_twist(w, t, eps=1e-8):
    theta = jnp.linalg.norm(w, axis=-1, keepdims=True)
    n = w / jnp.where(theta < eps, 1.0, theta)

    sinθ = jnp.sin(theta)
    cosθ = jnp.cos(theta)
    one_c = 1.0 - cosθ  # (1 - cosθ)

    n_cross_t = jnp.cross(n, t)
    n_dot_t = (n * t).sum(axis=-1, keepdims=True)

    return t * cosθ + n_cross_t * sinθ + n * n_dot_t * one_c


def rodrigues(k_unit, theta):
    K = jnp.array(
        [
            [0.0, -k_unit[2], k_unit[1]],
            [k_unit[2], 0.0, -k_unit[0]],
            [-k_unit[1], k_unit[0], 0.0],
        ]
    )
    I = jnp.eye(3, dtype=k_unit.dtype)
    return I + jnp.sin(theta) * K + (1.0 - jnp.cos(theta)) * K @ K


def rotate_frame_det(p, q):
    eps_norm = 1e-12
    eps_parallel = 1e-6
    eps_antiparallel = 1e-3
    v1 = jnp.asarray(p, dtype=jnp.float32)
    v2 = jnp.asarray(q, dtype=jnp.float32)

    n1 = jnp.linalg.norm(v1)
    n2 = jnp.linalg.norm(v2)

    # If either vector is (almost) zero-length, abort to identity.
    zero_case = (n1 < eps_norm) | (n2 < eps_norm)

    # Replace tiny norms with 1 so division is always safe.
    n1_safe = jnp.where(n1 < eps_norm, 1.0, n1)
    n2_safe = jnp.where(n2 < eps_norm, 1.0, n2)

    v1_u = v1 / n1_safe
    v2_u = v2 / n2_safe

    cos_theta = jnp.clip(jnp.dot(v1_u, v2_u), -1.0, 1.0)
    theta = jnp.arccos(cos_theta)

    k = jnp.cross(v1_u, v2_u)
    k_norm = jnp.linalg.norm(k)

    k_unit_general = jnp.where(
        k_norm < eps_norm, jnp.array([1.0, 0.0, 0.0]), k / k_norm
    )

    def identity_case():
        return jnp.eye(3, dtype=v1.dtype)

    def antiparallel_case():
        # Pick a reproducible axis ⟂ v1
        ref1 = jnp.array([1.0, 0.0, 0.0])
        ref2 = jnp.array([0.0, 1.0, 0.0])
        axis = jnp.where(jnp.abs(v1_u) < 0.9, ref1, ref2)
        k2 = jnp.cross(v1_u, axis)
        k2_norm = jnp.linalg.norm(k2)
        k2_unit = jnp.where(
            k2_norm < eps_norm, jnp.array([0.0, 0.0, 1.0]), k2 / k2_norm
        )
        return rodrigues(k2_unit, jnp.pi)

    def general_case():
        return rodrigues(k_unit_general, theta)

    R = jax.lax.cond(
        zero_case,
        identity_case,  # any zero-length input
        lambda: jax.lax.cond(
            theta < eps_parallel,
            identity_case,  # almost identical
            lambda: jax.lax.cond(
                jnp.abs(jnp.pi - theta) < eps_antiparallel,
                antiparallel_case,  # almost opposite
                general_case,  # generic rotation
            ),
        ),
    )

    return R


def frame_to_y(v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    eps_zero: float = 1e-12
    eps_anti: float = 1e-6

    v = jnp.asarray(v, dtype=jnp.float32)
    r = jnp.linalg.norm(v)

    # ---------- zero-length → identity ----------
    def zero_case():
        return jnp.eye(3, dtype=v.dtype)

    def nonzero_case():
        vhat = v / r
        e_y = jnp.array([0.0, 1.0, 0.0], dtype=v.dtype)

        cos_theta = jnp.dot(vhat, e_y)
        sin2 = 1.0 - cos_theta**2

        # ---------- parallel (θ≈0) & antiparallel (θ≈π) ----------
        def parallel_case():
            return jnp.eye(3, dtype=v.dtype)

        def antiparallel_case():
            # reproducible axis ⟂ e_y
            ref = jnp.array([1.0, 0.0, 0.0], dtype=v.dtype)
            axis = jnp.where(
                jnp.abs(vhat[0]) < 0.9, ref, jnp.array([0.0, 0.0, 1.0], dtype=v.dtype)
            )
            axis = axis / jnp.linalg.norm(axis)
            vx, vy, vz = axis
            K = jnp.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]], dtype=v.dtype)
            R = jnp.eye(3, dtype=v.dtype) + 2.0 * (K @ K)  # θ = π
            return R

        # ---------- general Rodrigues case ----------
        def general_case():
            axis = jnp.cross(vhat, e_y)
            axis = axis / jnp.linalg.norm(axis)
            vx, vy, vz = axis
            K = jnp.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]], dtype=v.dtype)
            theta = jnp.arctan2(jnp.sqrt(sin2), cos_theta)
            R = (
                jnp.eye(3, dtype=v.dtype)
                + jnp.sin(theta) * K
                + (1.0 - jnp.cos(theta)) * (K @ K)
            )
            return R

        # choose among three cases
        return jax.lax.cond(
            sin2 < eps_anti,
            lambda: jax.lax.cond(cos_theta > 0.0, parallel_case, antiparallel_case),
            general_case,
        )

    return jax.lax.cond(r < eps_zero, zero_case, nonzero_case)
