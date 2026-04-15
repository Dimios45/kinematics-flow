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
from jax.lax import fori_loop


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
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.
    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.
    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = jnp.unstack(a, axis=-1)
    bw, bx, by, bz = jnp.unstack(b, axis=-1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return jnp.stack((ow, ox, oy, oz), -1)


def quaternion_invert_jax(quaternion: jnp.ndarray) -> jnp.ndarray:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.
    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).
    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

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
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.
    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).
    Returns:
        Tensor of rotated points of shape (..., 3).
    """
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


def rotation_translation_vector_field(
    gripper_pcd: jnp.ndarray, segmentation: jnp.ndarray, gripper_type, dofs, dof_type
):
    translation_field = jnp.zeros_like(gripper_pcd)
    rotation_field = jnp.zeros_like(gripper_pcd)
    if gripper_type == 0:
        translation_field = jnp.where(
            segmentation[1][..., None], jnp.array([[4.0, 0, 0]]), translation_field
        )  # left finger
        translation_field = jnp.where(
            segmentation[2][..., None], jnp.array([[-4.0, 0, 0]]), translation_field
        )  # right finger

    return rotation_field, translation_field


def rotation_6d_to_matrix(d6: jnp.ndarray) -> jnp.ndarray:
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = a1 / jnp.linalg.norm(a1, axis=-1, keepdims=True)
    dot_product = jnp.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot_product * b1
    b2 = b2 / jnp.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = jnp.cross(b1, b2)
    return jnp.stack((b1, b2, b3), axis=-2)


def matrix_to_rotation_6d(matrix: jnp.ndarray) -> jnp.ndarray:
    batch_shape = matrix.shape[:-2]
    return matrix[..., :2, :].reshape(*batch_shape, 6)


@jax.jit
def compute_l2_distance_matrix(
    points_a: jnp.ndarray, points_b: jnp.ndarray
) -> jnp.ndarray:
    """Computes the pairwise L2 distance matrix between two sets of points.
    Args:
        points_a: (N, 3) JAX array.
        points_b: (M, 3) JAX array.
    Returns:
        dist_matrix: (N, M) JAX array where dist_matrix[i, j] is ||points_a[i] - points_b[j]||.
    """
    diff = points_a[:, None, :] - points_b[None, :, :]  # Shape (N, M, 3)
    dist_sq = jnp.sum(diff**2, axis=-1)  # Shape (N, M)
    return jnp.sqrt(dist_sq)


def farthest_point_sampling(x, num_samples):
    x = jax.lax.stop_gradient(x)
    farthest_points_idx = jnp.zeros(num_samples, dtype=jnp.int32)
    farthest_points_idx = farthest_points_idx.at[0].set(0)
    distances = jnp.full(x.shape[0], jnp.inf)

    def sampling_fn(i, val):
        farthest_points_idx, distances = val
        latest_point_idx = farthest_points_idx[i - 1]
        latest_point = x[latest_point_idx]
        new_dr = x - latest_point
        new_distances = jnp.sum(new_dr**2, axis=-1)
        distances = jnp.minimum(distances, new_distances)
        farthest_point_idx = jnp.argmax(distances)
        farthest_points_idx = farthest_points_idx.at[i].set(farthest_point_idx)
        return farthest_points_idx, distances

    farthest_points_idx, _ = fori_loop(
        1, num_samples, sampling_fn, (farthest_points_idx, distances)
    )
    return farthest_points_idx


@jax.jit
def find_best_assignment_and_reorder_targets(
    initial_fingertips_world: jnp.ndarray,
    offset_target_points: jnp.ndarray,
    idx_permutations: jnp.ndarray,
):
    num_points = initial_fingertips_world.shape[0]
    dist_matrix = compute_l2_distance_matrix(
        initial_fingertips_world, offset_target_points
    )
    finger_row_indices = jnp.arange(num_points)
    perm_distances = dist_matrix[finger_row_indices, idx_permutations]
    perm_loss = jnp.sum(perm_distances, axis=1)  # Shape (num_points!,)

    # Find the best permutation -> assign to targets
    best_perm_index = jnp.argmin(perm_loss)
    best_permutation_indices = idx_permutations[best_perm_index]
    assigned_target_points = offset_target_points[best_permutation_indices]

    return assigned_target_points


@jax.jit
def normalize_vector(v, axis=-1, epsilon=1e-6):
    norm = jnp.linalg.norm(v, axis=axis, keepdims=True)
    return v / (norm + epsilon)
