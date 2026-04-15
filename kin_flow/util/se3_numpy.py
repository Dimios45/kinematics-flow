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

import numpy as np
from scipy.spatial.transform import Rotation


def _to_scipy_quat(q: np.ndarray) -> np.ndarray:
    """Helper: Convert (w, x, y, z) -> (x, y, z, w)"""
    return np.roll(q, -1, axis=-1)


def _from_scipy_quat(q: np.ndarray) -> np.ndarray:
    """Helper: Convert (x, y, z, w) -> (w, x, y, z)"""
    return np.roll(q, 1, axis=-1)


def rot_mat_to_quat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R)
    orig_shape = R.shape[:-2]

    # Flatten batch dims for SciPy: (..., 3, 3) -> (N, 3, 3)
    R_flat = R.reshape(-1, 3, 3)

    # SciPy handles the complex trace checking and numerical stability internally
    rot_obj = Rotation.from_matrix(R_flat)
    q_scipy = rot_obj.as_quat()

    q_out = _from_scipy_quat(q_scipy)
    return q_out.reshape(orig_shape + (4,))


def quaternion_invert(quaternion: np.ndarray) -> np.ndarray:
    scaling = np.array([1, -1, -1, -1], dtype=quaternion.dtype)
    return quaternion * scaling


def quaternion_raw_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    b = np.asarray(b)

    # Shape handling
    out_shape = np.broadcast_shapes(a.shape, b.shape)
    a_flat = np.broadcast_to(a, out_shape).reshape(-1, 4)
    b_flat = np.broadcast_to(b, out_shape).reshape(-1, 4)

    # Convert to SciPy objects
    # Note: SciPy multiplication (r1 * r2) is composition: r1 applied AFTER r2.
    # Standard quaternion multiplication q1 * q2 usually corresponds to composition.
    rot_a = Rotation.from_quat(_to_scipy_quat(a_flat))
    rot_b = Rotation.from_quat(_to_scipy_quat(b_flat))

    # Result is a composition of rotations
    rot_out = rot_a * rot_b

    result = _from_scipy_quat(rot_out.as_quat())
    return result.reshape(out_shape)


def quaternion_apply(quaternion: np.ndarray, point: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion)
    point = np.asarray(point)

    # Broadcast shapes to allow (N,4) quat with (3,) point or vice versa
    # We treat the last dim as fixed (4 for quat, 3 for point)
    batch_shape = np.broadcast_shapes(quaternion.shape[:-1], point.shape[:-1])

    q_flat = np.broadcast_to(quaternion, batch_shape + (4,)).reshape(-1, 4)
    p_flat = np.broadcast_to(point, batch_shape + (3,)).reshape(-1, 3)

    rot = Rotation.from_quat(_to_scipy_quat(q_flat))
    rotated_points = rot.apply(p_flat)

    return rotated_points.reshape(batch_shape + (3,))


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    # Avoid div by zero
    return q / np.clip(norm, 1e-12, None)


def quat_to_rot_mat(
    q: np.ndarray, normalize: bool = False, eps: float = 1e-12
) -> np.ndarray:
    q = np.asarray(q)
    orig_shape = q.shape[:-1]
    q_flat = q.reshape(-1, 4)

    # Check normalization if requested, though SciPy usually normalizes internally.
    if normalize:
        n = np.linalg.norm(q_flat, axis=1, keepdims=True)
        q_flat = q_flat / np.clip(n, eps, None)

    rot = Rotation.from_quat(_to_scipy_quat(q_flat))
    mat = rot.as_matrix()

    return mat.reshape(orig_shape + (3, 3))


def random_quat(
    batch_shape=(),
    *,
    canonicalize: bool = True,
    dtype=np.float64,
    rng=None,
) -> np.ndarray:
    if isinstance(batch_shape, int):
        batch_shape = (batch_shape,)

    n_samples = int(np.prod(batch_shape)) if batch_shape else 1

    # Use passed RNG if available (SciPy accepts random_state)
    random_state = rng if rng is not None else None

    # SciPy generates uniform random rotations (Haar measure)
    rot = Rotation.random(n_samples, random_state=random_state)
    q = _from_scipy_quat(rot.as_quat())

    if dtype is not None:
        q = q.astype(dtype)

    if canonicalize:
        # Enforce w >= 0
        flip = q[..., :1] < 0
        q = np.where(flip, -q, q)

    return q.reshape(batch_shape + (4,))


def transform_points(points: np.ndarray, Ts: np.ndarray) -> np.ndarray:
    q, t = Ts[..., :4], Ts[..., 4:]
    points = quaternion_apply(q, points) + t
    return points


def se3_raw_mupltiply(Ts_one: np.ndarray, Ts_two: np.ndarray) -> np.ndarray:
    q_one, _ = Ts_one[..., :4], Ts_one[..., 4:]
    q_two, t_two = Ts_two[..., :4], Ts_two[..., 4:]
    q = quaternion_raw_multiply(q_one, q_two)
    t = transform_points(t_two, Ts_one)
    return np.concatenate([q, t], axis=-1)


def quaternion_from_axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis)
    angle = np.asarray(angle)

    half_angle = angle[..., np.newaxis] / 2.0
    s = np.sin(half_angle)
    c = np.cos(half_angle)

    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        axis_normalized = axis / norm

    q_xyz = s * axis_normalized
    q_w = c
    target_batch_shape = np.broadcast_shapes(q_w.shape[:-1], q_xyz.shape[:-1])

    q_w = np.broadcast_to(q_w, target_batch_shape + (1,))
    q_xyz = np.broadcast_to(q_xyz, target_batch_shape + (3,))

    q = np.concatenate([q_w, q_xyz], axis=-1)

    return q


def se3_invert(Ts: np.ndarray) -> np.ndarray:
    q, t = Ts[..., :4], Ts[..., 4:]
    q_inv = quaternion_invert(q)
    t = -quaternion_apply(q_inv, t)
    return np.concatenate([q_inv, t], axis=-1)


def similarity_transform(Ts_sim: np.ndarray, Ts: np.ndarray) -> np.ndarray:
    return se3_raw_mupltiply(se3_raw_mupltiply(Ts_sim, Ts), se3_invert(Ts_sim))
