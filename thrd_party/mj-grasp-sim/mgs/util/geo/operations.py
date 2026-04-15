# Copyright (c) 2025 Robert Bosch GmbH
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


# This source code is derived from Diffusion-EDFs
#   (https://github.com/tomato1mule/diffusion_edf/tree/d1ae47fa4bf6e0133c5c373b20939c603d327fc0)
# Copyright (c) Hyunwoo Ryu 2023 licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def quaternion_invert(quaternion: np.ndarray, type: str = "wxyz") -> np.ndarray:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.
    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).
    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = np.array([])
    if type == "wxyz":
        scaling = np.array([1, -1, -1, -1])
    elif type == "xyzw":
        scaling = np.array([-1, -1, -1, 1])
    return quaternion * scaling


# This source code is derived from Diffusion-EDFs
#   (https://github.com/tomato1mule/diffusion_edf/tree/d1ae47fa4bf6e0133c5c373b20939c603d327fc0)
# Copyright (c) Hyunwoo Ryu 2023 licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def quaternion_apply(
    quaternion: np.ndarray, point: np.ndarray, type: str = "wxyz"
) -> np.ndarray:
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

    real_parts = np.zeros(point.shape[:-1] + (1,))

    point_as_quaternion = np.array([])
    if type == "wxyz":
        point_as_quaternion = np.concatenate((real_parts, point), -1)
    elif type == "xyzw":
        point_as_quaternion = np.concatenate((point, real_parts), -1)

    qp = quaternion_raw_multiply(quaternion, point_as_quaternion, type=type)
    q_inv = quaternion_invert(quaternion, type=type)
    out = quaternion_raw_multiply(
        qp,
        q_inv,
        type=type,
    )
    return out[..., 1:]


# This source code is derived from Diffusion-EDFs
#   (https://github.com/tomato1mule/diffusion_edf/tree/d1ae47fa4bf6e0133c5c373b20939c603d327fc0)
# Copyright (c) Hyunwoo Ryu 2023 licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def quaternion_raw_multiply(
    a: np.ndarray, b: np.ndarray, type: str = "wxyz"
) -> np.ndarray:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.
    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.
    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    if type == "wxyz":
        aw, ax, ay, az = a[..., [0]], a[..., [1]], a[..., [2]], a[..., [3]]
        bw, bx, by, bz = b[..., [0]], b[..., [1]], b[..., [2]], b[..., [3]]
    elif type == "xyzw":
        aw, ax, ay, az = a[..., 3], a[..., 1], a[..., 2], a[..., 0]
        bw, bx, by, bz = b[..., 3], b[..., 1], b[..., 2], b[..., 0]
    else:
        raise ValueError

    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw

    out = np.array([])
    if type == "wxyz":
        out = np.concatenate([ow, ox, oy, oz], axis=-1)
    elif type == "xyzw":
        out = np.concatenate([ox, oy, oz, ow], axis=-1)
    return out
