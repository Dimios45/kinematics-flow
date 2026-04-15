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

import math

import numpy as np
from mgs.util.geo.convert import quat_xyzw_to_wxyz
from scipy.spatial.transform import Rotation


def rnd_direction():
    theta = 2 * np.pi * np.random.rand()
    phi = np.arccos(2 * np.random.rand() - 1)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.array([x, y, z])


def rnd_camera_pose(radius=2.0):
    direction = rnd_direction()
    direction /= np.linalg.norm(direction)
    position = -radius * direction

    up = np.array([0, 0, 1])
    right = np.cross(direction, up)
    up = np.cross(right, direction)

    transformation = np.eye(3)
    transformation[:3, 0] = direction
    transformation[:3, 1] = -right
    transformation[:3, 2] = up

    scipy_quat = Rotation.from_matrix(transformation).as_quat()
    mujoco_quat = quat_xyzw_to_wxyz(scipy_quat)

    return position, mujoco_quat


def rnd_direction_restricted(phi=0.125 * np.pi):
    theta = 2 * np.pi * np.random.rand()
    phi = np.pi - np.random.random() * phi

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.array([x, y, z])


def rnd_camera_pose_restricted(radius=2.0, phi=0.125 * np.pi):
    direction = rnd_direction_restricted(phi=phi)
    direction /= np.linalg.norm(direction)
    position = -radius * direction

    up = np.array([0, 0, 1])
    right = np.cross(direction, up)
    up = np.cross(right, direction)

    transformation = np.eye(3)
    transformation[:3, 0] = direction
    transformation[:3, 1] = -right
    transformation[:3, 2] = up

    scipy_quat = Rotation.from_matrix(transformation).as_quat()
    mujoco_quat = quat_xyzw_to_wxyz(scipy_quat)

    return position, mujoco_quat


def fibonacci_sphere(total_num, i):
    """
    Returns the (x, y, z) coordinates of the i-th point from a total of total_num points,
    distributed approximately evenly on the surface of a unit sphere using the Fibonacci method.

    Parameters:
      total_num (int): Total number of points.
      i (int): The index of the desired point (0 <= i < total_num).

    Returns:
      tuple: (x, y, z) coordinates on the unit sphere.
    """
    # Golden angle in radians
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    # Normalize i into range [0, total_num - 1], mapping to y in [-1, +1]
    y = 1 - (2.0 * i / (total_num - 1))
    radius = math.sqrt(max(0.0, 1.0 - y * y))

    # Compute the angle around the sphere
    theta = golden_angle * i

    x = math.cos(theta) * radius
    z = math.sin(theta) * radius

    return np.array([x, y, z])
