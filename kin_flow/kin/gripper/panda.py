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
import numpy as np
from flax import nnx

import kin_flow.util.const as CONST
from kin_flow.kin.base import KinematicsModel


class PandaKinematicsModel(nnx.Module, KinematicsModel):
    def __init__(self, num_emb_channels=64, is_static=False, *, rngs):
        self.is_static = is_static
        self.num_channels = num_emb_channels
        self.num_dofs = 2
        self.pose_embedding = nnx.Param(
            jax.random.normal(
                rngs(),
                shape=(
                    2,
                    num_emb_channels,
                    (CONST.L_MAX + 1) ** 2,
                ),
            )
        )
        self.dof_embedding = nnx.Param(
            jax.random.normal(
                rngs(),
                shape=(
                    self.num_dofs + 1,  # one for node base
                    num_emb_channels,
                    (CONST.L_MAX + 1) ** 2,
                ),
            )
        )
        self.kinematic_graph = [
            [0],  # left finger
            [1],  # right finger
        ]
        self.base_to_contact = nnx.Variable(
            jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

        self.kinematics_transforms = nnx.Variable(
            jnp.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.103],  # left
                    [0.0, 0.0, 0.0, 1.0, 0, -0.04, 0.103],  # right
                ]
            )
        )
        self.joint_transforms = nnx.Variable(
            jnp.array(
                [
                    [0, 1.0, 0, 0, 0, 0],
                    [0, 1.0, 0, 0, 0, 0],
                ],
                dtype=jnp.float32,
            )
        )
        self.joint_ranges = nnx.Variable(
            jnp.array(
                [
                    [0.0, 0.04],  # starts in the middle?
                    [-0.04, 0.0],  # starts on the end?... weird design
                ]
            )
        )


class PandaKinematicsModelNP(KinematicsModel):
    def __init__(self, is_static=False):
        self.is_static = is_static
        self.num_dofs = 2
        self.kinematic_graph = [
            [0],  # left finger
            [1],  # right finger
        ]
        self.base_to_contact = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.kinematics_transforms = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0, 0, 0.103],  # left
                [0.0, 0.0, 0.0, 1.0, 0, -0.04, 0.103],  # right
            ]
        )
        self.joint_transforms = np.array(
            [
                [0, 1.0, 0, 0, 0, 0],
                [0, 1.0, 0, 0, 0, 0],
            ],
        )
        self.joint_ranges = np.array(
            [
                [0.0, 0.04],  # starts in the middle?
                [-0.04, 0.0],  # starts on the end?... weird design
            ]
        )
