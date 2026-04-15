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


class Z0KinematicsModel(nnx.Module, KinematicsModel):
    """
    Used for the unconditional flow.
    """

    def __init__(self, num_emb_channels=64, *, rngs):
        self.num_channels = num_emb_channels
        self.num_dofs = 0
        self.is_static = True  # by default
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
            jnp.zeros(
                shape=(1, num_emb_channels, (CONST.L_MAX + 1) ** 2),  # DUMMY value
            )
        )
        self.kinematic_graph = []
        self.base_to_contact = nnx.Variable(
            jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

        self.kinematics_transforms = nnx.Variable(jnp.array([0.0]))
        self.joint_transforms = nnx.Variable(
            jnp.array(
                [0.0],
                dtype=jnp.float32,
            )
        )
        self.joint_ranges = nnx.Variable(jnp.zeros((1, 2)))  # DUMMY value


class Z0KinematicsModelNP(KinematicsModel):
    """
    Used for the unconditional flow.
    """

    def __init__(self):

        self.num_dofs = 0
        self.is_static = True  # by default
        self.kinematic_graph = []
        self.base_to_contact = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.kinematics_transforms = np.array((0,))
        self.joint_transforms = np.zeros(shape=(0,))
        self.joint_ranges = np.zeros((0, 2))  # DUMMY value
