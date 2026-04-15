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

import math

import jax.numpy as jnp
from flax import nnx
from mgs.sampler.kin.base import KinematicsModel
from mgs.sampler.kin.seg_op import kinematic_frames


class ShadowKinematicsModel(nnx.Module, KinematicsModel):
    def __init__(self):

        self.num_dofs = 22
        self.num_extra_dofs = 0
        self.kinematics_graph = [
            [0, 1, 2, 3],  # rh_FF
            [4, 5, 6, 7],  # rh_MF
            [8, 9, 10, 11],  # rh_RF
            [12, 13, 14, 15, 16],  # rh_LF
            [17, 18, 19, 20, 21],  # rh_TH
        ]
        self.base_to_contact = nnx.Variable(
            jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

        self.align_to_approach = nnx.Variable(
            (
                jnp.array([[0.0, 0, 1.0], [1.0, 0.0, 0], [0.0, 1.0, 0.0]]),
                jnp.array([-0.1, 0, 0]),
            )
        )

        self.kinematics_transforms = nnx.Variable(
            jnp.array(
                [
                    # rh_FF 4 (palm offset)
                    [1.0, 0.0, 0.0, 0.0, 0.033, 0, 0.095 + 0.034],
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.0],  # rh_FF 3
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.045],  # rh_FF 2
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.025],  # rh_FF 1
                    # rh_MF 4 (palm offset)
                    [1.0, 0.0, 0.0, 0.0, 0.011, 0, 0.099 + 0.034],
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.0],  # rh_MF 3
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.045],  # rh_MF 2
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.025],  # rh_MF 1
                    # rh_RF 4 (palm offset)
                    [1.0, 0.0, 0.0, 0.0, -0.011, 0, 0.095 + 0.034],
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.0],  # rh_RF 3
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.045],  # rh_RF 2
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.025],  # rh_RF 1
                    # rh_LF 5 (palm offset)
                    [1.0, 0.0, 0.0, 0.0, -0.033, 0, 0.02071 + 0.034],
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.06579],  # rh_LF 4
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.0],  # rh_RF 3
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.045],  # rh_LF 2
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.025],  # rh_LF 1
                    # rh_TH 5 (palm offset)
                    [0.92388, 0, 0.382683, 0, 0.034, -0.00858, 0.029 + 0.034],
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0],  # rh_TH 4
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.038],  # rh_TH 3
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0],  # rh_TH 2
                    [
                        1.0 / math.sqrt(2),
                        0.0,
                        0.0,
                        -1.0 / math.sqrt(2),
                        0,
                        0,
                        0.032,
                    ],  # rh_TH 1
                ]
            )
        )
        self.joint_transforms = nnx.Variable(
            jnp.array(
                [
                    # FF
                    [0, 0, 0, 0, -1, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    # MF
                    [0, 0, 0, 0, -1, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    # RF
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    # LF
                    [0, 0, 0, 0.573576, 0, 0.819152],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    # TH
                    [0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, -1, 0],
                    [0, 0, 0, 1, 0, 0],
                ],
                dtype=jnp.float32,
            )
        )
        self.joint_ranges = nnx.Variable(
            jnp.array(
                [
                    # FF
                    [-0.349066, 0.349066],
                    [-0.261799, 1.5708],
                    [0, 1.5708],
                    [0, 1.5708],
                    # MF
                    [-0.349066, 0.349066],
                    [-0.261799, 1.5708],
                    [0, 1.5708],
                    [0, 1.5708],
                    # RF
                    [-0.349066, 0.349066],
                    [-0.261799, 1.5708],
                    [0, 1.5708],
                    [0, 1.5708],
                    # LF
                    [0, 0.785398],
                    [-0.349066, 0.349066],
                    [-0.261799, 1.5708],
                    [0, 1.5708],
                    [0, 1.5708],
                    # TH
                    [-1.0472, 1.0472],
                    [0, 1.22173],
                    [-0.20944, 0.20944],
                    [-0.698132, 0.698132],
                    [-0.261799, 1.5708],
                ]
            )
        )

        self.fingertip_normals = nnx.Variable(
            jnp.array(
                [
                    [0.0, 1.0, 0],
                    [0.0, 1.0, 0],
                    [0.0, 1.0, 0],
                    [0.0, 1.0, 0],
                    [0.0, 1.0, 0],
                ]
            )
        )
        self.fingertip_idx = nnx.Variable(
            jnp.array([3, 7, 11, 16, 21], dtype=jnp.int32)
        )
        self.local_fingertip_contact_positions = nnx.Variable(
            jnp.array(
                [
                    [
                        [0, -0.01, 0],
                        [0, -0.01, 0.01],
                        [0, -0.01, -0.01],
                    ],
                    [
                        [0, -0.01, 0],
                        [0, -0.01, 0.01],
                        [0, -0.01, -0.01],
                    ],
                    [
                        [0, -0.01, 0],
                        [0, -0.01, 0.01],
                        [0, -0.01, -0.01],
                    ],
                    [
                        [0, -0.01, 0],
                        [0, -0.01, 0.01],
                        [0, -0.01, -0.01],
                    ],
                    [
                        [0, -0.01, 0],
                        [0, -0.01, 0.01],
                        [0, -0.01, -0.01],
                    ],
                ]
            )
        )
        self.init_pregrasp_joint = nnx.Variable(
            jnp.array(
                [
                    -0.350,
                    0.425,
                    0.015,
                    0.005,
                    -0.095,
                    0.415,
                    0.010,
                    0.0,
                    -0.075,
                    0.435,
                    0.015,
                    0.005,
                    0.0,
                    -0.220,
                    0.255,
                    0.0,
                    0.0,
                    -0.480,
                    1.05,
                    -0.19,
                    -0.080,
                    0.45,
                ]
            )
        )
