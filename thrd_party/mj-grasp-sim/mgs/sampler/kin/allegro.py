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

import os

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from flax import nnx
from mgs.sampler.kin.base import KinematicsModel


class AllegroKinematicsModel(nnx.Module, KinematicsModel):
    def __init__(self):
        self.num_dofs = 16
        self.num_extra_dofs = 0
        self.kinematics_graph = [
            [0, 1, 2, 3],  # ffa
            [4, 5, 6, 7],  # mfa
            [8, 9, 10, 11],  # rfa
            [12, 13, 14, 15],  # tha
        ]
        self.base_to_contact = nnx.Variable(
            jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

        self.align_to_approach = nnx.Variable(
            (
                jnp.array(
                    [
                        [np.cos(-np.pi / 2), 0, -np.sin(-np.pi / 2)],
                        [0, 1.0, 0],
                        [np.sin(-np.pi / 2), 0, np.cos(-np.pi / 2)],
                    ]
                ),
                jnp.array([+0.01, 0.0, +0.08]),
            )
        )

        self.kinematics_transforms = nnx.Variable(
            jnp.array(
                [
                    # FF
                    [0.999048, -0.0436194, 0, 0, 0, 0.0435, -0.001542],  # ffj0
                    [1, 0, 0, 0, 0, 0, 0.0164],  # ffj1
                    [1, 0, 0, 0, 0, 0, 0.054],  # ffj2
                    [1, 0, 0, 0, 0, 0, 0.0384],  # ffj3
                    # MF
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.0007],  # mfj0
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.0164],  # mfj1
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.054],  # mfj2
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.0384],  # mfj3
                    # RF
                    [0.999048, 0.0436194, 0, 0, 0, -0.0435, -0.001542],  # rfj0
                    [1, 0, 0, 0, 0, 0, 0.0164],  # rfj1
                    [1, 0, 0, 0, 0, 0, 0.054],  # rfj2
                    [1, 0, 0, 0, 0, 0, 0.0384],  # rfj3
                    # TH
                    [
                        0.477714,
                        -0.521334,
                        -0.521334,
                        -0.477714,
                        -0.0182,
                        0.019333,
                        -0.045987,
                    ],  # thj0
                    [1, 0, 0, 0, -0.027, 0.005, 0.0399],  # thj1
                    [1, 0, 0, 0, 0, 0, 0.0177],  # thj2
                    [1, 0, 0, 0, 0, 0, 0.0514],  # thj3
                ]
            )
        )

        self.joint_transforms = nnx.Variable(
            jnp.array(
                [
                    # FF
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    # MF
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    # RF
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    # TH
                    [0, 0, 0, -1, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                ],
                dtype=jnp.float32,
            )
        )
        self.joint_ranges = nnx.Variable(
            jnp.array(
                [
                    # FF
                    [-0.47, 0.47],
                    [-0.196, 1.61],
                    [-0.174, 1.709],
                    [-0.227, 1.618],
                    # MF
                    [-0.47, 0.47],
                    [-0.196, 1.61],
                    [-0.174, 1.709],
                    [-0.227, 1.618],
                    # RF
                    [-0.47, 0.47],
                    [-0.196, 1.61],
                    [-0.174, 1.709],
                    [-0.227, 1.618],
                    # TH
                    [0.263, 1.396],
                    [-0.105, 1.163],
                    [-0.189, 1.644],
                    [-0.162, 1.719],
                ]
            )
        )

        self.fingertip_normals = nnx.Variable(
            jnp.array(
                [
                    [-1.0, 0, 0],
                    [-1.0, 0, 0],
                    [-1.0, 0.0, 0],
                    [-1.0, 0.0, 0.0],
                ]
            )
        )

        self.fingertip_idx = nnx.Variable(jnp.array([3, 7, 11, 15], dtype=jnp.int32))

        self.local_fingertip_contact_positions = nnx.Variable(
            jnp.array(
                [
                    [
                        [0.0, 0, 0.023],
                        [0.002, 0, 0.02],
                        [0, 0.002, 0.02],
                    ],
                    [
                        [0, 0, 0.023],
                        [0.002, 0, 0.02],
                        [0.0, 0.002, 0.02],
                    ],
                    [
                        [0, 0, 0.023],
                        [0.002, 0, 0.02],
                        [0.0, 0.002, 0.02],
                    ],
                    [
                        [0, 0, 0.035],
                        [0.002, 0, 0.032],
                        [0.0, 0.002, 0.032],
                    ],
                ]
            )
        )
        self.init_pregrasp_joint = nnx.Variable(
            jnp.array(
                [
                    -0.08,
                    0.297,
                    0.710,
                    0.95,
                    0,
                    0.319,
                    0.71,
                    0.67,
                    0.08,
                    0.454,
                    0.710,
                    0.95,
                    1.06,
                    0.358,
                    0.251,
                    0.318,
                ]
            )
        )
