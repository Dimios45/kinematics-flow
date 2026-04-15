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

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from kin_flow.net.kinematics_flow import KinematicsFlow


class KinematicsFlowWOEncoding(nnx.Module):
    def __init__(
        self,
        net: KinematicsFlow,
        num_scene_points=11,
        *,
        rngs,
    ):
        self.net = net
        del self.net.unet  # for performance reasons

        N = num_scene_points
        channel_dims = net.multi_scale_feature_query.multiscale_channels
        self.scene = nnx.Variable(
            [
                (
                    jax.random.normal(nnx.Rngs(0)(), shape=(N, 3))
                    * 0.15
                    * net.position_scaling,
                    jnp.concatenate(
                        [
                            jax.random.normal(nnx.Rngs(1)(), shape=(N, dim, 9))
                            for dim in channel_dims
                        ],
                        axis=-2,
                    ),
                    None,
                ),  # 68.3 % are between +-0.15 * pos_scaling
            ],
        )

    def __call__(
        self,
        scene: Tuple[jnp.ndarray, jnp.ndarray],
        gripper_idx: jnp.ndarray,
        wigner_d: jnp.ndarray,
        q_pos_rot: jnp.ndarray,  # rotated, but not translated features -> numerical stability on SO3
        q_pos_rot_trans: jnp.ndarray,  # both rotated and translated
        t: jnp.ndarray,
        key,
    ):
        # Encode gripper
        q_feat, q_pos, valid = self.kinematic_encoding(
            gripper_idx, wigner_d, q_pos_rot, t
        )

        # Relate scene and query
        feat, _ = self.multi_scale_feature_query(
            self.scene.value, ((q_pos_rot_trans, q_pos), q_feat), t, valid
        )

        # Decode flow
        u = self.decoder(feat, q_pos, valid, t)
        dof_valid_mask = (
            self.kinematic_encoding.kinematic_net.valid_mask_with_static_flag(
                gripper_idx
            )[1:]
        )  # exclude base node

        return (u, dof_valid_mask), {}

    def get_kins(self):
        kins = self.net.kinematic_encoding.kinematic_net.kins
        return kins

    @property
    def position_scaling(self):
        return self.net.position_scaling

    @property
    def unet(self):
        return lambda _p, _rngs: (self.scene.value, None, None)

    @property
    def multi_scale_feature_query(self):
        return self.net.multi_scale_feature_query

    @property
    def kinematic_encoding(self):
        return self.net.kinematic_encoding

    @property
    def decoder(self):
        return self.net.decoder
