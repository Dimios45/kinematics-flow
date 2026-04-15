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
from flax import nnx
from jax.lax import switch

import kin_flow.util.const as CONST
from kin_flow.kin.base import KinematicsModel
from kin_flow.kin.helper import (pad_dof, pad_emb, pad_joint_idx,
                                 valid_mask_kinematic)


class KinematicEncoderSimple(nnx.Module):
    def __init__(
        self,
        kins: list[KinematicsModel],
        *,
        rngs,
    ):
        assert all(
            kin.num_channels == kins[0].num_channels for kin in kins
        ), "All kins must have the same num_channels"
        self.num_channels = kins[0].num_channels
        self.kins = kins

    def fn_builder(self, fn):
        kins = [nnx.split(kin) for kin in self.kins]
        return tuple(
            lambda x, g=g, s=s: fn(g, s, *x) for (g, s) in kins
        )  # <-- g, s captured here

    def dof_norm_ratio(self, idx):
        def switch_dof_norm_ratio(kin_g, kin_s):
            kin = nnx.merge(kin_g, kin_s)
            ranges = kin.joint_ranges.value
            ratio = jnp.abs(ranges[:, 0] - ranges[:, 1]) / 2.0
            ratio = pad_dof(ratio)
            return ratio

        out = switch(idx, self.fn_builder(switch_dof_norm_ratio), ())
        return out

    def kinematic_feature(self, idx):
        def switch_kinematic_feature(kin_g, kin_s):
            emb = pad_emb(kin_g, kin_s)
            mask = valid_mask_kinematic(kin_g, kin_s, include_static_flag=False)
            return emb, mask

        out = switch(idx, self.fn_builder(switch_kinematic_feature), ())
        return out

    def valid_mask(self, idx):
        def switch_valid_mask(kin_g, kin_s):
            mask = valid_mask_kinematic(
                kin_g,
                kin_s,
                include_static_flag=False,
            )
            return mask

        out = switch(idx, self.fn_builder(switch_valid_mask), ())
        return out

    def valid_mask_with_static_flag(self, idx):
        def switch_valid_mask(kin_g, kin_s):
            mask = valid_mask_kinematic(
                kin_g,
                kin_s,
                include_static_flag=True,
            )
            return mask

        out = switch(idx, self.fn_builder(switch_valid_mask), ())
        return out

    def pose_feature(self, idx):
        def switch_pose_feature(kin_g, kin_s):
            kin = nnx.merge(kin_g, kin_s)
            emb = kin.pose_embedding.value
            return emb

        out = switch(idx, self.fn_builder(switch_pose_feature), ())
        return out

    def j_idx(self, idx):
        def switch_j_idx(kin_g, kin_s):
            kin = nnx.merge(kin_g, kin_s)
            idx = jnp.asarray(kin.to_emb_idx(kin.kinematic_graph))
            pad_idx = pad_joint_idx(idx)
            return pad_idx

        out = switch(idx, self.fn_builder(switch_j_idx), ())
        return out

    def j_parent_idx(self, idx):
        def switch_j_parent_idx(kin_g, kin_s):
            kin = nnx.merge(kin_g, kin_s)
            idx = jnp.asarray(kin.to_emb_idx(kin.parent_joints()))
            pad_idx = pad_joint_idx(idx)
            return pad_idx

        out = switch(idx, self.fn_builder(switch_j_parent_idx), ())
        return out

    def __call__(
        self,
        gripper_idx: jnp.ndarray,
        wigner_d: jnp.ndarray,
    ):
        dof_feature, dof_valid_mask = self.kinematic_feature(gripper_idx)
        pose_feature = self.pose_feature(gripper_idx)
        feature = jnp.concatenate([pose_feature, dof_feature], axis=0)

        # the whole wigner_d computation outsourced to f64 precision
        feature_l = [feature[..., :1]]
        for wd, i in zip(wigner_d, range(CONST.L_MAX)):
            l = i + 1
            start_idx = l**2
            end_idx = (l + 1) ** 2
            f_l = feature[..., start_idx:end_idx]
            f_l_w = jnp.matmul(
                f_l, jnp.transpose(wd, [0, 2, 1]), precision=jax.lax.Precision.HIGHEST
            )
            feature_l.append(f_l_w)
        feature = jnp.concatenate(feature_l, axis=-1)

        pose_valid_mask = jnp.ones((2,), dtype=jnp.bool)
        valid_mask = jnp.concatenate(
            [pose_valid_mask, dof_valid_mask], axis=0, dtype=jnp.bool
        )

        j_idx = self.j_idx(gripper_idx)
        j_parent_idx = self.j_parent_idx(gripper_idx)

        return feature, (j_idx, j_parent_idx), valid_mask
