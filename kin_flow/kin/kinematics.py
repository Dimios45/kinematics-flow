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

import e3nn_jax
import jax
import jax.numpy as jnp
from flax import nnx
from jax.lax import switch

import kin_flow.util.const as CONST
from kin_flow.kin.base import KinematicsModel
from kin_flow.kin.helper import (pad_dof, pad_emb, pad_joint, pad_joint_idx,
                                 valid_mask_kinematic)
from kin_flow.kin.op import (denormalize_joints, kinematic_frames,
                             normalize_joints)
from kin_flow.util.irreps_helper import (convert_from_cl_to_irreps,
                                         convert_from_irreps_to_cl)


class KinematicEncoder(nnx.Module):
    def __init__(
        self,
        kins: list[KinematicsModel],
        position_scaling=1.0,
        *,
        rngs,
    ):
        assert all(
            kin.num_channels == kins[0].num_channels for kin in kins
        ), "All kins must have the same num_channels"
        self.num_channels = kins[0].num_channels
        self.kins = kins
        self.position_scaling = position_scaling

    def fn_builder(self, fn):
        kins = [nnx.split(kin) for kin in self.kins]
        return tuple(
            lambda x, g=g, s=s: fn(g, s, *x) for (g, s) in kins
        )  # <-- g, s captured here

    def denorm_dof(self, idx, dof):
        def switch_denorm_dof(kin_g, kin_s, dof):
            kin = nnx.merge(kin_g, kin_s)
            dof_denorm = denormalize_joints(dof[: kin.num_dofs], kin_g, kin_s)
            dof_denorm_pad = pad_dof(dof_denorm)
            return dof_denorm_pad

        out = switch(idx, self.fn_builder(switch_denorm_dof), (dof,))
        return out

    def norm_dof(self, idx, dof):
        def switch_norm_dof(kin_g, kin_s, dof):
            kin = nnx.merge(kin_g, kin_s)
            dof_norm = normalize_joints(dof[: kin.num_dofs], kin_g, kin_s)
            dof_norm_pad = pad_dof(dof_norm)
            return dof_norm_pad

        out = switch(idx, self.fn_builder(switch_norm_dof), (dof,))
        return out

    def dof_norm_ratio(self, idx):
        def switch_dof_norm_ratio(kin_g, kin_s):
            kin = nnx.merge(kin_g, kin_s)
            ranges = kin.joint_ranges.value
            ratio = jnp.abs(ranges[:, 0] - ranges[:, 1]) / 2.0
            ratio = pad_dof(ratio)
            return ratio

        out = switch(idx, self.fn_builder(switch_dof_norm_ratio), ())
        return out

    def joint_transforms(self, idx, dof):
        def switch_joint_transform(kin_g, kin_s, dof):
            j_frame, j_pos = kinematic_frames(dof, kin_g, kin_s)
            j_pos = pad_joint(j_pos)
            j_frame = jax.vmap(
                pad_joint,
                in_axes=(-1),
                out_axes=-1,
            )(j_frame)
            return j_frame, j_pos

        out = switch(idx, self.fn_builder(switch_joint_transform), (dof,))
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
        dof_norm: jnp.ndarray,
    ):
        dof_denorm = self.denorm_dof(gripper_idx, dof_norm)
        dof_frames, dof_joint_position = self.joint_transforms(gripper_idx, dof_denorm)
        dof_frames = jnp.concatenate([jnp.eye(3)[None, ...], dof_frames], axis=0)
        dof_joint_position = jnp.concatenate(
            [jnp.zeros((1, 3)), dof_joint_position * self.position_scaling],
            axis=0,
        )

        dof_feature, dof_valid_mask = self.kinematic_feature(gripper_idx)
        dof_frames = jnp.where(
            dof_valid_mask[..., None, None], dof_frames, jnp.eye(3)[None, ...]
        )

        dof_feature = e3nn_jax.vmap(convert_from_cl_to_irreps, in_axes=(0, None, None))(
            dof_feature, self.num_channels, CONST.L_MAX
        )
        dof_feature = e3nn_jax.vmap(
            e3nn_jax.IrrepsArray.transform_by_matrix, in_axes=(0, 0)
        )(dof_feature, dof_frames)
        dof_feature = e3nn_jax.vmap(convert_from_irreps_to_cl, in_axes=(0, None, None))(
            dof_feature, self.num_channels, CONST.L_MAX
        )

        pose_feature = self.pose_feature(gripper_idx)
        feature = jnp.concatenate([pose_feature, dof_feature], axis=0)

        pose_frames = jnp.broadcast_to(jnp.eye(3)[None, ...], (2, 3, 3))
        frames = jnp.concatenate(
            [pose_frames, dof_frames],
            axis=0,
        )
        pose_joint_position = jnp.zeros((2, 3))
        joint_position = jnp.concatenate(
            [pose_joint_position, dof_joint_position], axis=0
        )

        pose_valid_mask = jnp.ones((2,), dtype=jnp.bool)
        valid_mask = jnp.concatenate(
            [pose_valid_mask, dof_valid_mask], axis=0, dtype=jnp.bool
        )

        j_idx = self.j_idx(gripper_idx)
        j_parent_idx = self.j_parent_idx(gripper_idx)

        return (frames, joint_position), feature, (j_idx, j_parent_idx), valid_mask
