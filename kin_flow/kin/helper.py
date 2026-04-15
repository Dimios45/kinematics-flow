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

import jax.numpy as jnp
from flax import nnx

import kin_flow.util.const as CONST


def pad_emb(kin_g, kin_s):
    kin = nnx.merge(kin_g, kin_s)
    padding_dof = CONST.MAX_DOF - kin.num_dofs

    padding = jnp.zeros(
        shape=(
            padding_dof,
            kin.num_channels,
            (CONST.L_MAX + 1) ** 2,
        )
    )
    padded_emb = jnp.concatenate(
        [kin.dof_embedding.value[: kin.num_dofs + 1], padding], axis=0
    )
    return padded_emb


def pad_joint_idx(idx):
    padding_dof = CONST.MAX_DOF - len(idx)  # one for base node

    padding = jnp.zeros(
        shape=(padding_dof,),
        dtype=jnp.int32,
    )
    padded_emb = jnp.concatenate([idx, padding], axis=0, dtype=jnp.int32)
    return padded_emb


def valid_mask_kinematic(kin_g, kin_s, include_static_flag=False):
    kin = nnx.merge(kin_g, kin_s)
    if include_static_flag:
        valid_num = kin.num_dofs + 1 if not kin.is_static else 1  # 1 for base node
    else:
        valid_num = kin.num_dofs + 1  # 1 for base node

    in_valid_num = (CONST.MAX_DOF + 1) - valid_num  # +1 for base node
    valid = jnp.full(shape=(valid_num,), fill_value=True, dtype=jnp.bool)
    in_valid = jnp.full(shape=(in_valid_num,), fill_value=False, dtype=jnp.bool)
    mask = jnp.concatenate([valid, in_valid], axis=0, dtype=jnp.bool)
    return mask


def pad_dof(dof: jnp.ndarray):
    padding_dof = CONST.MAX_DOF - len(dof)
    padding = jnp.zeros(shape=(padding_dof,))
    padded_dof = jnp.concatenate([dof, padding])
    return padded_dof


def pad_seg(seg: jnp.ndarray):
    padding_dof = CONST.MAX_DOF - len(seg)
    padding_y = jnp.zeros(shape=(padding_dof, seg.shape[1]))
    padding_x = jnp.zeros(shape=(CONST.MAX_DOF, padding_dof))
    padded_seg = jnp.concatenate([seg, padding_y], axis=0)
    padded_seg = jnp.concatenate([padded_seg, padding_x], axis=1)
    return padded_seg


def pad_joint(
    joint_origins: jnp.ndarray,
):
    padding_dof = CONST.MAX_DOF - len(joint_origins)
    padding = jnp.zeros(shape=(padding_dof, 3))
    padded_joint_origins = jnp.concatenate([joint_origins, padding], axis=0)
    return padded_joint_origins
