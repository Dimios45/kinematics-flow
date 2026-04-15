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

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from mgs.sampler.kin.se3 import (quat_to_rot_mat, quaternion_apply_jax,
                                 quaternion_from_axis_angle, se3_raw_mupltiply,
                                 similarity_transform)


@jax.jit
def point_transform(
    points: jnp.ndarray,
    Ts: jnp.ndarray,
    _: jnp.ndarray,
):
    q, t = Ts[..., :4], Ts[..., 4:]
    return quaternion_apply_jax(q, points) + t


@jax.jit
def id_transform(
    _: jnp.ndarray,
    Ts: jnp.ndarray,
    Ts_world: jnp.ndarray,
):
    return se3_raw_mupltiply(Ts, Ts_world)


@partial(jax.jit, static_argnums=0)
def kinematic_transform(
    transform,
    data: jnp.ndarray,
    theta: jnp.ndarray,
    segmentation: jnp.ndarray,
    gripper_kinematics_g,
    gripper_kinematics_s,
):
    # unsqueeze segmentation to the data dimensions
    additional_data_shapes = len(data.shape) - 1
    segmentation = jnp.expand_dims(
        segmentation,
        axis=[
            i
            for i in range(
                -1,
                -additional_data_shapes - 1,
                -1,
            )
        ],
    )
    gripper_kinematics = nnx.merge(gripper_kinematics_g, gripper_kinematics_s)
    for chain in gripper_kinematics.kinematics_graph:
        current_transform = gripper_kinematics.base_to_contact
        current_delta = jnp.array([1.0, 0, 0, 0, 0, 0, 0])

        for index in chain:
            current_transform = se3_raw_mupltiply(
                se3_raw_mupltiply(current_transform, current_delta),
                gripper_kinematics.kinematics_transforms[index],
            )
            joint_theta = theta[index]
            translation = gripper_kinematics.joint_transforms[index][:3] * joint_theta

            joint_axis = gripper_kinematics.joint_transforms[index][3:]
            axis_norm = jnp.linalg.norm(joint_axis)
            rotation_quat = jnp.where(
                axis_norm > 0,
                quaternion_from_axis_angle(joint_axis, joint_theta),
                jnp.array([1.0, 0.0, 0.0, 0.0]),
            )
            current_delta = jnp.concatenate([rotation_quat, translation], axis=-1)
            mask = segmentation[index]
            Ts = similarity_transform(current_transform, current_delta)
            data = jnp.where(
                mask,
                transform(data, Ts, current_transform),
                data,
            )

    return data


@jax.jit
def joint_segmentation(kin_g, kin_s):
    kin = nnx.merge(kin_g, kin_s)
    segmentation = jnp.full(
        shape=(kin.num_dofs, kin.num_dofs),
        fill_value=False,
    )
    counter = 0
    for chain in kin.kinematics_graph:
        for i in range(len(chain)):
            segmentation = segmentation.at[counter, chain[i:]].set(True)
            counter += 1
    return segmentation


@jax.jit
def kinematic_frames(theta, kin_g, kin_s):
    seg = joint_segmentation(kin_g, kin_s)
    data = jnp.zeros(shape=(len(seg), 7))
    Ts = kinematic_transform(id_transform, data, theta, seg, kin_g, kin_s)
    quats = Ts[..., :4]
    joints = Ts[..., 4:]
    R = jax.vmap(quat_to_rot_mat)(quats)
    return R, joints


@jax.jit
def normalize_joints(dof: jnp.ndarray, kin_g, kin_s):
    kin = nnx.merge(kin_g, kin_s)
    if kin.num_dofs == 0:
        return jnp.zeros_like(dof)
    out = (dof - kin.joint_ranges[:, 0]) / (
        kin.joint_ranges[:, 1] - kin.joint_ranges[:, 0]
    ) * 2 - 1
    return out


@jax.jit
def denormalize_joints(dof: jnp.ndarray, kin_g, kin_s):
    kin = nnx.merge(kin_g, kin_s)
    out = ((dof + 1.0) * 0.5) * (
        kin.joint_ranges[:, 1] - kin.joint_ranges[:, 0]
    ) + kin.joint_ranges[:, 0]
    return out


@jax.jit
def decouple_segmention_masks(seg: jnp.ndarray, kin_g, kin_s):
    kin = nnx.merge(kin_g, kin_s)
    decoupled_seg = seg.copy()
    for chain in kin.kinematic_graph:
        for i in range(len(chain) - 1):
            current_idx = chain[i]
            for j in range(i + 1, len(chain)):
                distal_idx = chain[j]
                decoupled_seg = decoupled_seg.at[current_idx].set(
                    jnp.logical_and(
                        decoupled_seg[current_idx],
                        jnp.logical_not(
                            seg[distal_idx],
                        ),
                    )
                )
    return decoupled_seg
