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

import numpy as np

import kin_flow.util.const as CONST
from kin_flow.util.se3_numpy import (quat_to_rot_mat, quaternion_apply,
                                     quaternion_from_axis_angle,
                                     se3_raw_mupltiply, similarity_transform)


def point_transform(
    points: np.ndarray,
    Ts: np.ndarray,
    _: np.ndarray,
):
    q, t = Ts[..., :4], Ts[..., 4:]
    return quaternion_apply(q, points) + t


def id_transform(
    _: np.ndarray,
    Ts: np.ndarray,
    Ts_world: np.ndarray,
):
    return se3_raw_mupltiply(Ts, Ts_world)


def kinematic_transform(
    transform,
    data: np.ndarray,
    theta: np.ndarray,
    segmentation: np.ndarray,
    kinematics,
):
    B = theta.shape[0]
    segmentation = np.broadcast_to(
        segmentation[None, ..., None], shape=(B, *segmentation.shape, 1)
    )
    for chain in kinematics.kinematic_graph:
        current_transform = kinematics.base_to_contact
        current_transform = np.broadcast_to(current_transform[None, ...], shape=(B, 7))
        current_delta = np.broadcast_to(
            np.array([[1.0, 0, 0, 0, 0, 0, 0]]), shape=(B, 7)
        )

        for index in chain:
            current_transform = se3_raw_mupltiply(
                se3_raw_mupltiply(current_transform, current_delta),
                kinematics.kinematics_transforms[index],
            )
            joint_theta = theta[:, index]
            translation = (
                kinematics.joint_transforms[index][:3][None] * joint_theta[..., None]
            )

            joint_axis = kinematics.joint_transforms[index][3:][None]
            axis_norm = np.linalg.norm(joint_axis, keepdims=True)
            rotation_quat = np.where(
                axis_norm > 0,
                quaternion_from_axis_angle(joint_axis, joint_theta[...]),
                np.array([1.0, 0.0, 0.0, 0.0]),
            )
            current_delta = np.concatenate([rotation_quat, translation], axis=-1)
            mask = segmentation[:, index, ...]
            Ts = similarity_transform(current_transform, current_delta)
            data = np.where(
                mask,
                np.broadcast_to(
                    transform(data, Ts, current_transform)[:, None, :],
                    shape=(B, segmentation.shape[1], 7),
                ),
                data,
            )

    return data


def joint_segmentation(kin):
    segmentation = np.full(
        shape=(kin.num_dofs, kin.num_dofs),
        fill_value=False,
    )
    counter = 0
    for chain in kin.kinematic_graph:
        for i in range(len(chain)):
            segmentation[counter, chain[i:]] = True
            counter += 1
    return segmentation


def kinematic_frames(theta, kin):
    B = theta.shape[0]
    seg = joint_segmentation(kin)
    data = np.zeros(shape=(B, len(seg), 7))
    Ts = kinematic_transform(id_transform, data, theta, seg, kin)
    quats = Ts[..., :4]
    joints = Ts[..., 4:]
    R = quat_to_rot_mat(quats)
    return R, joints


def pad_joints(dof: np.ndarray):
    leading_dims = dof.shape[:-1]
    to_pad = np.zeros(shape=(*leading_dims, CONST.MAX_DOF - dof.shape[-1]))
    return np.concatenate([dof, to_pad], axis=-1)


def normalize_joints(dof: np.ndarray, kin):

    if kin.num_dofs == 0:
        return np.zeros_like(dof)
    out = (dof - kin.joint_ranges[:, 0]) / (
        kin.joint_ranges[:, 1] - kin.joint_ranges[:, 0]
    ) * 2 - 1
    return out


def denormalize_joints(dof: np.ndarray, kin):
    dof, dof_pad = dof[..., : kin.num_dofs], dof[..., kin.num_dofs :]
    out = ((dof + 1.0) * 0.5) * (
        kin.joint_ranges[:, 1] - kin.joint_ranges[:, 0]
    ) + kin.joint_ranges[:, 0]
    return np.concatenate([out, dof_pad], axis=-1)
