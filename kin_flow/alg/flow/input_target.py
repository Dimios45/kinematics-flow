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

import random

import jax
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import kin_flow.util.const as CONST
from kin_flow.kin.const import KINS_NP
from kin_flow.kin.op_numpy import (denormalize_joints, kinematic_frames,
                                   normalize_joints)
from kin_flow.kin.wigner_d import wigner_D_real


def compute_x1_targets(
    kins,  # Array of numpy kinematics desciptions [G]
    rot_scene,  # Clean Data R1 [(B, 3, 3)]_G
    pos_scene,  # Clean Data x1 [(B, 3)]_G
    joints,  # [(B, Q)]_G
    ts,  # shape (G, B)
    position_scaling=1.0,
):
    G, B = ts.shape[0], ts.shape[1]

    # ROTATION
    rot_scene = np.stack(rot_scene)
    Rs = Rotation.from_matrix(rot_scene.reshape(-1, 3, 3))
    ts_flat = ts.reshape(-1)
    rot_0_flat = Rotation.random(G * B).as_matrix()
    # Geodesic Rotation (SLERP)
    R0 = Rotation.from_matrix(rot_0_flat)
    R_rel = R0.inv() * Rs
    rel_vec = R_rel.as_rotvec()
    scaled_vec = rel_vec * ts_flat[:, None]
    R_step = Rotation.from_rotvec(scaled_vec)
    Rt = R0 * R_step
    rot_ts_flat = Rt.as_matrix()
    rot_ts = np.reshape(rot_ts_flat, shape=(G, B, 3, 3))
    target_rot_delta = Rt.apply((Rt.inv() * Rs).as_rotvec()).reshape(G, B, 3)

    # POSITION
    pos_scene = np.stack(pos_scene)
    pos_1 = pos_scene * position_scaling
    pos_0 = np.random.normal(size=pos_1.shape)
    pos_ts = (1 - ts[..., None]) * pos_0 + ts[..., None] * pos_1
    target_pos_delta = pos_1 - pos_ts

    # JOINTS
    joint_inputs, joint_targets = [], []
    for rot_t, pos_t, kin, theta, t in zip(rot_ts, pos_ts, kins, joints, ts):
        theta_t = normalize_joints(theta, kin)
        theta_1 = theta_t
        if not kin.is_static:
            theta_1 = normalize_joints(theta, kin)
            theta_0 = np.random.uniform(low=-1.0, high=1.0, size=theta_1.shape)
            theta_t = (1 - t[:, None]) * theta_0 + t[:, None] * theta_1

        joint_t = denormalize_joints(theta_t, kin)  # shape (B, J)
        joint_rot, joint_pos = kinematic_frames(
            joint_t, kin
        )  # shapes (B, J, 3, 3), (B, J, 3)
        rot_t = np.asarray(rot_t, dtype=np.float64)[:, None, :, :]
        joint_rot = np.asarray(joint_rot, dtype=np.float64)
        qpos_scene_rot = rot_t @ joint_rot  # shape (B, J, 3, 3)
        # append pose -> J + 3
        pose_rot = np.broadcast_to(rot_t, shape=(B, 3, 3, 3))
        qpos_scene_rot = np.concatenate(
            [pose_rot, qpos_scene_rot], axis=1
        )  # shape (B, J + 3, 3, 3)
        wigner_d = [
            wigner_D_real(l=i + 1, R=qpos_scene_rot) for i in range(CONST.L_MAX)
        ]  # shape ((B, J + 3, 2*l+1, 2*l+1)...)

        q_pos = np.concatenate(
            [np.zeros(shape=(B, 3, 3)), joint_pos], axis=1
        )  # (B, J + 3, 3)
        q_pos_rot = np.einsum("bij,bqj->bqi", rot_t[:, 0], q_pos) * position_scaling
        q_pos_rot_trans = q_pos_rot + pos_t[:, None, :]
        joint_inputs.append(
            (
                wigner_d,
                q_pos_rot,
                np.concatenate(
                    [q_pos_rot_trans[:, :2], q_pos_rot_trans[:, 3:]], axis=1
                ),
            )
        )
        joint_targets.append(theta_1 - theta_t)

    # PADDING
    padding_wd = [
        wigner_D_real(l=i + 1, R=np.eye(3)[None, None]) for i in range(CONST.L_MAX)
    ]
    pad_joint_inputs, pad_joint_targets = [], []
    for ji, jo in zip(joint_inputs, joint_targets):
        wd, qpr, qprt = ji
        jt = jo

        J_P = qpr.shape[1]
        to_pad = (CONST.MAX_DOF + 3) - J_P
        pad_wd = [
            np.concatenate(
                [
                    wd,
                    np.broadcast_to(
                        pwd, shape=(B, to_pad, 2 * (i + 1) + 1, 2 * (i + 1) + 1)
                    ),
                ],
                axis=1,
            )
            for (i, (pwd, wd)) in enumerate(zip(padding_wd, wd))
        ]

        pad_qpr = np.concatenate([qpr, np.zeros(shape=(B, to_pad, 3))], axis=1)
        pad_qprt = np.concatenate([qprt, np.zeros(shape=(B, to_pad, 3))], axis=1)
        pad_joint_inputs.append((pad_wd, pad_qpr, pad_qprt))

        pad_jt = np.concatenate([jt, np.zeros(shape=(B, to_pad))], axis=-1)
        pad_joint_targets.append(pad_jt)

    # STACKING
    joint_delta = np.stack(pad_joint_targets)
    q_pos_rot = np.stack([ins[1] for ins in pad_joint_inputs])
    q_pos_rot_trans = np.stack([ins[2] for ins in pad_joint_inputs])

    wigner_d = [wd[0] for wd in pad_joint_inputs]
    wigner_d_stacked = []
    for i in range(CONST.L_MAX):
        w_i = np.stack([w[i] for w in wigner_d])
        wigner_d_stacked.append(w_i)

    inputs = (wigner_d_stacked, q_pos_rot, q_pos_rot_trans)
    targets = np.concatenate([target_rot_delta, target_pos_delta, joint_delta], axis=-1)
    return inputs, targets


def scipy_slerp(times, rots):
    return Slerp(times, rots)


def input_target(sample, cfg, key):
    scene = (sample["scene_points"] * cfg.position_scaling, sample["scene_colors"])
    rot, pos, joints = (
        sample["rotations"],
        sample["positions"],
        sample["joints"],
    )

    kins = []
    for gripper in cfg.input_target.gripper:
        kins.append(KINS_NP[gripper])

    batch_size = rot[0].shape[0]
    num_gripper = len(rot)
    t = np.random.uniform(low=0.0, high=1.0 - 1e-2, size=(num_gripper, batch_size))

    (wigner_d, q_pos_rot, q_pos_rot_trans), targets = compute_x1_targets(
        kins,
        rot,
        pos,
        joints,
        t,
        position_scaling=cfg.position_scaling,
    )

    z0_id = -1
    z0_set = False
    for i, val in enumerate(cfg.input_target.gripper):
        if val == "z0" and not z0_set:
            z0_id = i
            z0_set = True
        elif val == "z0" and z0_set:
            raise ValueError("z0 set multiple times")
    if z0_id != -1:
        gripper_ids = np.concatenate(
            [
                np.arange(start=0, stop=z0_id),
                np.arange(start=z0_id + 1, stop=num_gripper + 1),
            ],
            axis=-1,
        )
        i = random.randint(0, num_gripper - 1)
        gripper_ids[i] = z0_id
    else:
        gripper_ids = np.arange(num_gripper)

    inputs = (
        scene,
        gripper_ids,
        wigner_d,
        q_pos_rot,
        q_pos_rot_trans,
        t,
        jax.random.split(key, num_gripper),
    )
    return inputs, targets
