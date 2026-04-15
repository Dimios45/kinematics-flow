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
from scipy.spatial.transform import Rotation

import kin_flow.util.const as CONST
from kin_flow.kin.const import KINS_NP
from kin_flow.kin.op_numpy import (denormalize_joints, kinematic_frames,
                                   normalize_joints)
from kin_flow.kin.wigner_d import wigner_D_real
from kin_flow.util.dist import SO3_uniform_R3_normal_np


@nnx.jit
def encode_scene(model_graph, model_state, scene):
    model = nnx.merge(model_graph, model_state)
    scene_points, _ = scene
    scene_points = scene_points * model.position_scaling
    multiscale_feature_pyramid, _, _ = model.unet(scene_points, jax.random.PRNGKey(0))
    return multiscale_feature_pyramid


@nnx.jit(static_argnames=["condition", "beta", "z0_id", "gripper_idx"])
def decode_flow(
    model_graph,
    model_state,
    scene_features,
    gripper_idx,
    wigner_d: jnp.ndarray,
    q_pos_rot: jnp.ndarray,
    q_pos_rot_trans: jnp.ndarray,
    t: jnp.ndarray,
    condition=False,
    beta=1.0,
    z0_id=None,
):
    def model_call(
        graph,
        state,
        scene_features,
        gripper_idx,
        wigner_d,
        q_pos_rot,
        q_pos_rot_trans,
        t,
    ):
        model = nnx.merge(graph, state)
        q_feat, q_pos, valid = model.kinematic_encoding(
            gripper_idx, wigner_d, q_pos_rot, t
        )

        # Relate scene and query
        feat, _ = model.multi_scale_feature_query(
            scene_features, ((q_pos_rot_trans, q_pos), q_feat), t, valid
        )

        # Decode flow
        u = model.decoder(feat, q_pos, valid, t)

        dof_valid_mask = (
            model.kinematic_encoding.kinematic_net.valid_mask_with_static_flag(
                gripper_idx
            )[1:]
        )  # exclude base node
        u = u.at[..., 6:].set(u[..., 6:] * dof_valid_mask)
        return u

    if not condition:
        N = len(t)
        gripper_idx = jnp.full(shape=(N,), fill_value=gripper_idx)
        u = jax.vmap(model_call, in_axes=(None, None, None, 0, 0, 0, 0, 0))(
            model_graph,
            model_state,
            scene_features,
            gripper_idx,
            wigner_d,
            q_pos_rot,
            q_pos_rot_trans,
            t,
        )
        return u
    else:
        N = len(t)
        gripper_idx = jnp.concatenate(
            [
                jnp.full(shape=(N,), fill_value=gripper_idx),
                jnp.full(shape=(N,), fill_value=z0_id),
            ],
        )
        u = jax.vmap(model_call, in_axes=(None, None, None, 0, 0, 0, 0, 0))(
            model_graph,
            model_state,
            scene_features,
            gripper_idx,
            [jnp.concatenate([l_w, l_w], axis=0) for l_w in wigner_d],
            jnp.concatenate([q_pos_rot, q_pos_rot], axis=0),
            jnp.concatenate([q_pos_rot_trans, q_pos_rot_trans], axis=0),
            jnp.concatenate([t, t], axis=0),
        )
        u_c, u_0 = u[:N], u[N:]
        beta = jnp.where(t[0, 0] <= 0.8, beta, 1.0)
        u_c = u_c.at[:, :6].set(beta * u_c[:, :6] + (1 - beta) * u_0[:, :6])
        return u_c


class FrameFlowModel:

    def __init__(self, model):
        self.mg, self.ms = nnx.split(model)

    def __call__(self, z, t, x_t):
        rot_t, pos_t, dof_t = x_t
        rot_t_obj = Rotation.from_matrix(rot_t)
        B = rot_t.shape[0]

        kin = z["gripper_list"][z["gripper_idx"]]
        denorm_dof_t = denormalize_joints(dof_t, kin)
        rot_t = np.asarray(rot_t, dtype=np.float64)[:, None, :, :]
        joint_rot, joint_pos = kinematic_frames(
            denorm_dof_t, kin
        )  # shapes (B, J, 3, 3), (B, J, 3)
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
        q_pos_rot = (
            np.einsum("bij,bqj->bqi", rot_t[:, 0], q_pos) * z["position_scaling"]
        )
        q_pos_rot_trans = q_pos_rot + pos_t[:, None, :]

        padding_wd = [
            wigner_D_real(l=i + 1, R=np.eye(3)[None, None]) for i in range(CONST.L_MAX)
        ]
        to_pad = (CONST.MAX_DOF + 3) - q_pos_rot.shape[1]
        wigner_d = [
            np.concatenate(
                [
                    wd,
                    np.broadcast_to(
                        pwd, shape=(B, to_pad, 2 * (i + 1) + 1, 2 * (i + 1) + 1)
                    ),
                ],
                axis=1,
            )
            for (i, (pwd, wd)) in enumerate(zip(padding_wd, wigner_d))
        ]
        q_pos_rot = np.concatenate([q_pos_rot, np.zeros(shape=(B, to_pad, 3))], axis=-2)
        q_pos_rot_trans = np.concatenate(
            [q_pos_rot_trans, np.zeros(shape=(B, to_pad, 3))], axis=-2
        )

        if z["scene_features"] is None:
            z["scene_features"] = encode_scene(self.mg, self.ms, z["scene"])

        u = decode_flow(
            self.mg,
            self.ms,
            z["scene_features"],
            z["gripper_idx"],
            wigner_d,
            q_pos_rot,
            np.concatenate([q_pos_rot_trans[:, :2], q_pos_rot_trans[:, 3:]], axis=1),
            t,
            condition=(z["z0_id"] != -1),
            z0_id=z["z0_id"],
            beta=2.0,
        )
        u = np.array(u)

        u_rot_scene = u[..., :3]
        u_rot_body = rot_t_obj.inv().apply(u_rot_scene)
        u_pos_scene = u[..., 3:6]

        u_dof = u[..., 6:]

        return u_rot_body, u_pos_scene, u_dof


class FrameFlowIntegrator:
    def __init__(self, num_steps, position_scaling=1.0):
        self.num_steps = num_steps
        self.position_scaling = position_scaling
        self.dof = CONST.MAX_DOF

        self.t = np.linspace(0.0, 1.0, num_steps + 1)
        self.t[-1] = 0.99

    def __call__(self, z, x_0, model_fn):
        rot_t, pos_t, dof_t = x_0
        batch_size = rot_t.shape[0]

        traj_se3 = np.zeros((batch_size, len(self.t) + 1, 4, 4))
        traj_dof = np.zeros((batch_size, len(self.t) + 1, self.dof))

        traj_se3[:, 0, :3, :3] = rot_t
        traj_se3[:, 0, :3, 3] = pos_t
        traj_se3[:, 0, 3, 3] = 1.0
        traj_dof[:, 0] = dof_t

        for i, t_curr in enumerate(self.t):
            dt = 1 / self.num_steps
            t_n = np.full((batch_size, 1), t_curr)
            u_rot, u_pos, u_dof = model_fn(z, t_n, (rot_t, pos_t, dof_t))
            denom = max(1.0 - t_curr, 1e-3)

            v_pos = u_pos / denom
            v_rot = u_rot / denom
            v_dof = u_dof / denom

            if i == len(self.t) - 1:
                # jump to last
                exp_u = Rotation.from_rotvec(u_rot).as_matrix()
                pred_rot_1 = np.einsum("bij,bjk->bik", rot_t, exp_u)

                traj_se3[:, i + 1, :3, :3] = pred_rot_1
                traj_se3[:, i + 1, :3, 3] = pos_t + u_pos  # Add remaining distance
                traj_se3[:, i + 1, 3, 3] = 1.0
                traj_dof[:, i + 1] = dof_t + u_dof
                break

            pos_t = pos_t + v_pos * dt
            dof_t = dof_t + v_dof * dt

            exp_step = Rotation.from_rotvec(v_rot * dt).as_matrix()
            rot_t = np.einsum("bij,bjk->bik", rot_t, exp_step)

            traj_se3[:, i + 1, :3, :3] = rot_t
            traj_se3[:, i + 1, :3, 3] = pos_t
            traj_se3[:, i + 1, 3, 3] = 1.0
            traj_dof[:, i + 1] = dof_t

        traj_se3[..., :3, 3] *= 1.0 / self.position_scaling
        return traj_se3, traj_dof


def inference(model, sample, num_samples, cfg):
    gripper_id = cfg.gripper_id
    gripper_kin = KINS_NP[cfg.gripper[gripper_id]]
    z0_id = cfg.z0_id
    scene_pcd = (
        jnp.array(sample["scene_points"][gripper_id]),
        jnp.array(sample["scene_colors"][gripper_id]),
    )

    # Initialize Noise
    rot_0 = SO3_uniform_R3_normal_np(num_samples)[:, :3, :3]
    pos_0 = np.random.normal(size=(num_samples, 3))

    # Initialize DOF (Random [-1, 1])
    dof_0 = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, CONST.MAX_DOF))
    flow_model = FrameFlowModel(model)

    integrator = FrameFlowIntegrator(
        num_steps=cfg.integrator_steps,
        position_scaling=cfg.position_scaling,
    )

    context = {
        "scene": scene_pcd,
        "gripper_idx": gripper_id,
        "z0_id": z0_id,
        "z0_beta": 2,
        "scene_features": None,
        "gripper_list": [KINS_NP[id] for id in cfg.gripper],
        "position_scaling": cfg.position_scaling,
    }

    se3_traj, dof_traj = integrator(context, (rot_0, pos_0, dof_0), flow_model)
    dof_traj_denorm = denormalize_joints(dof_traj, gripper_kin)
    dof_mask = model.kinematic_encoding.kinematic_net.valid_mask(
        jnp.asarray(gripper_id)
    )[1:]
    dof_traj_final = dof_traj_denorm[..., dof_mask]

    return se3_traj, np.array(dof_traj_final)
