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

import logging

logging.disable(logging.INFO)

import os
import warnings
from copy import deepcopy
from pathlib import Path

from mgs.env.selector import get_env_from_dict
from mgs.util.geo.transforms import SE3Pose

import hydra
import jax
import numpy as np
import plotly.io as pio
from flax import nnx
from omegaconf import DictConfig, OmegaConf

import kin_flow.util.const as CONST
from kin_flow.alg.selector import inference
from kin_flow.ctrl.trainer import Trainer
from kin_flow.data.selector import get_gripper_loaders
from kin_flow.data.zipped import ZippedGripperLoader
from kin_flow.kin.const import KINS_NP
from kin_flow.kin.op_numpy import normalize_joints
from kin_flow.net.debug_wrapper import KinematicsFlowWOEncoding
from kin_flow.net.kinematics_flow import (KinematicsFlow,
                                          KinematicsFlowConfiguration)
from kin_flow.util.viz.grasp_viz import viz_grasps

pio.renderers.default = "browser"

# mujoco grasp sim defines for certain grippers a contact point. We need to
# translate predicted grasps from gripper base to contact coordinate system.
B2C_TRANSFORM = {
    "allegro": np.eye(4),
    "shadow": np.eye(4),
    "dexee": np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -0.17],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "vx300": np.array(
        [
            [0.0, 0.0, 1.0, 0.12],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "panda": np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.102],
            [0.0, 0.0, 0.0, 1.0],
        ],
    ),
}


def collision_free_mask(scene_def, grasps, b2c_transform):
    class cfg:
        name: str = "ClutterTable"

    env = get_env_from_dict(cfg(), (deepcopy(scene_def)))
    pose, joints = grasps
    pose = np.einsum("nij,jk->nik", pose, b2c_transform)
    collision_free_mask = env.grasp_collision_mask(
        SE3Pose.from_mat(deepcopy(pose), type="wxyz"),
        deepcopy(joints),
    )
    return collision_free_mask


def eval_grasps(scene_def, grasps, b2c_transform):
    class cfg:
        name: str = "ClutterTable"

    env = get_env_from_dict(cfg(), (deepcopy(scene_def)))
    pose, joints = grasps
    pose = np.einsum("nij,jk->nik", pose, b2c_transform)

    stable_grasp_mask = env.grasp_stable_mask(
        SE3Pose.from_mat(deepcopy(pose), type="wxyz"),
        joints,
        deepcopy(scene_def["env_state"]["state"]),
        show_progress=True,
    )

    num_total = float(len(pose))
    return (
        sum(stable_grasp_mask) / num_total,
        {"num_objects": len(env.object_names), "succ_mask": stable_grasp_mask},
    )


@hydra.main(config_path="config", config_name="bench", version_base="1.2")
def main(cfg: DictConfig):
    CONST.configure(cfg)
    rngs = nnx.Rngs(0)
    conf: dict = OmegaConf.to_container(cfg, resolve=True)
    conf_net = KinematicsFlowConfiguration.from_dict_cls(conf["model"]["configuration"])

    model = KinematicsFlow(
        conf_net,
        rngs=rngs,
    )

    model_path = Path(CONST.MODEL_CHECKPOINT) / cfg.checkpoint
    model = Trainer.get_model_from_checkpoint(model, model_path)

    loader_list = get_gripper_loaders(
        conf["model"]["configuration"]["gripper"],
        num_scenes=cfg.num_scenes,
        sub_dir="test",
    )
    g_id = cfg.gripper_id
    loader = ZippedGripperLoader(loader_list, all_samples=True, rngs=rngs)

    all_succ_rates = []
    all_devs = []
    for sample in loader:
        se3, dof = None, None

        if cfg.viz:
            se3, dof = inference(
                name=cfg.algorithm.name,
                model=model,
                sample=sample,
                num_samples=cfg.num_samples,
                cfg=cfg.algorithm.inference,
            )
            se3 = se3[:, -1, ...]
            dof = dof[:, -1, ...]
            gripper_name = cfg.gripper[g_id]
            scene_pcd = np.array(sample["scene_points"][g_id])
            scene_colors = np.array(sample["scene_colors"][g_id])
            kin_np = KINS_NP[gripper_name]  # DON'T USE!
            fig = viz_grasps(
                gripper_name=gripper_name,
                scene_pcd=scene_pcd,
                se3=se3,
                dof=dof,
                kin=kin_np,
                scene_colors=scene_colors,
            )
            fig.show()

        if cfg.gs:
            print("=" * 80)
            scene_dir = sample["path"][g_id]
            scene_path = os.path.join(scene_dir, "scene.npz")
            scene_dict = np.load(scene_path, allow_pickle=True)[
                "scene_definition"
            ].item()
            b2c = B2C_TRANSFORM[cfg.gripper[g_id]]

            collision_free_se3 = []
            collision_free_dof = []

            total_grasps = 0
            while (
                sum([cse3.shape[0] for cse3 in collision_free_se3])
                < cfg.num_samples
            ):
                se3, dof = inference(
                    name=cfg.algorithm.name,
                    model=model,
                    sample=sample,
                    num_samples=cfg.num_samples,
                    cfg=cfg.algorithm.inference,
                )
                se3 = se3[:, -1, ...]
                dof = dof[:, -1, ...]
                total_grasps += len(se3)
                mask = collision_free_mask(
                    scene_dict,
                    (se3, dof),
                    b2c,
                )
                if float(sum(mask)) / cfg.num_samples < 0.1:
                    warnings.warn("Most predicted grasps contain collisions!")
                collision_free_se3.append(se3[mask])
                collision_free_dof.append(dof[mask])

            se3 = np.concatenate(collision_free_se3, axis=0)[: cfg.num_samples]
            dof = np.concatenate(collision_free_dof, axis=0)[: cfg.num_samples]
            success_rate, aux = eval_grasps(scene_dict, (se3, dof), b2c)
            joint_diversity = np.mean(
                np.std(
                    normalize_joints(
                        dof[aux["succ_mask"]], KINS_NP[cfg.gripper[g_id]]
                    ),
                    axis=0,
                )
            )
            all_succ_rates.append(float(success_rate))
            all_devs.append(float(joint_diversity))

            print("Scene: ", scene_dir)
            print(
                "Success rate: ",
                success_rate,
                "NJD: ",
                joint_diversity
            )
            print("=" * 80)

    if cfg.gs:
        all_succ_rates = np.asarray(all_succ_rates)
        all_devs = np.asarray(all_devs)
        mean_succ_rate = np.mean(all_succ_rates)
        mean_njd = np.mean(all_devs)
        print(
            "TOTAL Mean success rate: ",
            mean_succ_rate,
            "TOTAL Mean NJD: ",
            mean_njd,
        )


if __name__ == "__main__":
    main()
