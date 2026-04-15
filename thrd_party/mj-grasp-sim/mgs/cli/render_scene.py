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
from copy import deepcopy

import hydra
import jax.numpy as jnp
import numpy as np
from mgs.env.selector import get_env_from_dict
from mgs.gripper.base import MjScannableGripper
from mgs.gripper.selector import get_gripper
from mgs.sampler.helper import farthest_point_sampling
from mgs.util.img_proc import detect_outlier, rgbd_to_pcd, voxel_downsample_pcd
from omegaconf import DictConfig


def scan(cfg: DictConfig, scene_def):
    gripper = get_gripper(cfg.gripper)
    assert isinstance(gripper, MjScannableGripper)
    env = get_env_from_dict(cfg.env, (deepcopy(scene_def)))
    images, extrinsics, image_masks, _ = env.scan(num_images=cfg.num_images)
    intrinsics = env.get_camera_intrinsics()
    return images, extrinsics, intrinsics, image_masks


@hydra.main(config_path="config", config_name="render_scene")
def main(cfg: DictConfig):
    output_dir = os.getenv("MGS_OUTPUT_DIR")
    input_dir = os.getenv("MGS_INPUT_DIR")

    assert output_dir is not None
    assert input_dir is not None
    input_dir = os.path.join(input_dir, cfg.gripper.name)
    scene_dir = [
        d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
    ][cfg.id]
    input_dir = os.path.join(input_dir, scene_dir)
    print("Scene dir: ", input_dir)

    scene_path = os.path.join(input_dir, "scene.npz")
    scene = np.load(scene_path, allow_pickle=True)
    scene_dict = scene["scene_definition"].item()
    images, extrinsics, intrinsics, image_masks = scan(
        deepcopy(cfg), deepcopy(scene_dict)
    )
    pcd, feature = rgbd_to_pcd(images, intrinsics, extrinsics)
    pcd = pcd[image_masks]
    feature = feature[image_masks]

    region_mask = np.all(
        (pcd < np.array([[0.225, 0.225, 1.0]]))
        & (pcd > np.array([[-0.225, -0.225, -0.01]])),
        axis=-1,
    )
    pcd = pcd[region_mask]
    feature = feature[region_mask]

    pcd, feature = voxel_downsample_pcd(pcd, feature, voxel_size=0.002)
    mask = detect_outlier(pcd, radius=0.008, min_neighbors=2)
    pcd, feature = pcd[mask], feature[mask]
    idx = farthest_point_sampling(
        jnp.asarray(pcd, dtype=jnp.float32), num_samples=15000
    )
    pcd = pcd[idx]
    feature = feature[idx]

    output_dir = os.path.join(output_dir, cfg.gripper.name, scene_dir)
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, "scene_pcd"),
        **{
            "points": np.asarray(pcd, dtype=np.float32),
            "colors": np.asarray(feature, dtype=np.float32),
        },
    )
    print("Finished!")


if __name__ == "__main__":
    main()
