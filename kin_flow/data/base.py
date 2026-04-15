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
import random

import numpy as np
from scipy.spatial.transform import Rotation


class SceneLoader:
    def __init__(
        self,
        data_dir,
        gripper_dir,
        num_scenes,
        sub_dir="train",
        shuffle=True,
        shuffle_grasps=True,
        max_load_grasps=1500,
    ):
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.shuffle_grasps = shuffle_grasps

        scene_dir = os.path.join(data_dir, sub_dir, gripper_dir)
        self.scene_dir_path = scene_dir
        scene_dirs = [
            item
            for item in os.listdir(scene_dir)
            if os.path.isdir(os.path.join(scene_dir, item))
        ]

        self.scene_dirs = [
            sd
            for sd in scene_dirs
            if os.path.exists(os.path.join(scene_dir, sd, "scene_pcd.npz"))
        ][:num_scenes]
        self.scene_dirs.sort()
        if num_scenes <= 10:
            print(self.scene_dirs)
        self.max_load_grasps = max_load_grasps

    def __iter__(self):
        all_scenes = self.scene_dirs
        if self.shuffle:
            random.shuffle(all_scenes)

        for scene_dir in all_scenes:
            full_scene_dir_path = os.path.join(self.scene_dir_path, scene_dir)
            pcd_path = os.path.join(full_scene_dir_path, "scene_pcd.npz")
            pcd = np.load(pcd_path)
            points = pcd["points"]
            colors = pcd["colors"]

            grasp_files = os.listdir(full_scene_dir_path)
            grasp_files = [
                f
                for f in os.listdir(full_scene_dir_path)
                if f != "scene_pcd.npz"
                and f != "scene.npz"
                and f.endswith(".npz")
                and not f.endswith("_collision.npz")
            ]
            all_poses = []
            all_joints = []

            for grasp_file in grasp_files:
                grasp_path = os.path.join(full_scene_dir_path, grasp_file)
                grasps = np.load(grasp_path)
                poses = grasps["pose"] @ self.b2c
                joints = grasps["joints"]

                all_poses.append(poses)
                all_joints.append(joints)

            if len(all_poses) == 0:
                all_rotations = np.empty((0, 3, 3), dtype=np.float64)
                all_positions = np.empty((0, 3), dtype=np.float64)
                all_joints = np.empty((0, 0), dtype=np.float64)
            else:
                all_poses = np.concatenate(all_poses, axis=0)[: self.max_load_grasps]
                all_joints = np.concatenate(all_joints, axis=0)[: self.max_load_grasps]

                if self.shuffle_grasps:
                    indices = np.random.permutation(len(all_poses))
                    all_poses = all_poses[indices]
                    all_joints = all_joints[indices]

                all_rotations = all_poses[:, :3, :3]
                all_positions = all_poses[:, :3, 3]

            yield {
                "dir_path": full_scene_dir_path,
                "points": points,
                "colors": colors,
                "rot": all_rotations,
                "pos": all_positions,
                "joints": all_joints,
            }


class CachedSceneLoader:
    def __init__(self, rot_augmentation=False):
        self.rot_augmentation = rot_augmentation

    def __iter__(self):
        idx = np.arange(len(self.scenes))
        np.random.shuffle(idx)

        R = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        if self.rot_augmentation:
            angle = np.random.uniform(-180, 180)
            angle_rad = np.radians(angle)
            R = Rotation.from_euler("z", angle_rad).as_matrix()

        for i in idx:
            scene = self.scenes[i]
            scene = (np.einsum("ij,nj->ni", R, scene[0]), scene[1])
            rot, pos, joints = self.grasps[i]

            if rot.shape[0] > 1:
                g_idx = np.random.permutation(rot.shape[0])
                rot = rot[g_idx]
                rot = np.einsum("ij,njk->nik", R, rot)
                pos = pos[g_idx]
                pos = np.einsum("ij,nj->ni", R, pos)
                joints = joints[g_idx]

            dir_path = self.dir_paths[i]

            yield (scene, (rot, pos, joints), dir_path)
