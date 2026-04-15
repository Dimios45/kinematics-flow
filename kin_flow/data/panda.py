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

import numpy as np

from kin_flow.data.base import CachedSceneLoader, SceneLoader


class PandaSceneLoader(SceneLoader):
    def __init__(
        self,
        data_dir,
        num_scenes,
        shuffle=True,
        shuffle_grasps=True,
        sub_dir="train",
        max_load_grasps=1500,
    ):
        super().__init__(
            data_dir=data_dir,
            gripper_dir="PandaGripper",
            num_scenes=num_scenes,
            sub_dir=sub_dir,
            shuffle=shuffle,
            shuffle_grasps=shuffle_grasps,
            max_load_grasps=max_load_grasps,
        )
        self.b2c = np.array(
            [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -0.102],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )


class CachedPandaSceneLoader(CachedSceneLoader):
    def __init__(self, panda_scene_loader, rot_augmentation=False):

        super().__init__(rot_augmentation=rot_augmentation)
        self.scene_loader = panda_scene_loader

        pre_load = [data for data in self.scene_loader]
        self.scenes = [(data["points"], data["colors"]) for data in pre_load]
        self.grasps = [
            (
                data["rot"],
                data["pos"],
                data["joints"],
            )
            for data in pre_load
        ]
        self.dir_paths = [data["dir_path"] for data in pre_load]
