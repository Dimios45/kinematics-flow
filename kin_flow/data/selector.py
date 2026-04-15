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

import kin_flow.util.const as CONST
from kin_flow.data.allegro import AllegroSceneLoader, CachedAllegroSceneLoader
from kin_flow.data.dexee import CachedDexeeSceneLoader, DexeeSceneLoader
from kin_flow.data.panda import CachedPandaSceneLoader, PandaSceneLoader
from kin_flow.data.shadow import CachedShadowSceneLoader, ShadowSceneLoader
from kin_flow.data.vx import CachedVX300SceneLoader, VX300SceneLoader


def get_gripper_loaders(
    grippers: list[str], num_scenes, sub_dir="train", rot_augmentation=False
):

    shuffle = True if sub_dir == "train" else False
    loaders = []
    for i, gripper in enumerate(grippers):
        if gripper == "panda":
            loader = PandaSceneLoader(
                CONST.DATA_PATH,
                shuffle=shuffle,
                shuffle_grasps=shuffle,
                sub_dir=sub_dir,
                num_scenes=num_scenes,
            )
            loader = CachedPandaSceneLoader(loader, rot_augmentation=rot_augmentation)
            loaders.append(loader)
        elif gripper == "vx300":
            loader = VX300SceneLoader(
                CONST.DATA_PATH,
                shuffle=shuffle,
                shuffle_grasps=shuffle,
                sub_dir=sub_dir,
                num_scenes=num_scenes,
            )
            loader = CachedVX300SceneLoader(loader, rot_augmentation=rot_augmentation)
            loaders.append(loader)
        elif gripper == "dexee":
            loader = DexeeSceneLoader(
                CONST.DATA_PATH,
                num_scenes=num_scenes,
                shuffle=shuffle,
                shuffle_grasps=shuffle,
                sub_dir=sub_dir,
            )
            loader = CachedDexeeSceneLoader(loader, rot_augmentation=rot_augmentation)
            loaders.append(loader)
        elif gripper == "allegro":
            loader = AllegroSceneLoader(
                CONST.DATA_PATH,
                shuffle=shuffle,
                shuffle_grasps=shuffle,
                sub_dir=sub_dir,
                num_scenes=num_scenes,
            )
            loader = CachedAllegroSceneLoader(loader, rot_augmentation=rot_augmentation)
            loaders.append(loader)
        elif gripper == "shadow":
            loader = ShadowSceneLoader(
                CONST.DATA_PATH,
                shuffle=shuffle,
                shuffle_grasps=shuffle,
                sub_dir=sub_dir,
                num_scenes=num_scenes,
            )
            loader = CachedShadowSceneLoader(loader, rot_augmentation=rot_augmentation)
            loaders.append(loader)
        elif gripper == "z0":
            # z0 has to be the last gripper in the list
            assert (i + 1) == len(grippers)
        else:
            raise ValueError("Not a known gripper: " + gripper)
    return loaders
