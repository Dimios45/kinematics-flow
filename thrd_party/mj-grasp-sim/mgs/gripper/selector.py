# Copyright (c) 2025 Robert Bosch GmbH
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
from mgs.gripper.allegro import GripperAllegro
from mgs.gripper.base import MjShakableOpenCloseGripper
from mgs.gripper.dexee import GripperDexee
from mgs.gripper.panda import GripperPanda
from mgs.gripper.shadow import GripperShadowRight
from mgs.gripper.vx300 import GripperVX300
from mgs.util.geo.transforms import SE3Pose
from omegaconf import DictConfig


def get_gripper(cfg: DictConfig, default_pose=None) -> MjShakableOpenCloseGripper:
    pose = (
        SE3Pose(np.array([0, 0, 0]), np.array([1, 0, 0, 0]), type="wxyz")
        if default_pose is None
        else default_pose
    )
    if cfg.name == "PandaGripper":
        return GripperPanda(pose)

    if cfg.name == "ShadowHand":
        return GripperShadowRight(pose)

    if cfg.name == "VXGripper":
        return GripperVX300(pose)

    if cfg.name == "DexeeGripper":
        return GripperDexee(pose)

    if cfg.name == "AllegroGripper":
        return GripperAllegro(pose)

    raise ValueError(f"Unknown gripper: {cfg.name}")
