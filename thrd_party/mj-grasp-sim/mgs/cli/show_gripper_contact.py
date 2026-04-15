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

import math

import hydra
import numpy as np
from mgs.env.gravityless_object_grasping import GravitylessObjectGrasping
from mgs.gripper.selector import get_gripper
from mgs.obj.marker import Marker
from mgs.util.geo.transforms import SE3Pose
from omegaconf import DictConfig


def setup(gripper):
    marker_pose = SE3Pose(
        pos=np.array([0, 0, 0]), quat=np.array([1, 0, 0, 0]), type="wxyz"
    )
    marker = Marker(pose=marker_pose, name="contact_marker")
    scene = GravitylessObjectGrasping(gripper, marker)  # type: ignore
    qpos = np.array([0, -1.3963, 0, 0, 0, -1.3963, 0, 0, 0, -1.3963, 0, 0])
    qpos = np.array([0.0, 0.04])
    scene.idle_grasp(
        pose=SE3Pose(
            pos=np.array([0, 0, 0]),
            quat=np.array([1, 0, 0, 0]),
            type="wxyz",
        ),
        joints=qpos,
    )


@hydra.main(config_path="config", config_name="show_gripper_contact")
def main(cfg: DictConfig):
    gripper = get_gripper(cfg.gripper)
    setup(gripper)


if __name__ == "__main__":
    main()
