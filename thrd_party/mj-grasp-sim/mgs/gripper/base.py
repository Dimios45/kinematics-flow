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

from abc import abstractmethod
from typing import Any, Dict, List, Protocol, Tuple, runtime_checkable

import mujoco
import numpy as np
from mgs.core.mj_xml import MjXml
from mgs.core.simualtion import MjSimulation
from mgs.util.geo.transforms import SE3Pose


class MjGripper(MjXml, Protocol):
    pos: np.ndarray
    quat: np.ndarray
    base: str

    def __init__(self, pose: SE3Pose, base_body: str):
        pose_vec = pose.to_vec(layout="pq", type="wxyz")
        pos, quat = pose_vec[:3], pose_vec[3:]

        self.pos = pos
        self.quat = quat
        self.base = base_body

    def set_load_pose(self, pose: SE3Pose):
        pose_vec = pose.to_vec(layout="pq", type="wxyz")
        pos, quat = pose_vec[:3], pose_vec[3:]

        self.pos = pos
        self.quat = quat

    def set_pose(self, sim: MjSimulation, pose: SE3Pose):
        idxs = self.get_freejoint_idxs(sim)

        pose_vec = pose.to_vec(layout="pq", type="wxyz")
        pos, quat = pose_vec[:3], pose_vec[3:]

        sim.data.qpos[idxs[:3]] = pos
        sim.data.qpos[idxs[3:]] = quat
        sim.data.mocap_pos[0, :] = pos
        sim.data.mocap_quat[0, :] = quat

        mujoco.mj_forward(sim.model, sim.data)  # type: ignore

    def lift_up(self, sim, viewer=None):
        for i in range(10000):
            sim.data.mocap_pos[0, 2] += 0.00003
            mujoco.mj_step(sim.model, sim.data)  # type: ignore
            if viewer is not None and i % 100 == 0:
                viewer.sync()

    @abstractmethod
    def to_xml(self) -> Tuple[str, Dict[str, Any]]: ...

    @abstractmethod
    def get_actuator_joint_names(
        self,
    ) -> List[str]: ...

    @abstractmethod
    def get_freejoint_idxs(self, sim: MjSimulation) -> List[int]: ...

    @abstractmethod
    def base_to_contact_transform(self) -> SE3Pose: ...


class OpenCloseGripper(Protocol):
    @abstractmethod
    def open_gripper(self, sim: MjSimulation):
        pass

    @abstractmethod
    def close_gripper_at(self, sim: MjSimulation, pose: SE3Pose):
        pass


@runtime_checkable
class MjScannable(Protocol):
    def get_render_options(self):
        options = mujoco.MjvOption()  # type: ignore
        return options


@runtime_checkable
class MjScannableGripper(MjGripper, MjScannable, Protocol):
    def __init__(self, pose: SE3Pose, base_body: str):
        super().__init__(pose, base_body)


class Shakable(Protocol):
    def move_back(self, sim: MjSimulation, pose: SE3Pose, viewer=None):
        rot_mat = pose.to_mat()[:3, :3]
        lift_direction = rot_mat @ np.array([0, 0, -1.0])
        for _ in range(1000):
            sim.data.mocap_pos[0, :] += 0.0002 * lift_direction
            mujoco.mj_step(sim.model, sim.data)  # type: ignore
            if viewer is not None:
                viewer.sync()

    def move_right(self, sim: MjSimulation, pose: SE3Pose, viewer=None):
        rot_mat = pose.to_mat()[:3, :3]
        shake_direction = rot_mat @ np.array([0, 1.0, 0])

        for _ in range(500):
            sim.data.mocap_pos[0, :] += 0.0005 * shake_direction
            mujoco.mj_step(sim.model, sim.data)  # type: ignore
            if viewer is not None:
                viewer.sync()

    def move_left(self, sim: MjSimulation, pose: SE3Pose, viewer=None):
        rot_mat = pose.to_mat()[:3, :3]
        shake_direction = rot_mat @ np.array([0, -1.0, 0])
        for _ in range(500):
            sim.data.mocap_pos[0, :] += 0.0005 * shake_direction
            mujoco.mj_step(sim.model, sim.data)  # type: ignore
            if viewer is not None:
                viewer.sync()

    def shake_grasp_at(self, sim: MjSimulation, pose: SE3Pose):
        self.move_back(sim, pose)
        self.move_right(sim, pose)
        self.move_left(sim, pose)


class MjShakableOpenCloseGripper(MjGripper, OpenCloseGripper, Shakable, Protocol):
    def __init__(self, pose: SE3Pose, base_body: str):
        super().__init__(pose, base_body)
