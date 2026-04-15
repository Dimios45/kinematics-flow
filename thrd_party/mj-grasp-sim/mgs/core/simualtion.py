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

from dataclasses import dataclass
from typing import List

import mujoco
import mujoco.viewer
import numpy as np
from mujoco import MjData, MjModel  # type: ignore


@dataclass
class MjSimulation:
    data: MjData
    model: MjModel

    def idle(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while True:
                mujoco.mj_step(self.model, self.data)  # type: ignore
                viewer.sync()

    def get_joint_idxs(self, joint_list: List[str]) -> List[int]:
        id_list = []
        for j in joint_list:
            id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)  # type: ignore
            id_list.append(self.model.jnt_qposadr[id])

        return id_list

    def set_qpos(self, qpos: np.ndarray, idxs: List[int]):
        assert idxs is not None

        self.data.qpos[idxs] = qpos
        mujoco.mj_forward(self.model, self.data)  # type: ignore

    def get_state(self):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION  # type: ignore
        size = mujoco.mj_stateSize(self.model, spec)  # type: ignore
        current_state = np.empty(size, dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, current_state, spec)  # type: ignore
        return current_state

    def set_state(self, state):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION  # type: ignore
        mujoco.mj_setState(self.model, self.data, state, spec)  # type: ignore
        mujoco.mj_forward(self.model, self.data)  # type: ignore
