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

import os
from typing import Any, Dict, List, Tuple

import mujoco
import numpy as np
from mgs.core.simualtion import MjSimulation
from mgs.util.const import ASSET_PATH
from mgs.util.geo.transforms import SE3Pose

from .base import MjScannable, MjShakableOpenCloseGripper

# The XML template string is derived from MuJoCo Menagerie
# (https://github.com/google-deepmind/mujoco_menagerie/tree/469893211c41d5da9c314f5ab58059fa17c8e360)
# Copyright (c) 2016, SimLab License: BSD-2-Clause
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
XML = """
  <default>
    <default class="allegro_right">
      <joint axis="0 1 0" damping=".1"/>
      <position kp="1.5"/>
      <geom density="800"/>

      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2" material="black"/>
        <default class="palm_visual">
          <geom mesh="base_link"/>
        </default>
        <default class="base_visual">
          <geom mesh="link_0.0"/>
        </default>
        <default class="proximal_visual">
          <geom mesh="link_1.0"/>
        </default>
        <default class="medial_visual">
          <geom mesh="link_2.0"/>
        </default>
        <default class="distal_visual">
          <geom mesh="link_3.0"/>
        </default>
        <default class="fingertip_visual">
          <geom pos="0 0 0.0267" material="white" mesh="link_3.0_tip"/>
        </default>
        <default class="thumbtip_visual">
          <geom pos="0 0 0.0423" material="white" mesh="link_15.0_tip"/>
        </default>
      </default>

      <default class="collision">
        <geom group="3" type="box" mass="0" friction="3.0"/>
        <default class="palm_collision">
          <geom size="0.0204 0.0565 0.0475" pos="-0.0093 0 -0.0475"/>
        </default>
        <default class="base_collision">
          <geom size="0.0098 0.01375 0.0082" pos="0 0 0.0082"/>
          <default class="thumb_base_collision">
            <geom size="0.0179 0.017 0.02275" pos="-0.0179 0.009 0.0145"/>
          </default>
        </default>
        <default class="proximal_collision">
          <geom size="0.0098 0.01375 0.027" pos="0 0 0.027"/>
          <default class="thumb_proximal_collision">
            <geom size="0.0098 0.01375 0.00885" pos="0 0 0.00885"/>
          </default>
        </default>
        <default class="medial_collision">
          <geom size="0.0098 0.01375 0.0192" pos="0 0 0.0192"/>
          <default class="thumb_medial_collision">
            <geom size="0.0098 0.01375 0.0257" pos="0 0 0.0257"/>
          </default>
        </default>
        <default class="distal_collision">
          <geom size="0.0098 0.01375 0.008" pos="0 0 0.008"/>
          <default class="thumb_distal_collision">
            <geom size="0.0098 0.01375 0.0157" pos="0 0 0.0157"/>
          </default>
        </default>
        <default class="fingertip_collision">
          <geom type="capsule" size="0.012 0.01" pos="0 0 0.019"/>
          <default class="thumbtip_collision">
            <geom type="capsule" size="0.012 0.008" pos="0 0 0.035"/>
          </default>
        </default>
      </default>

      <default class="base">
        <joint axis="0 0 1" range="-0.47 0.47"/>
        <position ctrlrange="-0.47 0.47"/>
      </default>
      <default class="proximal">
        <joint range="-0.196 1.61"/>
        <position ctrlrange="-0.196 1.61"/>
      </default>
      <default class="medial">
        <joint range="-0.174 1.709"/>
        <position ctrlrange="-0.174 1.709"/>
      </default>
      <default class="distal">
        <joint range="-0.227 1.618"/>
        <position ctrlrange="-0.227 1.618"/>
      </default>
      <default class="thumb_base">
        <joint axis="-1 0 0" range="0.263 1.396"/>
        <position ctrlrange="0.263 1.396"/>
      </default>
      <default class="thumb_proximal">
        <joint axis="0 0 1" range="-0.105 1.163"/>
        <position ctrlrange="-0.105 1.163"/>
      </default>
      <default class="thumb_medial">
        <joint range="-0.189 1.644"/>
        <position ctrlrange="-0.189 1.644"/>
      </default>
      <default class="thumb_distal">
        <joint range="-0.162 1.719"/>
        <position ctrlrange="-0.162 1.719"/>
      </default>
    </default>
  </default>

  <asset>
    <material name="black" rgba="0.2 0.2 0.2 1"/>
    <material name="white" rgba="0.9 0.9 0.9 1"/>

    <mesh file="base_link.stl"/>
    <mesh file="link_0.0.stl"/>
    <mesh file="link_1.0.stl"/>
    <mesh file="link_2.0.stl"/>
    <mesh file="link_3.0.stl"/>
    <mesh file="link_3.0_tip.stl"/>
    <mesh file="link_12.0_right.stl"/>
    <mesh file="link_13.0.stl"/>
    <mesh file="link_14.0.stl"/>
    <mesh file="link_15.0.stl"/>
    <mesh file="link_15.0_tip.stl"/>
  </asset>

  <worldbody>
    <body name="mocap" mocap="true" pos="{position}" quat="{quaternion}"/>
    <body name="palm" pos="{position}" quat="{quaternion}" childclass="allegro_right">
      <freejoint name="freejoint" />
      <geom class="palm_visual" mesh="base_link"/>
      <geom class="palm_collision"/>
      <!-- First finger -->
      <body name="ff_base" pos="0 0.0435 -0.001542" quat="0.999048 -0.0436194 0 0">
        <joint name="ffj0" class="base"/>
        <geom class="base_visual"/>
        <geom class="base_collision"/>
        <body name="ff_proximal" pos="0 0 0.0164">
          <joint name="ffj1" class="proximal"/>
          <geom class="proximal_visual"/>
          <geom class="proximal_collision"/>
          <body name="ff_medial" pos="0 0 0.054">
            <joint name="ffj2" class="medial"/>
            <geom class="medial_visual"/>
            <geom class="medial_collision"/>
            <body name="ff_distal" pos="0 0 0.0384">
              <joint name="ffj3" class="distal"/>
              <geom class="distal_visual"/>
              <geom class="distal_collision"/>
              <body name="ff_tip">
                <geom class="fingertip_visual"/>
                <geom class="fingertip_collision"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <!-- Middle finger -->
      <body name="mf_base" pos="0 0 0.0007">
        <joint name="mfj0" class="base"/>
        <geom class="base_visual"/>
        <geom class="base_collision"/>
        <body name="mf_proximal" pos="0 0 0.0164">
          <joint name="mfj1" class="proximal"/>
          <geom class="proximal_visual"/>
          <geom class="proximal_collision"/>
          <body name="mf_medial" pos="0 0 0.054">
            <joint name="mfj2" class="medial"/>
            <geom class="medial_visual"/>
            <geom class="medial_collision"/>
            <body name="mf_distal" pos="0 0 0.0384">
              <joint name="mfj3" class="distal"/>
              <geom class="distal_visual"/>
              <geom class="distal_collision"/>
              <body name="mf_tip">
                <geom class="fingertip_visual"/>
                <geom class="fingertip_collision"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <!-- Ring finger -->
      <body name="rf_base" pos="0 -0.0435 -0.001542" quat="0.999048 0.0436194 0 0">
        <joint name="rfj0" class="base"/>
        <geom class="base_visual"/>
        <geom class="base_collision"/>
        <body name="rf_proximal" pos="0 0 0.0164">
          <joint name="rfj1" class="proximal"/>
          <geom class="proximal_visual"/>
          <geom class="proximal_collision"/>
          <body name="rf_medial" pos="0 0 0.054">
            <joint name="rfj2" class="medial"/>
            <geom class="medial_visual"/>
            <geom class="medial_collision"/>
            <body name="rf_distal" pos="0 0 0.0384">
              <joint name="rfj3" class="distal"/>
              <geom class="distal_visual"/>
              <geom class="distal_collision"/>
              <body name="rf_tip">
                <geom class="fingertip_visual"/>
                <geom class="fingertip_collision"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <!-- Thumb -->
      <body name="th_base" pos="-0.0182 0.019333 -0.045987" quat="0.477714 -0.521334 -0.521334 -0.477714">
        <joint name="thj0" class="thumb_base"/>
        <geom class="visual" mesh="link_12.0_right"/>
        <geom class="thumb_base_collision"/>
        <body name="th_proximal" pos="-0.027 0.005 0.0399">
          <joint name="thj1" class="thumb_proximal"/>
          <geom class="visual" mesh="link_13.0"/>
          <geom class="thumb_proximal_collision"/>
          <body name="th_medial" pos="0 0 0.0177">
            <joint name="thj2" class="thumb_medial"/>
            <geom class="visual" mesh="link_14.0"/>
            <geom class="thumb_medial_collision"/>
            <body name="th_distal" pos="0 0 0.0514">
              <joint name="thj3" class="thumb_distal"/>
              <geom class="visual" mesh="link_15.0"/>
              <geom class="thumb_distal_collision"/>
              <body name="th_tip">
                <geom class="thumbtip_visual"/>
                <geom class="thumbtip_collision"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <equality>
    <weld body1="mocap" body2="palm"/>
  </equality>

  <contact>
    <exclude body1="palm" body2="ff_base"/>
    <exclude body1="palm" body2="mf_base"/>
    <exclude body1="palm" body2="rf_base"/>
    <exclude body1="palm" body2="th_base"/>
    <exclude body1="palm" body2="th_proximal"/>
  </contact>

  <actuator>
    <position name="ffa0" joint="ffj0" class="base"/>
    <position name="ffa1" joint="ffj1" class="proximal"/>
    <position name="ffa2" joint="ffj2" class="medial"/>
    <position name="ffa3" joint="ffj3" class="distal"/>
    <position name="mfa0" joint="mfj0" class="base"/>
    <position name="mfa1" joint="mfj1" class="proximal"/>
    <position name="mfa2" joint="mfj2" class="medial"/>
    <position name="mfa3" joint="mfj3" class="distal"/>
    <position name="rfa0" joint="rfj0" class="base"/>
    <position name="rfa1" joint="rfj1" class="proximal"/>
    <position name="rfa2" joint="rfj2" class="medial"/>
    <position name="rfa3" joint="rfj3" class="distal"/>
    <position name="tha0" joint="thj0" class="thumb_base"/>
    <position name="tha1" joint="thj1" class="thumb_proximal"/>
    <position name="tha2" joint="thj2" class="thumb_medial"/>
    <position name="tha3" joint="thj3" class="thumb_distal"/>
  </actuator>
"""


class GripperAllegro(MjShakableOpenCloseGripper, MjScannable):
    def __init__(self, pose: SE3Pose):
        super().__init__(pose, "palm")
        self.open_pose = np.array(
            [
                -0.08,
                0.715,
                0.710,
                0.95,
                0,
                0.8,
                0.71,
                0.67,
                0.08,
                0.715,
                0.710,
                0.95,
                1.4,
                0.55,
                -0.19,
                1.45,
            ]
        )
        self.close_pose = np.array(
            [
                -0.08,
                0.95,
                1,
                0.95,
                0,
                0.95,
                1.2,
                0.85,
                0.08,
                0.95,
                1.2,
                0.9,
                1.4,
                0.55,
                0.29,
                1.45,
            ]
        )

    def base_to_contact_transform(self) -> SE3Pose:
        pos = np.array([0, 0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        return SE3Pose(pos, quat, type="wxyz")

    def open_gripper(self, sim: MjSimulation):
        gripper_idxs = sim.get_joint_idxs(self.get_actuator_joint_names())
        sim.set_qpos(np.copy(self.open_pose), gripper_idxs)  # type: ignore
        sim.data.ctrl[:] = np.copy(self.open_pose)

    def close_gripper_at(self, sim: MjSimulation, pose: SE3Pose):
        self.set_pose(sim, pose)
        sim.data.ctrl[:] = np.copy(self.close_pose)
        mujoco.mj_step(sim.model, sim.data, 3000)  # type: ignore

    def to_xml(self) -> Tuple[str, Dict[str, Any]]:
        pos = "{} {} {}".format(self.pos[0], self.pos[1], self.pos[2])
        quat = "{} {} {} {}".format(
            self.quat[0], self.quat[1], self.quat[2], self.quat[3]
        )
        xml = XML.format(
            **{
                "position": pos,
                "quaternion": quat,
            }
        )

        ASSETS = dict()
        base_path = os.path.join(ASSET_PATH, "allegro")
        for file_name in os.listdir(base_path):
            path = os.path.join(base_path, file_name)
            with open(path, "rb") as f:
                ASSETS[file_name] = f.read()

        return (xml, ASSETS)

    def get_actuator_joint_names(self) -> List[str]:
        return [
            "ffj0",
            "ffj1",
            "ffj2",
            "ffj3",
            "mfj0",
            "mfj1",
            "mfj2",
            "mfj3",
            "rfj0",
            "rfj1",
            "rfj2",
            "rfj3",
            "thj0",
            "thj1",
            "thj2",
            "thj3",
        ]

    def get_freejoint_idxs(self, sim: MjSimulation) -> List[int]:
        start_idx = sim.get_joint_idxs(["freejoint"])[0]
        return list(range(start_idx, start_idx + 7))
