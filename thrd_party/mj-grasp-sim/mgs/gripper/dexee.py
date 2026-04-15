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
# Copyright 2022 Shadow Robot Company Ltd License: Apache-2.0
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
XML = """
  <default>
    <site group="3" rgba="1 0 0 1" size="0.001"/>
    <default class="finger/visual">
      <geom type="mesh" contype="0" conaffinity="0"/>
    </default>
    <default class="finger/collision">
      <geom type="mesh" group="3" material="collision" mass="0"/>
      <default class="finger/collision_soft">
        <geom condim="6" friction="1 0.005 0.0001" solref="-2500 -100"/>
      </default>
      <default class="finger/collision_hard">
        <geom condim="4" friction="1 0.001 2e-05" solref="-7000 -167"/>
      </default>
    </default>
  </default>

  <asset>
    <material name="white_aluminium" rgba="0.6313 0.6313 0.62745 1"/>
    <material name="graphite_black" rgba="0.1529 0.16078 0.16862 1"/>
    <material name="jet_black" rgba="0.0549 0.0549 0.06274 1"/>
    <material name="collision" rgba="1 0.6 0.6 1"/>

    <mesh name="r3_config1_base" file="Asm-MRH-HB1-Visual,00-Plastic.stl"/>
    <mesh name="r3_config1_base_col" file="Asm-MRH-HB1-Visual,00-Plastic.stl"/>
    <mesh name="r3_config1_base_puck" file="Asm-MRH-HB1-Visual,00-Puck.stl"/>
    <mesh name="r3_config1_base_puck_col" file="Asm-MRH-HB1-Visual,00-Puck.stl"/>
    <mesh name="r3_finger_base" file="MRH-FB-MainALU-Visual,00.stl"/>
    <mesh name="r3_finger_base_tb" file="MRH-FB-TB-Main-Visual,00.stl"/>
    <mesh name="r3_finger_cover" file="MRH-FB-Main-Cover-Visual,00.stl"/>
    <mesh name="r3_finger_pulley_block" file="MRH-FB-PulleyBlock1-Visual,00.stl"/>
    <mesh name="r3_finger_lid_white" file="MRH-FB-TB-Lid-White-Visual,00.stl"/>
    <mesh name="r3_finger_lid_logo" file="MRH-FB-TB-Lid-Logo-Visual,00.stl"/>
    <mesh name="r3_finger_cover_small_left" file="MRH-FB-S-Cover-L-Visual,00.stl"/>
    <mesh name="r3_finger_cover_small_right" file="MRH-FB-S-Cover-R-Visual,00.stl"/>
    <mesh name="r3_finger_base_col" file="r3_finger_base_col.stl"/>
    <mesh name="r3_finger_knuckle" file="MRH-F-J0Link-Visual,00.stl" refquat="1 -1 0 0"/>
    <mesh name="r3_finger_knuckle_col" file="MRH-F-J0Link-Visual,00.stl" refquat="1 -1 0 0"/>
    <mesh name="r3_finger_proximal" file="MRH-F-Prox-Visual,00-Main.stl" refquat="1 -1 0 0"/>
    <mesh name="r3_finger_proximal_lid" file="MRH-F-Prox-Visual,00-Lid.stl" refquat="1 -1 0 0"/>
    <mesh name="r3_finger_proximal_magtac" file="MRH-F-Prox-MagTac-Sensor-Visual,00.stl" refquat="1 -1 0 0"/>
    <mesh name="r3_finger_proximal_col" file="Asm-MRH-F-Prox-Visual,00+Magtac,00.stl" refquat="1 -1 0 0"/>
    <mesh name="r3_finger_middle" file="MRH-F-Mid-Visual,00.stl" refquat="1 -1 0 0"/>
    <mesh name="r3_finger_middle_magtac" file="MRH-F-Mid-MagTac-Sensor-Visual,00.stl" refquat="1 -1 0 0"/>
    <mesh name="r3_finger_middle_col" file="Asm-MRH-F-Mid-Visual,00+MagTac,00.stl" refquat="1 -1 0 0"/>
    <mesh name="r3_finger_distal" file="MRH-F-Distal-Visual,00.stl" refquat="0 0 0 1"/>
    <mesh name="r3_finger_distal_sensor" file="MRH-F-Distal-Sensor-Visual,00.stl" refquat="0 -1 0 0"/>
    <mesh name="r3_finger_distal_col" file="MRH-F-Distal-Visual,00.stl" refquat="0 0 0 1"/>
    <mesh name="r3_finger_distal_tip_col" file="MRH-F-Distal-Sensor-Visual,00.stl" refquat="0 -1 0 0"/>
  </asset>

  <extension>
    <plugin plugin="mujoco.pid">
      <instance name="actuator_J0">
        <config key="kp" value="2.8"/>
        <config key="ki" value="4.0"/>
        <config key="kd" value="0.03"/>
        <config key="imax" value="0.1"/>
        <config key="slewmax" value="3.14159"/>
      </instance>
    </plugin>
    <plugin plugin="mujoco.pid">
      <instance name="actuator_J1">
        <config key="kp" value="2.5"/>
        <config key="ki" value="3.0"/>
        <config key="kd" value="0.02"/>
        <config key="imax" value="0.2"/>
        <config key="slewmax" value="3.14159"/>
      </instance>
    </plugin>
    <plugin plugin="mujoco.pid">
      <instance name="actuator_J2">
        <config key="kp" value="1.1"/>
        <config key="ki" value="3.0"/>
        <config key="kd" value="0.01"/>
        <config key="imax" value="0.2"/>
        <config key="slewmax" value="3.14159"/>
      </instance>
    </plugin>
    <plugin plugin="mujoco.pid">
      <instance name="actuator_J3">
        <config key="kp" value="0.6"/>
        <config key="ki" value="3.0"/>
        <config key="kd" value="0.008"/>
        <config key="imax" value="0.1"/>
        <config key="slewmax" value="3.14159"/>
      </instance>
    </plugin>
  </extension>

  <worldbody>
    <body name="mocap" mocap="true" pos="{position}" quat="{quaternion}"/>
    <body name="dexee_gripper" pos="{position}" quat="{quaternion}" gravcomp="1">
      <freejoint name="freejoint"/>
      <inertial mass="0.001" pos="0 0 0" diaginertia="5e-05 5e-05 5e-05"/>
      <site name="tcp" type="sphere" rgba="1 0.5 0 0.2" size="0.001" pos="0 0 0.3"/>
      <site name="attachment_site" type="sphere" group="3" rgba="1 0 0 1" size="0.001"/>
      <body name="hand_base" gravcomp="1">
        <geom name="hand_base_geom" material="jet_black" type="mesh" contype="0" conaffinity="0" mass="0.51"
          mesh="r3_config1_base"/>
        <geom name="hand_base_geom_col" type="mesh" group="3" rgba="1 0.6 0.6 1" mass="0" mesh="r3_config1_base_col"/>
        <geom name="hand_base_puck_geom" material="white_aluminium" type="mesh" contype="0" conaffinity="0" mass="0"
          mesh="r3_config1_base_puck"/>
        <geom name="hand_base_puck_geom_col" type="mesh" group="3" rgba="1 0.6 0.6 1" mass="0"
          mesh="r3_config1_base_puck_col"/>
        <geom name="hand_base_CollisionGeom_1" type="cylinder" contype="0" conaffinity="0" group="5" size="0.08"
          rgba="1 1 1 0.3" mass="0" fromto="0 0 -0.01 0 0 0.195"/>
      </body>
      <body pos="0 0.05 0.017" quat="1 0 0 0" name="F0/">
        <body name="F0/finger_base" gravcomp="1">
          <site name="F0/attachment_site"/>
          <geom name="F0/base_geom" class="finger/visual" mass="0.89937782" mesh="r3_finger_base"/>
          <geom name="F0/base_tb_geom" class="finger/visual" mass="0" mesh="r3_finger_base_tb"/>
          <geom name="F0/base_cover_geom" class="finger/visual" material="jet_black" mass="0" mesh="r3_finger_cover"/>
          <geom name="F0/base_pulley_block_geom" class="finger/visual" mass="0" mesh="r3_finger_pulley_block"/>
          <geom name="F0/base_lid_white_geom" class="finger/visual" material="white_aluminium" mass="0"
            mesh="r3_finger_lid_white"/>
          <geom name="F0/base_lid_logo_geom" class="finger/visual" material="jet_black" mass="0" mesh="r3_finger_lid_logo"/>
          <geom name="F0/base_cover_small_left_geom" class="finger/visual" material="jet_black" mass="0"
            mesh="r3_finger_cover_small_left"/>
          <geom name="F0/base_cover_small_right_geom" class="finger/visual" material="jet_black" mass="0"
            mesh="r3_finger_cover_small_right"/>
          <geom name="F0/base_geom_col" class="finger/collision_hard" quat="1 -1 0 0" mesh="r3_finger_base_col"/>
          <body name="F0/finger_knuckle" pos="0 0.015 0.17902" euler="-1.0472 0 0" gravcomp="1">
            <site name="F0/j0_site"/>
            <joint name="F0/J0" axis="0 0 -1" range="-0.8727 0.8727" armature="8e-05" damping="0.009" frictionloss="0.009"/>
            <geom name="F0/knuckle_geom" class="finger/visual" material="jet_black" mass="0.13077995"
              mesh="r3_finger_knuckle"/>
            <geom name="F0/knuckle_geom_col" class="finger/collision_hard" mesh="r3_finger_knuckle_col"/>
            <body name="F0/finger_proximal" pos="0 -0.03 0" gravcomp="1">
              <site name="F0/j1_site"/>
              <joint name="F0/J1" axis="1 0 0" range="-1.3963 0.7854" armature="8e-05" damping="0.009"
                frictionloss="0.009"/>
              <geom name="F0/proximal_geom" class="finger/visual" mass="0.09614332" mesh="r3_finger_proximal"/>
              <geom name="F0/proximal_lid_geom" class="finger/visual" material="jet_black" mass="0"
                mesh="r3_finger_proximal_lid"/>
              <geom name="F0/proximal_magtac_geom" class="finger/visual" material="graphite_black" mass="0"
                mesh="r3_finger_proximal_magtac"/>
              <geom name="F0/proximal_geom_col" class="finger/collision_hard" mesh="r3_finger_proximal_col"/>
              <body name="F0/finger_middle" pos="0 -0.05 0" gravcomp="1">
                <site name="F0/j2_site"/>
                <joint name="F0/J2" axis="1 0 0" range="0 1.3963" armature="8e-05" damping="0.009" frictionloss="0.009"/>
                <geom name="F0/middle_geom" class="finger/visual" material="jet_black" mass="0.05585897"
                  mesh="r3_finger_middle"/>
                <geom name="F0/middle_magtac_geom" class="finger/visual" material="graphite_black" mass="0"
                  mesh="r3_finger_middle_magtac"/>
                <geom name="F0/middle_geom_col" class="finger/collision_hard" mesh="r3_finger_middle_col"/>
                <body name="F0/finger_distal" pos="0 -0.035 0" quat="0 0 -1 1" gravcomp="1">
                  <site name="F0/j3_site"/>
                  <joint name="F0/J3" axis="-1 0 0" range="-0.5236 1.4835" armature="8e-05" damping="0.009"
                    frictionloss="0.009"/>
                  <geom name="F0/distal_geom" class="finger/visual" mass="0.02766365" mesh="r3_finger_distal"/>
                  <geom name="F0/distal_sensor_geom" class="finger/visual" material="graphite_black" mass="0"
                    mesh="r3_finger_distal_sensor"/>
                  <geom name="F0/distal_geom_col" class="finger/collision_hard" mesh="r3_finger_distal_col"/>
                  <geom name="F0/distal_geom_tip_col" class="finger/collision_soft" mesh="r3_finger_distal_tip_col"/>
                  <site name="F0/fingertip_site" pos="0 -0.00625 0.04"/>
                  <site name="F0/distal_site" pos="0 0.007 0.03"/>
                  <geom name="F0/finger_distal_CollisionGeom_1" type="capsule" contype="0" conaffinity="0" group="5"
                    size="0.018" rgba="1 1 1 0.3" mass="0" fromto="0 0 0.025 0 0 0"/>
                </body>
                <geom name="F0/finger_middle_CollisionGeom_1" type="cylinder" contype="0" conaffinity="0" group="5"
                  size="0.014" rgba="1 1 1 0.3" mass="0" fromto="0.015 0 0 -0.015 0 0"/>
                <geom name="F0/finger_middle_CollisionGeom_2" type="capsule" contype="0" conaffinity="0" group="5"
                  size="0.02" rgba="1 1 1 0.3" mass="0" fromto="0 -0.026 0 0 -0.01 0"/>
                <geom name="F0/finger_middle_CollisionGeom_3" type="cylinder" contype="0" conaffinity="0" group="5"
                  size="0.014" rgba="1 1 1 0.3" mass="0" fromto="0.015 -0.035 0 -0.015 -0.035 0"/>
              </body>
              <geom name="F0/finger_proximal_CollisionGeom_1" type="cylinder" contype="0" conaffinity="0" group="5"
                size="0.02" rgba="1 1 1 0.3" mass="0" fromto="0.02 0 0 -0.023 0 0"/>
              <geom name="F0/finger_proximal_CollisionGeom_2" type="capsule" contype="0" conaffinity="0" group="5"
                size="0.025" rgba="1 1 1 0.3" mass="0" fromto="0 -0.01 0.003 0 -0.04 0.003"/>
            </body>
            <geom name="F0/finger_knuckle_CollisionGeom_1" type="cylinder" contype="0" conaffinity="0" group="5"
              size="0.027" rgba="1 1 1 0.3" mass="0" fromto="0 0 0.008 0 0 -0.047"/>
            <geom name="F0/finger_knuckle_CollisionGeom_2" type="cylinder" contype="0" conaffinity="0" group="5"
              size="0.018" rgba="1 1 1 0.3" mass="0" fromto="0.018 -0.03 0 -0.018 -0.03 0"/>
            <geom name="F0/finger_knuckle_CollisionGeom_3" type="capsule" contype="0" conaffinity="0" group="5"
              size="0.025" rgba="1 1 1 0.3" mass="0" fromto="-0.0015 -0.02 -0.01 -0.0015 -0.02 -0.02"/>
          </body>
          <geom name="F0/finger_base_CollisionGeom_1" type="cylinder" contype="0" conaffinity="0" group="5" size="0.05"
            rgba="1 1 1 0.3" mass="0" fromto="0 0 -0.005 0 0 0.195"/>
        </body>
      </body>

      <body pos="0.039 -0.029 0.017" quat="-0.16212752892551119 0 0 0.98676981326168844" name="F1/">
        <body name="F1/finger_base" gravcomp="1">
          <site name="F1/attachment_site"/>
          <geom name="F1/base_geom" class="finger/visual" mass="0.89937782" mesh="r3_finger_base"/>
          <geom name="F1/base_tb_geom" class="finger/visual" mass="0" mesh="r3_finger_base_tb"/>
          <geom name="F1/base_cover_geom" class="finger/visual" material="jet_black" mass="0" mesh="r3_finger_cover"/>
          <geom name="F1/base_pulley_block_geom" class="finger/visual" mass="0" mesh="r3_finger_pulley_block"/>
          <geom name="F1/base_lid_white_geom" class="finger/visual" material="white_aluminium" mass="0"
            mesh="r3_finger_lid_white"/>
          <geom name="F1/base_lid_logo_geom" class="finger/visual" material="jet_black" mass="0" mesh="r3_finger_lid_logo"/>
          <geom name="F1/base_cover_small_left_geom" class="finger/visual" material="jet_black" mass="0"
            mesh="r3_finger_cover_small_left"/>
          <geom name="F1/base_cover_small_right_geom" class="finger/visual" material="jet_black" mass="0"
            mesh="r3_finger_cover_small_right"/>
          <geom name="F1/base_geom_col" class="finger/collision_hard" quat="1 -1 0 0" mesh="r3_finger_base_col"/>
          <body name="F1/finger_knuckle" pos="0 0.015 0.17902" euler="-1.0472 0 0" gravcomp="1">
            <site name="F1/j0_site"/>
            <joint name="F1/J0" axis="0 0 -1" range="-0.8727 0.8727" armature="8e-05" damping="0.009" frictionloss="0.009"/>
            <geom name="F1/knuckle_geom" class="finger/visual" material="jet_black" mass="0.13077995"
              mesh="r3_finger_knuckle"/>
            <geom name="F1/knuckle_geom_col" class="finger/collision_hard" mesh="r3_finger_knuckle_col"/>
            <body name="F1/finger_proximal" pos="0 -0.03 0" gravcomp="1">
              <site name="F1/j1_site"/>
              <joint name="F1/J1" axis="1 0 0" range="-1.3963 0.7854" armature="8e-05" damping="0.009"
                frictionloss="0.009"/>
              <geom name="F1/proximal_geom" class="finger/visual" mass="0.09614332" mesh="r3_finger_proximal"/>
              <geom name="F1/proximal_lid_geom" class="finger/visual" material="jet_black" mass="0"
                mesh="r3_finger_proximal_lid"/>
              <geom name="F1/proximal_magtac_geom" class="finger/visual" material="graphite_black" mass="0"
                mesh="r3_finger_proximal_magtac"/>
              <geom name="F1/proximal_geom_col" class="finger/collision_hard" mesh="r3_finger_proximal_col"/>
              <body name="F1/finger_middle" pos="0 -0.05 0" gravcomp="1">
                <site name="F1/j2_site"/>
                <joint name="F1/J2" axis="1 0 0" range="0 1.3963" armature="8e-05" damping="0.009" frictionloss="0.009"/>
                <geom name="F1/middle_geom" class="finger/visual" material="jet_black" mass="0.05585897"
                  mesh="r3_finger_middle"/>
                <geom name="F1/middle_magtac_geom" class="finger/visual" material="graphite_black" mass="0"
                  mesh="r3_finger_middle_magtac"/>
                <geom name="F1/middle_geom_col" class="finger/collision_hard" mesh="r3_finger_middle_col"/>
                <body name="F1/finger_distal" pos="0 -0.035 0" quat="0 0 -1 1" gravcomp="1">
                  <site name="F1/j3_site"/>
                  <joint name="F1/J3" axis="-1 0 0" range="-0.5236 1.4835" armature="8e-05" damping="0.009"
                    frictionloss="0.009"/>
                  <geom name="F1/distal_geom" class="finger/visual" mass="0.02766365" mesh="r3_finger_distal"/>
                  <geom name="F1/distal_sensor_geom" class="finger/visual" material="graphite_black" mass="0"
                    mesh="r3_finger_distal_sensor"/>
                  <geom name="F1/distal_geom_col" class="finger/collision_hard" mesh="r3_finger_distal_col"/>
                  <geom name="F1/distal_geom_tip_col" class="finger/collision_soft" mesh="r3_finger_distal_tip_col"/>
                  <site name="F1/fingertip_site" pos="0 -0.00625 0.04"/>
                  <site name="F1/distal_site" pos="0 0.007 0.03"/>
                  <geom name="F1/finger_distal_CollisionGeom_1" type="capsule" contype="0" conaffinity="0" group="5"
                    size="0.018" rgba="1 1 1 0.3" mass="0" fromto="0 0 0.025 0 0 0"/>
                </body>
                <geom name="F1/finger_middle_CollisionGeom_1" type="cylinder" contype="0" conaffinity="0" group="5"
                  size="0.014" rgba="1 1 1 0.3" mass="0" fromto="0.015 0 0 -0.015 0 0"/>
                <geom name="F1/finger_middle_CollisionGeom_2" type="capsule" contype="0" conaffinity="0" group="5"
                  size="0.02" rgba="1 1 1 0.3" mass="0" fromto="0 -0.026 0 0 -0.01 0"/>
                <geom name="F1/finger_middle_CollisionGeom_3" type="cylinder" contype="0" conaffinity="0" group="5"
                  size="0.014" rgba="1 1 1 0.3" mass="0" fromto="0.015 -0.035 0 -0.015 -0.035 0"/>
              </body>
              <geom name="F1/finger_proximal_CollisionGeom_1" type="cylinder" contype="0" conaffinity="0" group="5"
                size="0.02" rgba="1 1 1 0.3" mass="0" fromto="0.02 0 0 -0.023 0 0"/>
              <geom name="F1/finger_proximal_CollisionGeom_2" type="capsule" contype="0" conaffinity="0" group="5"
                size="0.025" rgba="1 1 1 0.3" mass="0" fromto="0 -0.01 0.003 0 -0.04 0.003"/>
            </body>
            <geom name="F1/finger_knuckle_CollisionGeom_1" type="cylinder" contype="0" conaffinity="0" group="5"
              size="0.027" rgba="1 1 1 0.3" mass="0" fromto="0 0 0.008 0 0 -0.047"/>
            <geom name="F1/finger_knuckle_CollisionGeom_2" type="cylinder" contype="0" conaffinity="0" group="5"
              size="0.018" rgba="1 1 1 0.3" mass="0" fromto="0.018 -0.03 0 -0.018 -0.03 0"/>
            <geom name="F1/finger_knuckle_CollisionGeom_3" type="capsule" contype="0" conaffinity="0" group="5"
              size="0.025" rgba="1 1 1 0.3" mass="0" fromto="-0.0015 -0.02 -0.01 -0.0015 -0.02 -0.02"/>
          </body>
          <geom name="F1/finger_base_CollisionGeom_1" type="cylinder" contype="0" conaffinity="0" group="5" size="0.0500"
            rgba="1 1 1 0.3" mass="0" fromto="0 0 -0.005 0 0 0.195"/>
        </body>
      </body>

      <body pos="-0.039 -0.029 0.017" quat="0.16212752892551119 0 0 0.98676981326168844" name="F2/">
        <body name="F2/finger_base" gravcomp="1">
          <site name="F2/attachment_site"/>
          <geom name="F2/base_geom" class="finger/visual" mass="0.89937782" mesh="r3_finger_base"/>
          <geom name="F2/base_tb_geom" class="finger/visual" mass="0" mesh="r3_finger_base_tb"/>
          <geom name="F2/base_cover_geom" class="finger/visual" material="jet_black" mass="0" mesh="r3_finger_cover"/>
          <geom name="F2/base_pulley_block_geom" class="finger/visual" mass="0" mesh="r3_finger_pulley_block"/>
          <geom name="F2/base_lid_white_geom" class="finger/visual" material="white_aluminium" mass="0"
            mesh="r3_finger_lid_white"/>
          <geom name="F2/base_lid_logo_geom" class="finger/visual" material="jet_black" mass="0" mesh="r3_finger_lid_logo"/>
          <geom name="F2/base_cover_small_left_geom" class="finger/visual" material="jet_black" mass="0"
            mesh="r3_finger_cover_small_left"/>
          <geom name="F2/base_cover_small_right_geom" class="finger/visual" material="jet_black" mass="0"
            mesh="r3_finger_cover_small_right"/>
          <geom name="F2/base_geom_col" class="finger/collision_hard" quat="1 -1 0 0" mesh="r3_finger_base_col"/>
          <body name="F2/finger_knuckle" pos="0 0.015 0.17902" euler="-1.0472 0 0" gravcomp="1">
            <site name="F2/j0_site"/>
            <joint name="F2/J0" axis="0 0 -1" range="-0.8727 0.8727" armature="8e-05" damping="0.009" frictionloss="0.009"/>
            <geom name="F2/knuckle_geom" class="finger/visual" material="jet_black" mass="0.13077995"
              mesh="r3_finger_knuckle"/>
            <geom name="F2/knuckle_geom_col" class="finger/collision_hard" mesh="r3_finger_knuckle_col"/>
            <body name="F2/finger_proximal" pos="0 -0.03 0" gravcomp="1">
              <site name="F2/j1_site"/>
              <joint name="F2/J1" axis="1 0 0" range="-1.3963 0.7854" armature="8e-05" damping="0.009"
                frictionloss="0.009"/>
              <geom name="F2/proximal_geom" class="finger/visual" mass="0.09614332" mesh="r3_finger_proximal"/>
              <geom name="F2/proximal_lid_geom" class="finger/visual" material="jet_black" mass="0"
                mesh="r3_finger_proximal_lid"/>
              <geom name="F2/proximal_magtac_geom" class="finger/visual" material="graphite_black" mass="0"
                mesh="r3_finger_proximal_magtac"/>
              <geom name="F2/proximal_geom_col" class="finger/collision_hard" mesh="r3_finger_proximal_col"/>
              <body name="F2/finger_middle" pos="0 -0.05 0" gravcomp="1">
                <site name="F2/j2_site"/>
                <joint name="F2/J2" axis="1 0 0" range="0 1.3963" armature="8e-05" damping="0.009" frictionloss="0.009"/>
                <geom name="F2/middle_geom" class="finger/visual" material="jet_black" mass="0.05585897"
                  mesh="r3_finger_middle"/>
                <geom name="F2/middle_magtac_geom" class="finger/visual" material="graphite_black" mass="0"
                  mesh="r3_finger_middle_magtac"/>
                <geom name="F2/middle_geom_col" class="finger/collision_hard" mesh="r3_finger_middle_col"/>
                <body name="F2/finger_distal" pos="0 -0.035 0" quat="0 0 -1 1" gravcomp="1">
                  <site name="F2/j3_site"/>
                  <joint name="F2/J3" axis="-1 0 0" range="-0.5236 1.4835" armature="8e-05" damping="0.009"
                    frictionloss="0.009"/>
                  <geom name="F2/distal_geom" class="finger/visual" mass="0.02766365" mesh="r3_finger_distal"/>
                  <geom name="F2/distal_sensor_geom" class="finger/visual" material="graphite_black" mass="0"
                    mesh="r3_finger_distal_sensor"/>
                  <geom name="F2/distal_geom_col" class="finger/collision_hard" mesh="r3_finger_distal_col"/>
                  <geom name="F2/distal_geom_tip_col" class="finger/collision_soft" mesh="r3_finger_distal_tip_col"/>
                  <site name="F2/fingertip_site" pos="0 -0.00625 0.04"/>
                  <site name="F2/distal_site" pos="0 0.007 0.03"/>
                  <geom name="F2/finger_distal_CollisionGeom_1" type="capsule" contype="0" conaffinity="0" group="5"
                    size="0.018" rgba="1 1 1 0.3" mass="0" fromto="0 0 0.025 0 0 0"/>
                </body>
                <geom name="F2/finger_middle_CollisionGeom_1" type="cylinder" contype="0" conaffinity="0" group="5"
                  size="0.014" rgba="1 1 1 0.3" mass="0" fromto="0.015 0 0 -0.015 0 0"/>
                <geom name="F2/finger_middle_CollisionGeom_2" type="capsule" contype="0" conaffinity="0" group="5"
                  size="0.02" rgba="1 1 1 0.3" mass="0" fromto="0 -0.026 0 0 -0.01 0"/>
                <geom name="F2/finger_middle_CollisionGeom_3" type="cylinder" contype="0" conaffinity="0" group="5"
                  size="0.014" rgba="1 1 1 0.3" mass="0" fromto="0.015 -0.035 0 -0.015 -0.035 0"/>
              </body>
              <geom name="F2/finger_proximal_CollisionGeom_1" type="cylinder" contype="0" conaffinity="0" group="5"
                size="0.02" rgba="1 1 1 0.3" mass="0" fromto="0.02 0 0 -0.023 0 0"/>
              <geom name="F2/finger_proximal_CollisionGeom_2" type="capsule" contype="0" conaffinity="0" group="5"
                size="0.025" rgba="1 1 1 0.3" mass="0" fromto="0 -0.01 0.003 0 -0.04 0.003"/>
            </body>
            <geom name="F2/finger_knuckle_CollisionGeom_1" type="cylinder" contype="0" conaffinity="0" group="5"
              size="0.027" rgba="1 1 1 0.3" mass="0" fromto="0 0 0.008 0 0 -0.047"/>
            <geom name="F2/finger_knuckle_CollisionGeom_2" type="cylinder" contype="0" conaffinity="0" group="5"
              size="0.018" rgba="1 1 1 0.3" mass="0" fromto="0.018 -0.03 0 -0.018 -0.03 0"/>
            <geom name="F2/finger_knuckle_CollisionGeom_3" type="capsule" contype="0" conaffinity="0" group="5"
              size="0.025" rgba="1 1 1 0.3" mass="0" fromto="-0.0015 -0.02 -0.01 -0.0015 -0.02 -0.02"/>
          </body>
          <geom name="F2/finger_base_CollisionGeom_1" type="cylinder" contype="0" conaffinity="0" group="5" size="0.05"
            rgba="1 1 1 0.3" mass="0" fromto="0 0 -0.005 0 0 0.195"/>
        </body>
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude name="hand_base:F0_finger_knuckle" body1="hand_base" body2="F0/finger_knuckle"/>
    <exclude name="hand_base:F1_finger_knuckle" body1="hand_base" body2="F1/finger_knuckle"/>
    <exclude name="hand_base:F2_finger_knuckle" body1="hand_base" body2="F2/finger_knuckle"/>
    <exclude name="F0/base:knuckle" body1="F0/finger_base" body2="F0/finger_knuckle"/>
    <exclude name="F1/base:knuckle" body1="F1/finger_base" body2="F1/finger_knuckle"/>
    <exclude name="F2/base:knuckle" body1="F2/finger_base" body2="F2/finger_knuckle"/>
  </contact>
  <equality>
    <weld body1="mocap" body2="dexee_gripper" torquescale="1.0"/>
  </equality>
  <actuator>
    <plugin name="F0/J0_actuator" plugin="mujoco.pid" instance="actuator_J0" ctrlrange="-0.8727 0.8727"
      forcerange="-0.9 0.53" dyntype="none" joint="F0/J0" actdim="2"/>
    <plugin name="F0/J1_actuator" plugin="mujoco.pid" instance="actuator_J1" ctrlrange="-1.3963 0.7854"
      forcerange="-0.35 1.2" dyntype="none" joint="F0/J1" actdim="2"/>
    <plugin name="F0/J2_actuator" plugin="mujoco.pid" instance="actuator_J2" ctrlrange="0 1.3963" forcerange="-0.52 0.7"
      dyntype="none" joint="F0/J2" actdim="2"/>
    <plugin name="F0/J3_actuator" plugin="mujoco.pid" instance="actuator_J3" ctrlrange="-0.5236 1.4835"
      forcerange="-0.3 0.3" dyntype="none" joint="F0/J3" actdim="2"/>
    <plugin name="F1/J0_actuator" plugin="mujoco.pid" instance="actuator_J0" ctrlrange="-0.8727 0.8727"
      forcerange="-0.9 0.53" dyntype="none" joint="F1/J0" actdim="2"/>
    <plugin name="F1/J1_actuator" plugin="mujoco.pid" instance="actuator_J1" ctrlrange="-1.3963 0.7854"
      forcerange="-0.35 1.2" dyntype="none" joint="F1/J1" actdim="2"/>
    <plugin name="F1/J2_actuator" plugin="mujoco.pid" instance="actuator_J2" ctrlrange="0 1.3963" forcerange="-0.52 0.7"
      dyntype="none" joint="F1/J2" actdim="2"/>
    <plugin name="F1/J3_actuator" plugin="mujoco.pid" instance="actuator_J3" ctrlrange="-0.5236 1.4835"
      forcerange="-0.3 0.3" dyntype="none" joint="F1/J3" actdim="2"/>
    <plugin name="F2/J0_actuator" plugin="mujoco.pid" instance="actuator_J0" ctrlrange="-0.8727 0.8727"
      forcerange="-0.9 0.53" dyntype="none" joint="F2/J0" actdim="2"/>
    <plugin name="F2/J1_actuator" plugin="mujoco.pid" instance="actuator_J1" ctrlrange="-1.3963 0.7854"
      forcerange="-0.35 1.2" dyntype="none" joint="F2/J1" actdim="2"/>
    <plugin name="F2/J2_actuator" plugin="mujoco.pid" instance="actuator_J2" ctrlrange="0 1.3963" forcerange="-0.52 0.7"
      dyntype="none" joint="F2/J2" actdim="2"/>
    <plugin name="F2/J3_actuator" plugin="mujoco.pid" instance="actuator_J3" ctrlrange="-0.5236 1.4835"
      forcerange="-0.3 0.3" dyntype="none" joint="F2/J3" actdim="2"/>
  </actuator>

"""


class GripperDexee(MjShakableOpenCloseGripper, MjScannable):
    def __init__(self, pose: SE3Pose):
        super().__init__(pose, "dexee_gripper")

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
        base_path = os.path.join(ASSET_PATH, "dexee")
        for file_name in os.listdir(base_path):
            path = os.path.join(base_path, file_name)
            with open(path, "rb") as f:
                ASSETS[file_name] = f.read()

        return (xml, ASSETS)

    def base_to_contact_transform(self) -> SE3Pose:
        pos = np.array([0.0, 0.0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        return SE3Pose(pos, quat, type="wxyz")

    def open_gripper(self, sim: MjSimulation):
        idxs = sim.get_joint_idxs(self.get_actuator_joint_names())

        qpos = np.array([0, -1.3963, 0, 0, 0, -1.3963, 0, 0, 0, -1.3963, 0, 0])
        sim.set_qpos(np.copy(qpos), idxs)
        sim.data.ctrl[:] = np.copy(qpos)

    def close_gripper_at(self, sim: MjSimulation, pose: SE3Pose):
        self.set_pose(sim, pose)
        qpos = np.array(
            [0, -0.0325, 0, 0.00143, 0.0655, -0.0369, 0, 0, -0.0654, -0.0337, 0, 0]
        )
        sim.data.ctrl[:] = np.copy(qpos)
        mujoco.mj_step(sim.model, sim.data, 500)  # type: ignore

    def get_actuator_joint_names(self) -> List[str]:
        return [
            "F0/J0",
            "F0/J1",
            "F0/J2",
            "F0/J3",
            "F1/J0",
            "F1/J1",
            "F1/J2",
            "F1/J3",
            "F2/J0",
            "F2/J1",
            "F2/J2",
            "F2/J3",
        ]

    def get_freejoint_idxs(self, sim: MjSimulation) -> List[int]:
        start_idx = sim.get_joint_idxs(["freejoint"])[0]
        return list(range(start_idx, start_idx + 7))
