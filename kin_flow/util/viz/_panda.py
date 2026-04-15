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

from __future__ import annotations

from typing import List

import numpy as np

from kin_flow.util.viz._shared import (GEOM_MESH, GEOM_PRIM,
                                       GripperCollisionDef, _LimbSpec)


def collision_def() -> GripperCollisionDef:
    """
    Panda gripper collision geometry.

    Joint mapping:
      joint 0 = base (hand body)
      joint 1 = left finger  (slide along +y)
      joint 2 = right finger (slide along +y, body rotated 180 deg about z)
    """
    pad_pos = [
        (0.0, 0.0055, 0.0445),
        (0.0055, 0.0020, 0.0500),
        (-0.0055, 0.0020, 0.0500),
        (0.0055, 0.0020, 0.0395),
        (-0.0055, 0.0020, 0.0395),
    ]
    pad_boxes = [
        (0.0085, 0.004, 0.0085),
        (0.003, 0.002, 0.003),
        (0.003, 0.002, 0.003),
        (0.003, 0.002, 0.0035),
        (0.003, 0.002, 0.0035),
    ]

    limbs: List[_LimbSpec] = []

    # Limb 0: hand mesh (base)
    limbs.append(_LimbSpec(joint_idx=0, geom_kind=GEOM_MESH, mesh_name="hand"))
    # Limb 1: left finger mesh
    limbs.append(_LimbSpec(joint_idx=1, geom_kind=GEOM_MESH, mesh_name="finger_0"))
    # Limbs 2-6: left finger pads (boxes)
    for k in range(5):
        limbs.append(
            _LimbSpec(
                joint_idx=1,
                geom_kind=GEOM_PRIM,
                prim_type="box",
                prim_params=pad_boxes[k],
                pos_xyz=pad_pos[k],
            )
        )
    # Limb 7: right finger mesh
    limbs.append(_LimbSpec(joint_idx=2, geom_kind=GEOM_MESH, mesh_name="finger_0"))
    # Limbs 8-12: right finger pads (boxes)
    for k in range(5):
        limbs.append(
            _LimbSpec(
                joint_idx=2,
                geom_kind=GEOM_PRIM,
                prim_type="box",
                prim_params=pad_boxes[k],
                pos_xyz=pad_pos[k],
            )
        )

    return GripperCollisionDef(
        limbs=limbs,
        mesh_files={"hand": "hand.stl", "finger_0": "finger_0.obj"},
        mesh_scale=1.0,
        asset_subdir="panda",
    )


def joint_world_transforms(
    base_se3: np.ndarray,
    dof: np.ndarray,
    kin=None,
) -> np.ndarray:
    """
    Panda: 3 joint frames (base, left finger, right finger) in world.

    MJCF body chain:
      - base: identity (hand body)
      - left finger:  body pos=(0, 0, 0.0584), R=I, slide along +y by dof[0]
      - right finger: body pos=(0, -0.04, 0.0584), R=diag(-1,-1,1), slide along +y by dof[1]

    DOFs from kin_flow:
      dof[0] = left finger   [0 .. 0.04]    body-frame y += dof[0]
      dof[1] = right finger  [-0.04 .. 0]   body-frame y += dof[1]
    """
    T_body = np.asarray(base_se3, dtype=np.float64)

    T = np.zeros((3, 4, 4), dtype=np.float64)
    T[:] = np.eye(4)

    # joint 0: hand base
    T[0] = T_body

    # joint 1: left finger — body offset + slide
    T_left = np.eye(4, dtype=np.float64)
    T_left[:3, 3] = [0.0, float(dof[0]), 0.0584]
    T[1] = T_body @ T_left

    # joint 2: right finger (180 deg about z -> diag(-1,-1,1))
    T_right = np.eye(4, dtype=np.float64)
    T_right[0, 0] = -1.0
    T_right[1, 1] = -1.0
    T_right[:3, 3] = [0.0, -0.04 - float(dof[1]), 0.0584]
    T[2] = T_body @ T_right

    return T.astype(np.float32)
