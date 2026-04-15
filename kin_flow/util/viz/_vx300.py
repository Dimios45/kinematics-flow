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
                                       GripperCollisionDef, _LimbSpec,
                                       _T44_from_pos_quat)


def collision_def() -> GripperCollisionDef:
    """
    VX300 gripper collision geometry (meshes + box primitives).

    Joint mapping (kin_flow, 2 DOFs):
      joint 0 = base (freejoint / gripper_link)
      joint 1 = left_finger
      joint 2 = right_finger
    """
    BASE_QUAT = (1.0, 0.0, 0.0, 1.0)  # Un-normalized; _quat_wxyz_to_R33 handles it

    limbs: List[_LimbSpec] = []

    # --- Base collision meshes on gripper_link (freejoint) ---
    limbs.append(
        _LimbSpec(
            joint_idx=0,
            geom_kind=GEOM_MESH,
            mesh_name="gripper",
            pos_xyz=(-0.02, 0.0, 0.0),
            quat_wxyz=BASE_QUAT,
        )
    )
    limbs.append(
        _LimbSpec(
            joint_idx=0,
            geom_kind=GEOM_MESH,
            mesh_name="gripper_bar",
            pos_xyz=(-0.020175, 0.0, 0.0),
            quat_wxyz=BASE_QUAT,
        )
    )
    limbs.append(
        _LimbSpec(
            joint_idx=0,
            geom_kind=GEOM_MESH,
            mesh_name="gripper_prop",
            pos_xyz=(0.0485 - 0.0685, 0.0, 0.0),
            quat_wxyz=BASE_QUAT,
        )
    )

    # --- Finger pad boxes ---
    # Left finger pads (joint 1)
    limbs += [
        _LimbSpec(
            1,
            GEOM_PRIM,
            prim_type="box",
            prim_params=(0.01405, 0.01405, 0.001),
            pos_xyz=(0.0478, -0.0125, 0.0106),
            quat_wxyz=(0.65, 0.65, -0.27, 0.27),
        ),
        _LimbSpec(
            1,
            GEOM_PRIM,
            prim_type="box",
            prim_params=(0.01405, 0.01405, 0.001),
            pos_xyz=(0.0478, -0.0125, -0.0106),
            quat_wxyz=(0.65, 0.65, -0.27, 0.27),
        ),
        _LimbSpec(
            1,
            GEOM_PRIM,
            prim_type="box",
            prim_params=(0.01058, 0.01058, 0.001),
            pos_xyz=(0.0571, -0.0125, 0.0),
            quat_wxyz=(1.0, 1.0, 0.0, 0.0),
        ),
        _LimbSpec(
            1,
            GEOM_PRIM,
            prim_type="box",
            prim_params=(0.01, 0.0105, 0.001),
            pos_xyz=(0.0378, -0.0125, 0.0),
            quat_wxyz=(1.0, 1.0, 0.0, 0.0),
        ),
        _LimbSpec(
            1,
            GEOM_PRIM,
            prim_type="box",
            prim_params=(0.015, 0.0105, 0.001),
            pos_xyz=(0.0128, -0.0125, 0.0),
            quat_wxyz=(1.0, 1.0, 0.0, 0.0),
        ),
        _LimbSpec(
            1,
            GEOM_PRIM,
            prim_type="box",
            prim_params=(0.01, 0.0105, 0.001),
            pos_xyz=(0.0378, -0.0125, 0.02),
            quat_wxyz=(1.0, 1.0, 0.0, 0.0),
        ),
        _LimbSpec(
            1,
            GEOM_PRIM,
            prim_type="box",
            prim_params=(0.015, 0.0105, 0.001),
            pos_xyz=(0.0128, -0.0125, 0.02),
            quat_wxyz=(1.0, 1.0, 0.0, 0.0),
        ),
        _LimbSpec(
            1,
            GEOM_PRIM,
            prim_type="box",
            prim_params=(0.01, 0.0105, 0.001),
            pos_xyz=(0.0378, -0.0125, -0.02),
            quat_wxyz=(1.0, 1.0, 0.0, 0.0),
        ),
        _LimbSpec(
            1,
            GEOM_PRIM,
            prim_type="box",
            prim_params=(0.015, 0.0105, 0.001),
            pos_xyz=(0.0128, -0.0125, -0.02),
            quat_wxyz=(1.0, 1.0, 0.0, 0.0),
        ),
    ]

    # Right finger pads (joint 2)
    limbs += [
        _LimbSpec(
            2,
            GEOM_PRIM,
            prim_type="box",
            prim_params=(0.01405, 0.01405, 0.001),
            pos_xyz=(0.0478, 0.0125, 0.0106),
            quat_wxyz=(0.65, 0.65, -0.27, 0.27),
        ),
        _LimbSpec(
            2,
            GEOM_PRIM,
            prim_type="box",
            prim_params=(0.01405, 0.01405, 0.001),
            pos_xyz=(0.0478, 0.0125, -0.0106),
            quat_wxyz=(0.65, 0.65, -0.27, 0.27),
        ),
        _LimbSpec(
            2,
            GEOM_PRIM,
            prim_type="box",
            prim_params=(0.01058, 0.01058, 0.001),
            pos_xyz=(0.0571, 0.0125, 0.0),
            quat_wxyz=(1.0, 1.0, 0.0, 0.0),
        ),
        _LimbSpec(
            2,
            GEOM_PRIM,
            prim_type="box",
            prim_params=(0.01, 0.0105, 0.001),
            pos_xyz=(0.0378, 0.0125, 0.0),
            quat_wxyz=(1.0, 1.0, 0.0, 0.0),
        ),
        _LimbSpec(
            2,
            GEOM_PRIM,
            prim_type="box",
            prim_params=(0.015, 0.0105, 0.001),
            pos_xyz=(0.0128, 0.0125, 0.0),
            quat_wxyz=(1.0, 1.0, 0.0, 0.0),
        ),
        _LimbSpec(
            2,
            GEOM_PRIM,
            prim_type="box",
            prim_params=(0.01, 0.0105, 0.001),
            pos_xyz=(0.0378, 0.0125, 0.02),
            quat_wxyz=(1.0, 1.0, 0.0, 0.0),
        ),
        _LimbSpec(
            2,
            GEOM_PRIM,
            prim_type="box",
            prim_params=(0.015, 0.0105, 0.001),
            pos_xyz=(0.0128, 0.0125, 0.02),
            quat_wxyz=(1.0, 1.0, 0.0, 0.0),
        ),
        _LimbSpec(
            2,
            GEOM_PRIM,
            prim_type="box",
            prim_params=(0.01, 0.0105, 0.001),
            pos_xyz=(0.0378, 0.0125, -0.02),
            quat_wxyz=(1.0, 1.0, 0.0, 0.0),
        ),
        _LimbSpec(
            2,
            GEOM_PRIM,
            prim_type="box",
            prim_params=(0.015, 0.0105, 0.001),
            pos_xyz=(0.0128, 0.0125, -0.02),
            quat_wxyz=(1.0, 1.0, 0.0, 0.0),
        ),
    ]

    return GripperCollisionDef(
        limbs=limbs,
        mesh_files={
            "gripper": "vx300s_7_gripper.stl",
            "gripper_bar": "vx300s_9_gripper_bar.stl",
            "gripper_prop": "vx300s_8_gripper_prop.stl",
        },
        mesh_scale=0.001,
        asset_subdir="vx300",
    )


def joint_world_transforms(
    base_se3: np.ndarray,
    dof: np.ndarray,
    kin=None,
) -> np.ndarray:
    """
    VX300: 3 joint frames in world (1 base + 2 fingers).

    MJCF body chain:
      - base: identity (gripper_link)
      - left_finger:  body pos=(0.0687, q_left, 0.0), identity rotation
      - right_finger: body pos=(0.0687, q_right, 0.0), identity rotation

    DOFs from kin_flow (prismatic, slide along Y):
      dof[0] = left finger   [0.021 .. 0.057]   Y += dof[0]
      dof[1] = right finger  [-0.057 .. -0.021]  Y += dof[1]
    """
    T_body = np.asarray(base_se3, dtype=np.float64)

    T = np.zeros((3, 4, 4), dtype=np.float64)
    T[:] = np.eye(4)

    # joint 0: gripper base
    T[0] = T_body

    # joint 1: left finger — fixed X offset 0.0687 + slide along Y
    T_left = _T44_from_pos_quat((0.0687, float(dof[0]), 0.0)).astype(np.float64)
    T[1] = T_body @ T_left

    # joint 2: right finger — fixed X offset 0.0687 + slide along Y
    T_right = _T44_from_pos_quat((0.0687, float(dof[1]), 0.0)).astype(np.float64)
    T[2] = T_body @ T_right

    return T.astype(np.float32)
