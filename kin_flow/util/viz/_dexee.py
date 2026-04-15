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

from typing import List, Tuple

import numpy as np

from kin_flow.util.viz._shared import (GEOM_MESH, GripperCollisionDef,
                                       _insert_revolute, _LimbSpec,
                                       _R33_to_quat_wxyz, _T44_from_pos_euler,
                                       _T44_from_pos_quat)

_FINGER_MOUNTS: List[Tuple[Tuple[float, ...], Tuple[float, ...]]] = [
    # F0
    ((0.0, 0.05, 0.017), (1.0, 0.0, 0.0, 0.0)),
    # F1
    (
        (0.039, -0.029, 0.017),
        (-0.16212752892551119, 0.0, 0.0, 0.98676981326168844),
    ),
    # F2
    (
        (-0.039, -0.029, 0.017),
        (0.16212752892551119, 0.0, 0.0, 0.98676981326168844),
    ),
]

# MJCF body-chain offsets (shared across all 3 fingers)
_T_KNUCKLE = _T44_from_pos_euler((0.0, 0.015, 0.17902), (-1.0472, 0.0, 0.0))
_T_PROX = _T44_from_pos_quat((0.0, -0.03, 0.0))
_T_MID = _T44_from_pos_quat((0.0, -0.05, 0.0))
_T_DIST = _T44_from_pos_quat((0.0, -0.035, 0.0), (0.0, 0.0, -1.0, 1.0))


def collision_def() -> GripperCollisionDef:
    """
    Dexee gripper collision geometry (all mesh, no primitives).

    Joint mapping (kin_flow, 12 DOFs):
      joint 0   = base (freejoint / dexee_gripper root)
      joint 1-4  = F0 (J0..J3)
      joint 5-8  = F1 (J0..J3)
      joint 9-12 = F2 (J0..J3)

    Collision meshes use MJCF refquat baked into vertices at load time.
    Finger base geoms have a compound offset = finger_mount_T @ base_geom_quat_T.
    """
    # The only geom-level rotation: Fi/base_geom_col quat="1 -1 0 0"
    base_geom_col_quat = (1.0, -1.0, 0.0, 0.0)
    T_base_geom = _T44_from_pos_quat((0.0, 0.0, 0.0), base_geom_col_quat)

    limbs: List[_LimbSpec] = []

    # Hand base + puck (on freejoint, identity offset)
    limbs.append(_LimbSpec(joint_idx=0, geom_kind=GEOM_MESH, mesh_name="base_col"))
    limbs.append(_LimbSpec(joint_idx=0, geom_kind=GEOM_MESH, mesh_name="base_puck_col"))

    finger_joint_offsets = [0, 4, 8]

    for fi, (pos, quat) in enumerate(_FINGER_MOUNTS):
        mount_T = _T44_from_pos_quat(pos, quat)
        compound_T = mount_T @ T_base_geom

        # Extract compound transform back to pos + quat for _LimbSpec
        compound_pos = tuple(float(v) for v in compound_T[:3, 3])
        compound_quat = _R33_to_quat_wxyz(compound_T[:3, :3])

        limbs.append(
            _LimbSpec(
                joint_idx=0,
                geom_kind=GEOM_MESH,
                mesh_name="finger_base_col",
                pos_xyz=compound_pos,
                quat_wxyz=compound_quat,
            )
        )

        j_off = finger_joint_offsets[fi]
        # J0 = j_off + 1, J1 = j_off + 2, J2 = j_off + 3, J3 = j_off + 4
        limbs.append(
            _LimbSpec(
                joint_idx=j_off + 1,
                geom_kind=GEOM_MESH,
                mesh_name="knuckle_col",
            )
        )
        limbs.append(
            _LimbSpec(
                joint_idx=j_off + 2,
                geom_kind=GEOM_MESH,
                mesh_name="proximal_col",
            )
        )
        limbs.append(
            _LimbSpec(
                joint_idx=j_off + 3,
                geom_kind=GEOM_MESH,
                mesh_name="middle_col",
            )
        )
        limbs.append(
            _LimbSpec(
                joint_idx=j_off + 4,
                geom_kind=GEOM_MESH,
                mesh_name="distal_col",
            )
        )
        limbs.append(
            _LimbSpec(
                joint_idx=j_off + 4,
                geom_kind=GEOM_MESH,
                mesh_name="distal_tip_col",
            )
        )

    return GripperCollisionDef(
        limbs=limbs,
        mesh_files={
            "base_col": "Asm-MRH-HB1-Visual,00-Plastic.stl",
            "base_puck_col": "Asm-MRH-HB1-Visual,00-Puck.stl",
            "finger_base_col": "r3_finger_base_col.stl",
            "knuckle_col": "MRH-F-J0Link-Visual,00.stl",
            "proximal_col": "Asm-MRH-F-Prox-Visual,00+Magtac,00.stl",
            "middle_col": "Asm-MRH-F-Mid-Visual,00+MagTac,00.stl",
            "distal_col": "MRH-F-Distal-Visual,00.stl",
            "distal_tip_col": "MRH-F-Distal-Sensor-Visual,00.stl",
        },
        mesh_scale=1.0,
        asset_subdir="dexee",
        mesh_refquats={
            "base_col": (1.0, 0.0, 0.0, 0.0),
            "base_puck_col": (1.0, 0.0, 0.0, 0.0),
            "finger_base_col": (1.0, 0.0, 0.0, 0.0),
            "knuckle_col": (1.0, -1.0, 0.0, 0.0),
            "proximal_col": (1.0, -1.0, 0.0, 0.0),
            "middle_col": (1.0, -1.0, 0.0, 0.0),
            "distal_col": (0.0, 0.0, 0.0, 1.0),
            "distal_tip_col": (0.0, -1.0, 0.0, 0.0),
        },
    )


# ===================================================================
# Joint axes (from MJCF, per finger: J0..J3)
# ===================================================================
# Each finger has the same axis pattern:
#   J0 axis=(0, 0, -1)   knuckle/yaw
#   J1 axis=(1, 0, 0)    proximal/pitch
#   J2 axis=(1, 0, 0)    middle/pitch
#   J3 axis=(-1, 0, 0)   distal/pitch

_AXES_PER_FINGER = [
    (0, 0, -1),  # J0
    (1, 0, 0),  # J1
    (1, 0, 0),  # J2
    (-1, 0, 0),  # J3
]


def joint_world_transforms(
    base_se3: np.ndarray,
    dof: np.ndarray,
    kin=None,
) -> np.ndarray:
    T_se3 = np.asarray(base_se3, dtype=np.float64)
    d = np.asarray(dof, dtype=np.float64)

    # Undo the kin_flow origin rebase: the MJCF hand_base body is 0.17 m
    # below the kin_flow origin in the gripper's local Z direction.
    T_rebase = np.eye(4, dtype=np.float64)
    T_rebase[2, 3] = -0.17
    T_body = T_se3 @ T_rebase

    Jp1 = 13  # 1 base + 12 DOF joints
    T = np.zeros((Jp1, 4, 4), dtype=np.float64)
    T[:] = np.eye(4)

    T[0] = T_body

    for fi in range(3):
        pos, quat = _FINGER_MOUNTS[fi]
        T_root = _T44_from_pos_quat(pos, quat).astype(np.float64)

        jb = 1 + 4 * fi  # base joint index for this finger
        dof_base = 4 * fi  # base DOF index for this finger

        # J0: T_base @ root @ knuckle, then insert J0 rotation
        Tw_j0 = T_body @ T_root @ _T_KNUCKLE.astype(np.float64)
        Tw_j0 = _insert_revolute(Tw_j0, _AXES_PER_FINGER[0], d[dof_base + 0]).astype(
            np.float64
        )
        T[jb + 0] = Tw_j0

        # J1: Tw_j0 @ prox, then insert J1 rotation
        Tw_j1 = Tw_j0 @ _T_PROX.astype(np.float64)
        Tw_j1 = _insert_revolute(Tw_j1, _AXES_PER_FINGER[1], d[dof_base + 1]).astype(
            np.float64
        )
        T[jb + 1] = Tw_j1

        # J2: Tw_j1 @ mid, then insert J2 rotation
        Tw_j2 = Tw_j1 @ _T_MID.astype(np.float64)
        Tw_j2 = _insert_revolute(Tw_j2, _AXES_PER_FINGER[2], d[dof_base + 2]).astype(
            np.float64
        )
        T[jb + 2] = Tw_j2

        # J3: Tw_j2 @ dist, then insert J3 rotation
        Tw_j3 = Tw_j2 @ _T_DIST.astype(np.float64)
        Tw_j3 = _insert_revolute(Tw_j3, _AXES_PER_FINGER[3], d[dof_base + 3]).astype(
            np.float64
        )
        T[jb + 3] = Tw_j3

    return T.astype(np.float32)
