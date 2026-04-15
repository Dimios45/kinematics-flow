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

from kin_flow.util.viz._shared import (GEOM_PRIM, GripperCollisionDef,
                                       _insert_revolute, _LimbSpec,
                                       _T44_from_pos_quat)


def collision_def() -> GripperCollisionDef:
    """
    Allegro hand collision geometry (all primitives, no meshes).

    Joint mapping (kin_flow, 16 DOFs):
      joint 0  = base (palm)
      joint 1-4   = FF (ffj0..ffj3)
      joint 5-8   = MF (mfj0..mfj3)
      joint 9-12  = RF (rfj0..rfj3)
      joint 13-16 = TH (thj0..thj3)
    """
    J_BASE = 0

    BOX_PALM = (0.0204, 0.0565, 0.0475)
    POS_PALM = (-0.0093, 0.0, -0.0475)

    BOX_FINGER_BASE = (0.0098, 0.01375, 0.0082)
    POS_FINGER_BASE = (0.0, 0.0, 0.0082)

    BOX_PROX = (0.0098, 0.01375, 0.027)
    POS_PROX = (0.0, 0.0, 0.027)

    BOX_MED = (0.0098, 0.01375, 0.0192)
    POS_MED = (0.0, 0.0, 0.0192)

    BOX_DIST = (0.0098, 0.01375, 0.008)
    POS_DIST = (0.0, 0.0, 0.008)

    CAP_FINGERTIP = (0.012, 0.01)
    POS_FINGERTIP = (0.0, 0.0, 0.019)

    BOX_TH_BASE = (0.0179, 0.017, 0.02275)
    POS_TH_BASE = (-0.0179, 0.009, 0.0145)

    BOX_TH_PROX = (0.0098, 0.01375, 0.00885)
    POS_TH_PROX = (0.0, 0.0, 0.00885)

    BOX_TH_MED = (0.0098, 0.01375, 0.0257)
    POS_TH_MED = (0.0, 0.0, 0.0257)

    BOX_TH_DIST = (0.0098, 0.01375, 0.0157)
    POS_TH_DIST = (0.0, 0.0, 0.0157)

    CAP_THUMBTIP = (0.012, 0.008)
    POS_THUMBTIP = (0.0, 0.0, 0.035)

    limbs: List[_LimbSpec] = []

    # Palm
    limbs.append(
        _LimbSpec(
            joint_idx=J_BASE,
            geom_kind=GEOM_PRIM,
            prim_type="box",
            prim_params=BOX_PALM,
            pos_xyz=POS_PALM,
        )
    )

    def _add_finger(j0, j1, j2, j3):
        limbs.append(
            _LimbSpec(
                joint_idx=j0,
                geom_kind=GEOM_PRIM,
                prim_type="box",
                prim_params=BOX_FINGER_BASE,
                pos_xyz=POS_FINGER_BASE,
            )
        )
        limbs.append(
            _LimbSpec(
                joint_idx=j1,
                geom_kind=GEOM_PRIM,
                prim_type="box",
                prim_params=BOX_PROX,
                pos_xyz=POS_PROX,
            )
        )
        limbs.append(
            _LimbSpec(
                joint_idx=j2,
                geom_kind=GEOM_PRIM,
                prim_type="box",
                prim_params=BOX_MED,
                pos_xyz=POS_MED,
            )
        )
        limbs.append(
            _LimbSpec(
                joint_idx=j3,
                geom_kind=GEOM_PRIM,
                prim_type="box",
                prim_params=BOX_DIST,
                pos_xyz=POS_DIST,
            )
        )
        limbs.append(
            _LimbSpec(
                joint_idx=j3,
                geom_kind=GEOM_PRIM,
                prim_type="capsule",
                prim_params=CAP_FINGERTIP,
                pos_xyz=POS_FINGERTIP,
            )
        )

    _add_finger(1, 2, 3, 4)  # FF
    _add_finger(5, 6, 7, 8)  # MF
    _add_finger(9, 10, 11, 12)  # RF

    # Thumb
    limbs.append(
        _LimbSpec(
            joint_idx=13,
            geom_kind=GEOM_PRIM,
            prim_type="box",
            prim_params=BOX_TH_BASE,
            pos_xyz=POS_TH_BASE,
        )
    )
    limbs.append(
        _LimbSpec(
            joint_idx=14,
            geom_kind=GEOM_PRIM,
            prim_type="box",
            prim_params=BOX_TH_PROX,
            pos_xyz=POS_TH_PROX,
        )
    )
    limbs.append(
        _LimbSpec(
            joint_idx=15,
            geom_kind=GEOM_PRIM,
            prim_type="box",
            prim_params=BOX_TH_MED,
            pos_xyz=POS_TH_MED,
        )
    )
    limbs.append(
        _LimbSpec(
            joint_idx=16,
            geom_kind=GEOM_PRIM,
            prim_type="box",
            prim_params=BOX_TH_DIST,
            pos_xyz=POS_TH_DIST,
        )
    )
    limbs.append(
        _LimbSpec(
            joint_idx=16,
            geom_kind=GEOM_PRIM,
            prim_type="capsule",
            prim_params=CAP_THUMBTIP,
            pos_xyz=POS_THUMBTIP,
        )
    )

    return GripperCollisionDef(
        limbs=limbs,
        mesh_files={},
        mesh_scale=1.0,
        asset_subdir="",
    )


# ===================================================================
# Joint axes (from MJCF)
# ===================================================================
# Joint ordering (16 DOFs):
#   0  ffj0  axis=(0,0,1)     1  ffj1  axis=(0,1,0)
#   2  ffj2  axis=(0,1,0)     3  ffj3  axis=(0,1,0)
#   4  mfj0  axis=(0,0,1)     5  mfj1  axis=(0,1,0)
#   6  mfj2  axis=(0,1,0)     7  mfj3  axis=(0,1,0)
#   8  rfj0  axis=(0,0,1)     9  rfj1  axis=(0,1,0)
#  10  rfj2  axis=(0,1,0)    11  rfj3  axis=(0,1,0)
#  12  thj0  axis=(-1,0,0)   13  thj1  axis=(0,0,1)
#  14  thj2  axis=(0,1,0)    15  thj3  axis=(0,1,0)

_AXES = [
    (0, 0, 1),  # 0  ffj0
    (0, 1, 0),  # 1  ffj1
    (0, 1, 0),  # 2  ffj2
    (0, 1, 0),  # 3  ffj3
    (0, 0, 1),  # 4  mfj0
    (0, 1, 0),  # 5  mfj1
    (0, 1, 0),  # 6  mfj2
    (0, 1, 0),  # 7  mfj3
    (0, 0, 1),  # 8  rfj0
    (0, 1, 0),  # 9  rfj1
    (0, 1, 0),  # 10 rfj2
    (0, 1, 0),  # 11 rfj3
    (-1, 0, 0),  # 12 thj0
    (0, 0, 1),  # 13 thj1
    (0, 1, 0),  # 14 thj2
    (0, 1, 0),  # 15 thj3
]


def joint_world_transforms(
    base_se3: np.ndarray,
    dof: np.ndarray,
    kin=None,
) -> np.ndarray:
    T_body = np.asarray(base_se3, dtype=np.float64)
    d = np.asarray(dof, dtype=np.float64)

    Jp1 = 17  # 1 base + 16 DOF joints
    T = np.zeros((Jp1, 4, 4), dtype=np.float64)
    T[:] = np.eye(4)

    # index 0: base (palm)
    T[0] = T_body

    # --- FF (dof 0-3, joint indices 1-4) ---
    T_ff0 = T_body @ _T44_from_pos_quat(
        (0.0, 0.0435, -0.001542), (0.999048, -0.0436194, 0.0, 0.0)
    ).astype(np.float64)
    T_ff0 = _insert_revolute(T_ff0, _AXES[0], d[0]).astype(np.float64)  # ffj0
    T[1] = T_ff0

    T_ff1 = T_ff0 @ _T44_from_pos_quat((0.0, 0.0, 0.0164)).astype(np.float64)
    T_ff1 = _insert_revolute(T_ff1, _AXES[1], d[1]).astype(np.float64)  # ffj1
    T[2] = T_ff1

    T_ff2 = T_ff1 @ _T44_from_pos_quat((0.0, 0.0, 0.054)).astype(np.float64)
    T_ff2 = _insert_revolute(T_ff2, _AXES[2], d[2]).astype(np.float64)  # ffj2
    T[3] = T_ff2

    T_ff3 = T_ff2 @ _T44_from_pos_quat((0.0, 0.0, 0.0384)).astype(np.float64)
    T_ff3 = _insert_revolute(T_ff3, _AXES[3], d[3]).astype(np.float64)  # ffj3
    T[4] = T_ff3

    # --- MF (dof 4-7, joint indices 5-8) ---
    T_mf0 = T_body @ _T44_from_pos_quat((0.0, 0.0, 0.0007)).astype(np.float64)
    T_mf0 = _insert_revolute(T_mf0, _AXES[4], d[4]).astype(np.float64)  # mfj0
    T[5] = T_mf0

    T_mf1 = T_mf0 @ _T44_from_pos_quat((0.0, 0.0, 0.0164)).astype(np.float64)
    T_mf1 = _insert_revolute(T_mf1, _AXES[5], d[5]).astype(np.float64)  # mfj1
    T[6] = T_mf1

    T_mf2 = T_mf1 @ _T44_from_pos_quat((0.0, 0.0, 0.054)).astype(np.float64)
    T_mf2 = _insert_revolute(T_mf2, _AXES[6], d[6]).astype(np.float64)  # mfj2
    T[7] = T_mf2

    T_mf3 = T_mf2 @ _T44_from_pos_quat((0.0, 0.0, 0.0384)).astype(np.float64)
    T_mf3 = _insert_revolute(T_mf3, _AXES[7], d[7]).astype(np.float64)  # mfj3
    T[8] = T_mf3

    # --- RF (dof 8-11, joint indices 9-12) ---
    T_rf0 = T_body @ _T44_from_pos_quat(
        (0.0, -0.0435, -0.001542), (0.999048, 0.0436194, 0.0, 0.0)
    ).astype(np.float64)
    T_rf0 = _insert_revolute(T_rf0, _AXES[8], d[8]).astype(np.float64)  # rfj0
    T[9] = T_rf0

    T_rf1 = T_rf0 @ _T44_from_pos_quat((0.0, 0.0, 0.0164)).astype(np.float64)
    T_rf1 = _insert_revolute(T_rf1, _AXES[9], d[9]).astype(np.float64)  # rfj1
    T[10] = T_rf1

    T_rf2 = T_rf1 @ _T44_from_pos_quat((0.0, 0.0, 0.054)).astype(np.float64)
    T_rf2 = _insert_revolute(T_rf2, _AXES[10], d[10]).astype(np.float64)  # rfj2
    T[11] = T_rf2

    T_rf3 = T_rf2 @ _T44_from_pos_quat((0.0, 0.0, 0.0384)).astype(np.float64)
    T_rf3 = _insert_revolute(T_rf3, _AXES[11], d[11]).astype(np.float64)  # rfj3
    T[12] = T_rf3

    # --- TH (dof 12-15, joint indices 13-16) ---
    T_th0 = T_body @ _T44_from_pos_quat(
        (-0.0182, 0.019333, -0.045987),
        (0.477714, -0.521334, -0.521334, -0.477714),
    ).astype(np.float64)
    T_th0 = _insert_revolute(T_th0, _AXES[12], d[12]).astype(np.float64)  # thj0
    T[13] = T_th0

    T_th1 = T_th0 @ _T44_from_pos_quat((-0.027, 0.005, 0.0399)).astype(np.float64)
    T_th1 = _insert_revolute(T_th1, _AXES[13], d[13]).astype(np.float64)  # thj1
    T[14] = T_th1

    T_th2 = T_th1 @ _T44_from_pos_quat((0.0, 0.0, 0.0177)).astype(np.float64)
    T_th2 = _insert_revolute(T_th2, _AXES[14], d[14]).astype(np.float64)  # thj2
    T[15] = T_th2

    T_th3 = T_th2 @ _T44_from_pos_quat((0.0, 0.0, 0.0514)).astype(np.float64)
    T_th3 = _insert_revolute(T_th3, _AXES[15], d[15]).astype(np.float64)  # thj3
    T[16] = T_th3

    return T.astype(np.float32)
