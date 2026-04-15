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
                                       GripperCollisionDef, _insert_revolute,
                                       _LimbSpec, _T44_from_pos_quat)


def collision_def() -> GripperCollisionDef:
    """Shadow hand collision geometry (36 limbs)."""
    PALM_OFFSET = (0.0, 0.0, 0.034)

    def _palm_pos(xyz):
        return (
            PALM_OFFSET[0] + xyz[0],
            PALM_OFFSET[1] + xyz[1],
            PALM_OFFSET[2] + xyz[2],
        )

    limbs: List[_LimbSpec] = []

    # --- Wrist collision primitives (attached to base = joint 0) ---
    limbs += [
        _LimbSpec(
            joint_idx=0,
            geom_kind=GEOM_PRIM,
            prim_type="cylinder",
            prim_params=(0.0135, 0.015),
            pos_xyz=(0.0, 0.0, 0.0),
            quat_wxyz=(0.499998, 0.5, 0.5, -0.500002),
        ),
        _LimbSpec(
            joint_idx=0,
            geom_kind=GEOM_PRIM,
            prim_type="cylinder",
            prim_params=(0.011, 0.005),
            pos_xyz=(-0.026, 0.0, 0.034),
            quat_wxyz=(1.0, 0.0, 1.0, 0.0),
        ),
        _LimbSpec(
            joint_idx=0,
            geom_kind=GEOM_PRIM,
            prim_type="cylinder",
            prim_params=(0.011, 0.005),
            pos_xyz=(0.031, 0.0, 0.034),
            quat_wxyz=(1.0, 0.0, 1.0, 0.0),
        ),
        _LimbSpec(
            joint_idx=0,
            geom_kind=GEOM_PRIM,
            prim_type="box",
            prim_params=(0.0135, 0.009, 0.005),
            pos_xyz=(-0.021, 0.0, 0.011),
            quat_wxyz=(0.923879, 0.0, 0.382684, 0.0),
        ),
        _LimbSpec(
            joint_idx=0,
            geom_kind=GEOM_PRIM,
            prim_type="box",
            prim_params=(0.0135, 0.009, 0.005),
            pos_xyz=(0.026, 0.0, 0.010),
            quat_wxyz=(0.923879, 0.0, -0.382684, 0.0),
        ),
    ]

    # --- Palm collision primitives (folded into base joint 0) ---
    limbs += [
        _LimbSpec(
            joint_idx=0,
            geom_kind=GEOM_PRIM,
            prim_type="box",
            prim_params=(0.031, 0.0035, 0.049),
            pos_xyz=_palm_pos((0.011, 0.0085, 0.038)),
        ),
        _LimbSpec(
            joint_idx=0,
            geom_kind=GEOM_PRIM,
            prim_type="box",
            prim_params=(0.018, 0.0085, 0.049),
            pos_xyz=_palm_pos((-0.002, -0.0035, 0.038)),
        ),
        _LimbSpec(
            joint_idx=0,
            geom_kind=GEOM_PRIM,
            prim_type="box",
            prim_params=(0.013, 0.0085, 0.005),
            pos_xyz=_palm_pos((0.029, -0.0035, 0.082)),
        ),
        _LimbSpec(
            joint_idx=0,
            geom_kind=GEOM_PRIM,
            prim_type="box",
            prim_params=(0.013, 0.007, 0.009),
            pos_xyz=_palm_pos((0.0265, -0.001, 0.070)),
            quat_wxyz=(0.987241, 0.0990545, 0.0124467, 0.124052),
        ),
        _LimbSpec(
            joint_idx=0,
            geom_kind=GEOM_PRIM,
            prim_type="box",
            prim_params=(0.0105, 0.0135, 0.012),
            pos_xyz=_palm_pos((0.0315, -0.0085, 0.001)),
        ),
        _LimbSpec(
            joint_idx=0,
            geom_kind=GEOM_PRIM,
            prim_type="box",
            prim_params=(0.011, 0.0025, 0.015),
            pos_xyz=_palm_pos((0.0125, -0.015, 0.004)),
            quat_wxyz=(0.971338, 0.0, 0.0, -0.237703),
        ),
        _LimbSpec(
            joint_idx=0,
            geom_kind=GEOM_PRIM,
            prim_type="box",
            prim_params=(0.009, 0.012, 0.002),
            pos_xyz=_palm_pos((0.011, 0.0, 0.089)),
        ),
        _LimbSpec(
            joint_idx=0,
            geom_kind=GEOM_PRIM,
            prim_type="box",
            prim_params=(0.010, 0.012, 0.020),
            pos_xyz=_palm_pos((-0.030, 0.0, 0.009)),
        ),
    ]

    # --- FF / MF / RF fingers ---
    for finger_offset in [0, 4, 8]:
        j4 = finger_offset + 1
        j3 = finger_offset + 2
        j2 = finger_offset + 3
        j1 = finger_offset + 4
        limbs += [
            _LimbSpec(
                joint_idx=j4,
                geom_kind=GEOM_PRIM,
                prim_type="cylinder",
                prim_params=(0.009, 0.009),
                quat_wxyz=(1.0, 0.0, 1.0, 0.0),
            ),
            _LimbSpec(
                joint_idx=j3,
                geom_kind=GEOM_PRIM,
                prim_type="capsule",
                prim_params=(0.009, 0.02),
                pos_xyz=(0.0, 0.0, 0.025),
            ),
            _LimbSpec(
                joint_idx=j2,
                geom_kind=GEOM_PRIM,
                prim_type="capsule",
                prim_params=(0.009, 0.0125),
                pos_xyz=(0.0, 0.0, 0.0125),
            ),
            _LimbSpec(
                joint_idx=j1,
                geom_kind=GEOM_MESH,
                mesh_name="f_distal_pst",
            ),
        ]

    # --- LF (little finger): DOF 12..16, joint_idx 13..17 ---
    limbs += [
        _LimbSpec(
            joint_idx=13,
            geom_kind=GEOM_PRIM,
            prim_type="box",
            prim_params=(0.011, 0.012, 0.025),
            pos_xyz=(0.002, 0.0, 0.033),
        ),
        _LimbSpec(
            joint_idx=14,
            geom_kind=GEOM_PRIM,
            prim_type="cylinder",
            prim_params=(0.009, 0.009),
            quat_wxyz=(1.0, 0.0, 1.0, 0.0),
        ),
        _LimbSpec(
            joint_idx=15,
            geom_kind=GEOM_PRIM,
            prim_type="capsule",
            prim_params=(0.009, 0.02),
            pos_xyz=(0.0, 0.0, 0.025),
        ),
        _LimbSpec(
            joint_idx=16,
            geom_kind=GEOM_PRIM,
            prim_type="capsule",
            prim_params=(0.009, 0.0125),
            pos_xyz=(0.0, 0.0, 0.0125),
        ),
        _LimbSpec(
            joint_idx=17,
            geom_kind=GEOM_MESH,
            mesh_name="f_distal_pst",
        ),
    ]

    # --- Thumb: DOF 17..21, joint_idx 18..22 ---
    limbs += [
        _LimbSpec(
            joint_idx=18,
            geom_kind=GEOM_PRIM,
            prim_type="sphere",
            prim_params=(0.013,),
        ),
        _LimbSpec(
            joint_idx=19,
            geom_kind=GEOM_PRIM,
            prim_type="capsule",
            prim_params=(0.0105, 0.009),
            pos_xyz=(0.0, 0.0, 0.020),
        ),
        _LimbSpec(
            joint_idx=20,
            geom_kind=GEOM_PRIM,
            prim_type="sphere",
            prim_params=(0.011,),
        ),
        _LimbSpec(
            joint_idx=21,
            geom_kind=GEOM_PRIM,
            prim_type="capsule",
            prim_params=(0.009, 0.009),
            pos_xyz=(0.0, 0.0, 0.012),
        ),
        _LimbSpec(
            joint_idx=21,
            geom_kind=GEOM_PRIM,
            prim_type="sphere",
            prim_params=(0.010,),
            pos_xyz=(0.0, 0.0, 0.030),
        ),
        _LimbSpec(
            joint_idx=22,
            geom_kind=GEOM_MESH,
            mesh_name="th_distal_pst",
        ),
    ]

    return GripperCollisionDef(
        limbs=limbs,
        mesh_files={
            "f_distal_pst": "f_distal_pst.obj",
            "th_distal_pst": "th_distal_pst.obj",
        },
        mesh_scale=0.001,
        asset_subdir="shadow",
    )


# ===================================================================
# Joint axes (from MJCF)
# ===================================================================
# Joint ordering (22 DOFs):
#   0  FFJ4  axis=(0,-1,0)     1  FFJ3  axis=(1,0,0)
#   2  FFJ2  axis=(1,0,0)      3  FFJ1  axis=(1,0,0)
#   4  MFJ4  axis=(0,-1,0)     5  MFJ3  axis=(1,0,0)
#   6  MFJ2  axis=(1,0,0)      7  MFJ1  axis=(1,0,0)
#   8  RFJ4  axis=(0,1,0)      9  RFJ3  axis=(1,0,0)
#  10  RFJ2  axis=(1,0,0)     11  RFJ1  axis=(1,0,0)
#  12  LFJ5  axis=(0.573576,0,0.819152)
#  13  LFJ4  axis=(0,1,0)     14  LFJ3  axis=(1,0,0)
#  15  LFJ2  axis=(1,0,0)     16  LFJ1  axis=(1,0,0)
#  17  THJ5  axis=(0,0,-1)    18  THJ4  axis=(1,0,0)
#  19  THJ3  axis=(1,0,0)     20  THJ2  axis=(0,-1,0)
#  21  THJ1  axis=(1,0,0)

_AXES = [
    (0, -1, 0),  # 0  FFJ4
    (1, 0, 0),  # 1  FFJ3
    (1, 0, 0),  # 2  FFJ2
    (1, 0, 0),  # 3  FFJ1
    (0, -1, 0),  # 4  MFJ4
    (1, 0, 0),  # 5  MFJ3
    (1, 0, 0),  # 6  MFJ2
    (1, 0, 0),  # 7  MFJ1
    (0, 1, 0),  # 8  RFJ4
    (1, 0, 0),  # 9  RFJ3
    (1, 0, 0),  # 10 RFJ2
    (1, 0, 0),  # 11 RFJ1
    (0.573576, 0, 0.819152),  # 12 LFJ5
    (0, 1, 0),  # 13 LFJ4
    (1, 0, 0),  # 14 LFJ3
    (1, 0, 0),  # 15 LFJ2
    (1, 0, 0),  # 16 LFJ1
    (0, 0, -1),  # 17 THJ5
    (1, 0, 0),  # 18 THJ4
    (1, 0, 0),  # 19 THJ3
    (0, -1, 0),  # 20 THJ2
    (1, 0, 0),  # 21 THJ1
]


def joint_world_transforms(
    base_se3: np.ndarray,
    dof: np.ndarray,
    kin=None,
) -> np.ndarray:
    """
    Shadow hand: 23 joint frames in world (1 base + 22 DOFs).

    Uses the MJCF body chain with joint-angle insertion (Rodrigues rotation
    right-multiplied into the parent frame at each joint location).

    MJCF body chain:
      T_free (wrist) -> T_palm(0,0,0.034) [fixed body, no joint]
      FF: palm -> knuckle(0.033, 0, 0.095) -> proximal(same) -> middle(+0.045z) -> distal(+0.025z)
      MF: palm -> knuckle(0.011, 0, 0.099) -> proximal(same) -> middle(+0.045z) -> distal(+0.025z)
      RF: palm -> knuckle(-0.011, 0, 0.095) -> proximal(same) -> middle(+0.045z) -> distal(+0.025z)
      LF: palm -> metacarpal(-0.033, 0, 0.02071) -> knuckle(+0.06579z) -> proximal(same) -> middle(+0.045z) -> distal(+0.025z)
      TH: palm -> thbase(0.034, -0.00858, 0.029) quat=(0.92388,0,0.382683,0) -> proximal(same)
          -> hub(+0.038z) -> middle(same) -> distal(+0.032z) quat=(1,0,0,-1)
    """
    T_body = np.asarray(base_se3, dtype=np.float64)
    d = np.asarray(dof, dtype=np.float64)

    Jp1 = 23  # 1 base + 22 DOF joints
    T = np.zeros((Jp1, 4, 4), dtype=np.float64)
    T[:] = np.eye(4)

    # index 0: freejoint (wrist base)
    T[0] = T_body

    # Palm is a fixed body at (0,0,0.034) from wrist — not a joint index
    T_palm = T_body @ _T44_from_pos_quat((0.0, 0.0, 0.034)).astype(np.float64)

    # --- FF (dof 0-3, joint indices 1-4) ---
    T_ff_kn = T_palm @ _T44_from_pos_quat((0.033, 0.0, 0.095)).astype(np.float64)
    T_ff_kn = _insert_revolute(T_ff_kn, _AXES[0], d[0]).astype(np.float64)  # FFJ4
    T[1] = T_ff_kn

    T_ff_prox = _insert_revolute(T_ff_kn, _AXES[1], d[1]).astype(np.float64)  # FFJ3
    T[2] = T_ff_prox

    T_ff_mid = T_ff_prox @ _T44_from_pos_quat((0.0, 0.0, 0.045)).astype(np.float64)
    T_ff_mid = _insert_revolute(T_ff_mid, _AXES[2], d[2]).astype(np.float64)  # FFJ2
    T[3] = T_ff_mid

    T_ff_dis = T_ff_mid @ _T44_from_pos_quat((0.0, 0.0, 0.025)).astype(np.float64)
    T_ff_dis = _insert_revolute(T_ff_dis, _AXES[3], d[3]).astype(np.float64)  # FFJ1
    T[4] = T_ff_dis

    # --- MF (dof 4-7, joint indices 5-8) ---
    T_mf_kn = T_palm @ _T44_from_pos_quat((0.011, 0.0, 0.099)).astype(np.float64)
    T_mf_kn = _insert_revolute(T_mf_kn, _AXES[4], d[4]).astype(np.float64)  # MFJ4
    T[5] = T_mf_kn

    T_mf_prox = _insert_revolute(T_mf_kn, _AXES[5], d[5]).astype(np.float64)  # MFJ3
    T[6] = T_mf_prox

    T_mf_mid = T_mf_prox @ _T44_from_pos_quat((0.0, 0.0, 0.045)).astype(np.float64)
    T_mf_mid = _insert_revolute(T_mf_mid, _AXES[6], d[6]).astype(np.float64)  # MFJ2
    T[7] = T_mf_mid

    T_mf_dis = T_mf_mid @ _T44_from_pos_quat((0.0, 0.0, 0.025)).astype(np.float64)
    T_mf_dis = _insert_revolute(T_mf_dis, _AXES[7], d[7]).astype(np.float64)  # MFJ1
    T[8] = T_mf_dis

    # --- RF (dof 8-11, joint indices 9-12) ---
    T_rf_kn = T_palm @ _T44_from_pos_quat((-0.011, 0.0, 0.095)).astype(np.float64)
    T_rf_kn = _insert_revolute(T_rf_kn, _AXES[8], d[8]).astype(np.float64)  # RFJ4
    T[9] = T_rf_kn

    T_rf_prox = _insert_revolute(T_rf_kn, _AXES[9], d[9]).astype(np.float64)  # RFJ3
    T[10] = T_rf_prox

    T_rf_mid = T_rf_prox @ _T44_from_pos_quat((0.0, 0.0, 0.045)).astype(np.float64)
    T_rf_mid = _insert_revolute(T_rf_mid, _AXES[10], d[10]).astype(np.float64)  # RFJ2
    T[11] = T_rf_mid

    T_rf_dis = T_rf_mid @ _T44_from_pos_quat((0.0, 0.0, 0.025)).astype(np.float64)
    T_rf_dis = _insert_revolute(T_rf_dis, _AXES[11], d[11]).astype(np.float64)  # RFJ1
    T[12] = T_rf_dis

    # --- LF (dof 12-16, joint indices 13-17) ---
    T_lf_meta = T_palm @ _T44_from_pos_quat((-0.033, 0.0, 0.02071)).astype(np.float64)
    T_lf_meta = _insert_revolute(T_lf_meta, _AXES[12], d[12]).astype(np.float64)  # LFJ5
    T[13] = T_lf_meta

    T_lf_kn = T_lf_meta @ _T44_from_pos_quat((0.0, 0.0, 0.06579)).astype(np.float64)
    T_lf_kn = _insert_revolute(T_lf_kn, _AXES[13], d[13]).astype(np.float64)  # LFJ4
    T[14] = T_lf_kn

    T_lf_prox = _insert_revolute(T_lf_kn, _AXES[14], d[14]).astype(np.float64)  # LFJ3
    T[15] = T_lf_prox

    T_lf_mid = T_lf_prox @ _T44_from_pos_quat((0.0, 0.0, 0.045)).astype(np.float64)
    T_lf_mid = _insert_revolute(T_lf_mid, _AXES[15], d[15]).astype(np.float64)  # LFJ2
    T[16] = T_lf_mid

    T_lf_dis = T_lf_mid @ _T44_from_pos_quat((0.0, 0.0, 0.025)).astype(np.float64)
    T_lf_dis = _insert_revolute(T_lf_dis, _AXES[16], d[16]).astype(np.float64)  # LFJ1
    T[17] = T_lf_dis

    # --- TH (dof 17-21, joint indices 18-22) ---
    T_th_base = T_palm @ _T44_from_pos_quat(
        (0.034, -0.00858, 0.029), (0.92388, 0.0, 0.382683, 0.0)
    ).astype(np.float64)
    T_th_base = _insert_revolute(T_th_base, _AXES[17], d[17]).astype(np.float64)  # THJ5
    T[18] = T_th_base

    T_th_prox = _insert_revolute(T_th_base, _AXES[18], d[18]).astype(np.float64)  # THJ4
    T[19] = T_th_prox

    T_th_hub = T_th_prox @ _T44_from_pos_quat((0.0, 0.0, 0.038)).astype(np.float64)
    T_th_hub = _insert_revolute(T_th_hub, _AXES[19], d[19]).astype(np.float64)  # THJ3
    T[20] = T_th_hub

    T_th_mid = _insert_revolute(T_th_hub, _AXES[20], d[20]).astype(np.float64)  # THJ2
    T[21] = T_th_mid

    T_th_dis = T_th_mid @ _T44_from_pos_quat(
        (0.0, 0.0, 0.032), (1.0, 0.0, 0.0, -1.0)
    ).astype(np.float64)
    T_th_dis = _insert_revolute(T_th_dis, _AXES[21], d[21]).astype(np.float64)  # THJ1
    T[22] = T_th_dis

    return T.astype(np.float32)
