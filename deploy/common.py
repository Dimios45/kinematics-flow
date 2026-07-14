"""Shared constants and frame helpers for real-hardware deployment.

Frames
------
world  : workspace frame defined by the calibration board on the table.
         Origin at the board origin (workspace center), z up, table plane = z0.
         This matches the sim frame the model was trained in (see
         mgs/cli/render_scene.py crop box).
cam    : RealSense color-optical frame (z forward, x right, y down).
base   : Franka base frame (link0).
hand   : the model's raw output pose IS the Franka/menagerie hand-base frame
         ("panda_hand"): origin at the hand base, +z toward the fingertips,
         fingers opening along +-y, grasp point ~0.102 m along +z.
         (bench.py's B2C_TRANSFORM and the sim gripper's
         base_to_contact_transform are exact inverses and cancel — verified
         empirically against the training data; see PANDA_B2C below.)

extrinsics.yaml stores
  T_world_cam : p_world = T_world_cam @ p_cam   (what rgbd_to_pcd expects)
  T_base_world: p_base  = T_base_world @ p_world
"""

from pathlib import Path

import numpy as np
import yaml

# --- observation contract (must match mgs/cli/render_scene.py exactly) -----
CROP_MIN = np.array([-0.225, -0.225, -0.01])
CROP_MAX = np.array([0.225, 0.225, 1.0])
VOXEL_SIZE = 0.002
OUTLIER_RADIUS = 0.008
OUTLIER_MIN_NEIGHBORS = 2
NUM_POINTS = 15000

# --- gripper constants ------------------------------------------------------
# Copied from kin_flow/cli/bench.py B2C_TRANSFORM["panda"] for reference.
# NOT applied anywhere in deploy/: the sim pipeline computes
# se3 @ PANDA_B2C @ GripperPanda.base_to_contact_transform(), and those two
# transforms are exact inverses (rot z -/+90 deg, z +/-0.102 m) — they
# cancel, so the raw model output already is the hand-base pose. Verified
# against the training data: grasp point (= hand origin + 0.102 z) lands ~1cm
# from the scene cloud, and ground-truth grasps are collision-free only with
# fingers along the hand y axis.
PANDA_B2C = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.102],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

PANDA_MAX_WIDTH = 0.08  # m, mgs/gripper/panda.py
PANDA_MIN_WIDTH = 0.003
SIM_TCP_OFFSET = 0.102  # m, sim grasp point along hand z (kin/data convention)
HAND_TCP_OFFSET = 0.1034  # m, real Franka Hand TCP along hand z

# franka_description: panda_hand mounted on the flange (link8) rotated -45deg
# about z, no translation.
_C45 = np.cos(np.pi / 4)


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


T_FLANGE_HAND = rot_z(-np.pi / 4)


def se3_inv(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def load_extrinsics(path: str | Path) -> dict[str, np.ndarray]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    out = {}
    for key in ("T_world_cam", "T_base_world"):
        if key in raw and raw[key] is not None:
            mat = np.asarray(raw[key], dtype=np.float64)
            assert mat.shape == (4, 4), f"{key} must be 4x4"
            out[key] = mat
    return out


def save_extrinsics(path: str | Path, **mats: np.ndarray) -> None:
    path = Path(path)
    existing = {}
    if path.exists():
        existing = load_extrinsics(path)
    existing.update(mats)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(
            {k: np.asarray(v).tolist() for k, v in existing.items()},
            f,
            sort_keys=True,
        )


def grasp_width(dof: np.ndarray) -> np.ndarray:
    """Panda finger joints -> opening width.

    Inverse of mgs GripperPanda.width_to_joints (j1 = w/2, j2 = w/2 - 0.04):
    the right finger body sits at y=-0.04 in the hand frame, so
    w = j1 + j2 + 0.04 (NOT j1 - j2).
    """
    dof = np.atleast_2d(dof)
    return np.clip(dof[:, 0] + dof[:, 1] + 0.04, 0.0, PANDA_MAX_WIDTH)
