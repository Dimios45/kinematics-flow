"""Grasp filtering + ranking against the scene point cloud.

Replaces the sim's grasp_collision_mask (which needs a MuJoCo scene) with the
paper's analytical point-cloud approach: the hand and fingers are approximated
by boxes in the hand frame; any scene point inside them is a collision. The
raw sampler produces ~10-20% colliding grasps, so this stage is mandatory.

Input se3 = raw model output in world frame — which IS the hand-base pose
(origin at the hand base, +z = approach, fingers along +-y, grasp point at
+0.102 z; empirically verified against the training data, see common.py).
"""

import argparse

import numpy as np

from deploy.common import (CROP_MIN, PANDA_MAX_WIDTH, SIM_TCP_OFFSET,
                           grasp_width, se3_inv)

# Approximate Franka Hand geometry in the hand-base frame (+z toward the
# fingertips, y = finger opening axis). Sized from the menagerie panda
# hand/finger meshes, validated against the sim's ground-truth grasps (see
# deploy/README.md). Fingers are checked AT the predicted opening width — the
# sim validates grasps that way, and the predicted width already includes the
# ~8 mm clearance added at data generation (mgs GripperPanda._clamp_width).
PALM_X_HALF = 0.035
PALM_Y_HALF = 0.103  # wide hand housing (wrist side, along the finger axis)
PALM_Z = (-0.06, 0.042)  # includes 6 cm of wrist/flange behind the hand base
NECK_X_HALF = 0.022
NECK_Y_HALF = 0.045  # the housing tapers toward the finger mount
NECK_Z = (0.042, 0.0584)
FINGER_X_HALF = 0.011
FINGER_THICKNESS = 0.012  # each finger, along y beyond the opening width
FINGER_Z = (0.0584, 0.115)  # finger mount to just past the fingertip pads
CLOSING_X_HALF = 0.015  # region between the fingertips that must hold points
CLOSING_Z_HALF = 0.02
SAFETY_MARGIN = 0.002  # inflate collision boxes by this much on every side


def _in_box(p, x_half, y_lo, y_hi, z_lo, z_hi):
    return (
        (np.abs(p[:, 0]) < x_half)
        & (p[:, 1] > y_lo)
        & (p[:, 1] < y_hi)
        & (p[:, 2] > z_lo)
        & (p[:, 2] < z_hi)
    )


def evaluate_grasps(se3, dof, points, min_closing_points=20):
    """Score each contact-frame grasp against the scene cloud.

    Returns a dict of per-grasp arrays:
      valid        collision-free, in-workspace, approaching from above,
                   with enough points between the fingers
      score        higher = better (only meaningful where valid)
      n_collision  scene points inside palm/finger boxes
      n_closing    scene points inside the closing region
      hand_pose    (N,4,4) world hand-base poses == se3 (execute.py input)
    """
    se3 = np.asarray(se3)
    widths = grasp_width(dof)
    m = SAFETY_MARGIN

    n = len(se3)
    n_collision = np.zeros(n, dtype=int)
    n_closing = np.zeros(n, dtype=int)
    in_workspace = np.zeros(n, dtype=bool)
    from_above = np.zeros(n, dtype=bool)
    downwardness = np.zeros(n)

    for i in range(n):
        T_wh = se3[i]
        approach = T_wh[:3, 2]  # hand z axis, toward the fingertips
        grasp_pos = T_wh[:3, 3] + SIM_TCP_OFFSET * approach

        # sim workspace bounds (mgs grasp_collision_mask uses +-0.20)
        in_workspace[i] = (
            np.all(np.abs(grasp_pos[:2]) < 0.20)
            and CROP_MIN[2] < grasp_pos[2] < 0.5
        )
        # never approach from below the table; prefer more top-down grasps
        downwardness[i] = -approach[2]
        from_above[i] = approach[2] < 0.2

        p = (se3_inv(T_wh)[:3, :3] @ points.T).T + se3_inv(T_wh)[:3, 3]
        half_w = min(widths[i], PANDA_MAX_WIDTH) / 2.0

        palm = _in_box(
            p, PALM_X_HALF + m, -(PALM_Y_HALF + m), PALM_Y_HALF + m,
            PALM_Z[0] - m, PALM_Z[1] + m,
        ) | _in_box(
            p, NECK_X_HALF + m, -(NECK_Y_HALF + m), NECK_Y_HALF + m,
            NECK_Z[0] - m, NECK_Z[1] + m,
        )
        fingers = _in_box(
            p, FINGER_X_HALF + m, half_w, half_w + FINGER_THICKNESS + m,
            FINGER_Z[0] - m, FINGER_Z[1] + m,
        ) | _in_box(
            p, FINGER_X_HALF + m, -(half_w + FINGER_THICKNESS + m), -half_w,
            FINGER_Z[0] - m, FINGER_Z[1] + m,
        )
        closing = _in_box(
            p, CLOSING_X_HALF, -half_w, half_w,
            SIM_TCP_OFFSET - CLOSING_Z_HALF, SIM_TCP_OFFSET + CLOSING_Z_HALF,
        )
        n_collision[i] = int(np.count_nonzero(palm | fingers))
        n_closing[i] = int(np.count_nonzero(closing))

    valid = (
        (n_collision == 0)
        & (n_closing >= min_closing_points)
        & in_workspace
        & from_above
    )
    # rank: prefer top-down approaches; mild bonus for a fuller closing region
    score = downwardness + 0.001 * np.minimum(n_closing, 200)
    return {
        "valid": valid,
        "score": score,
        "n_collision": n_collision,
        "n_closing": n_closing,
        "in_workspace": in_workspace,
        "from_above": from_above,
        "hand_pose": se3,
    }


def select_best(se3, dof, points, top_k=5, min_closing_points=20):
    """Return indices of the top_k valid grasps, best first, plus diagnostics."""
    diag = evaluate_grasps(se3, dof, points, min_closing_points)
    valid_idx = np.flatnonzero(diag["valid"])
    order = valid_idx[np.argsort(-diag["score"][valid_idx])][:top_k]
    return order, diag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasps", required=True, help="grasps.npz from deploy.infer")
    parser.add_argument("--pcd", required=True, help="scene_pcd.npz")
    parser.add_argument("--out", default="grasps_ranked.npz")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    grasps = np.load(args.grasps)
    points = np.load(args.pcd)["points"]
    se3, dof = grasps["se3"], grasps["dof"]

    order, diag = select_best(se3, dof, points, top_k=args.top_k)
    n = len(se3)
    print(
        f"{n} grasps: {int(diag['valid'].sum())} valid | "
        f"{int((diag['n_collision'] > 0).sum())} colliding, "
        f"{int((~diag['from_above']).sum())} from below, "
        f"{int((diag['n_closing'] < 20).sum())} empty closing region, "
        f"{int((~diag['in_workspace']).sum())} out of workspace"
    )
    if len(order) == 0:
        raise SystemExit("No valid grasp — re-sample or adjust the scene.")

    np.savez(
        args.out,
        se3=se3[order],
        dof=dof[order],
        width=grasp_width(dof[order]),
        hand_pose=diag["hand_pose"][order],
        score=diag["score"][order],
    )
    print(f"Saved top {len(order)} grasps to {args.out} (best first)")


if __name__ == "__main__":
    main()
