"""End-to-end grasp: capture -> infer -> select -> (visualize) -> execute.

Examples:
  # dry run: capture, sample grasps, write viz html, no robot motion
  python -m deploy.run_grasp --checkpoint me-full_25000_130 --dry-run

  # frame-convention gate: hover 10 cm above the best grasp, fingers open
  python -m deploy.run_grasp --checkpoint ... --host <ip> --hover --step

  # full grasp, confirming each motion
  python -m deploy.run_grasp --checkpoint ... --host <ip> --step
"""

import argparse
import time
from pathlib import Path

import numpy as np

from deploy.common import grasp_width, load_extrinsics
from deploy.infer import load_model, sample_grasps
from deploy.select import select_best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--extrinsics", default="deploy/config/extrinsics.yaml")
    parser.add_argument("--host", help="Franka controller IP (omit with --dry-run)")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="re-sample up to this many times if nothing is valid")
    parser.add_argument("--z0-id", type=int, default=None)
    parser.add_argument("--offline", help="use a saved scene_pcd.npz, skip camera")
    parser.add_argument("--out-dir", default="deploy_runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hover", action="store_true")
    parser.add_argument("--step", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not args.host:
        parser.error("--host is required unless --dry-run")

    out_dir = Path(args.out_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    extr = load_extrinsics(args.extrinsics)

    # ---- observe -----------------------------------------------------------
    if args.offline:
        pcd = np.load(args.offline)
        points, colors = pcd["points"], pcd["colors"]
    else:
        from deploy.perception import RealSenseCamera, capture_scene_cloud

        camera = RealSenseCamera()
        try:
            points, colors, rgbd = capture_scene_cloud(camera, extr["T_world_cam"])
            np.savez(out_dir / "raw.npz", rgbd=rgbd, intrinsics=camera.intrinsics())
        finally:
            camera.close()
    np.savez(out_dir / "scene_pcd.npz", points=points, colors=colors)
    print(f"cloud: {len(points)} pts, z [{points[:, 2].min():+.3f}, "
          f"{points[:, 2].max():+.3f}] m")

    # ---- infer + select (re-sample if no valid grasp) ----------------------
    model, conf = load_model(args.checkpoint, args.config_name)
    order = np.array([], dtype=int)
    for round_i in range(args.max_rounds):
        t0 = time.time()
        se3, dof, width = sample_grasps(
            model, conf, points, colors, args.num_samples, args.z0_id
        )
        order, diag = select_best(se3, dof, points, top_k=10)
        print(f"round {round_i + 1}: {len(se3)} grasps in {time.time() - t0:.1f}s, "
              f"{int(diag['valid'].sum())} valid after filtering")
        if len(order) > 0:
            break
    if len(order) == 0:
        raise SystemExit("No valid grasp found — adjust scene/camera and retry.")

    np.savez(
        out_dir / "grasps_ranked.npz",
        se3=se3[order], dof=dof[order], width=grasp_width(dof[order]),
        hand_pose=diag["hand_pose"][order], score=diag["score"][order],
    )

    try:
        from kin_flow.kin.const import KINS_NP
        from kin_flow.util.viz.grasp_viz import viz_grasps

        fig = viz_grasps(
            gripper_name="panda", scene_pcd=points, se3=se3[order], dof=dof[order],
            kin=KINS_NP["panda"], scene_colors=colors,
        )
        fig.write_html(out_dir / "grasps.html")
        print(f"viz: {out_dir / 'grasps.html'}")
    except Exception as e:  # viz must never block execution
        print(f"viz skipped: {e}")

    if args.dry_run:
        print(f"dry run complete, artifacts in {out_dir}")
        return

    # ---- execute: first ranked grasp that is IK-reachable ------------------
    from deploy.execute import FrankaExecutor, execute_grasp

    executor = FrankaExecutor(args.host)
    widths = grasp_width(dof[order])
    for rank, idx in enumerate(order):
        print(f"executing ranked grasp {rank} "
              f"(score {diag['score'][idx]:.2f}, width {widths[rank] * 1000:.0f} mm)")
        ok = execute_grasp(
            executor,
            diag["hand_pose"][idx],
            float(widths[rank]),
            extr["T_base_world"],
            hover=args.hover,
            step=args.step,
        )
        if ok:
            print("SUCCESS")
            return
        if args.hover:
            return  # hover mode: only inspect the best grasp
        print("failed/unreachable, trying next candidate")
    raise SystemExit("all candidates failed")


if __name__ == "__main__":
    main()
