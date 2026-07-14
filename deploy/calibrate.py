"""World-frame calibration: ChArUco board on the table defines the world.

Place the board flat at the workspace center. The world frame is the board
center: z up out of the board, table plane = z0 — matching the sim frame.

Three subcommands:
  camera  detect the board with the RealSense -> T_world_cam
  robot   touch known board points with the closed fingertips -> T_base_world
  check   report table-plane z statistics of the live depth in world frame

Examples:
  python -m deploy.calibrate camera --extrinsics deploy/config/extrinsics.yaml
  python -m deploy.calibrate robot --host <panda-ip> --extrinsics ...
  python -m deploy.calibrate check --extrinsics ...
"""

import argparse

import numpy as np

from deploy.common import load_extrinsics, save_extrinsics, se3_inv

# Default board: 7x5 ChArUco, 40 mm squares, 30 mm markers, DICT_5X5_100.
# Print at 100% scale and verify square size with a ruler.
BOARD = dict(squares_x=7, squares_y=5, square_len=0.04, marker_len=0.03)

# Points the robot touches for base<->world calibration, in world coordinates
# (board-center frame, meters). Mark them on the printout.
TOUCH_POINTS_WORLD = np.array(
    [
        [0.0, 0.0, 0.0],  # board center
        [0.10, 0.0, 0.0],  # +x
        [0.0, 0.08, 0.0],  # +y
        [-0.10, -0.08, 0.0],
    ]
)


def _make_board(cv2):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    return cv2.aruco.CharucoBoard(
        (BOARD["squares_x"], BOARD["squares_y"]),
        BOARD["square_len"],
        BOARD["marker_len"],
        dictionary,
    )


def calibrate_camera(args):
    import cv2

    from deploy.perception import RealSenseCamera

    camera = RealSenseCamera()
    try:
        rgbd = camera.capture_rgbd(num_frames=5)
        intrinsics = camera.intrinsics()
    finally:
        camera.close()

    gray = cv2.cvtColor((rgbd[..., :3] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    board = _make_board(cv2)
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) < 6:
        raise SystemExit("Board not detected (or too few corners) — check view/light.")

    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, intrinsics, None)
    if not ok:
        raise SystemExit("solvePnP failed.")

    proj, _ = cv2.projectPoints(obj_points, rvec, tvec, intrinsics, None)
    err = np.linalg.norm(proj.squeeze(1) - img_points.squeeze(1), axis=1)
    print(f"{len(charuco_ids)} corners, reprojection error "
          f"mean {err.mean():.2f}px max {err.max():.2f}px")
    if err.mean() > 1.0:
        print("WARNING: high reprojection error — recheck board flatness/print scale.")

    # solvePnP gives board(corner frame) -> cam; shift origin to board center
    T_cam_board = np.eye(4)
    T_cam_board[:3, :3], _ = cv2.Rodrigues(rvec)
    T_cam_board[:3, 3] = tvec.squeeze()
    center = np.array(
        [
            BOARD["squares_x"] * BOARD["square_len"] / 2,
            BOARD["squares_y"] * BOARD["square_len"] / 2,
            0.0,
        ]
    )
    T_board_world = np.eye(4)
    T_board_world[:3, 3] = center
    T_world_cam = se3_inv(T_cam_board @ T_board_world)

    save_extrinsics(args.extrinsics, T_world_cam=T_world_cam)
    cam_pos = T_world_cam[:3, 3]
    print(f"Camera at {np.round(cam_pos, 3)} m in world "
          f"(distance {np.linalg.norm(cam_pos):.3f} m). Saved to {args.extrinsics}")


def calibrate_robot(args):
    import panda_py

    panda = panda_py.Panda(args.host)
    measured = []
    print(
        "Guide the arm so the CLOSED fingertips touch each marked point.\n"
        "The robot is in free-guiding mode; press Enter to record each point."
    )
    panda.teaching_mode(True)
    try:
        for p_world in TOUCH_POINTS_WORLD:
            input(f"  touch world point {p_world} then press Enter... ")
            pose = np.asarray(panda.get_pose())
            measured.append(pose[:3, 3])
            print(f"    recorded EE position {np.round(measured[-1], 4)}")
    finally:
        panda.teaching_mode(False)
    p_base = np.asarray(measured)

    # Kabsch: p_base = R @ p_world + t
    p_w, p_b = TOUCH_POINTS_WORLD, p_base
    cw, cb = p_w.mean(0), p_b.mean(0)
    H = (p_w - cw).T @ (p_b - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = cb - R @ cw

    T_base_world = np.eye(4)
    T_base_world[:3, :3] = R
    T_base_world[:3, 3] = t
    resid = np.linalg.norm((p_w @ R.T + t) - p_b, axis=1)
    print(f"Fit residual mean {resid.mean() * 1000:.1f} mm "
          f"max {resid.max() * 1000:.1f} mm")
    if resid.max() > 0.005:
        print("WARNING: residual > 5 mm — repeat the touch points carefully.")

    save_extrinsics(args.extrinsics, T_base_world=T_base_world)
    print(f"Saved T_base_world to {args.extrinsics}")


def check(args):
    from deploy.perception import RealSenseCamera, rgbd_to_world_cloud

    T_world_cam = load_extrinsics(args.extrinsics)["T_world_cam"]
    camera = RealSenseCamera()
    try:
        rgbd = camera.capture_rgbd(num_frames=5)
        points, _ = rgbd_to_world_cloud(rgbd, camera.intrinsics(), T_world_cam)
    finally:
        camera.close()

    # empty-table assumption: everything in the crop box near z=0 is table
    near = points[
        np.all(np.abs(points[:, :2]) < 0.2, axis=1) & (np.abs(points[:, 2]) < 0.05)
    ]
    if len(near) < 100:
        raise SystemExit("Too few table points — is the board/table in view?")
    z = near[:, 2]
    print(
        f"Table plane in world frame ({len(near)} pts): "
        f"z mean {z.mean() * 1000:+.1f} mm, std {z.std() * 1000:.1f} mm, "
        f"p5/p95 {np.percentile(z, 5) * 1000:+.1f}/{np.percentile(z, 95) * 1000:+.1f} mm"
    )
    print("Expect |mean| < 5 mm on an empty table (clear objects before checking).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in [("camera", calibrate_camera), ("robot", calibrate_robot),
                     ("check", check)]:
        p = sub.add_parser(name)
        p.add_argument("--extrinsics", default="deploy/config/extrinsics.yaml")
        if name == "robot":
            p.add_argument("--host", required=True, help="Franka controller IP")
        p.set_defaults(fn=fn)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
