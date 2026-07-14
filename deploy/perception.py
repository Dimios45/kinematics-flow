"""RealSense -> 15k-point world-frame cloud, replicating the sim contract.

The processing chain mirrors mgs/cli/render_scene.py exactly:
deproject -> world frame -> crop box -> 2 mm voxel downsample -> radius
outlier removal -> FPS to 15000 points.

Usage (robot PC):
    python -m deploy.perception --extrinsics deploy/config/extrinsics.yaml \
        --out scene_pcd.npz [--save-raw raw.npz] [--frames 10]
Offline (re-process a saved raw capture):
    python -m deploy.perception --offline raw.npz --extrinsics ... --out scene_pcd.npz
"""

import argparse
import warnings

import numpy as np

from mgs.util.img_proc import detect_outlier, voxel_downsample_pcd

from deploy.common import (CROP_MAX, CROP_MIN, NUM_POINTS,
                           OUTLIER_MIN_NEIGHBORS, OUTLIER_RADIUS, VOXEL_SIZE,
                           load_extrinsics)

MAX_DEPTH = 1.5  # m, discard far background before cropping


class RealSenseCamera:
    """Thin pyrealsense2 wrapper: aligned RGB-D in meters + color intrinsics."""

    def __init__(self, width=848, height=480, fps=30, warmup_frames=15):
        import pyrealsense2 as rs  # lazy: only needed on the robot PC

        self._rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        profile = self.pipeline.start(config)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        for _ in range(warmup_frames):  # let auto-exposure settle
            self.pipeline.wait_for_frames()

    def intrinsics(self) -> np.ndarray:
        """3x3 color intrinsics (depth is aligned to the color frame)."""
        stream = self.pipeline.get_active_profile().get_stream(self._rs.stream.color)
        i = stream.as_video_stream_profile().get_intrinsics()
        return np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1.0]])

    def capture_rgbd(self, num_frames: int = 10) -> np.ndarray:
        """(H, W, 4) float array: RGB in [0,1], depth in meters (median of
        num_frames; pixels invalid in more than half of the frames get 0)."""
        depths, color = [], None
        for _ in range(num_frames):
            frames = self.align.process(self.pipeline.wait_for_frames())
            depth = np.asanyarray(frames.get_depth_frame().get_data())
            depths.append(depth.astype(np.float32) * self.depth_scale)
            color = np.asanyarray(frames.get_color_frame().get_data())
        depth_stack = np.stack(depths)
        depth_stack[depth_stack == 0.0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels
            depth = np.nanmedian(depth_stack, axis=0)
        depth = np.nan_to_num(depth, nan=0.0)
        rgb = color.astype(np.float32) / 255.0
        return np.concatenate([rgb, depth[..., None]], axis=-1)

    def close(self):
        self.pipeline.stop()


def rgbd_to_world_cloud(rgbd, intrinsics, T_world_cam):
    """Deproject one (H, W, 4) RGB-D image to valid world-frame points/colors.

    Same pinhole math as mgs.util.img_proc.rgbd_to_pcd, but correct for
    non-square images (that helper swaps H/W and only works for the sim's
    square 480x480 renders).
    """
    h, w = rgbd.shape[:2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = rgbd[..., -1]
    u = np.arange(w)[None, :] - cx
    v = np.arange(h)[:, None] - cy
    points_cam = np.stack([(z * u) / fx, (z * v) / fy, z], axis=-1)
    points = points_cam @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
    valid = (z > 0.0) & (z < MAX_DEPTH)
    return points[valid], rgbd[..., :3][valid]


def process_cloud(points, colors, num_points=NUM_POINTS, seed=None):
    """Apply the sim's crop/voxel/outlier/FPS chain to a raw world-frame cloud."""
    region = np.all((points < CROP_MAX) & (points > CROP_MIN), axis=-1)
    points, colors = points[region], colors[region]
    if len(points) == 0:
        raise RuntimeError(
            "No points inside the workspace crop box — check extrinsics/calibration."
        )

    points, colors = voxel_downsample_pcd(points, colors, voxel_size=VOXEL_SIZE)
    inliers = detect_outlier(
        points, radius=OUTLIER_RADIUS, min_neighbors=OUTLIER_MIN_NEIGHBORS
    )
    points, colors = points[inliers], colors[inliers]

    if len(points) < num_points:
        warnings.warn(
            f"Only {len(points)} points after filtering (< {num_points}); "
            "padding by duplication. Consider a closer camera or more views."
        )
        rng = np.random.default_rng(seed)
        pad = rng.choice(len(points), size=num_points - len(points))
        idx = np.concatenate([np.arange(len(points)), pad])
    else:
        from kin_flow.util.fps import farthest_point_sampling_np

        idx = farthest_point_sampling_np(points, num_samples=num_points, seed=seed)
    return points[idx].astype(np.float32), colors[idx].astype(np.float32)


def capture_scene_cloud(camera: RealSenseCamera, T_world_cam, num_frames=10):
    rgbd = camera.capture_rgbd(num_frames)
    points, colors = rgbd_to_world_cloud(rgbd, camera.intrinsics(), T_world_cam)
    return (*process_cloud(points, colors), rgbd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extrinsics", required=True)
    parser.add_argument("--out", default="scene_pcd.npz")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--save-raw", help="also save the raw RGB-D + K (npz)")
    parser.add_argument(
        "--offline", help="process a previously saved raw npz instead of the camera"
    )
    args = parser.parse_args()

    T_world_cam = load_extrinsics(args.extrinsics)["T_world_cam"]

    if args.offline:
        raw = np.load(args.offline)
        points, colors = rgbd_to_world_cloud(
            raw["rgbd"], raw["intrinsics"], T_world_cam
        )
        points, colors = process_cloud(points, colors)
    else:
        camera = RealSenseCamera()
        try:
            points, colors, rgbd = capture_scene_cloud(
                camera, T_world_cam, args.frames
            )
            if args.save_raw:
                np.savez(
                    args.save_raw, rgbd=rgbd, intrinsics=camera.intrinsics()
                )
        finally:
            camera.close()

    np.savez(args.out, points=points, colors=colors)
    print(
        f"Saved {len(points)} points to {args.out} | "
        f"x [{points[:, 0].min():+.3f}, {points[:, 0].max():+.3f}] "
        f"y [{points[:, 1].min():+.3f}, {points[:, 1].max():+.3f}] "
        f"z [{points[:, 2].min():+.3f}, {points[:, 2].max():+.3f}] m"
    )


if __name__ == "__main__":
    main()
