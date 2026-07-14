# Real-hardware deployment: Franka Panda + 1 front-mounted RealSense

Runs the trained kinematics-flow grasp generator on a real Franka. The model
is a **single-shot grasp-pose sampler** (point cloud in → SE(3) grasp poses +
finger width out); this package supplies everything the repo doesn't:
camera capture, calibration, grasp filtering, and arm execution.

## Pipeline

```
RealSense RGB-D ──deproject──> world-frame cloud ──crop/voxel/FPS──> 15k points
      │  (T_world_cam from board calibration)          (same chain as sim)
      v
 deploy.infer (KinematicsFlow, CPU JAX) ──> N contact-frame grasp poses + widths
      v
 deploy.select: collision filter (hand-box vs cloud), workspace, approach dir
      v
 deploy.execute (panda-py): pre-grasp → approach → close → lift 0.3 m
      (T_base_world from robot touch-point calibration)
```

## Robot-PC setup

1. PC with RT kernel (`PREEMPT_RT`) on the Franka's network (see libfranka docs).
2. Clone this repo, then:
   ```bash
   uv venv && uv pip install -e . -e thrd_party/mj-grasp-sim
   uv pip install pyrealsense2 opencv-contrib-python panda-python pyyaml
   ```
   Plain CPU `jax==0.8.0` (pulled in by the project) is enough — no GPU needed.
3. Copy a checkpoint directory from the training box to `../checkpoint/<name>`
   (relative to the repo root), e.g. `me-full_25000_130`.

## One-time calibration

1. Print the 7×5 ChArUco board (40 mm squares, DICT_5X5_100) at 100 % scale,
   verify with a ruler, and tape it flat at the **workspace center**. Its
   center becomes the world origin; the model's workspace is ±0.225 m around
   it, so the robot must be able to reach that area.
2. Mount the camera in front of the scene, pitched down ~30–45°, ~0.7–0.9 m
   from the board (matches the sim's 0.75 m camera sphere).
3. `python -m deploy.calibrate camera --extrinsics deploy/config/extrinsics.yaml`
4. `python -m deploy.calibrate robot --host <robot-ip> --extrinsics ...`
   (guide the closed fingertips onto the 4 marked board points)
5. `python -m deploy.calibrate check --extrinsics ...` on the empty table:
   the table plane must sit at |z| < 5 mm.

Remove the board before grasping (or leave it — it's flat and gets cropped
into the table plane either way).

## Staged bring-up (do these in order)

1. **Offline**: `python -m deploy.run_grasp --checkpoint <ckpt> --dry-run
   --offline <a test scene_pcd.npz>` — sanity-check grasps in `grasps.html`.
2. **Perception**: place 2–3 objects, `python -m deploy.perception
   --extrinsics ... --out scene.npz`, then a dry-run on it; check the cloud
   and grasps visually.
3. **Hover test** (frame-convention gate): `python -m deploy.run_grasp
   --checkpoint <ckpt> --host <ip> --hover --step` — arm stops 10 cm above
   the grasp with fingers open. The fingers must straddle the object when you
   lower the arm by hand. Do not skip this on a new setup.
4. **Full grasp**: same command without `--hover` (keep `--step` at first).

## Safety notes

- `--step` asks for confirmation before every motion — use it until trusted.
- Speed is limited (`SPEED_FACTOR = 0.15` in `execute.py`); keep the
  operator's hand on the external enabling/e-stop device.
- The selector rejects grasps approaching from below and grasps whose hand
  volume intersects the cloud, but the cloud only shows what the camera sees:
  **keep the area behind the objects clear** (single front view = occlusion).

## Frame conventions (empirically verified — do not "fix" these)

- The model's raw output pose IS the Franka Hand **base** pose: origin at the
  hand base, +z = approach, fingers along ±y, grasp point at +0.102 z.
  `bench.py`'s `B2C_TRANSFORM["panda"]` and the sim gripper's
  `base_to_contact_transform()` are exact inverses that cancel inside the sim
  pipeline — neither is applied here.
- Verified against training scenes: the grasp point (hand origin + 0.102·ẑ)
  lands ~1 cm from the scene cloud, and ground-truth grasps are only
  collision-free with fingers along ±y of that frame.
- Finger width from joints: `w = j1 + j2 + 0.04` (right finger body sits at
  y=−0.04), NOT `j1 − j2`.

## Known limitations / tuning knobs

- Single view vs the sim's 10 fused views: expect degraded success on heavily
  occluded scenes. First knob: camera pitch; second: capture 2–3 static views
  once at setup and fuse them in `perception.py`.
- `num_samples=32` keeps CPU inference fast; raise it (or `--max-rounds`) if
  the filter rejects too many grasps.
- Hand collision boxes in `select.py` are conservative approximations —
  loosen `SAFETY_MARGIN` only after real-world validation.
- The paper's learned grasp-stability ranker was not released; ranking here is
  geometric (top-downness + closing-region fill).
