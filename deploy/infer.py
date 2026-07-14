"""Checkpoint loading + grasp sampling for deployment (CPU-friendly).

Reuses the training Hydra config to build the model, restores the orbax
checkpoint exactly like kin_flow/cli/bench.py, and wraps
kin_flow.alg.flow.inference.inference for a raw (points, colors) cloud.

Usage:
    python -m deploy.infer --pcd scene_pcd.npz --checkpoint me-full_25000_130 \
        --num-samples 32 --out grasps.npz [--viz grasps.html] [--z0-id 5]
"""

import argparse
from pathlib import Path

import numpy as np
from flax import nnx
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import kin_flow.util.const as CONST
from deploy.common import grasp_width

CONFIG_DIR = Path(__file__).resolve().parent.parent / "kin_flow" / "cli" / "config"


def load_model(checkpoint: str, config_name: str = "train"):
    """Build KinematicsFlow from the repo Hydra config and restore weights."""
    from kin_flow.ctrl.trainer import Trainer
    from kin_flow.net.kinematics_flow import (KinematicsFlow,
                                              KinematicsFlowConfiguration)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2"):
        cfg = compose(config_name=config_name)
    CONST.configure(cfg)
    conf = OmegaConf.to_container(cfg, resolve=True)
    conf_net = KinematicsFlowConfiguration.from_dict_cls(conf["model"]["configuration"])
    model = KinematicsFlow(conf_net, rngs=nnx.Rngs(0))
    model_path = Path(CONST.MODEL_CHECKPOINT) / checkpoint
    if not model_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {model_path}")
    model = Trainer.get_model_from_checkpoint(model, model_path)
    return model, conf


def sample_grasps(
    model,
    conf: dict,
    points: np.ndarray,
    colors: np.ndarray,
    num_samples: int = 32,
    z0_id: int | None = None,
):
    """Sample grasps for the panda on a single world-frame cloud.

    Returns (se3, dof, width): contact-frame poses (N,4,4) in meters/world,
    finger joints (N,2), and opening width (N,) in meters.
    """
    from kin_flow.alg.selector import inference

    gripper_list = conf["model"]["configuration"]["gripper"]
    inference_cfg = OmegaConf.create(
        {
            "gripper_id": 0,  # panda
            "z0_id": (
                z0_id
                if z0_id is not None
                else (gripper_list.index("z0") if "z0" in gripper_list else -1)
            ),
            "integrator_steps": conf["algorithm"]["inference"]["integrator_steps"],
            "position_scaling": conf["algorithm"]["position_scaling"],
            "gripper": gripper_list,
        }
    )
    sample = {
        "scene_points": [np.asarray(points, dtype=np.float32)],
        "scene_colors": [np.asarray(colors, dtype=np.float32)],
    }
    se3, dof = inference(
        name=conf["algorithm"]["name"],
        model=model,
        sample=sample,
        num_samples=num_samples,
        cfg=inference_cfg,
    )
    se3, dof = se3[:, -1], dof[:, -1]
    return se3, dof, grasp_width(dof)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", required=True, help="scene_pcd.npz (points, colors)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument(
        "--z0-id", type=int, default=None,
        help="override guidance embedding id (-1 disables; default: from config)",
    )
    parser.add_argument("--out", default="grasps.npz")
    parser.add_argument("--viz", help="write a plotly html of cloud + grasps")
    args = parser.parse_args()

    pcd = np.load(args.pcd)
    points, colors = pcd["points"], pcd["colors"]

    import time

    model, conf = load_model(args.checkpoint, args.config_name)
    t0 = time.time()
    se3, dof, width = sample_grasps(
        model, conf, points, colors, args.num_samples, args.z0_id
    )
    print(f"Sampled {len(se3)} grasps in {time.time() - t0:.1f}s")

    np.savez(args.out, se3=se3, dof=dof, width=width)
    print(f"Saved grasps to {args.out}")

    if args.viz:
        from kin_flow.kin.const import KINS_NP
        from kin_flow.util.viz.grasp_viz import viz_grasps

        fig = viz_grasps(
            gripper_name="panda",
            scene_pcd=points,
            se3=se3,
            dof=dof,
            kin=KINS_NP["panda"],
            scene_colors=colors,
        )
        fig.write_html(args.viz)
        print(f"Wrote visualization to {args.viz}")


if __name__ == "__main__":
    main()
