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

import os

# respect an externally set fraction (needed on GPUs shared with other tenants)
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".99")
import gc
import glob
import shutil
import time

import hydra
import jax
import jax.numpy as jnp
import omegaconf
import wandb
from flax import nnx
from omegaconf import DictConfig

import kin_flow.util.const as CONST
from kin_flow.alg.selector import get_inputs_targets, get_train_fn
from kin_flow.ctrl.trainer import Trainer
from kin_flow.data.selector import get_gripper_loaders
from kin_flow.data.zipped import ZippedGripperLoader
from kin_flow.net.debug_wrapper import KinematicsFlowWOEncoding
from kin_flow.net.kinematics_flow import (KinematicsFlow,
                                          KinematicsFlowConfiguration)

BATCH_SIZE = 128
N_DEVICES = jax.local_device_count()

# ROCm: warm up the BLAS library on each device sequentially before pmap runs
# them concurrently — parallel first-use of rocBLAS across devices in one
# process can fail with rocblas_status_internal_error.
for _d in jax.local_devices():
    _x = jax.device_put(jnp.ones((4, 32, 32)), _d)
    jnp.einsum("bij,bjk->bik", _x, _x).block_until_ready()
del _x, _d


@hydra.main(config_path="config", config_name="train", version_base="1.2")
def train(cfg: DictConfig):
    CONST.configure(cfg)
    rngs = nnx.Rngs(cfg.seed)
    conf: dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    conf_net = KinematicsFlowConfiguration.from_dict_cls(conf["model"]["configuration"])

    gripper_list = list(conf_net.gripper)
    num_grippers = len(gripper_list) - 1 if "z0" in gripper_list else len(gripper_list)
    net = KinematicsFlow(
        conf_net,
        rngs=rngs,
    )
    if conf["debug_wo_encoding"]:
        net = KinematicsFlowWOEncoding(net, rngs=rngs)

    wandb.init(
        project=cfg.project_name,
        name=f"{cfg.run_name}_{cfg.num_scenes}",
        config={**conf, "BATCH_SIZE": BATCH_SIZE, "N_DEVICES": N_DEVICES},
    )

    loss_fn, stat_fn = get_train_fn(cfg.algorithm.name)
    trainer = Trainer(
        net,
        loss_fn=loss_fn,
        stat_fn=stat_fn,
        alg_cfg=cfg.algorithm.train,
        train_cfg=cfg.trainer_cfg,
        grippers=gripper_list,
    )
    key = jax.random.PRNGKey(0)
    train_counter = 0
    sample_stash = []

    # resume from the newest rolling step-checkpoint if one exists (written every
    # RESUME_EVERY_STEPS below; restores model + optimizer + LR-schedule step).
    # The data loader restarts at a fresh shuffled epoch, which is fine.
    resume_pattern = os.path.join(
        CONST.MODEL_CHECKPOINT, f"{cfg.run_name}_{cfg.num_scenes}_resume_step*"
    )

    def _resume_step(path):
        # dirs are named ..._resume_step<N> or ..._resume_step<N>_epoch<E>
        return int(path.rsplit("step", 1)[1].split("_epoch")[0])

    def _resume_epoch(path):
        return int(path.rsplit("_epoch", 1)[1]) if "_epoch" in path else None

    resume_dirs = sorted(glob.glob(resume_pattern), key=_resume_step)

    RESUME_EVERY_STEPS = 1000
    steps_per_epoch = max(1, cfg.num_scenes // cfg.num_scenes_per_batch)
    start_epoch = 0
    if resume_dirs:
        latest = resume_dirs[-1]
        trainer.restore(latest)
        train_counter = _resume_step(latest)
        resumed_epoch = _resume_epoch(latest)
        # Prefer the epoch recorded in the checkpoint name itself: deriving it
        # from train_counter // steps_per_epoch silently breaks if
        # num_scenes_per_batch (and therefore steps_per_epoch) ever changed
        # between when those steps were taken and now — train_counter is an
        # absolute, never-reset counter, so old steps taken under a different
        # batch size get misattributed to the new steps_per_epoch and inflate
        # the epoch count (bit us once: 5->15 batch resume mislabeled epoch
        # ~31 as epoch 91). Fall back to the derived value only for older
        # checkpoints saved before this fix.
        start_epoch = (
            resumed_epoch if resumed_epoch is not None else train_counter // steps_per_epoch
        )
        print(
            f"Resumed from {latest} (train step {train_counter}, epoch {start_epoch})",
            flush=True,
        )

    if conf["debug_wo_encoding"]:
        loader_list = get_gripper_loaders(
            gripper_list,
            num_scenes=cfg.num_scenes,
            rot_augmentation=cfg.rot_augmentation,
        )
        loader = ZippedGripperLoader(loader_list, batch_size=BATCH_SIZE, rngs=rngs)
        sample = next(iter(loader))
        loader = [sample]
    else:
        loader_list = get_gripper_loaders(
            gripper_list,
            num_scenes=cfg.num_scenes,
            rot_augmentation=cfg.rot_augmentation,
        )

    for epoch in range(start_epoch, cfg.epochs):
        if (epoch + 1) % cfg.save_every_epoch == 0:
            save_dir = os.path.join(
                CONST.MODEL_CHECKPOINT,
                f"{cfg.run_name}_{cfg.num_scenes}_{epoch+1}",
            )
            trainer.save(save_dir)
            gc.collect()

        if not conf["debug_wo_encoding"]:
            loader = ZippedGripperLoader(loader_list, batch_size=BATCH_SIZE, rngs=rngs)

        for sample in loader:
            key, subkey = jax.random.split(key)
            inputs, targets = get_inputs_targets(
                sample,
                cfg=cfg.algorithm,
                key=subkey,
            )
            inputs, targets = jax.tree.map(lambda x: jnp.asarray(x), (inputs, targets))

            sample_stash.append((inputs, targets))

            if len(sample_stash) == (cfg.num_scenes_per_batch) // num_grippers:
                targets_big = jnp.concatenate([t for (_, t) in sample_stash], axis=0)
                inputs_list = [ins for (ins, _) in sample_stash]
                inputs_big = jax.tree_util.tree_map(
                    lambda *xs: jnp.concatenate(xs, axis=0), *inputs_list
                )

                # Reshape the scene axis S_GLOBAL -> [D, S_LOCAL]
                def split_scene_axis(x):
                    if isinstance(x, jnp.ndarray) and x.ndim >= 1:
                        return x.reshape(
                            N_DEVICES,
                            cfg.num_scenes_per_batch // N_DEVICES,
                            *x.shape[1:],
                        )
                    return x

                inputs_sharded = jax.tree_util.tree_map(split_scene_axis, inputs_big)
                targets_sharded = split_scene_axis(targets_big)

                t0 = time.time()
                loss, aux = trainer.train_step(inputs_sharded, targets_sharded)
                loss = jax.block_until_ready(loss)
                step_time = time.time() - t0

                sample_stash.clear()
                train_counter += 1

                if (train_counter % RESUME_EVERY_STEPS) == 0:
                    resume_dir = os.path.join(
                        CONST.MODEL_CHECKPOINT,
                        f"{cfg.run_name}_{cfg.num_scenes}_resume_step{train_counter}_epoch{epoch}",
                    )
                    trainer.save(resume_dir)
                    for old in sorted(glob.glob(resume_pattern), key=_resume_step)[:-2]:
                        shutil.rmtree(old, ignore_errors=True)
                    gc.collect()

                wandb.log(
                    {
                        "loss": float(loss),
                        "epoch": epoch,
                        "step_time": step_time,
                        **{k: float(v) for k, v in aux.items()},
                    },
                    step=train_counter,
                )

                if (train_counter % 10) == 0:
                    print(f"Train Step: {step_time:.2f}s, Loss: {float(loss):.4f}")


if __name__ == "__main__":
    train()
