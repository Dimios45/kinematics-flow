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

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
import gc
import time

import hydra
import jax
import jax.numpy as jnp
import omegaconf
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

    for epoch in range(cfg.epochs):
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

                if (train_counter % 10) == 0:
                    print(f"Train Step: {step_time:.2f}s, Loss: {float(loss):.4f}")


if __name__ == "__main__":
    train()
