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

import functools

import jax
import optax
import orbax.checkpoint as ocp
from flax import nnx

import kin_flow.util.const as CONST

AXIS_NAME = "devices"

ID_DOF_LOOKUP = {
    # FILLED BY TRAINER
}
ID_NAME_LOOKUP = {
    # FILLED BY TRAINER
}
LOSS_FN = None
STAT_FN = None
STAT_CONTEXT = None


def _pmean_tree(x):
    return jax.tree_util.tree_map(lambda y: jax.lax.pmean(y, AXIS_NAME), x)


@functools.partial(jax.pmap, axis_name=AXIS_NAME, in_axes=(0, 0, 0, 0))
def update(graph, state, inputs, target):
    model, optimizer = nnx.merge(graph, state)

    def loss_fn_wrt(model):
        model_over_grasp = nnx.vmap(
            model,
            in_axes=((None, None), None, [0] * CONST.L_MAX, 0, 0, 0, None),
        )
        outputs = nnx.vmap(
            model_over_grasp,
            in_axes=((0, 0), 0, [0] * CONST.L_MAX, 0, 0, 0, 0),
        )(*inputs)

        loss, aux = LOSS_FN(inputs, outputs, target)  # type: ignore
        stats = STAT_FN(  # type: ignore
            model,
            inputs,
            outputs,
            target,
            STAT_CONTEXT,
        )
        return loss, {**aux, **stats}

    (loss, aux), grad = nnx.value_and_grad(loss_fn_wrt, has_aux=True)(model)
    loss = jax.lax.pmean(loss, AXIS_NAME)
    aux = _pmean_tree(aux)
    grad = _pmean_tree(grad)

    optimizer.update(grad)

    # Unfortunately required, since jax is non-deterministic -> needs sync
    graph, params, rest = nnx.split((model, optimizer), nnx.Param, ...)
    params = jax.lax.pmean(params, AXIS_NAME)
    model, optimizer = nnx.merge(graph, params, rest)
    new_state = nnx.state((model, optimizer))

    return loss, aux, new_state


DOF_LOOKUP = {
    "z0": 0,
    "panda": 2,
    "panda_orbit": 0,
    "panda_static": 0,
    "vx300": 2,
    "dexee": 12,
    "dexee_static": 0,
    "allegro": 16,
    "shadow": 22,
    "shadow_static": 0,
}


class Trainer:
    def __init__(
        self,
        model,
        loss_fn,
        stat_fn,
        alg_cfg,
        train_cfg,
        grippers=[],
    ):
        self.model = model
        for i, gripper in enumerate(grippers):
            ID_NAME_LOOKUP[i] = gripper
            ID_DOF_LOOKUP[i] = DOF_LOOKUP[gripper]

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=train_cfg.init_lr,
            peak_value=train_cfg.peak_lr,
            warmup_steps=train_cfg.warmup_steps,
            decay_steps=train_cfg.total_steps,  # includes warmup
            end_value=train_cfg.end_lr,
        )
        tx = optax.adamw(learning_rate=lr_schedule)
        # flax>=0.11: nnx.Optimizer requires wrt; ModelAndOptimizer keeps the old semantics
        self.optimizer = nnx.ModelAndOptimizer(model, tx)
        self.train_graph, self.train_state = nnx.split((self.model, self.optimizer))

        self.devices = jax.local_devices()
        self.n_devices = len(self.devices)

        # put on all devices
        self.train_graph = jax.device_put_replicated(self.train_graph, self.devices)
        self.train_state = jax.device_put_replicated(self.train_state, self.devices)

        self.checkpointer = ocp.StandardCheckpointer()
        global LOSS_FN, STAT_FN, STAT_CONTEXT
        LOSS_FN = loss_fn
        STAT_FN = stat_fn
        STAT_CONTEXT = {
            "id_dof_lookup": ID_DOF_LOOKUP,
            "id_name_lookup": ID_NAME_LOOKUP,
            "cfg": alg_cfg,
        }

    def train_step(self, inputs, target):
        # position_scaling must have a leading device axis under pmap
        loss, aux, self.train_state = update(
            self.train_graph,
            self.train_state,
            inputs,
            target,
        )
        # take replica 0 for host logging (already pmean'd across devices)
        loss0 = loss[0]
        aux0 = jax.tree_util.tree_map(lambda x: x[0], aux)
        return loss0, aux0

    def save(self, directory: str):
        single_state = jax.tree_util.tree_map(lambda x: x[0], self.train_state)
        state = single_state.to_pure_dict()
        self.checkpointer.save(directory, state)
        self.checkpointer.wait_until_finished()

    def restore(self, directory: str):
        # inverse of save(): load pure dict into an unreplicated copy of the
        # state (model + optimizer, incl. schedule step), then re-replicate
        single_state = jax.tree_util.tree_map(lambda x: x[0], self.train_state)
        state = self.checkpointer.restore(directory)
        single_state.replace_by_pure_dict(state)  # type: ignore
        self.train_state = jax.device_put_replicated(single_state, self.devices)

    @classmethod
    def get_model_from_checkpoint(cls, model, path):

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=0,
            warmup_steps=0,
            decay_steps=1.0,
            end_value=0,
        )
        tx = optax.adamw(learning_rate=lr_schedule)
        optimizer = nnx.ModelAndOptimizer(model, tx)
        _train_graph, train_state = nnx.split((model, optimizer))
        checkpointer = ocp.StandardCheckpointer()
        # pass an explicit target so Orbax remaps the checkpoint's saved device
        # topology (e.g. 3-way sharded from multi-GPU training) onto whatever
        # devices are available here, instead of replaying the original sharding.
        state = checkpointer.restore(path, target=train_state.to_pure_dict())
        train_state.replace_by_pure_dict(state)  # type: ignore
        model, _opt = nnx.merge(_train_graph, train_state)
        return model

    def get_optimzer(self):
        _, opt = nnx.merge(self.train_graph, self.train_state[0])
        return opt
