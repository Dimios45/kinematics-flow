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

import jax
import jax.numpy as jnp


def loss(model_in, model_out, target):
    (pred_twist, valid_mask), _ = model_out
    pred_u_rot = pred_twist[..., :3]
    pred_u_pos = pred_twist[..., 3:6]
    pred_u_dof = pred_twist[..., 6:]

    t = model_in[-2]

    # Targets: Clean State (x_1)
    target_rot_delta = target[..., :3]
    target_pos_delta = target[..., 3:6]
    target_dof = target[..., 6:]

    t_clipped = jnp.minimum(t, 0.9)
    scale = 1.0 - t_clipped
    scale = scale[..., None]

    diff_pos = pred_u_pos - target_pos_delta

    scaled_diff_pos = diff_pos / scale
    loss_pos = jnp.mean(scaled_diff_pos**2)
    scaled_w_diff = (target_rot_delta - pred_u_rot) / scale
    loss_rot = jnp.mean(scaled_w_diff**2)

    diff_dof = pred_u_dof - target_dof
    scaled_diff_dof = diff_dof / scale

    loss_dof = jnp.mean(scaled_diff_dof**2, where=valid_mask)
    loss_dof = jnp.where(jnp.isnan(loss_dof), 0.0, loss_dof)
    total_loss = loss_pos + loss_rot + loss_dof

    return total_loss, {
        "loss_pos": loss_pos,
        "loss_rot": loss_rot,
        "loss_dof": loss_dof,
    }


def stats(model, model_in, model_out, target, context):
    t = model_in[-2]
    denom = (1.0 - t)[..., None]
    (flow, valid_mask), stats = model_out
    flow_rot, flow_pos, flow_dof = flow[..., :3], flow[..., 3:6], flow[..., 6:]
    target_rot, target_pos, target_dof = (
        target[..., :3],
        target[..., 3:6],
        target[..., 6:],
    )
    position_scaling = context["cfg"].position_scaling
    intervals = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.9), (0.9, 1.0)]
    t_flat = t.flatten()

    rot_error = jax.lax.stop_gradient(
        jnp.sqrt(jnp.sum(((flow_rot - target_rot) / denom) ** 2, axis=-1))
    )
    pos_error = jax.lax.stop_gradient(
        jnp.sqrt(jnp.sum(((flow_pos - target_pos) / denom) ** 2, axis=-1))
        / position_scaling
    )
    dof_error = jax.lax.stop_gradient(
        jnp.sqrt(
            jnp.sum(((flow_dof - target_dof) / denom) ** 2, where=valid_mask, axis=-1)
        )
    )
    dof_error = jnp.where(jnp.isnan(dof_error), 0.0, dof_error)

    gripper_metrics = {}
    id_dof_lookup = context["id_dof_lookup"]
    id_name_lookup = context["id_name_lookup"]
    for i in range(len(id_dof_lookup)):
        ratio = model.kinematic_encoding.kinematic_net.dof_norm_ratio(jnp.array(i))[
            None
        ]
        gripper_dist = jnp.mean(
            jnp.sqrt(((flow[i, :, 6:] - target[i, :, 6:]) * ratio / denom[i]) ** 2),
            where=valid_mask[i],
            axis=0,
        )
        for dof in range(id_dof_lookup[i]):
            gripper_metrics[
                f"gripper-mean-dof-dist-{id_name_lookup[i]}-dof_id-{dof}"
            ] = gripper_dist[dof]

    rot_err_flat, pos_err_flat, dof_err_flat = (
        rot_error.flatten(),
        pos_error.flatten(),
        dof_error.flatten(),
    )
    delta_err_intervals = {}
    for lo, hi in intervals:
        mask = (t_flat >= lo) & (t_flat < hi)
        count = jnp.sum(mask)
        rot_avg_err = jnp.where(count > 0, jnp.sum(rot_err_flat * mask) / count, 0.0)
        pos_avg_err = jnp.where(count > 0, jnp.sum(pos_err_flat * mask) / count, 0.0)
        dof_avg_err = jnp.where(count > 0, jnp.sum(dof_err_flat * mask) / count, 0.0)
        delta_err_intervals[f"delta_rot_{lo:.2f}_{hi:.2f}"] = rot_avg_err
        delta_err_intervals[f"delta_pos_{lo:.2f}_{hi:.2f}"] = pos_avg_err
        delta_err_intervals[f"delta_dof_{lo:.2f}_{hi:.2f}"] = dof_avg_err

    for k in stats.keys():
        stats[k] = jnp.mean(stats[k])

    aux = {
        **gripper_metrics,
        **delta_err_intervals,
        **stats,
    }
    return aux
