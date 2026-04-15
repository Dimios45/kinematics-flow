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

import math

import e3nn_jax
import jax
import jax.numpy as jnp
from flax import nnx

import kin_flow.util.const as CONST
from kin_flow.net.ffn import FeedForwardNetwork
from kin_flow.net.module.layer_norm import EquivariantRMSLayerNorm
from kin_flow.net.module.so2_ops import (SO2_Convolution,
                                         init_edge_rot_mat_deterministic)
from kin_flow.util.irreps_helper import (convert_from_cl_to_irreps,
                                         convert_from_irreps_to_cl)


def segment_softmax_jax(logits, segment_ids, num_segments, valid=None):
    # Shapes must match; segment_ids must be integer
    segment_ids = segment_ids.astype(jnp.int32)
    if valid is None:
        valid = jnp.ones_like(logits, dtype=jnp.bool_)
    logits = jnp.where(jnp.isnan(logits), -jnp.inf, logits)
    masked = jnp.where(valid, logits, -jnp.inf)
    is_posinf = jnp.isposinf(masked)
    posinf_count = jax.ops.segment_sum(
        is_posinf.astype(jnp.int32), segment_ids, num_segments=num_segments
    )  # (num_segments,)
    has_posinf = posinf_count > 0
    max_per_seg = jax.ops.segment_max(masked, segment_ids, num_segments=num_segments)
    max_per_seg = jnp.where(jnp.isneginf(max_per_seg), 0.0, max_per_seg)

    centered_regular = masked - max_per_seg[segment_ids]
    centered_with_posinf = jnp.where(is_posinf, 0.0, -jnp.inf)
    centered = jnp.where(
        has_posinf[segment_ids], centered_with_posinf, centered_regular
    )
    exp_centered = jnp.exp(centered)
    denom = jax.ops.segment_sum(exp_centered, segment_ids, num_segments=num_segments)
    denom_safe = jnp.where(denom > 0, denom, 1.0)
    alpha = exp_centered / denom_safe[segment_ids]
    alpha = jnp.where(valid, alpha, 0.0)

    return alpha


class SO2GraphLayer(nnx.Module):
    def __init__(
        self,
        num_channels,
        edge_attr_channels,
        num_heads=1,
        *,
        rngs,
    ):
        self.num_channels = num_channels
        self.num_heads = num_heads
        assert num_channels % num_heads == 0
        self.attn_weights = nnx.Param(
            jax.random.uniform(
                rngs(),
                shape=(num_heads, num_channels),
                minval=-1 / math.sqrt(num_channels),
                maxval=1 / math.sqrt(num_channels),
            )
        )
        self.ln_d_dot = nnx.LayerNorm(num_features=num_channels, rngs=rngs)
        self.ln_s_dot = nnx.LayerNorm(num_features=num_channels, rngs=rngs)

        mlp_in = edge_attr_channels + num_channels * 2
        self.so2_tp1 = SO2_Convolution(
            in_channel=num_channels * 2,
            out_channel=num_channels,
            lmax=2,
            mlp=[mlp_in, mlp_in, mlp_in],
            rngs=rngs,
            extra_channels=(num_channels * num_heads),
        )
        self.ffn_ln = EquivariantRMSLayerNorm(num_channels)
        self.ffn = FeedForwardNetwork(
            num_channels,
            num_channels,
            scalar_mlp=[num_channels, num_channels],
            grid_mlp=[num_channels, num_channels],
            rngs=rngs,
        )
        self.tp_ln = EquivariantRMSLayerNorm(num_channels)
        self.so2_tp2 = SO2_Convolution(
            in_channel=num_channels,
            out_channel=num_channels,
            lmax=2,
            mlp=[mlp_in, mlp_in, mlp_in],
            rngs=rngs,
            extra_channels=0,
        )

    def __call__(
        self, s_feat, d_feat, s_pos, d_pos, edge_attr, edge_src, edge_dst, valid
    ):
        num_nodes = d_pos.shape[0]
        C = self.num_channels

        edge_dst = edge_dst.astype(jnp.int32)
        edge_src = edge_src.astype(jnp.int32)

        pos_src = jnp.where(valid[:, None], s_pos[edge_src], 0.0)
        pos_dst = jnp.where(valid[:, None], d_pos[edge_dst], 0.0)
        edge_vec_raw = pos_dst - pos_src
        rot, valid_rot = jax.vmap(init_edge_rot_mat_deterministic)(
            edge_vec_raw
        )  # (E,3,3), (E,)
        safe_dir = jnp.array([1.0, 0.0, 0.0], dtype=s_pos.dtype)
        edge_vec_safe = jnp.where(valid_rot[:, None], edge_vec_raw, safe_dir)  # (E,3)
        s_feat = s_feat[edge_src]  # (E, C, 9)
        d_feat = d_feat[edge_dst]  # (E, C, 9)

        sh = e3nn_jax.spherical_harmonics(
            "1x0e+1x1e+1x2e", edge_vec_safe, True
        ).array  # (E, 9)

        # Scalar projections per edge/channel
        d_dot = jnp.einsum("ecl,el->ec", d_feat, sh)  # (E, C)
        s_dot = jnp.einsum("ecl,el->ec", s_feat, sh)  # (E, C)
        scalars = jnp.concatenate(
            [edge_attr, self.ln_d_dot(d_dot), self.ln_s_dot(s_dot)], axis=-1
        )
        m = jnp.concatenate([s_feat, d_feat], axis=1)  # (E, 2C, 9)
        m = e3nn_jax.vmap(convert_from_cl_to_irreps, in_axes=(0, None, None))(
            m, 2 * C, CONST.L_MAX
        )
        m = e3nn_jax.vmap(e3nn_jax.IrrepsArray.transform_by_matrix)(m, rot)
        m = e3nn_jax.vmap(convert_from_irreps_to_cl, in_axes=(0, None, None))(
            m, 2 * C, CONST.L_MAX
        )
        m, extra = nnx.vmap(self.so2_tp1)(m, scalars)  # m: (E, C, 9), extra: (E, C)
        m = nnx.vmap(self.ffn_ln)(m)
        m = nnx.vmap(self.ffn)(m)
        m = nnx.vmap(self.tp_ln)(m)
        m, _ = nnx.vmap(self.so2_tp2)(m, scalars)  # m: (E, C, 9), extra: (E, C)
        m = e3nn_jax.vmap(convert_from_cl_to_irreps, in_axes=(0, None, None))(
            m, C, CONST.L_MAX
        )
        m = e3nn_jax.vmap(e3nn_jax.IrrepsArray.transform_by_matrix)(
            m, jnp.transpose(rot, (0, 2, 1))
        )
        m = e3nn_jax.vmap(convert_from_irreps_to_cl, in_axes=(0, None, None))(
            m, C, CONST.L_MAX
        )  # (E, C, 9)

        # Attention logits and mask; exclude tiny edges in the softmax too
        extra = jnp.reshape(extra, shape=(-1, self.num_heads, self.num_channels))
        attn_logits = jnp.einsum("ehc,hc->eh", extra, self.attn_weights.value)  # (E,)
        attn_valid = valid & valid_rot
        alpha_edge = jax.vmap(
            segment_softmax_jax, in_axes=(1, None, None, None), out_axes=1
        )(
            attn_logits, edge_dst, num_nodes, attn_valid
        )  # (E, 4)

        # Attention Heads
        m = jnp.reshape(
            m,
            shape=(
                -1,
                self.num_heads,
                self.num_channels // self.num_heads,
                (CONST.L_MAX + 1) ** 2,
            ),
        )  # (num_nodes, num_heads, ...)
        weight = (alpha_edge * attn_valid[..., None])[..., None, None]  # (E,heads,1)
        m = jnp.reshape(
            m * weight, shape=(-1, self.num_channels, (CONST.L_MAX + 1) ** 2)
        )

        # Aggregate to destination nodes
        node_feat = jax.ops.segment_sum(m, edge_dst, num_nodes)  # (num_nodes, C, 9)

        return node_feat


class SO2GraphLayerSimple(nnx.Module):
    def __init__(
        self,
        num_channels,
        edge_attr_channels,
        num_heads=1,
        *,
        rngs,
    ):
        self.num_channels = num_channels
        self.num_heads = num_heads
        assert num_channels % num_heads == 0
        self.attn_weights = nnx.Param(
            jax.random.uniform(
                rngs(),
                shape=(num_heads, num_channels),
                minval=-1 / math.sqrt(num_channels),
                maxval=1 / math.sqrt(num_channels),
            )
        )
        self.ln_d_dot = nnx.LayerNorm(num_features=num_channels, rngs=rngs)
        self.ln_s_dot = nnx.LayerNorm(num_features=num_channels, rngs=rngs)

        mlp_in = edge_attr_channels + num_channels * 2
        self.lmax = 1
        self.so2_tp1 = SO2_Convolution(
            in_channel=num_channels * 2,
            out_channel=num_channels,
            lmax=self.lmax,
            mlp=[mlp_in, mlp_in, mlp_in],
            rngs=rngs,
            extra_channels=(num_channels * num_heads),
        )

    def __call__(
        self, s_feat, d_feat, s_pos, d_pos, edge_attr, edge_src, edge_dst, valid
    ):
        num_nodes = d_pos.shape[0]
        C = self.num_channels

        edge_dst = edge_dst.astype(jnp.int32)
        edge_src = edge_src.astype(jnp.int32)

        pos_src = jnp.where(valid[:, None], s_pos[edge_src], 0.0)
        pos_dst = jnp.where(valid[:, None], d_pos[edge_dst], 0.0)
        edge_vec_raw = pos_dst - pos_src
        rot, valid_rot = jax.vmap(init_edge_rot_mat_deterministic)(
            edge_vec_raw
        )  # (E,3,3), (E,)
        safe_dir = jnp.array([1.0, 0.0, 0.0], dtype=s_pos.dtype)
        edge_vec_safe = jnp.where(valid_rot[:, None], edge_vec_raw, safe_dir)  # (E,3)
        s_feat = s_feat[edge_src]  # (E, C, 9)
        d_feat = d_feat[edge_dst]  # (E, C, 9)

        sh = e3nn_jax.spherical_harmonics(
            "1x0e+1x1e+1x2e", edge_vec_safe, True
        ).array  # (E, 4)

        # Scalar projections per edge/channel
        d_dot = jnp.einsum("ecl,el->ec", d_feat, sh)  # (E, C)
        s_dot = jnp.einsum("ecl,el->ec", s_feat, sh)  # (E, C)

        # Conditioning vector
        scalars = jnp.concatenate(
            [edge_attr, self.ln_d_dot(d_dot), self.ln_s_dot(s_dot)], axis=-1
        )
        m = jnp.concatenate([s_feat[..., :4], d_feat[..., :4]], axis=1)  # (E, 2C, 4)
        m = e3nn_jax.vmap(convert_from_cl_to_irreps, in_axes=(0, None, None))(
            m, 2 * C, self.lmax
        )
        m = e3nn_jax.vmap(e3nn_jax.IrrepsArray.transform_by_matrix)(m, rot)
        m = e3nn_jax.vmap(convert_from_irreps_to_cl, in_axes=(0, None, None))(
            m, 2 * C, self.lmax
        )
        m, extra = nnx.vmap(self.so2_tp1)(m, scalars)  # m: (E, C, 4), extra: (E, C)
        m = e3nn_jax.vmap(convert_from_cl_to_irreps, in_axes=(0, None, None))(
            m, C, self.lmax
        )
        m = e3nn_jax.vmap(e3nn_jax.IrrepsArray.transform_by_matrix)(
            m, jnp.transpose(rot, (0, 2, 1))
        )
        m = e3nn_jax.vmap(convert_from_irreps_to_cl, in_axes=(0, None, None))(
            m, C, self.lmax
        )  # (E, C, 4)
        extra = jnp.reshape(extra, shape=(-1, self.num_heads, self.num_channels))
        attn_logits = jnp.einsum("ehc,hc->eh", extra, self.attn_weights.value)  # (E,)
        attn_valid = valid & valid_rot
        alpha_edge = jax.vmap(
            segment_softmax_jax, in_axes=(1, None, None, None), out_axes=1
        )(
            attn_logits, edge_dst, num_nodes, attn_valid
        )  # (E, 4)

        # Attention Heads
        m = jnp.reshape(
            m,
            shape=(
                -1,
                self.num_heads,
                self.num_channels // self.num_heads,
                (self.lmax + 1) ** 2,
            ),
        )  # (num_nodes, num_heads, ...)
        weight = (alpha_edge * attn_valid[..., None])[..., None, None]  # (E,heads,1)
        m = jnp.reshape(m * weight, shape=(-1, self.num_channels, (self.lmax + 1) ** 2))
        node_feat = jax.ops.segment_sum(m, edge_dst, num_nodes)  # (num_nodes, C, 4)

        return node_feat
