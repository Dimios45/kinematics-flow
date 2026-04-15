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


class GraphLayer(nnx.Module):
    def __init__(
        self,
        num_channels,
        edge_attr_channels,
        num_heads=1,
        output_channels=None,
        *,
        rngs,
    ):
        if output_channels is None:
            output_channels = num_channels
        self.output_channels = output_channels
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
        self.ln_sd_dot = nnx.LayerNorm(num_features=num_channels, rngs=rngs)
        self.msg_mlp = nnx.Sequential(
            *[
                nnx.Linear(
                    edge_attr_channels + 2 * num_channels, 4 * num_channels, rngs=rngs
                ),
                nnx.gelu,
                nnx.Linear(4 * num_channels, 4 * num_channels, rngs=rngs),
                nnx.gelu,
                nnx.Linear(4 * num_channels, 4 * num_channels, rngs=rngs),
                nnx.gelu,
                nnx.Linear(4 * num_channels, output_channels, rngs=rngs),
            ]
        )
        self.attn_mlp = nnx.Sequential(
            *[
                nnx.Linear(
                    edge_attr_channels + 2 * num_channels, 4 * num_channels, rngs=rngs
                ),
                nnx.gelu,
                nnx.Linear(4 * num_channels, 4 * num_channels, rngs=rngs),
                nnx.gelu,
                nnx.Linear(4 * num_channels, num_heads * num_channels, rngs=rngs),
            ]
        )

        # these are similar weights as in a weighted per channel tensor product
        # for scalar features
        self.sd_dot_modulation = nnx.Param(
            jax.random.uniform(rngs(), shape=(num_channels, CONST.L_MAX + 1))
        )
        self.d_dot_modulation = nnx.Param(
            jax.random.uniform(rngs(), shape=(num_channels, CONST.L_MAX + 1))
        )
        expand_index = jnp.zeros(shape=((CONST.L_MAX + 1) ** 2), dtype=jnp.int32)
        for l in range(CONST.L_MAX + 1):
            start_idx = l**2
            length = 2 * l + 1
            expand_index = expand_index.at[start_idx : (start_idx + length)].set(l)
        self.expand_index = nnx.Variable(expand_index)

    def __call__(
        self, s_feat, d_feat, s_pos, d_pos, edge_attr, edge_src, edge_dst, valid
    ):
        num_nodes = d_pos.shape[0]

        edge_dst = edge_dst.astype(jnp.int32)
        edge_src = edge_src.astype(jnp.int32)

        pos_src = jnp.where(valid[:, None], s_pos[edge_src], 0.0)
        pos_dst = jnp.where(valid[:, None], d_pos[edge_dst], 0.0)

        edge_vec = pos_dst - pos_src  # type: ignore

        s_feat = s_feat[edge_src]  # (E, C, 9)
        d_feat = d_feat[edge_dst]  # (E, C, 9)

        # Real SH up to l=2 -> 9 comps
        sh = e3nn_jax.spherical_harmonics(
            "1x0e+1x1e+1x2e", edge_vec, True
        ).array  # (E, 9)

        # Scalar projections per edge/channel
        d_dot_modulation = self.d_dot_modulation[
            None, :, self.expand_index.value
        ]  # (1, C, 9)
        d_dot = jnp.einsum("ecl,el->ec", d_feat * d_dot_modulation, sh)  # (E, C)

        sd_dot_modulation = self.sd_dot_modulation[
            None, :, self.expand_index.value
        ]  # (1, C, 9)
        sd_dot = jnp.einsum("ecl,ecl->ec", d_feat * sd_dot_modulation, s_feat)  # (E, C)

        scalars = jnp.concatenate(
            [edge_attr, self.ln_d_dot(d_dot), self.ln_sd_dot(sd_dot)], axis=-1
        )
        msg = self.msg_mlp(scalars)
        extra = self.attn_mlp(scalars)

        extra = jnp.reshape(extra, shape=(-1, self.num_heads, self.num_channels))
        attn_logits = jnp.einsum(
            "ehc,hc->eh", extra, self.attn_weights.value
        )  # (E,heads)

        alpha_edge = jax.vmap(
            segment_softmax_jax, in_axes=(1, None, None, None), out_axes=1
        )(
            attn_logits, edge_dst, num_nodes, valid
        )  # (E, heads)

        # Attention Heads
        msg = jnp.reshape(
            msg,
            shape=(
                -1,
                self.num_heads,
                self.output_channels // self.num_heads,
            ),
        )  # (num_nodes, num_heads, ...)
        weight = (alpha_edge * valid[..., None])[..., None]  # (E,heads,1)
        msg = jnp.reshape(msg * weight, shape=(-1, self.output_channels))

        # Aggregate to destination nodes
        node_feat = jax.ops.segment_sum(msg, edge_dst, num_nodes)  # (num_nodes, C)

        return node_feat
