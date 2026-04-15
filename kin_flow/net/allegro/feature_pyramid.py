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
from e3nn_jax import Irreps
from flax import nnx

from kin_flow.net.allegro.weighted_channels import MakeWeightedChannelsNNX
from kin_flow.net.ffn import FeedForwardNetwork, SO3LinearCL
from kin_flow.net.module.connectivity import KNNConnect, PoolingConnect
from kin_flow.net.module.layer_norm import EquivariantRMSLayerNorm
from kin_flow.net.module.radial_function import SinusoidalPositionEmbeddings
from kin_flow.net.module.so2_graph_layer import (SO2GraphLayer,
                                                 segment_softmax_jax)
from kin_flow.util.fps import farthest_point_sampling


class ProjectGraphLayer(nnx.Module):
    def __init__(self, num_channels, edge_attr_dim, out_num_channels=None, *, rngs):
        self.ln_graph = EquivariantRMSLayerNorm(num_channels)
        self.ln_ffn = EquivariantRMSLayerNorm(num_channels)

        self.ln_attr = []
        for dim in edge_attr_dim:
            self.ln_attr.append(nnx.LayerNorm(dim, rngs=rngs))

        self.graph = SO2GraphLayer(
            num_channels, sum(edge_attr_dim), num_heads=4, rngs=rngs
        )
        self.fnn = FeedForwardNetwork(
            num_channels,
            num_channels,
            [num_channels] * 2,
            [num_channels] * 2,
            rngs=rngs,
        )
        self.out_num_channels = out_num_channels
        if out_num_channels is not None:
            self.proj = SO3LinearCL(num_channels, out_num_channels, rngs=rngs)

    def __call__(
        self, pos, feat, dst_idx, edge_src, edge_dst, valid_edges, edge_attr_list
    ):
        x = nnx.vmap(self.ln_graph)(feat)

        attr_list = []
        for ln, attr in zip(self.ln_attr, edge_attr_list):
            attr_list.append(ln(attr))
        edge_attr = jnp.concatenate(attr_list, axis=-1)

        x_delta = self.graph(
            x,
            x,
            pos,
            pos,
            edge_attr,
            edge_src,
            edge_dst,
            valid_edges,
        )
        feat = feat[dst_idx] + x_delta[dst_idx]
        x = nnx.vmap(self.ln_ffn)(feat)
        x_delta = nnx.vmap(self.fnn)(x)
        feat = feat + x_delta

        if self.out_num_channels is not None:
            feat = nnx.vmap(self.proj)(feat)

        return feat


class MPLayer(nnx.Module):
    def __init__(self, num_channels, edge_attr_dim, *, rngs):
        self.ln_graph = EquivariantRMSLayerNorm(num_channels)
        self.ln_ffn = EquivariantRMSLayerNorm(num_channels)

        self.ln_attr = []
        for dim in edge_attr_dim:
            self.ln_attr.append(nnx.LayerNorm(dim, rngs=rngs))

        self.graph = SO2GraphLayer(
            num_channels, sum(edge_attr_dim), num_heads=4, rngs=rngs
        )
        self.fnn = FeedForwardNetwork(
            num_channels,
            num_channels,
            [num_channels] * 2,
            [num_channels] * 2,
            rngs=rngs,
        )

    def __call__(self, pos, feat, edge_src, edge_dst, valid_edges, edge_attr_list):
        x = nnx.vmap(self.ln_graph)(feat)

        attr_list = []
        for ln, attr in zip(self.ln_attr, edge_attr_list):
            attr_list.append(ln(attr))
        edge_attr = jnp.concatenate(attr_list, axis=-1)

        x_delta = self.graph(
            x,
            x,
            pos,
            pos,
            edge_attr,
            edge_src,
            edge_dst,
            valid_edges,
        )
        feat = feat + x_delta
        x = nnx.vmap(self.ln_ffn)(feat)
        x_delta = nnx.vmap(self.fnn)(x)
        feat = feat + x_delta
        return feat


def masked_rms(x, mask):
    # x: (N, C, M), mask: (N, 1, 1) boolean
    w = mask.astype(jnp.float32)
    num = jnp.sum(w * (x**2))
    den = jnp.sum(w) * x.shape[1] * x.shape[2]  # count × channels × components
    return jnp.sqrt(jnp.where(den > 0, num / den, 0.0))


class PCDPyramid(nnx.Module):

    def __init__(
        self,
        time_freq,
        rngs,
        *,
        lmax: int = 2,
        max_neighbors: tuple[int, ...] = (5, 5, 5),
        pool_count: tuple[int, ...] = (2000, 200, 40),
        layer_channels: tuple[int, ...] = (32, 64, 64),
        num_message_passing_layers: tuple[int, ...] = (2, 2, 2),
    ):
        super().__init__()
        self.pool_count = pool_count
        self.dist_enc = SinusoidalPositionEmbeddings(64, 1.0, n=time_freq)
        self.graph_parser = [
            (PoolingConnect(), KNNConnect(k=K_i, r_cap=None, disallow_self_loops=True))
            for K_i in max_neighbors
        ]

        self.attn_weights = nnx.Param(
            jax.random.uniform(
                rngs(),
                shape=(layer_channels[0],),
                minval=-1 / math.sqrt(layer_channels[0]),
                maxval=1 / math.sqrt(layer_channels[0]),
            )
        )

        self.pool_blocks: list[ProjectGraphLayer] = []
        for in_c, out_c in zip(layer_channels[:-1], layer_channels[1:]):
            self.pool_blocks.append(
                ProjectGraphLayer(in_c, (64,), out_num_channels=out_c, rngs=rngs)
            )

        self.mp_blocks: list[list[MPLayer]] = []
        for channels, num_layers in zip(layer_channels, num_message_passing_layers):
            layers = []
            for _ in range(num_layers):
                layers.append(MPLayer(channels, edge_attr_dim=(64,), rngs=rngs))
            self.mp_blocks.append(layers)

        first = layer_channels[0]
        self.layer_0_in_channels = first
        self.weight_mlp = nnx.Sequential(
            *[
                nnx.Linear(64, 64, rngs=rngs),
                nnx.gelu,
                nnx.Linear(64, 4 * first, rngs=rngs),
            ]
        )
        self.weight_irrep = MakeWeightedChannelsNNX(
            irreps_in=Irreps("1x0e + 1x1e + 1x2e"),
            multiplicity_out=first,
            alpha=1.0,
            weight_individual_irreps=True,
        )
        self.lmax = lmax
        self.layer_channel = layer_channels

    def __call__(
        self,
        pos: jnp.ndarray,
        key: jnp.ndarray,
    ):
        """Build multi-scale features with FPS pyramid and per-level MP."""
        _stats = {}

        # ----------------------
        # L0: initial edge-based aggregation to build features
        # ----------------------
        valid = jnp.ones(shape=pos[..., 0].shape, dtype=jnp.bool_)
        idx_fps0 = farthest_point_sampling(pos, self.pool_count[0], key=key)
        pooling, mp = self.graph_parser[0]
        e_src0, e_dst0, e_val0 = pooling(pos, pos[idx_fps0], valid, valid[idx_fps0])

        edge_vec0 = pos[idx_fps0][e_dst0] - pos[e_src0]
        edge_len0 = jnp.linalg.norm(edge_vec0, axis=-1)
        # length -> encoding -> weights
        len_enc0 = nnx.vmap(self.dist_enc)(edge_len0)
        weights_and_attention_0 = self.weight_mlp(len_enc0)  # (E, 4*first)
        weights_0 = weights_and_attention_0[
            :, : 3 * self.layer_channel[0]
        ]  # (E, 3*first)
        attention_0 = weights_and_attention_0[
            :, 3 * self.layer_channel[0] :
        ]  # (E,     first)
        # SH on directions, then weighted mixing into (C, 1+3+5)
        sh0 = e3nn_jax.vmap(e3nn_jax.spherical_harmonics, in_axes=(None, 0, None))(
            "1x0e+1x1e+1x2e", edge_vec0, True
        )
        dir_emb0 = self.weight_irrep(sh0.array, weights_0)  # [E, C, 9]

        # attention over incoming edges per dst
        # be careful with precedence in the mask:
        valid_dir0 = (edge_len0 > 1e-5) & e_val0
        attn_logits0 = jnp.einsum("ec,c->e", attention_0, self.attn_weights.value)
        alpha0 = segment_softmax_jax(
            attn_logits0, e_dst0, num_segments=pos.shape[0], valid=valid_dir0
        )
        feat = jax.ops.segment_sum(
            dir_emb0 * (alpha0 * valid_dir0)[..., None, None],
            e_dst0,
            num_segments=self.pool_count[0],
        )  # [N0, C, 9]

        # a few reference stats
        _ref_rms_0 = jnp.sqrt(jnp.mean((feat[..., :1].ravel()) ** 2))
        _ref_rms_1 = jnp.sqrt(jnp.mean((feat[..., 1:4].ravel()) ** 2))
        _stats["ref_rms_l0"] = _ref_rms_0
        _stats["ref_rms_l1"] = _ref_rms_1
        _stats["ref_ratio_rms_l0_l1"] = _ref_rms_1 / (_ref_rms_0 + 1e-12)

        # L0 message passing (optional but keeps interface uniform)
        e_mp_src0, e_mp_dst0, e_mp_val0 = mp(
            pos[idx_fps0], pos[idx_fps0], valid[idx_fps0], valid[idx_fps0]
        )
        edge0 = pos[idx_fps0][e_mp_dst0] - pos[idx_fps0][e_mp_src0]
        edge0_len_enc = nnx.vmap(self.dist_enc)(jnp.linalg.norm(edge0, axis=-1))
        for mp in self.mp_blocks[0]:
            feat = mp(
                pos=pos[idx_fps0],
                feat=feat,
                edge_src=e_mp_src0,
                edge_dst=e_mp_dst0,
                valid_edges=e_mp_val0,
                edge_attr_list=[edge0_len_enc],
            )

        _after0_l0 = jnp.sqrt(jnp.mean((feat[..., :1].ravel()) ** 2))
        _after0_l1 = jnp.sqrt(jnp.mean((feat[..., 1:4].ravel()) ** 2))
        _stats["after_L0_rms_l0"] = _after0_l0
        _stats["after_L0_rms_l1"] = _after0_l1
        _stats["after_L0_ratio_rms_l0_l1"] = _after0_l1 / (_after0_l0 + 1e-12)

        # Stage-0 outputs
        hierarchical_features = [(pos[idx_fps0], feat, None)]
        abs_indices_per_level = [jnp.arange(pos[idx_fps0].shape[0], dtype=jnp.int32)]

        # ----------------------
        # Down path: for each subsequent level
        # ----------------------
        cur_pos = pos[idx_fps0]
        cur_feat = feat
        cur_valid = valid[idx_fps0]
        cur_abs = abs_indices_per_level[0]

        keys = jax.random.split(key, max(1, len(self.pool_count) - 1))

        for i, (num_dst_nodes, pool_block, mp_layers) in enumerate(
            zip(self.pool_count[1:], self.pool_blocks, self.mp_blocks[1:])
        ):
            # ---- FPS on current layer to pick the next layer's nodes (relative indices)
            fps_rel = farthest_point_sampling(
                cur_pos, num_dst_nodes, key=keys[i]
            )  # [Nd] in [0..Ns-1]
            fps_rel = fps_rel.astype(jnp.int32)
            # next-layer arrays (relative)
            nxt_pos = cur_pos[fps_rel]
            nxt_valid = cur_valid[fps_rel]
            nxt_abs = cur_abs[fps_rel]
            nxt_feat = cur_feat[fps_rel]

            # ---- Pooling connections
            pool_src, pool_dst_rel, pool_mask = self.graph_parser[i][0](
                cur_pos, nxt_pos, cur_valid, nxt_valid
            )  # each [Ns]
            pool_src = pool_src.astype(jnp.int32)
            pool_dst_rel = pool_dst_rel.astype(jnp.int32)

            # Mapping from current to next-layer *relative* index, used by QueryAtMultiLayer.
            pooling_edges_map = jnp.where(pool_mask, pool_dst_rel, jnp.int32(0))

            # Build edge attributes with *sanitized* vectors
            edge_vec = nxt_pos[pool_dst_rel] - cur_pos[pool_src]
            edge_len = jnp.linalg.norm(edge_vec, axis=-1)
            edge_len_enc = nnx.vmap(self.dist_enc)(edge_len)

            # IMPORTANT: pass pool_dst_rel (0..Nd-1), not absolute indices
            nxt_feat = pool_block(
                pos=cur_pos,
                feat=cur_feat,
                dst_idx=fps_rel,  # [Nd], absolute indices into cur_pos/cur_feat
                edge_src=pool_src,  # [Ns], absolute in current level
                edge_dst=fps_rel[pool_dst_rel],  # [Ns], *relative* in next level
                valid_edges=pool_mask,
                edge_attr_list=[edge_len_enc],
            )

            # ---- Message passing within the next layer
            # Build edges entirely **within** the next layer, all relative to that layer
            e_src, e_dst, e_val = self.graph_parser[i][1](
                nxt_pos, nxt_pos, nxt_valid, nxt_valid
            )
            e_src = e_src.astype(jnp.int32)
            e_dst = e_dst.astype(jnp.int32)
            ev = nxt_pos[e_dst] - nxt_pos[e_src]
            el = jnp.linalg.norm(ev, axis=-1)
            el_enc = nnx.vmap(self.dist_enc)(el)
            for mp in mp_layers:
                nxt_feat = mp(
                    nxt_pos,
                    nxt_feat,
                    edge_src=e_src,
                    edge_dst=e_dst,
                    valid_edges=e_val,
                    edge_attr_list=[el_enc],
                )

            # stats
            _rms_l0 = jnp.sqrt(jnp.mean((nxt_feat[..., :1].ravel()) ** 2))
            _rms_l1 = jnp.sqrt(jnp.mean((nxt_feat[..., 1:4].ravel()) ** 2))
            _stats[f"after_L{i+1}_rms_l0"] = _rms_l0
            _stats[f"after_L{i+1}_rms_l1"] = _rms_l1
            _stats[f"after_L{i+1}_ratio_rms_l0_l1"] = _rms_l1 / (_rms_l0 + 1e-12)

            # append and advance
            hierarchical_features.append((nxt_pos, nxt_feat, pooling_edges_map))
            abs_indices_per_level.append(nxt_abs)
            cur_pos, cur_feat, cur_valid, cur_abs = (
                nxt_pos,
                nxt_feat,
                nxt_valid,
                nxt_abs,
            )

        return hierarchical_features, {}, _stats
