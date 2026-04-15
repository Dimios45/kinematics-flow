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
from dataclasses import dataclass, is_dataclass
from typing import List, Tuple

import e3nn_jax
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps
from flax import nnx

import kin_flow.util.const as CONST
from kin_flow.net.allegro.weighted_channels import MakeWeightedChannelsNNX
from kin_flow.net.module.cu_project import CUELinearProject
from kin_flow.net.module.cu_vec_condition import CUEScalarDirectionConditioning
from kin_flow.net.module.fctp import FullyConnectedTP
from kin_flow.net.module.film import EquiFiLM
from kin_flow.net.module.graph_layer import GraphLayer
from kin_flow.net.module.layer_norm import EquivariantRMSLayerNorm
from kin_flow.net.module.radial_function import SinusoidalPositionEmbeddings
from kin_flow.util.irreps_helper import (convert_from_cl_to_irreps,
                                         convert_from_irreps_to_cl)


@dataclass
class MultiscaleTensorFieldConfiguration:
    blocks: int = 1
    num_scene_points: int = 5
    position_scaling: float = 10.0
    time_freq: float = 400.0
    dist_freq: float = 50.0
    max_dist_length: float = 5.0
    num_channels: int = 64
    multiscale_channels: Tuple[int, ...] = (32, 64, 64)

    def from_dict(self, d: dict):
        for k, v in d.items():
            if not hasattr(self, k):
                continue
            cur = getattr(self, k)

            if is_dataclass(cur) and isinstance(v, dict):
                cur.from_dict(v)
                continue

            if isinstance(cur, tuple) and isinstance(v, list):
                setattr(self, k, type(cur)(v))
            else:
                setattr(self, k, v)
        return self

    @classmethod
    def from_dict_cls(cls, d: dict):
        return cls().from_dict(d)


class QueryAtMultiLayer(nnx.Module):
    def __call__(self, multi_scale_feature_pcd, query_pos):
        per_layer_feature = []
        p0, f0, _ = multi_scale_feature_pcd[0]
        dist_to_positions = jnp.linalg.norm(query_pos[None] - p0, axis=-1)
        idx_min = jnp.argmin(dist_to_positions)
        per_layer_feature.append(f0[idx_min])
        for _, features, pooling_edges in multi_scale_feature_pcd[1:]:
            idx_min = pooling_edges[idx_min]
            per_layer_feature.append(features[idx_min])
        feat = jnp.concatenate(per_layer_feature, axis=-2)
        return feat


class FeatureProjectLayer(nnx.Module):
    def __init__(self, multiscale_channels, num_channels, *, rngs):
        self.film = EquiFiLM(
            num_channels,
            num_channels,
            hidden_mlp=[
                64,
                64,
            ],
            rngs=rngs,
        )
        self.s_proj = CUELinearProject(
            sum(multiscale_channels),
            num_channels,
            rngs=rngs,
        )
        self.per_scale_ln = []
        for c in multiscale_channels:
            self.per_scale_ln.append((c, EquivariantRMSLayerNorm(c)))

    def __call__(self, x, time_emb):
        current_c = 0
        for c, ln in self.per_scale_ln:
            x = x.at[current_c : current_c + c].set(ln(x[current_c : current_c + c]))
            current_c += c
        x = self.s_proj(x)
        x = self.film(x, time_emb)
        return x


class FeatureExtractionLayer(nnx.Module):
    def __init__(self, num_channels, k, num_heads=4, *, rngs) -> None:
        self.k = k
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.attn_mlp = nnx.Sequential(
            *[
                nnx.Linear(num_channels * 4, 256, rngs=rngs),
                nnx.gelu,
                nnx.Linear(256, 256, rngs=rngs),
                nnx.gelu,
                nnx.Linear(256, num_channels * 4, rngs=rngs),
            ]
        )
        self.attn_weights = nnx.Param(
            jax.random.uniform(
                rngs(),
                shape=(num_heads, num_channels),
                minval=-1.0 / math.sqrt(num_channels),
                maxval=1.0 / math.sqrt(num_channels),
            )
        )
        self.ln_scalar = nnx.LayerNorm(num_features=64, rngs=rngs)
        self.ln_time = nnx.LayerNorm(num_features=64, rngs=rngs)
        self.ln_q_feat = nnx.LayerNorm(num_features=num_channels, rngs=rngs)
        self.ln_q_dot = nnx.LayerNorm(num_features=num_channels, rngs=rngs)

        self.q_dot_sh_modulation = nnx.Param(
            jax.random.uniform(rngs(), shape=(num_channels, CONST.L_MAX + 1))
        )
        self.q_s_modulation = nnx.Param(
            jax.random.uniform(rngs(), shape=(num_channels, CONST.L_MAX + 1))
        )
        expand_index = jnp.zeros(shape=((CONST.L_MAX + 1) ** 2), dtype=jnp.int32)
        for l in range(CONST.L_MAX + 1):
            start_idx = l**2
            length = 2 * l + 1
            expand_index = expand_index.at[start_idx : (start_idx + length)].set(l)
        self.expand_index = nnx.Variable(expand_index)
        self.mha = nnx.MultiHeadAttention(
            num_heads=8,
            in_features=num_channels * 4,
            qkv_features=num_channels * 4,
            out_features=num_channels * 4,
            rngs=rngs,
        )
        self.weight_irrep = MakeWeightedChannelsNNX(
            irreps_in=Irreps("1x0e+1x1e+1x2e"),
            multiplicity_out=num_channels,
            alpha=1.0,
            weight_individual_irreps=True,
        )
        self.mlp = nnx.Sequential(
            *[
                nnx.gelu,
                nnx.Linear(4 * num_channels, 4 * num_channels, rngs=rngs),
                nnx.gelu,
                nnx.Linear(4 * num_channels, 3 * num_channels, rngs=rngs),
            ]
        )
        self.layer = CUEScalarDirectionConditioning(
            "64x0+64x1+64x2",
            rngs=rngs,
        )

    def __call__(self, query, source, edges, len_emb, time_emb):
        # --- Feature Extraction ---
        # Scalars for gating
        C = self.num_channels
        q_s_modulation = self.q_s_modulation[:, self.expand_index.value]  # (C, 9)
        q_s_scalars = jnp.einsum("cl,kcl->kc", query * q_s_modulation, source)
        sh = e3nn_jax.vmap(e3nn_jax.spherical_harmonics, in_axes=(None, 0, None))(
            "1x0e+1x1e+1x2e", edges, True
        ).array  # (k, 9)
        q_dot_sh_modulation = self.q_dot_sh_modulation[
            :, self.expand_index.value
        ]  # (C, 9)
        q_dot_sh = jnp.einsum("cl,kl->kc", query * q_dot_sh_modulation, sh)

        # Gating weights and weighted directional irreps
        scalars = jnp.concatenate(
            [
                self.ln_scalar(len_emb),
                self.ln_time(time_emb),
                self.ln_q_feat(q_s_scalars),
                self.ln_q_dot(q_dot_sh),
            ],
            axis=-1,
        )  # (k, 256)
        weights = self.mha(
            scalars,
            deterministic=True,
            decode=False,
        )
        weights = self.mlp(weights)
        source = self.weight_irrep(source, weights, True) + source
        source = self.layer(source, edges)
        attn = self.attn_mlp(scalars)  # (k, num_channels * 4)
        attn = jnp.reshape(attn, shape=(-1, self.num_heads, self.num_channels))
        attn_logits = jnp.einsum("ehc,hc->eh", attn, self.attn_weights.value)  # (k,H)
        alpha_k = jax.nn.softmax(attn_logits, axis=0)[
            ..., None, None
        ]  # shape (k,H,1,1)
        message = jnp.reshape(
            source,
            shape=(
                -1,
                self.num_heads,
                self.num_channels // self.num_heads,
                (CONST.L_MAX + 1) ** 2,
            ),
        )  # (num_nodes, num_heads, ...)
        message = (message * alpha_k).sum(axis=0)  # (H, C/H, 9)
        message = jnp.reshape(
            message, shape=(self.num_channels, (CONST.L_MAX + 1) ** 2)
        )
        return message


class MergeFeatureLayer(nnx.Module):
    def __init__(self, num_channels, *, rngs):
        self.num_channels = num_channels
        in1 = Irreps(f"{num_channels}x0e+{num_channels}x1e+{num_channels}x2e")
        in2 = Irreps(f"{num_channels}x0e+{num_channels}x1e+{num_channels}x2e")
        out_irreps = Irreps(f"{num_channels}x0e+{num_channels}x1e+{num_channels}x2e")
        self.tp = FullyConnectedTP(in1, in2, out_irreps, lmax=2, rngs=rngs)

        self.ln_feat = EquivariantRMSLayerNorm(num_channels)
        self.ln_query = EquivariantRMSLayerNorm(num_channels)

    def __call__(self, q_feat, feat):
        # Final TP with query features and rotate back
        C = self.num_channels
        feat = self.ln_feat(feat)
        q_feat = self.ln_query(q_feat)
        feat = convert_from_cl_to_irreps(feat, C, CONST.L_MAX)
        q_feat_ir = convert_from_cl_to_irreps(q_feat, C, CONST.L_MAX)
        q_feat_ir = self.tp(q_feat_ir, feat)
        q_feat = convert_from_irreps_to_cl(q_feat_ir, C, CONST.L_MAX)
        return q_feat


class QueryMessagePassing(nnx.Module):
    def __init__(
        self, num_channels, position_scaling, time_freq, num_attn_layer=2, *, rngs
    ):
        self.num_channels = num_channels
        self.weight_irrep = MakeWeightedChannelsNNX(
            irreps_in=Irreps("1x0e+1x1e+1x2e"),
            multiplicity_out=num_channels,
            alpha=1.0,
            weight_individual_irreps=True,
        )
        self.ln_dist_graph = nnx.LayerNorm(num_channels, rngs=rngs)
        self.ln_time_graph = nnx.LayerNorm(num_channels, rngs=rngs)
        self.dist_enc = SinusoidalPositionEmbeddings(
            num_channels, position_scaling / 4.0, n=time_freq  # empirical value
        )
        self.time_enc = SinusoidalPositionEmbeddings(num_channels, 1.0, n=400.0)
        self.ln_graph = EquivariantRMSLayerNorm(num_channels)
        self.gl = GraphLayer(
            num_channels, 64 + 64, output_channels=3 * num_channels, rngs=rngs
        )
        self.attn_layers = []
        num_l_channels = 3 * num_channels
        for _ in range(num_attn_layer):
            self.attn_layers.append(
                nnx.MultiHeadAttention(
                    num_heads=6,
                    in_features=num_l_channels,
                    qkv_features=num_l_channels,
                    out_features=num_l_channels,
                    rngs=rngs,
                )
            )

    def __call__(self, q_pos, q_feat, valid, t):
        # q_pos needs to be in scene coordinates for this one
        # transform back afterwards
        edge_src = jnp.arange(start=0, stop=CONST.MAX_DOF + 2, dtype=jnp.int32)
        edge_dst = jnp.arange(start=0, stop=CONST.MAX_DOF + 2, dtype=jnp.int32)
        edge_src, edge_dst = jnp.meshgrid(edge_src, edge_dst, indexing="ij")
        edge_src = edge_src.reshape(-1)
        edge_dst = edge_dst.reshape(-1)
        edge_vec = q_pos[edge_dst] - q_pos[edge_src]
        edge_dist = jnp.linalg.norm(edge_vec, axis=-1)
        edge_dist_enc = nnx.vmap(self.dist_enc)(edge_dist)
        no_self_loop_mask = edge_src != edge_dst
        valid_mask = valid[edge_dst] & valid[edge_src] & no_self_loop_mask
        time_enc = self.time_enc(t)
        time_enc = jnp.broadcast_to(
            time_enc[None, :], (len(edge_src), time_enc.shape[-1])
        )
        edge_attr = jnp.concatenate(
            [
                self.ln_dist_graph(edge_dist_enc),
                self.ln_time_graph(time_enc),
            ],
            axis=-1,
        )
        q_feat_ln = nnx.vmap(self.ln_graph)(q_feat)
        weights = self.gl(
            q_feat_ln,
            q_feat_ln,
            q_pos,
            q_pos,
            edge_attr,
            edge_src,
            edge_dst,
            valid_mask,
        )

        # additional scalar attention layers
        attn_mask = valid[None, :] & valid[:, None]
        L = len(valid)
        attn_mask = jnp.broadcast_to(attn_mask, (6, L, L))  # (Heads, L, L)

        for attn in self.attn_layers:
            weights = attn(weights, mask=attn_mask, deterministic=True, decode=False)
        q_feat = self.weight_irrep(q_feat, weights, True)
        return q_feat


class MultiscaleTensorField(nnx.Module):
    def __init__(self, conf: MultiscaleTensorFieldConfiguration, *, rngs):
        super().__init__()
        self.lmax = CONST.L_MAX
        self.num_channels = conf.num_channels
        self.k = conf.num_scene_points
        self.multiscale_channels = conf.multiscale_channels

        self.query_at = QueryAtMultiLayer()
        self.readout_blocks = []
        for _ in range(conf.blocks):
            ln = EquivariantRMSLayerNorm(conf.num_channels)
            proj = FeatureProjectLayer(
                multiscale_channels=conf.multiscale_channels,
                num_channels=conf.num_channels,
                rngs=rngs,
            )
            extract = FeatureExtractionLayer(conf.num_channels, self.k, rngs=rngs)
            merge = MergeFeatureLayer(conf.num_channels, rngs=rngs)
            mp = QueryMessagePassing(
                conf.num_channels,
                position_scaling=conf.position_scaling,
                time_freq=conf.time_freq,
                rngs=rngs,
            )
            self.readout_blocks.append((ln, proj, extract, merge, mp))

        self.pos_emb = SinusoidalPositionEmbeddings(64, max_val=5.0, n=conf.dist_freq)
        self.time_emb = SinusoidalPositionEmbeddings(64, max_val=1.0, n=conf.time_freq)

    def __call__(
        self,
        featurized_multiscale_source_pcd: List[Tuple[jnp.ndarray, jnp.ndarray]],
        featurized_query_pcd: Tuple[jnp.ndarray, jnp.ndarray],
        t: jnp.ndarray,
        valid: jnp.ndarray,
    ):
        (q_pos, q_pos_rot), q_feat = featurized_query_pcd
        s_pos_first = featurized_multiscale_source_pcd[0][0]

        # k-NN indices via identical argpartition logic
        D = jnp.sum(
            (q_pos[:, None, :] - s_pos_first[None, :, :]) ** 2, axis=2
        )  # (Q, S)
        Q, _S = D.shape
        k = self.k
        part_idx = jnp.argpartition(D, k, axis=1)
        idx = part_idx[:, :k]  # (Q, k)

        # shuffle idx
        query_at = s_pos_first[idx]  # (Q, k, 3)

        # Query multi-scale features at those positions: vmap over k within each Q, then over Q
        vm_query = nnx.vmap(self.query_at, in_axes=(None, 0))  # map over k
        s_feat = nnx.vmap(vm_query, in_axes=(None, 0))(  # map over Q
            featurized_multiscale_source_pcd, query_at
        )  # (Q, k, feat_dim)

        # Time conditioning
        time_emb = self.time_emb(t)  # (64,)
        time_emb = jnp.broadcast_to(time_emb, (Q, k, 64))  # cheap broadcast view
        query_scene_edges = q_pos[:, None, :] - query_at
        r_len = jnp.linalg.norm(query_scene_edges, axis=-1)  # (Q, k)
        len_emb = nnx.vmap(nnx.vmap(self.pos_emb))(r_len)  # (Q, k, 64)

        # --- Readout Blocks ---
        for ln, proj, extract, merge, mp in self.readout_blocks:
            s_feat_p = nnx.vmap(nnx.vmap(proj))(
                s_feat,
                time_emb,  # we use different k for each block
            )
            q_feat_ln = nnx.vmap(ln)(q_feat)
            feat = nnx.vmap(extract)(
                q_feat_ln,
                s_feat_p,
                query_scene_edges,
                len_emb,
                time_emb,
            )
            q_feat_delta = nnx.vmap(merge)(q_feat_ln, feat)
            q_feat = q_feat + q_feat_delta  # skip

            # NOTE: Again only relative positions here
            q_feat = mp(q_pos_rot, q_feat, valid, t) + q_feat

        return q_feat, {}
