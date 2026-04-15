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

from dataclasses import dataclass, field, is_dataclass
from typing import Tuple

import e3nn_jax
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps
from flax import nnx

import kin_flow.util.const as CONST
from kin_flow.kin.const import KINS
from kin_flow.kin.kinematics_simple import KinematicEncoderSimple
from kin_flow.net.allegro.feature_pyramid import PCDPyramid
from kin_flow.net.allegro.weighted_channels import MakeWeightedChannelsNNX
from kin_flow.net.ffn import FeedForwardNetwork
from kin_flow.net.module.graph_layer import GraphLayer
from kin_flow.net.module.multi_scale_tensor_field import (
    MultiscaleTensorField, MultiscaleTensorFieldConfiguration)
from kin_flow.net.module.radial_function import SinusoidalPositionEmbeddings
from kin_flow.net.module.so2_graph_layer import SO2GraphLayerSimple


class KinematicEncoding(nnx.Module):
    def __init__(
        self,
        num_channels,
        positional_scaling,
        time_freq=400.0,
        kin_list=[],
        num_attn_layer=2,
        *,
        rngs,
    ):
        self.kinematic_net = KinematicEncoderSimple(
            rngs=rngs,
            kins=kin_list,
        )
        self.dist_enc = SinusoidalPositionEmbeddings(
            num_channels, positional_scaling / 4.0, n=time_freq  # empirical value
        )
        self.time_enc = SinusoidalPositionEmbeddings(num_channels, 1.0, n=400.0)
        self.mlp = nnx.Sequential(
            *[
                nnx.Linear(4 * num_channels, 4 * num_channels, rngs=rngs),
                nnx.gelu,
                nnx.Linear(4 * num_channels, 4 * num_channels, rngs=rngs),
                nnx.gelu,
                nnx.Linear(4 * num_channels, 4 * num_channels, rngs=rngs),
                nnx.gelu,
                nnx.Linear(4 * num_channels, 3 * num_channels, rngs=rngs),
            ]
        )
        self.gl = GraphLayer(
            num_channels, 64 + 64, output_channels=3 * num_channels, rngs=rngs
        )

        self.weight_irrep = MakeWeightedChannelsNNX(
            irreps_in=Irreps("1x0e+1x1e+1x2e"),
            multiplicity_out=num_channels,
            alpha=1.0,
            weight_individual_irreps=True,
        )
        self.ln_dist = nnx.LayerNorm(num_channels, rngs=rngs)
        self.ln_time = nnx.LayerNorm(num_channels, rngs=rngs)
        self.ln_q_dot = nnx.LayerNorm(num_channels, rngs=rngs)
        self.ln_q_dot_sh = nnx.LayerNorm(num_channels, rngs=rngs)
        self.ln_dist_graph = nnx.LayerNorm(num_channels, rngs=rngs)
        self.ln_time_graph = nnx.LayerNorm(num_channels, rngs=rngs)

        self.attn_layers = []
        num_l_channels = 3 * num_channels
        for _ in range(num_attn_layer):
            self.attn_layers.append(
                (
                    nnx.LayerNorm(num_l_channels, rngs=rngs),
                    nnx.MultiHeadAttention(
                        num_heads=6,
                        in_features=num_l_channels,
                        qkv_features=num_l_channels,
                        out_features=num_l_channels,
                        rngs=rngs,
                    ),
                    nnx.Sequential(
                        *[
                            nnx.gelu,
                            nnx.Linear(num_l_channels, num_l_channels, rngs=rngs),
                        ]
                    ),
                )
            )

        # these are similar weights as in a weighted per channel tensor product
        # for scalar features
        self.q_dot_modulation = nnx.Param(
            jax.random.uniform(rngs(), shape=(num_channels, CONST.L_MAX + 1))
        )
        self.q_dot_sh_modulation = nnx.Param(
            jax.random.uniform(rngs(), shape=(num_channels, CONST.L_MAX + 1))
        )
        expand_index = jnp.zeros(shape=((CONST.L_MAX + 1) ** 2), dtype=jnp.int32)
        for l in range(CONST.L_MAX + 1):
            start_idx = l**2
            length = 2 * l + 1
            expand_index = expand_index.at[start_idx : (start_idx + length)].set(l)
        self.expand_index = nnx.Variable(expand_index)

    def __call__(self, gripper_idx, wigner_d, q_pos, t):
        # compute joint positions after dof transform
        q_feat, (idx, idx_p), valid = self.kinematic_net(gripper_idx, wigner_d)
        assert (
            q_pos.shape == (CONST.MAX_DOF + 3, 3)  # 2 pose + 1 base + 22 DoF
            and q_feat.shape == (CONST.MAX_DOF + 3, 64, 9)
            and valid.shape == (CONST.MAX_DOF + 3,)
            and idx.shape == (CONST.MAX_DOF,)
            and idx_p.shape == (CONST.MAX_DOF,)
        )
        q_vec = q_pos[2:][idx] - q_pos[2:][idx_p]
        q_vec_sh = e3nn_jax.vmap(e3nn_jax.spherical_harmonics, in_axes=(None, 0, None))(
            "1x0e+1x1e+1x2e", q_vec, True
        ).array
        q_dist = jnp.linalg.norm(q_vec, axis=-1)
        q_dist_enc = nnx.vmap(self.dist_enc)(q_dist)
        time_enc = self.time_enc(t)[None]
        time_enc = jnp.broadcast_to(time_enc, shape=q_dist_enc.shape)
        q_dot_modulation = self.q_dot_modulation[
            None, :, self.expand_index.value
        ]  # (1, C, 9)
        q_dot = jnp.einsum(
            "dcl,dcl->dc", q_feat[2:][idx_p] * q_dot_modulation, q_feat[2:][idx]
        )
        q_dot_sh_modulation = self.q_dot_sh_modulation[
            None, :, self.expand_index.value
        ]  # (1, C, 9)
        q_dot_sh = jnp.einsum(
            "dcl,dl->dc", q_feat[2:][idx] * q_dot_sh_modulation, q_vec_sh
        )
        scalars = jnp.concatenate(
            [
                self.ln_q_dot_sh(q_dot_sh),
                self.ln_q_dot(q_dot),
                self.ln_dist(q_dist_enc),
                self.ln_time(time_enc),
            ],
            axis=-1,
        )
        weights = self.mlp(scalars)
        q_feat_dof = self.weight_irrep(q_feat[2:][idx], weights, True)

        # drop the base node: only required to encode joint scalars
        q_feat = jnp.concatenate([q_feat[:2], q_feat_dof], axis=0)
        q_pos = jnp.concatenate([q_pos[:2], q_pos[3:]], axis=0)
        valid = jnp.concatenate([valid[:2], valid[3:]], axis=0)
        assert len(q_feat) == CONST.MAX_DOF + 2
        assert len(q_pos) == CONST.MAX_DOF + 2
        assert len(valid) == CONST.MAX_DOF + 2

        # graph layer over whole gripper configuration
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
        weights = self.gl(
            q_feat,
            q_feat,
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

        for ln, attn, ffn in self.attn_layers:
            dw = attn(ln(weights), mask=attn_mask, deterministic=True, decode=False)
            dw = ffn(dw)
            weights = weights + dw
        q_feat = self.weight_irrep(q_feat, weights, True) + q_feat
        return q_feat, q_pos, valid


class GraphDecoder(nnx.Module):
    def __init__(
        self,
        num_channels,
        positional_scaling,
        time_freq=400.0,
        num_attn_layer=2,
        *,
        rngs,
    ):
        self.dof_mlp = nnx.Sequential(
            *[
                nnx.Linear(64, 64, rngs=rngs),
                nnx.gelu,
                nnx.Linear(64, 64, rngs=rngs),
                nnx.gelu,
                nnx.Linear(64, 1, rngs=rngs),
            ]
        )
        max_dist = positional_scaling / 4.0
        self.dist_enc = SinusoidalPositionEmbeddings(
            num_channels, max_dist, n=time_freq
        )
        self.time_enc = SinusoidalPositionEmbeddings(num_channels, 1.0, n=time_freq)

        self.ln_dist_pose = nnx.LayerNorm(64, rngs=rngs)
        self.ln_tmp_pose = nnx.LayerNorm(64, rngs=rngs)
        self.ln_dist_dof = nnx.LayerNorm(64, rngs=rngs)
        self.ln_tmp_dof = nnx.LayerNorm(64, rngs=rngs)
        self.pose_gl = SO2GraphLayerSimple(
            num_channels, 64 + 64, num_heads=4, rngs=rngs
        )
        self.dof_gl = GraphLayer(num_channels, 64 + 64, num_heads=4, rngs=rngs)
        self.eps = 1e-5
        self.rot_head = FeedForwardNetwork(
            num_channels,
            1,
            l_max=1,
            scalar_mlp=[num_channels, num_channels, num_channels],
            grid_mlp=[num_channels, num_channels, num_channels],
            rngs=rngs,
        )
        self.pos_head = FeedForwardNetwork(
            num_channels,
            1,
            l_max=1,
            scalar_mlp=[num_channels, num_channels, num_channels],
            grid_mlp=[num_channels, num_channels, num_channels],
            rngs=rngs,
        )

        self.attn_layers = []
        for _ in range(num_attn_layer):
            self.attn_layers.append(
                (
                    nnx.LayerNorm(num_channels, rngs=rngs),
                    nnx.MultiHeadAttention(
                        num_heads=4,
                        in_features=num_channels,
                        qkv_features=num_channels,
                        out_features=num_channels,
                        rngs=rngs,
                    ),
                    nnx.Sequential(
                        *[
                            nnx.gelu,
                            nnx.Linear(num_channels, num_channels, rngs=rngs),
                        ]
                    ),
                )
            )

    def __call__(self, feat, q_pos, valid, t):
        ### DOF ###
        feat_rev = jnp.concatenate([feat[2:], feat[:2]], axis=0)  # dof + pose
        q_pos_rev = jnp.concatenate([q_pos[2:], q_pos[:2]], axis=0)  # dof + pose
        valid_rev = jnp.concatenate([valid[2:], valid[:2]], axis=0)  # dof + pose
        edge_src = jnp.arange(start=0, stop=CONST.MAX_DOF + 2, dtype=jnp.int32)
        edge_dst = jnp.arange(start=0, stop=CONST.MAX_DOF, dtype=jnp.int32)
        edge_src, edge_dst = jnp.meshgrid(edge_src, edge_dst, indexing="ij")
        edge_src = edge_src.reshape(-1)
        edge_dst = edge_dst.reshape(-1)

        edge_vec = q_pos[2:][edge_dst] - q_pos_rev[edge_src]
        edge_dist = jnp.linalg.norm(edge_vec, axis=-1)
        edge_dist_enc = nnx.vmap(self.dist_enc)(edge_dist)
        min_dist_mask = edge_dist > 1e-5
        no_self_loop_mask = edge_src != edge_dst
        valid_mask = (
            valid[2:][edge_dst]
            & valid_rev[edge_src]
            & no_self_loop_mask
            & min_dist_mask
        )
        time_enc = self.time_enc(t)
        time_enc = jnp.broadcast_to(
            time_enc[None, :], (len(edge_src), time_enc.shape[-1])
        )
        edge_attr = [
            self.ln_dist_dof(edge_dist_enc),
            self.ln_tmp_dof(time_enc),
        ]
        edge_attr = jnp.concatenate(edge_attr, axis=-1)
        scalars = (
            self.dof_gl(
                feat_rev,
                feat[2:],
                q_pos_rev,
                q_pos[2:],
                edge_attr,
                edge_src,
                edge_dst,
                valid_mask,
            )
            + feat[2:, ..., 0]
        )
        # Final DoF Attention Layers
        attn_mask = valid[2:][None, :] & valid[2:][:, None]
        L = len(valid[2:])
        attn_mask = jnp.broadcast_to(attn_mask, (4, L, L))  # (Heads, L, L)
        for ln, attn, ffn in self.attn_layers:
            ds = attn(ln(scalars), mask=attn_mask, deterministic=True, decode=False)
            ds = ffn(ds)
            scalars = scalars + ds

        ### POSE ###
        edge_src = jnp.arange(start=0, stop=CONST.MAX_DOF, dtype=jnp.int32)
        edge_src = jnp.concatenate([edge_src, edge_src])
        edge_dst = jnp.array([0] * CONST.MAX_DOF + [1] * CONST.MAX_DOF, dtype=jnp.int32)

        edge_vec = q_pos[:2][edge_dst] - q_pos[2:][edge_src]
        edge_dist = jnp.linalg.norm(edge_vec, axis=-1)
        edge_dist_enc = nnx.vmap(self.dist_enc)(edge_dist)
        min_dist_mask = edge_dist > 1e-5
        valid_mask = valid[:2][edge_dst] & valid[2:][edge_src] & min_dist_mask
        time_enc = self.time_enc(t)
        time_enc = jnp.broadcast_to(
            time_enc[None, :], (len(edge_src), time_enc.shape[-1])
        )
        edge_attr = [
            self.ln_dist_pose(edge_dist_enc),
            self.ln_tmp_pose(time_enc),
        ]
        edge_attr = jnp.concatenate(edge_attr, axis=-1)
        feat = (
            self.pose_gl(
                feat[2:],
                feat[:2],
                q_pos[2:],
                q_pos[:2],
                edge_attr,
                edge_src,
                edge_dst,
                valid_mask,
            )
            + feat[:2, ..., :4]
        )

        emb_rot, emb_pos = feat[0], feat[1]
        u_rot = self.rot_head(emb_rot)[0, 1:4]
        u_pos = self.pos_head(emb_pos)[0, 1:4]
        u_dof = self.dof_mlp(scalars)[..., 0]
        u = jnp.concatenate([u_rot, u_pos, u_dof], axis=0)
        return u


@dataclass
class KinematicsFlowConfiguration:
    time_freq: float = 400.0
    dist_freq: float = 50.0
    position_scaling: float = 10.0
    num_channels: int = 64
    num_mp_layer: tuple[int, int, int] = (2, 2, 2)
    multi_scale_tensor_field: MultiscaleTensorFieldConfiguration = field(
        default_factory=MultiscaleTensorFieldConfiguration
    )
    gripper: tuple[str, ...] = ("panda",)

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


class KinematicsFlow(nnx.Module):
    def __init__(
        self,
        config: KinematicsFlowConfiguration,
        *,
        rngs,
    ):
        cfg = config
        num_channels = cfg.num_channels
        self.lmax = CONST.L_MAX
        self.num_channels = cfg.num_channels
        self.position_scaling = cfg.position_scaling

        kin_list = []
        for gripper in config.gripper:
            kin_list.append(KINS[gripper](rngs))

        self.kinematic_encoding = KinematicEncoding(
            num_channels,
            cfg.position_scaling,
            cfg.time_freq,
            kin_list=kin_list,
            rngs=rngs,
        )

        self.unet = PCDPyramid(
            time_freq=cfg.time_freq,
            layer_channels=cfg.multi_scale_tensor_field.multiscale_channels,
            num_message_passing_layers=cfg.num_mp_layer,
            rngs=rngs,
        )
        self.multi_scale_feature_query = MultiscaleTensorField(
            cfg.multi_scale_tensor_field,
            rngs=rngs,
        )
        self.decoder = GraphDecoder(
            num_channels,
            cfg.position_scaling,
            cfg.time_freq,
            rngs=rngs,
        )

    def __call__(
        self,
        scene: Tuple[jnp.ndarray, jnp.ndarray],
        gripper_idx: jnp.ndarray,
        wigner_d: jnp.ndarray,
        q_pos_rot: jnp.ndarray,  # rotated, but not translated features -> numerical stability on SO3
        q_pos_rot_trans: jnp.ndarray,  # both rotated and translated
        t: jnp.ndarray,
        key,
    ):
        # Encode scene
        scene_points, _ = scene
        multiscale_feature_pyramid, _, stats = self.unet(scene_points, key)

        # Encode gripper
        # NOTE: We only require relative distances between joints
        q_feat, q_pos, valid = self.kinematic_encoding(
            gripper_idx, wigner_d, q_pos_rot, t
        )

        # Relate scene and query
        # NOTE: Actual pose of gripper in scene space required
        feat, _ = self.multi_scale_feature_query(
            multiscale_feature_pyramid, ((q_pos_rot_trans, q_pos), q_feat), t, valid
        )

        # Decode flow
        # NOTE: We only require relative distances between joints
        u = self.decoder(feat, q_pos, valid, t)
        dof_valid_mask = (
            self.kinematic_encoding.kinematic_net.valid_mask_with_static_flag(
                gripper_idx
            )[1:]
        )  # exclude base node
        return (u, dof_valid_mask), {
            **stats,
        }

    def get_kins(self):
        kins = self.kinematic_encoding.kinematic_net.kins
        return kins
