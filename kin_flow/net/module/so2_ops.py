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

import jax.numpy as jnp
from flax import nnx

from kin_flow.net.module.so3_coefficient_mapping import PrecomputedMappings
from kin_flow.util.irreps_helper import (convert_l_to_m_major,
                                         convert_m_to_l_major)


class SO2_m_Convolution(nnx.Module):
    def __init__(
        self,
        rngs,
        m: int,
        in_channel: int,
        out_channel: int,
        lmax: int,
    ):
        self.m = m
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.lmax = lmax

        num_coefficients = self.lmax - m + 1
        self.num_channels = num_coefficients * self.in_channel
        self.out_channels = 2 * self.out_channel * num_coefficients
        self.fc = nnx.Linear(
            self.num_channels,
            self.out_channels,
            use_bias=False,
            rngs=rngs,
        )
        self.fc.kernel.value *= 1.0 / math.sqrt(2.0)

    def __call__(self, x_m: jnp.ndarray) -> jnp.ndarray:
        y = self.fc(x_m.T)  # -> (2, out_channels)

        half = self.out_channels // 2
        x_r = y[:, :half]  # (2, out_channels/2)
        x_i = y[:, half:]  # (2, out_channels/2)

        x_m_r = x_r[0:1, :] - x_i[1:2, :]  # (1, out_channels/2)
        x_m_i = x_r[1:2, :] + x_i[0:1, :]  # (1, out_channels/2)
        x_out = jnp.concatenate([x_m_r, x_m_i], axis=0).T
        return x_out


class SO2_Convolution(nnx.Module):

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        lmax: int,
        *,
        mlp: list[int],
        rngs,
        extra_channels: int = 0,
    ):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.lmax = lmax
        self.mlp = list(mlp)
        self.extra_channels = extra_channels

        num_coefficients = self.lmax + 1
        num_channels_m0 = num_coefficients * self.in_channel
        self.m0_in = num_channels_m0
        m0_output_channels = num_coefficients * self.out_channel + self.extra_channels
        self.fc_0 = nnx.Linear(
            num_channels_m0, m0_output_channels, use_bias=True, rngs=rngs
        )

        self.so2_m_conv = []
        num_channels_rad = num_channels_m0
        for m in range(1, self.lmax + 1):
            conv = SO2_m_Convolution(
                rngs=rngs,
                m=m,
                in_channel=self.in_channel,
                out_channel=self.out_channel,
                lmax=self.lmax,
            )
            self.so2_m_conv.append(conv)
            num_channels_rad += conv.num_channels
        self.num_channels_rad = num_channels_rad

        self.map = PrecomputedMappings(lmax=self.lmax)

        self.mlp.append(int(num_channels_rad))
        mlp_layers = []
        for in_feat, out_feat in zip(self.mlp[:-1], self.mlp[1:]):
            mlp_layers.append(nnx.silu)
            mlp_layers.append(nnx.Linear(in_feat, out_feat, rngs=rngs))
        self.mlp = nnx.Sequential(*mlp_layers[1:])

    def __call__(self, x: jnp.ndarray, x_edge: jnp.ndarray):
        out_blocks = []

        x_array = convert_l_to_m_major(x, self.map.to_m())  # (C_in, total_m_len)
        x_edge = self.mlp(x_edge)  # (num_channels_rad,)
        x_edge = x_edge.reshape(-1)  # ensure 1-D
        x_0 = x_array[:, : self.lmax + 1]  # (C_in, lmax+1)
        x_0 = x_0.reshape(-1)  # (C_in * (lmax+1),)
        x_0 = x_0 * x_edge[: self.m0_in]  # gate
        x_0 = self.fc_0(x_0)  # (m0_output_channels,)
        if self.extra_channels > 0:
            x_0_extra = x_0[: self.extra_channels]  # (extra_channels,)
            x_0 = x_0[self.extra_channels :]  # (C_out*(lmax+1),)
        else:
            x_0_extra = jnp.zeros((0,), dtype=x_0.dtype)

        x_0 = x_0.reshape(self.out_channel, -1)  # (C_out, lmax+1)
        out_blocks.append(x_0)
        offset = self.lmax + 1
        offset_rad = self.m0_in
        for m in range(1, self.lmax + 1):
            m_size = self.lmax + 1 - m

            x_m = x_array[:, offset : offset + 2 * m_size]
            x_m = x_m.reshape(-1, 2)  # (C_in * m_size, 2)

            nrad = self.so2_m_conv[m - 1].num_channels  # C_in * m_size
            x_edge_m = x_edge[offset_rad : offset_rad + nrad]  # (C_in * m_size,)
            x_m = x_m * x_edge_m[:, None]

            x_m = self.so2_m_conv[m - 1](x_m)  # (C_out * m_size, 2)
            x_m = x_m.reshape(self.out_channel, -1)  # (C_out, 2*m_size)
            out_blocks.append(x_m)

            offset += 2 * m_size
            offset_rad += nrad

        out = jnp.concatenate(out_blocks, axis=1)  # (C_out, total_m_len)
        out = convert_m_to_l_major(out, self.map.to_m())  # (C_out, total_l_len)

        return out, x_0_extra


def init_edge_rot_mat_deterministic(
    edge: jnp.ndarray, tol: float = 1e-4, eps: float = 1e-5
):
    length = jnp.linalg.norm(edge)
    valid = length >= tol

    norm_x = edge / jnp.maximum(length, eps)
    fixed_vecs = jnp.eye(3, dtype=edge.dtype)  # shape (3, 3)
    dots = jnp.abs(fixed_vecs @ norm_x)  # (3,)
    min_idx = jnp.argmin(dots)
    selected_fixed = fixed_vecs[min_idx]  # (3,)
    z_raw = jnp.cross(norm_x, selected_fixed)
    z = z_raw / jnp.maximum(jnp.linalg.norm(z_raw), eps)
    y_raw = jnp.cross(norm_x, z)
    y = y_raw / jnp.maximum(jnp.linalg.norm(y_raw), eps)
    rot_inv = jnp.stack([z, norm_x, -y], axis=1)  # (3, 3), columns are z, x, -y
    rot = rot_inv.T  # (3, 3)
    rot = jnp.where(valid, rot, jnp.eye(3, dtype=edge.dtype))

    return rot, valid
