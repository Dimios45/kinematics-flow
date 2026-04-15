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

import jax.numpy as jnp
from flax import nnx

import kin_flow.util.const as CONST
from kin_flow.util.irreps_helper import convert_cl_to_lc, convert_lc_to_cl


class EquiFiLM(nnx.Module):
    def __init__(self, in_channels, time_channels, hidden_mlp, *, rngs):
        self.in_channels = in_channels
        self.time_channels = time_channels
        self.lmax = CONST.L_MAX

        scalar_mlp_sizes = list(hidden_mlp) + [2 * in_channels]
        scalar_layers = []
        in_dim = time_channels
        for k, out_dim in enumerate(scalar_mlp_sizes):
            scalar_layers.append(nnx.Linear(in_dim, out_dim, use_bias=True, rngs=rngs))
            if k < len(scalar_mlp_sizes) - 1:
                scalar_layers.append(nnx.gelu)
            in_dim = out_dim

        self.time_scalar_scales = nnx.Sequential(*scalar_layers)

        vec_mlp_sizes = list(hidden_mlp) + [CONST.L_MAX * in_channels]
        vec_layers = []
        in_dim = time_channels
        for k, out_dim in enumerate(vec_mlp_sizes):
            vec_layers.append(nnx.Linear(in_dim, out_dim, use_bias=True, rngs=rngs))
            if k < len(vec_mlp_sizes) - 1:
                vec_layers.append(nnx.gelu)
            in_dim = out_dim

        self.time_vec_scales = nnx.Sequential(*vec_layers)

        expand_index = jnp.zeros(shape=((CONST.L_MAX + 1) ** 2), dtype=jnp.int32)
        for l_irrep in range(CONST.L_MAX + 1):
            start_idx = l_irrep**2
            length = 2 * l_irrep + 1
            expand_index = expand_index.at[start_idx : (start_idx + length)].set(
                l_irrep - 1
            )
        self.expand_index = nnx.Variable(expand_index[1:])

    def __call__(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:

        assert x.shape == (self.in_channels, (self.lmax + 1) ** 2)
        x = convert_cl_to_lc(x, self.in_channels, self.lmax)

        x_0, x_vec = x[:1], x[1:]
        assert t.shape == (self.time_channels,)
        t_emb = t
        weight = jnp.reshape(
            self.time_vec_scales(t_emb), shape=(self.lmax, self.in_channels)
        )
        weight = weight[self.expand_index.value, ...]
        x_vec_out = jnp.einsum("mi,mi->mi", x_vec, (1.0 + weight))

        gamma_beta = self.time_scalar_scales(t_emb)  # (2C,)
        gamma, beta = jnp.split(gamma_beta, 2, axis=-1)  # (C,), (C,)
        x_0 = x_0 * (1.0 + gamma)[None, :] + beta[None, :]

        x_out = jnp.concatenate([x_0, x_vec_out], axis=0)

        assert x_out.shape == ((self.lmax + 1) ** 2, self.in_channels)
        x_out = convert_lc_to_cl(x_out, self.in_channels, self.lmax)
        return x_out
