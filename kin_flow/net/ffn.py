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

import jax
import jax.numpy as jnp
from flax import nnx

import kin_flow.util.const as CONST
from kin_flow.net.module.so3_coefficient_mapping import \
    CoefficientMappingModule
from kin_flow.net.module.so3_grid import SO3_Grid, from_grid, to_grid
from kin_flow.util.irreps_helper import convert_cl_to_lc, convert_lc_to_cl


class SO3LinearCL(nnx.Module):
    def __init__(self, in_features, out_features, l_max=CONST.L_MAX, *, rngs):
        self.in_features = in_features
        self.out_features = out_features

        bound = 1 / math.sqrt(self.in_features)
        self.weight = nnx.Param(
            jax.random.uniform(
                rngs(),
                (l_max + 1, out_features, in_features),
                minval=-bound,
                maxval=bound,
            )
        )
        self.bias = nnx.Param(jnp.zeros(out_features))
        expand_index = jnp.zeros(shape=((l_max + 1) ** 2), dtype=jnp.int32)
        for l in range(l_max + 1):
            start_idx = l**2
            length = 2 * l + 1
            expand_index = expand_index.at[start_idx : (start_idx + length)].set(l)
        self.expand_index = nnx.Variable(expand_index)

    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        weight = self.weight[
            self.expand_index.value, ...
        ]  # [(l_max + 1) ** 2, C_out, C_in]
        x_out = jnp.einsum("im,moi->om", x, weight)
        bias_out = x_out[..., 0] + self.bias
        x_out = x_out.at[..., 0].set(bias_out)
        return x_out


class SO3Linear(nnx.Module):
    def __init__(self, in_features, out_features, l_max=CONST.L_MAX, *, rngs):
        self.in_features = in_features
        self.out_features = out_features

        bound = 1 / math.sqrt(self.in_features)
        self.weight = nnx.Param(
            jax.random.uniform(
                rngs(),
                (l_max + 1, out_features, in_features),
                minval=-bound,
                maxval=bound,
            )
        )
        self.bias = nnx.Param(jnp.zeros(out_features))
        expand_index = jnp.zeros(shape=((l_max + 1) ** 2), dtype=jnp.int32)
        for l in range(l_max + 1):
            start_idx = l**2
            length = 2 * l + 1
            expand_index = expand_index.at[start_idx : (start_idx + length)].set(l)
        self.expand_index = nnx.Variable(expand_index)

    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        weight = self.weight[
            self.expand_index.value, ...
        ]  # [(l_max + 1) ** 2, C_out, C_in]
        x_out = jnp.einsum("mi,moi->mo", x, weight)
        bias_out = x_out[0, ...] + self.bias
        x_out = x_out.at[0, ...].set(bias_out)
        return x_out


def make_mlp(in_dim, sizes, *, rngs):
    layers, cur = [], in_dim
    for i, out_dim in enumerate(sizes):
        layers.append(nnx.Linear(cur, out_dim, use_bias=True, rngs=rngs))
        if i < len(sizes) - 1:
            layers.append(nnx.gelu)
        cur = out_dim
    return nnx.Sequential(*layers)


class FeedForwardNetwork(nnx.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scalar_mlp,
        grid_mlp,
        l_max=CONST.L_MAX,
        *,
        rngs,
    ):
        assert scalar_mlp[-1] == grid_mlp[-1]
        mlp_last = scalar_mlp[-1]
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.grid: SO3_Grid = SO3_Grid(l_max + 1, l_max + 1)
        self.so3_linear_1 = SO3Linear(
            self.input_channels,
            grid_mlp[0],
            l_max=l_max,
            rngs=rngs,
        )

        self.scalar_mlp = make_mlp(self.input_channels, list(scalar_mlp), rngs=rngs)
        self.grid_mlp = make_mlp(grid_mlp[0], list(grid_mlp)[1:], rngs=rngs)
        self.pre_grid_act = nnx.gelu
        self.pose_grid_act = nnx.gelu

        self.so3_linear_2 = SO3Linear(
            mlp_last,
            self.output_channels,
            l_max=l_max,
            rngs=rngs,
        )
        self.coefficient_idx = nnx.Variable(
            jnp.array(
                CoefficientMappingModule(
                    [l_max + 1],
                    [l_max + 1],
                ).coefficient_idx(l_max, l_max),
                jnp.int32,
            )
        )
        self.lmax = l_max

    def __call__(self, input: jnp.ndarray):
        # shape of input is assumed to be (c, (lmax + 1) ** 2)
        # this code requires ((lmax + 1) ** 2, c)
        assert input.shape == (self.input_channels, (self.lmax + 1) ** 2)
        input = convert_cl_to_lc(input, self.input_channels, self.lmax)

        gating_scalars = self.scalar_mlp(input[0])
        gating_scalars = gating_scalars[None, :]
        input = self.so3_linear_1(input)

        input_grid = to_grid(
            input,
            self.grid.get_to_grid_mat().value,
            self.coefficient_idx.value,
        )
        input_grid = self.pre_grid_act(input_grid)
        input_grid = self.grid_mlp(input_grid)
        input_grid = self.pose_grid_act(input_grid)
        input = from_grid(
            input_grid,
            self.grid.get_from_grid_mat().value,
            self.coefficient_idx.value,
        )
        input = jnp.concatenate([gating_scalars, input[1:, ...]], axis=0)
        input = self.so3_linear_2(input)

        # we need to convert back for consistency
        assert input.shape == ((self.lmax + 1) ** 2, self.output_channels)
        input = convert_lc_to_cl(input, self.output_channels, self.lmax)
        return input
