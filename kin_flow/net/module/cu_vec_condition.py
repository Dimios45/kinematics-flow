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

import cuequivariance as cue
import cuequivariance_jax as cuex
import jax
import jax.numpy as jnp
from flax import nnx

import kin_flow.util.const as CONST


class CUEScalarDirectionConditioning(nnx.Module):
    def __init__(
        self,
        irreps_in: str | cue.Irreps,
        rngs: nnx.Rngs,
    ):
        self.irreps_in = cue.Irreps("SO3", irreps_in)
        self.num_channels = self.irreps_in.muls[0]  # e.g., 64
        self.irrep_1ch = cue.Irreps("SO3", "1x0+1x1+1x2")
        self.irrep_vec = cue.Irreps("SO3", "1x1")
        self.depth_wise = cue.descriptors.channelwise_tensor_product(
            self.irreps_in, self.irrep_1ch, simplify_irreps3=True
        )

        self.linear = cue.descriptors.linear(
            self.depth_wise.outputs[0].irreps, self.irreps_in
        )

        limit = 1.0 / jnp.sqrt(15)
        self.d_weights = nnx.Param(
            jax.random.uniform(
                rngs(),
                (self.depth_wise.inputs[0].dim,),
                minval=-limit,
                maxval=limit,
            )
        )

        limit = 1.0 / jnp.sqrt(3 * self.num_channels)
        self.l_weights = nnx.Param(
            jax.random.uniform(
                rngs(),
                (self.linear.inputs[0].dim,),
                minval=-limit,
                maxval=limit,
            )
        )

    def __call__(self, x: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
        """
        x: (E, C, L)
        vec: (E, 3)
        weights: (E, C * L_SH) - Modulation weights
        """
        _E, C, L = x.shape

        vec_cue = cuex.RepArray(self.irrep_vec, vec, layout=cue.ir_mul)
        sh = cuex.spherical_harmonics(range(CONST.L_MAX + 1), vec_cue)
        x_flat = jnp.transpose(x, axes=[0, 2, 1]).reshape(*x.shape[:-2], -1)
        x_cue = cuex.RepArray(self.depth_wise.inputs[1], x_flat, layout=cue.ir_mul)
        w_cue = cuex.RepArray(
            self.depth_wise.inputs[0], self.d_weights.value, layout=cue.ir_mul
        )
        depth_wise_out = cuex.equivariant_polynomial(
            self.depth_wise,
            [w_cue, x_cue, sh],
            # method="uniform_1d", # does not work on v100 -> only on h100/200
            method="naive",
            math_dtype="float32",
        )
        w_cue = cuex.RepArray(
            self.linear.inputs[0], self.l_weights.value, layout=cue.ir_mul
        )
        out = cuex.equivariant_polynomial(
            self.linear,
            [w_cue, depth_wise_out],
            method="naive",
            math_dtype="float32",
        ).array
        out = out.reshape(*out.shape[:-1], L, C).transpose(0, 2, 1)
        return out
