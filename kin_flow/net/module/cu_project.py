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


class CUELinearProject(nnx.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        rngs: nnx.Rngs,
    ):
        self.irreps_in = cue.Irreps(
            "SO3", f"{channels_in}x0+{channels_in}x1+{channels_in}x2"
        )
        self.irreps_out = cue.Irreps(
            "SO3", f"{channels_out}x0+{channels_out}x1+{channels_out}x2"
        )
        self.linear = cue.descriptors.linear(self.irreps_in, self.irreps_out)

        limit = 1.0 / jnp.sqrt(channels_in)
        self.l_weights = nnx.Param(
            jax.random.uniform(
                rngs(),
                (self.linear.inputs[0].dim,),
                minval=-limit,
                maxval=limit,
            )
        )
        self.channels_out = channels_out

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        _, L = x.shape

        x_cue = cuex.RepArray(self.linear.inputs[1], x.T.ravel(), layout=cue.ir_mul)
        w_cue = cuex.RepArray(
            self.linear.inputs[0], self.l_weights.value, layout=cue.mul_ir
        )
        out = cuex.equivariant_polynomial(
            self.linear,
            [w_cue, x_cue],
            method="naive",
            math_dtype="float32",
        ).array
        out = out.reshape(L, self.channels_out).T
        return out
