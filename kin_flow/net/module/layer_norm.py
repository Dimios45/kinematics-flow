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
import numpy as np
from flax import nnx

import kin_flow.util.const as CONST


def get_l_to_all_m_expand_index(lmax: int) -> np.ndarray:
    # length ( (lmax+1)^2 ), value at position i is the degree ℓ for that i
    expand_index = np.zeros((lmax + 1) ** 2, dtype=np.int32)
    for l in range(lmax + 1):
        start = l * l
        length = 2 * l + 1
        expand_index[start : start + length] = l
    return expand_index


class EquivariantRMSLayerNorm(nnx.Module):
    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        centering: bool = True,
        std_balance_degrees: bool = True,
    ):
        super().__init__()
        lmax = CONST.L_MAX
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees

        self.affine_weight = nnx.Param(jnp.ones((num_channels, lmax + 1)))  # (C, L+1)
        self.affine_bias = nnx.Param(jnp.zeros((num_channels,)))  # (C,)

        self.expand_index = nnx.Variable(
            jnp.asarray(get_l_to_all_m_expand_index(lmax), dtype=jnp.int32)
        )  # (I,)

        # degree-balancing weights: each ℓ contributes 1/(L+1) of the total, split evenly across its (2ℓ+1) components
        # row vector shape (1, I) to match einsum "ci,ai->ca"
        I = (lmax + 1) ** 2
        w = np.zeros((1, I), dtype=np.float32)
        for l in range(lmax + 1):
            start = l * l
            length = 2 * l + 1
            w[:, start : start + length] = 1.0 / length
        w /= lmax + 1
        self.balance_degree_weight = nnx.Variable(jnp.asarray(w))  # (1, I)

    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        """
        feature: (C, I) with I = (lmax+1)**2 in CL order [ℓ=0 | ℓ=1 | ℓ=2 | ...].
        Returns: (C, I), normalized and (optionally) affine-transformed.
        """
        x = feature

        if self.centering:
            x_l0, x_rest = x[:, :1], x[:, 1:]  # (C,1), (C,I-1)
            mu = jnp.mean(x_l0, axis=0, keepdims=True)  # (1,1) ← mean over channels
            x_l0 = x_l0 - mu
            x = jnp.concatenate([x_l0, x_rest], axis=1)

        if self.std_balance_degrees:
            x2_I = x * x
            comp = jnp.einsum(
                "ci,ai->ca", x2_I, self.balance_degree_weight.value
            )  # (C,1)
        else:
            comp = jnp.mean(x * x, axis=1, keepdims=True)  # (C,1)
        rms_inv = (jnp.mean(comp, axis=0, keepdims=True) + self.eps) ** (-0.5)  # (1,1)

        if self.affine:
            # (C,L+1) → (C,I)
            wL = self.affine_weight.value[:, self.expand_index.value]
            scale = rms_inv * wL  # broadcast (1,1) across (C,I) → (C,I)
        else:
            scale = rms_inv  # broadcast (1,1) across (C,I)

        y = x * scale  # (C,I)

        if self.affine and self.centering:
            y = y.at[:, :1].add(self.affine_bias.value[:, None])

        return y
