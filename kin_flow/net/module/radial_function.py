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


class SinusoidalPositionEmbeddings(nnx.Module):
    def __init__(self, dim: int, max_val: float, n: float = 10000.0):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "dim must be an even number!"
        self.n = float(n)
        self.max_val = float(max_val)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x / self.max_val * self.n  # Scale time to 0~10000

        half_dim = self.dim // 2
        emb_factor = math.log(self.n) / (half_dim - 1)
        emb_factor = jnp.array(emb_factor)
        arange = jnp.arange(half_dim)
        emb = jnp.exp(-emb_factor * arange)
        embeddings = x * emb
        embeddings = jnp.concatenate(
            [jnp.sin(embeddings), jnp.cos(embeddings)], axis=-1
        )

        return embeddings
