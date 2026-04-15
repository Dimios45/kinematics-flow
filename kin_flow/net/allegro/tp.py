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
from typing import List, Optional, Tuple

import e3nn_jax
import jax
import jax.numpy as jnp
from flax import nnx


def wigner_3j(l1: int, l2: int, l3: int, dtype=None):
    assert abs(l2 - l3) <= l1 <= l2 + l3
    assert isinstance(l1, int) and isinstance(l2, int) and isinstance(l3, int)
    C = e3nn_jax.clebsch_gordan(l1, l2, l3)
    return C


class ContracterNNX(nnx.Module):
    _weight_w3j_einstr: str
    _contract_einstr: str
    mul: int
    base_dim1: int
    base_dim2: int
    base_dim_out: int
    num_paths: int

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        mul: int,
        instructions: Optional[List[Tuple[int, int, int]]] = None,
        path_channel_coupling: bool = True,
        scatter_factor: Optional[float] = None,
        irrep_normalization: str = "component",
        *,
        rngs,
    ):
        self.scatter_factor = scatter_factor

        # -- Instruction management --
        self.instructions = instructions
        if instructions is None:
            instructions = []
            for i_out, (_, ir_out) in enumerate(irreps_out):
                for i_1, (_, ir_1) in enumerate(irreps_in1):
                    for i_2, (_, ir_2) in enumerate(irreps_in2):
                        if ir_out in ir_1 * ir_2:
                            instructions.append((i_1, i_2, i_out))

        assert mul > 0
        self_irreps_in1 = e3nn_jax.Irreps(irreps_in1)
        base_irreps1 = e3nn_jax.Irreps((1, ir) for _, ir in self_irreps_in1)
        dim1 = base_irreps1.dim
        assert all(m == 1 for m, ir in self_irreps_in1)
        self_irreps_in2 = e3nn_jax.Irreps(irreps_in2)
        base_irreps2 = e3nn_jax.Irreps((1, ir) for _, ir in self_irreps_in2)
        dim2 = base_irreps2.dim
        assert all(m == 1 for m, ir in irreps_in2)
        self_irreps_out = e3nn_jax.Irreps(irreps_out)
        base_irreps_out = e3nn_jax.Irreps((1, ir) for _, ir in self_irreps_out)
        dimout = base_irreps_out.dim
        assert all(m == 1 for m, ir in self_irreps_out)
        self.base_dim1 = dim1
        self.base_dim2 = dim2
        self.base_dim_out = dimout
        self.mul = mul
        self.num_paths: int = len(instructions)
        assert self.num_paths > 0, "No TP paths available"

        # -- Make the w3j --
        # list of tensors of shape [N, 3] containing i,j,k indexes
        w3j_index: List[jnp.ndarray] = []
        # list of tensors of shape [N] containing w3j values
        w3j_values: List[jnp.ndarray] = []

        for ins_i, ins in enumerate(instructions):
            ir_in1 = base_irreps1[ins[0]].ir
            ir_in2 = base_irreps2[ins[1]].ir
            ir_out = base_irreps_out[ins[2]].ir

            # Check instruction against the O3 selection rules
            assert ir_in1.p * ir_in2.p == ir_out.p
            assert abs(ir_in1.l - ir_in2.l) <= ir_out.l <= ir_in1.l + ir_in2.l

            this_w3j = wigner_3j(ir_in1.l, ir_in2.l, ir_out.l)
            this_w3j_index = jnp.stack(jnp.nonzero(this_w3j), axis=1)
            w3j_values.append(
                this_w3j[
                    this_w3j_index[:, 0], this_w3j_index[:, 1], this_w3j_index[:, 2]
                ]
            )

            self.irrep_normalization = irrep_normalization
            if self.irrep_normalization is None:
                w3j_norm_term = 1
            elif self.irrep_normalization == "component":
                w3j_norm_term = math.sqrt(2 * ir_out.l + 1)
            else:
                raise NotImplementedError(
                    f"`{self.irrep_normalization}` `irrep_normalization` is not implemented"
                )
            w3j_values[-1] = w3j_values[-1] * w3j_norm_term

            this_w3j_index = this_w3j_index.at[:, 0].add(base_irreps1[: ins[0]].dim)
            this_w3j_index = this_w3j_index.at[:, 1].add(base_irreps2[: ins[1]].dim)
            this_w3j_index = this_w3j_index.at[:, 2].add(base_irreps_out[: ins[2]].dim)
            w3j_index.append(this_w3j_index)
            del ir_in1, ir_in2, ir_out, w3j_norm_term, this_w3j, this_w3j_index

            # pijk
        w3j = jnp.zeros(
            shape=(
                self.num_paths,
                base_irreps1.dim,
                base_irreps2.dim,
                base_irreps_out.dim,
            )
        )

        for path_index, (path_w3j_indexes, path_w3j_values) in enumerate(
            zip(w3j_index, w3j_values)
        ):
            w3j = w3j.at[
                path_index,  # p
                path_w3j_indexes[:, 0],  # i
                path_w3j_indexes[:, 1],  # j
                path_w3j_indexes[:, 2],  # k
            ].set(path_w3j_values)

        self.w3j = nnx.Variable(w3j)

        self.path_channel_coupling = path_channel_coupling
        weight_shape = (self.mul,) if self.path_channel_coupling else tuple()
        if self.num_paths > 1:
            weight_shape = weight_shape + (self.num_paths,)
        self.weights = nnx.Param(
            jax.random.uniform(
                rngs(), shape=weight_shape, minval=-math.sqrt(3), maxval=math.sqrt(3)
            )
        )

        ij = "ij"
        p = "p" if self.num_paths > 1 else ""
        u = "u" if self.path_channel_coupling else ""
        self._weight_w3j_einstr = f"{u}{p},{p}{ij}k->{u}{ij}k"

    def __call__(
        self,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
        idxs: jnp.ndarray,
        scatter_dim_size: int,
    ) -> jnp.ndarray:

        if self.scatter_factor is not None:
            x2 = self.scatter_factor * x2

        x2_scatter = jax.ops.segment_sum(
            x2,
            idxs,
            num_segments=scatter_dim_size,
        )

        x2 = jnp.take(x2_scatter, idxs, axis=0)

        x1 = x1.reshape((-1, self.mul, self.base_dim1))
        x2 = x2.reshape((-1, self.mul, self.base_dim2))
        return self._contract(x1, x2)

    def _contract(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        # for shared weights, we can precontract weights and w3j so they can be frozen together
        # this is usually advantageous for inference, since the weights would have to be
        # multiplied in anyway at some point
        # `up, pijk -> uijk`` or `p, pijk -> ijk`
        ww3j = jnp.einsum(self._weight_w3j_einstr, self.weights.value, self.w3j.value)

        outer = jnp.expand_dims(x1, -1) * jnp.expand_dims(x2, -2)  # (Z, U, I, J)
        if self.path_channel_coupling:
            # zuij, uijk → zuk
            out = jnp.sum(
                jnp.expand_dims(outer, -1) * ww3j,  # (Z, U, I, J, K)
                axis=(2, 3),  # sum over i & j
            )  # → (Z, U, K)
        else:
            # (zu)(ij) · (ij)k  →  (zu)k  →  zuk
            out = jnp.matmul(
                outer.reshape(
                    outer.shape[0] * outer.shape[1],  # Z·U
                    outer.shape[2] * outer.shape[3],  # I·J
                ),  # (Z·U, I·J)
                ww3j.reshape(-1, ww3j.shape[2]),  # (I·J, K)
            ).reshape(
                -1, self.mul, ww3j.shape[2]
            )  # (Z, U, K)

        return out

    def extra_repr(self):
        return f"{self.irreps_in1} x {self.irreps_in2} -> {self.irreps_out} | {self.mul} channels | {self.num_paths} paths"
