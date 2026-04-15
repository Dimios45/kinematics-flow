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


class MakeWeightedChannelsNNX(nnx.Module):
    weight_numel: int
    multiplicity_out: int
    weight_individual_irreps: bool
    alpha: float
    _num_irreps: int

    def __init__(
        self,
        irreps_in,
        multiplicity_out: int,
        alpha: float = 1.0,
        weight_individual_irreps: bool = True,
    ):
        assert all([mul == 1 for mul, ir in irreps_in])
        assert multiplicity_out >= 1
        self._num_irreps = len(irreps_in)
        self.multiplicity_out = multiplicity_out
        self.weight_individual_irreps = weight_individual_irreps
        self.alpha = alpha
        self.weight_numel = len(irreps_in) * multiplicity_out
        # Each edgewise output multiplicity is a per-irrep weighted sum over the input
        # So we need to apply the weight for the ith irrep to all DOF in that irrep
        rtoi = jnp.zeros(shape=(self._num_irreps, irreps_in.dim))
        for i, this_slice in enumerate(irreps_in.slices()):
            rtoi = rtoi.at[i, this_slice].set(alpha)
        self.rtoi = nnx.Variable(rtoi)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n  num_irreps: {self._num_irreps}\n  multiplicity_out: {self.multiplicity_out}\n  weight_numel: {self.weight_numel}\n)"

    def __call__(self, edge_attr, weights, multi_edge=False):
        # weights are [z, u, r]
        # edge_attr are [z, i]
        # r runs over all irreps, which is why the weights need
        # to be indexed in order to go from [r] to [i]
        # [zu]r @ ri -> [zu]i -> zui
        aux = jnp.matmul(
            weights.reshape(-1, self._num_irreps), self.rtoi.value
        ).reshape(
            (
                edge_attr.shape[0],
                self.multiplicity_out,
                self.rtoi.value.shape[1],
            )
        )
        if multi_edge:
            # zui,zui->zui
            out = edge_attr * aux
        else:
            # zi,zui->zui
            out = edge_attr[..., None, :] * aux
        return out
