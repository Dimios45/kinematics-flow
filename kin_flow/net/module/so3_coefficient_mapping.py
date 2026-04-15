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
import numpy as np
from flax import nnx


class CoefficientMappingModule:
    def __init__(self, lmax_list, mmax_list):
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)

        # Compute the degree (l) and order (m) for each entry of the embedding.
        self.l_harmonic = np.array([], dtype=int)
        self.m_harmonic = np.array([], dtype=int)
        self.m_complex = np.array([], dtype=int)

        self.res_size = np.zeros(self.num_resolutions, dtype=int)

        offset = 0
        for i in range(self.num_resolutions):
            for l in range(0, self.lmax_list[i] + 1):
                # For each degree l, the allowed orders are in the range [-min(mmax, l), min(mmax, l)].
                current_mmax = min(self.mmax_list[i], l)
                m = np.arange(-current_mmax, current_mmax + 1, dtype=int)
                self.m_complex = np.concatenate([self.m_complex, m])
                self.m_harmonic = np.concatenate([self.m_harmonic, np.abs(m)])
                self.l_harmonic = np.concatenate(
                    [self.l_harmonic, np.full(m.shape, l, dtype=int)]
                )
            self.res_size[i] = len(self.l_harmonic) - offset
            offset = len(self.l_harmonic)

        num_coefficients = len(self.l_harmonic)
        self.to_m = np.zeros((num_coefficients, num_coefficients), dtype=float)
        self.m_size = np.zeros(max(self.mmax_list) + 1, dtype=int)

        offset = 0
        for m in range(max(self.mmax_list) + 1):
            idx_r, idx_i = self.complex_idx(m, -1, self.m_complex, self.l_harmonic)
            for idx_out, idx_in in enumerate(idx_r):
                self.to_m[idx_out + offset, idx_in] = 1.0
            offset += len(idx_r)
            self.m_size[m] = len(idx_r)
            for idx_out, idx_in in enumerate(idx_i):
                self.to_m[idx_out + offset, idx_in] = 1.0
            offset += len(idx_i)

        self.lmax_cache = None
        self.mmax_cache = None
        self.mask_indices_cache = None
        self.rotate_inv_rescale_cache = None

    def complex_idx(self, m, lmax, m_complex, l_harmonic):
        if lmax == -1:
            lmax = max(self.lmax_list)
        indices = np.arange(len(l_harmonic))
        mask_r = (l_harmonic <= lmax) & (m_complex == m)
        mask_idx_r = indices[mask_r]

        mask_idx_i = np.array([], dtype=int)
        if m != 0:
            mask_i = (l_harmonic <= lmax) & (m_complex == -m)
            mask_idx_i = indices[mask_i]

        return mask_idx_r, mask_idx_i

    def coefficient_idx(self, lmax, mmax):
        if (self.lmax_cache is not None) and (self.mmax_cache is not None):
            if (self.lmax_cache == lmax) and (self.mmax_cache == mmax):
                if self.mask_indices_cache is not None:
                    return self.mask_indices_cache

        mask = (self.l_harmonic <= lmax) & (self.m_harmonic <= mmax)
        indices = np.arange(len(self.l_harmonic))
        self.mask_indices_cache = indices[mask]
        self.lmax_cache, self.mmax_cache = lmax, mmax
        return self.mask_indices_cache

    def get_rotate_inv_rescale(self, lmax, mmax):
        if (self.lmax_cache is not None) and (self.mmax_cache is not None):
            if (self.lmax_cache == lmax) and (self.mmax_cache == mmax):
                if self.rotate_inv_rescale_cache is not None:
                    return self.rotate_inv_rescale_cache

        if self.mask_indices_cache is None:
            self.coefficient_idx(lmax, mmax)

        rotate_inv_rescale = np.ones((1, (lmax + 1) ** 2, (lmax + 1) ** 2), dtype=float)
        for l in range(lmax + 1):
            if l <= mmax:
                continue
            start_idx = l**2
            length = 2 * l + 1
            rescale_factor = math.sqrt(length / (2 * mmax + 1))
            rotate_inv_rescale[
                :, start_idx : start_idx + length, start_idx : start_idx + length
            ] = rescale_factor
        rotate_inv_rescale = rotate_inv_rescale[:, :, self.mask_indices_cache]
        self.rotate_inv_rescale_cache = rotate_inv_rescale
        return self.rotate_inv_rescale_cache


class PrecomputedMappings(nnx.Module):
    def __init__(self, lmax=2) -> None:

        self.map = CoefficientMappingModule(lmax_list=[lmax], mmax_list=[lmax])
        self.to_m_var = nnx.Variable(jnp.array(self.map.to_m))

    def to_m(self):
        return self.to_m_var.value
