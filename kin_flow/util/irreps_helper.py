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

import e3nn_jax
import jax.numpy as jnp


def convert_lc_to_cl(data: jnp.ndarray, num_channels: int, lmax: int):
    l, c = data.shape[0], data.shape[1]
    assert l == (lmax + 1) ** 2, "Data shape does not match expected lmax"
    assert c == num_channels, "Data shape does not match expected num_channels"

    return jnp.transpose(data, (1, 0))


def convert_cl_to_lc(data: jnp.ndarray, num_channels: int, lmax: int):
    c, l = data.shape[0], data.shape[1]
    assert l == (lmax + 1) ** 2, "Data shape does not match expected lmax"
    assert c == num_channels, "Data shape does not match expected num_channels"

    return jnp.transpose(data, (1, 0))


def convert_from_irreps_to_cl(
    irreps: e3nn_jax.IrrepsArray,
    num_channels: int,
    lmax: int,
):
    l_total = (lmax + 1) ** 2
    assert (
        irreps.array.shape[0] == l_total * num_channels
    ), "Wrong number of channels -> possible l-type leak!"
    raw_data = irreps.array
    out_array = jnp.empty((num_channels, l_total), dtype=raw_data.dtype)

    current_l_type_start = 0
    for l_type in range(lmax + 1):
        out_array = out_array.at[
            :, current_l_type_start : (current_l_type_start + 2 * l_type + 1)
        ].set(
            raw_data[
                num_channels
                * current_l_type_start : num_channels
                * (current_l_type_start + 2 * l_type + 1)
            ].reshape(num_channels, 2 * l_type + 1)
        )
        current_l_type_start += 2 * l_type + 1
    return out_array


def convert_from_irreps_to_lc(
    irreps: e3nn_jax.IrrepsArray,
    num_channels: int,
    lmax: int,
):
    cl = convert_from_irreps_to_cl(irreps, num_channels, lmax)
    return convert_cl_to_lc(cl, num_channels, lmax)


def convert_from_cl_to_irreps(
    data: jnp.ndarray,
    num_channels: int,
    lmax: int,
) -> e3nn_jax.IrrepsArray:
    c, l_data = data.shape[0], data.shape[1]
    l_total = (lmax + 1) ** 2
    assert c == num_channels, "Wrong number of channels -> possible l-type leak!"
    assert l_data == l_total, "Data shape does not match expected lmax"

    into_irreps = " + ".join([f"{c}x{l}e" for l in range(lmax + 1)])

    result = []
    current_l_type_start = 0
    for l_type in range(lmax + 1):
        result.append(
            data[
                :, current_l_type_start : (current_l_type_start + 2 * l_type + 1)
            ].ravel()
        )
        current_l_type_start += 2 * l_type + 1

    return e3nn_jax.IrrepsArray(into_irreps, jnp.concatenate(result))


def convert_from_lc_to_irreps(
    data: jnp.ndarray,
    num_channels: int,
    lmax: int,
) -> e3nn_jax.IrrepsArray:
    l_data, c = data.shape[0], data.shape[1]
    l_total = (lmax + 1) ** 2
    assert c == num_channels, "Wrong number of channels -> possible l-type leak!"
    assert l_data == l_total, "Data shape does not match expected lmax"

    cl_data = convert_lc_to_cl(data, num_channels, lmax)
    return convert_from_cl_to_irreps(cl_data, num_channels, lmax)


def convert_l_to_m_major(irreps: jnp.ndarray, to_m_matrix):
    res = jnp.einsum("cl, bl -> cb", irreps, to_m_matrix)
    return res


def convert_m_to_l_major(irreps: jnp.ndarray, to_m_matrix):
    res = jnp.einsum("cl, lb -> cb", irreps, to_m_matrix)
    return res


def complex_idx(m, lmax, m_complex, l_harmonic):
    # Create an array of indices for each coefficient
    indices = jnp.arange(l_harmonic.shape[0])

    # Build mask for the "real" part: coefficients with l <= lmax and m_complex == m
    mask_r = jnp.logical_and(l_harmonic <= lmax, m_complex == m)
    mask_idx_r = indices[mask_r]

    # For m != 0, also get the "imaginary" part: coefficients with l <= lmax and m_complex == -m
    mask_idx_i = jnp.array([], dtype=indices.dtype)
    if m != 0:
        mask_i = jnp.logical_and(l_harmonic <= lmax, m_complex == -m)
        mask_idx_i = indices[mask_i]

    return mask_idx_r, mask_idx_i


def expand_to_l_type(data: jnp.ndarray, lmax, mmax):
    channels = data.shape[-1]

    expanded_data = jnp.empty(((lmax + 1) ** 2, channels))
    for l in range(lmax + 1):
        start_idx = l**2
        length = 2 * l + 1
        expanded_data[start_idx : (start_idx + length)] = data[l]
    return expanded_data
