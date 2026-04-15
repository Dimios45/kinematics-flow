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
from typing import List

import jax.numpy as jnp
import numpy as np
from flax import nnx
from sympy import Integer, Poly, diff, factorial, pi, sqrt, symbols

from kin_flow.net.module.so3_coefficient_mapping import \
    CoefficientMappingModule as CM


def s2_grid_np(res_beta, res_alpha):
    i = np.arange(res_beta)
    betas = (i + 0.5) / res_beta * math.pi

    i = np.arange(res_alpha)
    alphas = i / res_alpha * 2 * math.pi
    return betas, alphas


def _sympy_legendre(l, m) -> float:
    l = Integer(l)
    m = Integer(abs(m))
    z, y = symbols("z y", real=True)
    ex = 1 / (2**l * factorial(l)) * y**m * diff((z**2 - 1) ** l, z, l + m)  # type: ignore
    ex *= sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m))
    return ex


def _poly_legendre(l, m):
    z, y = symbols("z y", real=True)
    return Poly(_sympy_legendre(l, m), domain="R", gens=(z, y)).as_dict()  # type: ignore


def legendre(ls: List, z: np.ndarray, y: np.ndarray) -> np.ndarray:
    total_channels = sum(2 * l + 1 for l in ls)
    out = np.zeros(z.shape + (total_channels,))

    i = 0
    for l in ls:
        leg = []
        for m in range(l + 1):
            poly = _poly_legendre(l, m)
            poly_items = list(poly.items())

            (zn, yn), c = poly_items[0]
            x = float(c) * (z**zn) * (y**yn)
            for (zn, yn), c in poly_items[1:]:
                x += float(c) * (z**zn) * (y**yn)
            leg.append(x[..., None])
        for m in range(-l, l + 1):
            out[..., i : i + 1] = leg[abs(m)]  # slow, but only used for precomputation
            i += 1

    return out


def spherical_harmonics_alpha(l: int, alpha: np.ndarray) -> np.ndarray:
    alpha = alpha[..., None]
    m = np.arange(1, l + 1)
    cos = np.cos(m * alpha)  # [..., m]

    m = np.arange(l, 0, -1)
    sin = np.sin(m * alpha)  # [..., m]
    out = np.concatenate(
        [
            math.sqrt(2) * sin,
            np.ones_like(alpha),
            math.sqrt(2) * cos,
        ],
        axis=alpha.ndim - 1,
    )

    return out  # [..., m]


def _expand_matrix(ls):
    lmax = max(ls)
    m = np.zeros((len(ls), 2 * lmax + 1, sum(2 * l + 1 for l in ls)))
    i = 0
    for j, l in enumerate(ls):
        m[j, lmax - l : lmax + l + 1, i : i + 2 * l + 1] = jnp.eye(2 * l + 1)
        i += 2 * l + 1
    return m


def _quadrature_weights(b):
    k = np.arange(b)
    j = np.arange(2 * b)

    factor = 2.0 / b
    sin_j = np.sin(np.pi * (2 * j + 1) / (4 * b))
    k_weights = 1.0 / (2 * k + 1)  # shape: (b,)

    inner = np.sin((2 * j[:, None] + 1) * (2 * k[None, :] + 1) * np.pi / (4 * b))
    inner_sum = np.sum(k_weights * inner, axis=1)
    w = factor * sin_j * inner_sum
    w = w / (2.0 * ((2 * b) ** 2))

    return w


class SO3_Grid(nnx.Module):
    def __init__(
        self,
        lmax,
        mmax,
    ):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.lat_resolution = 2 * (self.lmax + 1)
        if lmax == mmax:
            self.long_resolution = 2 * (self.mmax + 1) + 1
        else:
            self.long_resolution = 2 * (self.mmax) + 1

        self.mapping = CM([self.lmax], [self.lmax])

        # TO
        res_beta = self.lat_resolution
        res_alpha = self.long_resolution
        betas, alphas = s2_grid_np(res_beta, res_alpha)
        to_grid_sha = spherical_harmonics_alpha(lmax, alphas)  # [a, m]

        shb = legendre(list(range(lmax + 1)), np.cos(betas), np.abs(np.sin(betas)))
        n = (
            math.sqrt(4 * math.pi)
            * np.array([1 / math.sqrt(2 * l + 1) for l in range(lmax + 1)])
            / math.sqrt(lmax + 1)
        )
        m = _expand_matrix(range(lmax + 1))
        to_grid_shb = np.einsum("lmj,bj,lmi,l->mbi", m, shb, m, n)
        to_grid_mat = np.einsum("mbi, am -> bai", to_grid_shb, to_grid_sha)
        # rescale based on mmax
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l**2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                to_grid_mat[:, :, start_idx : (start_idx + length)] = (
                    to_grid_mat[:, :, start_idx : (start_idx + length)] * rescale_factor
                )
        to_grid_mat = to_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        # FROM
        betas, alphas = s2_grid_np(res_beta, res_alpha)
        from_grid_sha = spherical_harmonics_alpha(lmax, alphas)  # [a, m]
        shb = legendre(list(range(lmax + 1)), np.cos(betas), np.abs(np.sin(betas)))
        m = _expand_matrix(range(lmax + 1))
        qw = _quadrature_weights(res_beta // 2) * res_beta**2 / res_alpha
        n = (
            math.sqrt(4 * math.pi)
            * np.array([math.sqrt(2 * l + 1) for l in range(lmax + 1)])
            * math.sqrt(lmax + 1)
        )
        from_grid_shb = np.einsum("lmj,bj,lmi,l,b->mbi", m, shb, m, n, qw)  # [m, b, i]

        from_grid_mat = np.einsum("am, mbi -> bai", from_grid_sha, from_grid_shb)
        # rescale based on mmax
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l**2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                from_grid_mat[:, :, start_idx : (start_idx + length)] = (
                    from_grid_mat[:, :, start_idx : (start_idx + length)]
                    * rescale_factor
                )
        from_grid_mat = from_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        self.to_grid_mat = nnx.Cache(jnp.array(to_grid_mat))
        self.from_grid_mat = nnx.Cache(jnp.array(from_grid_mat))

    def get_to_grid_mat(self):
        return self.to_grid_mat

    def get_from_grid_mat(self):
        return self.from_grid_mat


def to_grid(x, to_grid_mat, coefficient_idx):
    to_grid_mat = to_grid_mat[:, :, coefficient_idx]
    x_grid = jnp.einsum("bai, ic -> bac", to_grid_mat, x)
    return x_grid


def from_grid(x, from_grid_mat, coefficient_idx):
    from_grid_mat = from_grid_mat[:, :, coefficient_idx]
    x = jnp.einsum("bai, bac -> ic", from_grid_mat, x)
    return x


def precompute_fibonacci_l2_matrices(
    dirs: np.ndarray,
    *,
    l=2,
):
    mapping = CM([l], [l])
    x, y, z = np.asarray(dirs.T)
    betas = np.arccos(jnp.clip(z, -1.0, 1.0))
    alphas = np.mod(jnp.arctan2(y, x), 2.0 * math.pi)
    to_grid_sha = spherical_harmonics_alpha(l, alphas)
    shb = legendre(list(range(l + 1)), np.cos(betas), np.abs(np.sin(betas)))
    n = (
        math.sqrt(4 * math.pi)
        * np.array([1 / math.sqrt(2 * l + 1) for l in range(l + 1)])
        / math.sqrt(l + 1)
    )
    m = _expand_matrix(range(l + 1))
    to_grid_shb = np.einsum("lmj,bj,lmi,l->mbi", m, shb, m, n)
    to_grid_mat = np.einsum("mbi, am -> bai", to_grid_shb, to_grid_sha)
    to_grid_mat = to_grid_mat[:, :, mapping.coefficient_idx(l, l)]

    from_grid_sha = spherical_harmonics_alpha(l, alphas)  # [a, m]
    shb = legendre(list(range(l + 1)), np.cos(betas), np.abs(np.sin(betas)))
    m = _expand_matrix(range(l + 1))
    qw = _quadrature_weights(len(dirs) // 2) * len(dirs) ** 2 / len(dirs)
    n = (
        math.sqrt(4 * math.pi)
        * np.array([math.sqrt(2 * l + 1) for l in range(l + 1)])
        * math.sqrt(l + 1)
    )
    from_grid_shb = np.einsum("lmj,bj,lmi,l,b->mbi", m, shb, m, n, qw)  # [m, b, i]
    from_grid_mat = np.einsum("am, mbi -> bai", from_grid_sha, from_grid_shb)
    idx_diag = np.arange(len(dirs), dtype=np.int32)
    l2_readout = jnp.asarray(from_grid_mat[idx_diag, idx_diag, 4:])
    return l2_readout
