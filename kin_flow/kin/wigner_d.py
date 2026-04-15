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

import functools
import time

import numpy as np
from scipy.spatial.transform import Rotation


def wigner_D_real(l: int, R: np.ndarray) -> np.ndarray:
    """
    Compute Real Wigner D-matrices matching e3nn_jax conventions.

    Implementation:
    - Converts R to axis-angle coordinates.
    - Constructs the Lie Algebra element A = sum(angle_i * Generator_i).
    - Computes exp(A) using a vectorized Scaling and Squaring method with Taylor expansion.

    This avoids the slow python loop of scipy.linalg.expm and guarantees exact
    basis compatibility by using the authoritative e3nn generators.

    Args:
        l: Harmonic order.
        R: Batch of rotation matrices (..., 3, 3).

    Returns:
        D: (..., 2l+1, 2l+1) with dtype float64.
    """
    # 1. Prepare Inputs
    R = np.asarray(R, dtype=np.float64)
    prefix_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    B = R_flat.shape[0]

    if l == 0:
        return np.ones(prefix_shape + (1, 1), dtype=np.float64)

    # 2. Get Generators (Cached)
    # X_basis: (3, dim, dim)
    X_basis = _get_e3nn_generators(l)
    map_indices = _get_generator_mapping()  # Map e3nn generators to x,y,z
    X_xyz = X_basis[map_indices]

    # 3. Convert R to Log Coordinates (Axis-Angle)
    # This is the standard mapping: R = exp(theta * n . J)
    # SciPy's as_rotvec returns vector v = theta * n
    rot_vecs = Rotation.from_matrix(R_flat).as_rotvec()  # (B, 3)

    # 4. Construct Exponent Matrices
    # A = v_x * X + v_y * Y + v_z * Z
    # Einsum: (B, 3) x (3, D, D) -> (B, D, D)
    A = np.einsum("bi,ijk->bjk", rot_vecs, X_xyz)

    # 5. Compute Matrix Exponential (Vectorized Scaling & Squaring)
    D = _expm_taylor_schematic(A, l)

    return D.reshape(prefix_shape + (2 * l + 1, 2 * l + 1))


def _expm_taylor_schematic(A: np.ndarray, l: int) -> np.ndarray:
    """
    Computes exp(A) for a batch of skew-symmetric matrices A using
    scaling and squaring with Taylor approximation.
    """
    B, dim, _ = A.shape

    # 1. Determine Scaling Factor
    # We want ||A / 2^k|| < theta_max
    # Max spectral radius for order l is approx l * pi.
    # We target a norm of ~0.25 for fast convergence of Taylor series.
    # k = log2(max_norm / 0.25)

    max_eigenval = l * np.pi
    target_norm = 0.25
    if max_eigenval > target_norm:
        k_scaling = int(np.ceil(np.log2(max_eigenval / target_norm)))
    else:
        k_scaling = 0

    scaling_factor = 2.0**k_scaling
    A_scaled = A / scaling_factor

    # 2. Taylor Expansion (Horner's Method)
    # Degree 12 is sufficient for double precision at norm 0.25
    # P(X) = I + X + X^2/2! + ...
    #      = I + X(I + X/2(I + X/3(...)))

    res = np.eye(dim, dtype=A.dtype).reshape(1, dim, dim).repeat(B, axis=0)

    # Coefficients 1/k!
    # We iterate backwards from degree d down to 1
    degree = 12

    for k in range(degree, 0, -1):
        # res = I + (A_scaled * res) / k
        # We can write: res = I + A_scaled @ (res / k)
        res = res / float(k)
        res = np.matmul(A_scaled, res)
        # Add Identity (broadcasted)
        res[:, np.arange(dim), np.arange(dim)] += 1.0

    # 3. Squaring
    # exp(A) = exp(A/s)^s
    for _ in range(k_scaling):
        res = np.matmul(res, res)

    return res


# ==========================================
# Caching & Generator Logic
# ==========================================


@functools.lru_cache(maxsize=None)
def _get_e3nn_generators(l: int) -> np.ndarray:
    """Reproduces e3nn_jax.generators(l) logic."""
    X_su2 = _su2_generators(l)
    Q = _change_basis_real_to_complex(l)
    Q_H = np.conjugate(Q.T)
    # Transform: X = real( Q.H @ X_su2 @ Q )
    X_transformed = Q_H @ X_su2 @ Q
    # e3nn asserts small imaginary part; we take real
    return np.real(X_transformed)


def _su2_generators(j: int) -> np.ndarray:
    """Reproduces e3nn_jax.su2_generators(j) logic."""
    dim = 2 * j + 1
    m = np.arange(-j, j)
    raising = np.diag(-np.sqrt(j * (j + 1) - m * (m + 1)), k=-1)

    m = np.arange(-j + 1, j + 1)
    lowering = np.diag(np.sqrt(j * (j + 1) - m * (m - 1)), k=1)

    m = np.arange(-j, j + 1)

    val_0 = 0.5 * (raising + lowering)
    val_1 = np.diag(1j * m)
    val_2 = -0.5j * (raising - lowering)

    return np.stack([val_0, val_1, val_2], axis=0)


def _change_basis_real_to_complex(l: int) -> np.ndarray:
    """Reproduces e3nn_jax.change_basis_real_to_complex(l) logic."""
    dim = 2 * l + 1
    q = np.zeros((dim, dim), dtype=np.complex128)

    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / np.sqrt(2)
        q[l + m, l - abs(m)] = -1j / np.sqrt(2)
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** m / np.sqrt(2)
        q[l + m, l - abs(m)] = 1j * (-1) ** m / np.sqrt(2)

    return (-1j) ** l * q


@functools.lru_cache(maxsize=1)
def _get_generator_mapping():
    """Maps e3nn generator indices to (x, y, z) axes."""
    gens_e3nn = _get_e3nn_generators(1)

    # Standard SO(3) generators
    L_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float64)
    L_x = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    L_y = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=np.float64)

    target_gens = [L_x, L_y, L_z]
    mapping = [-1, -1, -1]

    for i, target in enumerate(target_gens):
        for j in range(3):
            if np.max(np.abs(gens_e3nn[j] - target)) < 1e-5:
                mapping[i] = j
                break

    if -1 in mapping:
        raise RuntimeError(
            "Mapping failed: e3nn generators do not match standard SO(3) axes."
        )
    return mapping


# ==========================================
# Benchmark + Testing
# ==========================================
# Enable x64 for precision
import os

# comment this out to see the reason for this implementation
os.environ["JAX_ENABLE_X64"] = "True"


def run_benchmark():
    B = 10000
    ls = [
        1,
        2,
    ]

    print(f"=== Wigner-D Optimized Benchmark (Batch: {B}, float64) ===\n")
    print(f"{'L':<4} | {'NumPy (ms)':<12} | {'JAX (ms)':<12} | {'Speedup':<15}")
    print("-" * 55)

    rng = np.random.default_rng(42)
    R_np = Rotation.random(B, random_state=rng).as_matrix()

    # Trigger JIT and cache once
    for l in ls:
        wigner_D_real(l, R_np[:10])

    for l in ls:
        # NumPy Run
        t0 = time.perf_counter()
        y_np = wigner_D_real(l, R_np)
        t_np = (time.perf_counter() - t0) * 1000

        print(f"{l:<4} | {t_np:<12.2f}")


def check_compatibility():
    import e3nn_jax
    import jax.numpy as jnp

    print("--- Starting e3nn_jax Compatibility Check ---")

    # Test multiple l values
    for l in [1, 2, 3, 4]:
        dim = 2 * l + 1
        B = 100
        threshold = 1e-12  # f64 precision is high, but matrix exp adds some error

        # 1. Random Data
        rng = np.random.default_rng(42 + l)
        R_np = Rotation.random(B, random_state=rng).as_matrix()
        x_np = rng.normal(size=(B, dim)).astype(np.float64)

        # 2. NumPy Implementation
        # Start timer
        t0 = time.time()
        D_np = wigner_D_real(l, R_np)
        y_np = np.einsum("bij,bj->bi", D_np, x_np)
        dt = time.time() - t0

        # 3. JAX Implementation
        R_jax = jnp.array(R_np)
        x_jax = jnp.array(x_np)

        irreps = e3nn_jax.Irreps(f"{l}e")
        x_e3nn = e3nn_jax.IrrepsArray(irreps, x_jax)

        y_e3nn = e3nn_jax.vmap(e3nn_jax.IrrepsArray.transform_by_matrix)(x_e3nn, R_jax)
        y_jax_out = np.array(y_e3nn.array)

        # 4. Compare
        abs_diff = np.abs(y_np - y_jax_out)
        max_diff = np.max(abs_diff)

        print(f"L={l}: Max Discrepancy = {max_diff:.4e} (Time: {dt*1000:.2f}ms)")

        if max_diff < threshold:
            print(f"[SUCCESS] L={l} matches e3nn_jax.")
        else:
            print(f"[FAILURE] L={l} mismatch.")


if __name__ == "__main__":
    run_benchmark()
    check_compatibility()
