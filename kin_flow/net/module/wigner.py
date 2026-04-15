import jax.numpy as jnp

from kin_flow.net.module.w3j import _Jd_jax


def _z_rot_mat(angle: jnp.ndarray, l_irrep: int) -> jnp.ndarray:
    M = jnp.zeros(shape=(2 * l_irrep + 1, 2 * l_irrep + 1))

    inds = jnp.arange(0, 2 * l_irrep + 1)
    reversed_inds = jnp.arange(2 * l_irrep, -1, -1)
    frequencies = jnp.arange(l_irrep, -l_irrep - 1, -1)

    sin_term = jnp.sin(frequencies * angle)
    cos_term = jnp.cos(frequencies * angle)

    M = M.at[inds, reversed_inds].set(sin_term)
    M = M.at[inds, inds].set(cos_term)
    return M


def wigner_D(
    l_irrep: int, alpha: jnp.ndarray, beta: jnp.ndarray, gamma: jnp.ndarray, J
) -> jnp.ndarray:
    Xa = _z_rot_mat(alpha, l_irrep)
    Xb = _z_rot_mat(beta, l_irrep)
    Xc = _z_rot_mat(gamma, l_irrep)
    # Compute Xa @ J
    temp1 = jnp.matmul(Xa, J)
    # Compute temp1 @ Xb
    temp2 = jnp.matmul(temp1, Xb)
    # Compute temp2 @ J
    temp3 = jnp.matmul(temp2, J)
    # Compute temp3 @ Xc
    D = jnp.matmul(temp3, Xc)
    return D


def _angle_from_tan(axis: str, other_axis: str, data, horizontal: bool) -> jnp.ndarray:
    i1, i2 = 0, 2
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return jnp.atan2(data[..., i1], data[..., i2])
    return jnp.atan2(data[..., i2], -data[..., i1])


def matrix_to_euler_angles(matrix: jnp.ndarray) -> jnp.ndarray:
    central_angle = jnp.acos(jnp.clip(matrix[..., 1, 1], min=-1.0, max=1.0))

    o = (
        _angle_from_tan("Y", "X", matrix[..., 1], False),
        central_angle,
        _angle_from_tan("Y", "X", matrix[..., 1, :], True),
    )
    return jnp.stack(o, -1)


def matrix_to_wigner(matrix: jnp.ndarray, lmax: int) -> jnp.ndarray:
    alpha_beta_gamma = matrix_to_euler_angles(matrix).T
    alpha, beta, gamma = (
        alpha_beta_gamma[0],
        alpha_beta_gamma[1],
        alpha_beta_gamma[2],
    )

    # Calculate the size of the Wigner D-matrix (assuming start_lmax=0)
    size = (lmax + 1) ** 2

    # Initialize the output Wigner D-matrix
    wigner = jnp.zeros((size, size))

    start = 0
    for l_irrep in range(lmax + 1):
        J = _Jd_jax[l_irrep]  # Get the J matrix for this l value
        block = wigner_D(l_irrep, alpha, beta, gamma, J)
        dim_l = 2 * l_irrep + 1

        # Update the wigner matrix with this block
        end = start + dim_l
        wigner = wigner.at[start:end, start:end].set(block)
        start = end

    return wigner


def rotate_embedding(emb, rot_mat, lmax=2):
    wigner = matrix_to_wigner(rot_mat, lmax)
    out = jnp.einsum("ol,lc->oc", wigner, emb)
    return out
