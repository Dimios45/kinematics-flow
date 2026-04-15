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

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx


class PoolingConnect(nnx.Module):
    def __call__(
        self,
        src: jnp.ndarray,  # [Ns, 3]
        dst: jnp.ndarray,  # [Nd, 3]
        valid_mask_src: jnp.ndarray,  # [Ns] bool
        valid_mask_dst: jnp.ndarray,  # [Nd] bool
    ):
        Ns = src.shape[0]
        any_dst = jnp.any(valid_mask_dst)  # scalar bool (jnp.bool_)

        def connect_one(p, v_src):
            """Return index of nearest valid dst to p; if src invalid or no valid dst, return 0."""

            def pick(_):
                diff = dst - p  # safe because we only enter if v_src is True
                dist2 = jnp.sum(diff * diff, axis=-1)  # [Nd]
                dist2 = jnp.where(valid_mask_dst, dist2, jnp.inf)
                return jnp.argmin(dist2).astype(jnp.int32)

            return jax.lax.cond(
                v_src & any_dst, pick, lambda _: jnp.int32(0), operand=None
            )

        dst_edges = jax.vmap(connect_one, in_axes=(0, 0))(src, valid_mask_src)  # [Ns]
        src_edges = jnp.arange(Ns, dtype=jnp.int32)  # [Ns]
        edge_mask = (valid_mask_src & any_dst).astype(jnp.bool_)  # [Ns]
        dst_edges = jnp.where(edge_mask, dst_edges, jnp.int32(0))  # [Ns]

        return src_edges, dst_edges, edge_mask


@partial(jax.jit, static_argnames=("k", "eps", "use_cap", "disallow_self_loops"))
def knn_graph(
    x: jnp.ndarray,  # [N_src, 3]
    y: jnp.ndarray,  # [N_dst, 3]
    k: int,
    eps: float = 1e-6,
    r_cap: float | None = None,  # edge-length cap (meters); None == no cap
    use_cap: bool = False,  # static: r_cap is set or not
    disallow_self_loops: bool = True,
    valid_mask_x: jnp.ndarray | None = None,  # [N_src] bool
    valid_mask_y: jnp.ndarray | None = None,  # [N_dst] bool
):
    """
    Returns
    -------
    edge_src : int32 [N_dst * k]   (indices in x, -1 for padded)
    edge_dst : int32 [N_dst * k]   (indices in y, -1 for padded)
    valid    : bool  [N_dst * k]   (True where the edge is real)
    """
    N_src = x.shape[0]
    N_dst = y.shape[0]

    if valid_mask_x is None:
        valid_mask_x = jnp.isfinite(x).all(axis=-1)
    if valid_mask_y is None:
        valid_mask_y = jnp.isfinite(y).all(axis=-1)

    # Sanitize coordinates so ops on invalid rows are harmless.
    x_sanit = jnp.where(valid_mask_x[:, None], x, 0.0)
    y_sanit = jnp.where(valid_mask_y[:, None], y, 0.0)

    r_cap_sq = (r_cap * r_cap) if use_cap else 0.0

    def query_one(dst_idx, y_pt, y_valid):
        """Return k nearest neighbors in x for this y[dst_idx]."""

        def _do_valid_dst(_):
            # squared distances to all sources
            diff = x_sanit - y_pt  # [N_src, 3]
            d2 = jnp.sum(diff * diff, axis=-1)  # [N_src]

            # candidate mask: valid sources, optional self-loop removal, optional radius cap
            cand = valid_mask_x
            if disallow_self_loops:
                # only correct when x and y index spaces align (e.g., within-level x==y)
                same_len = N_src == N_dst
                self_mask = jnp.where(
                    same_len, (jnp.arange(N_src, dtype=jnp.int32) == dst_idx), False
                )
                cand = cand & ~self_mask
            if use_cap:
                cand = cand & (d2 < r_cap_sq)

            # exclude tiny distances to avoid numerical self-duplicates
            cand = cand & (d2 > eps)

            # set non-candidates to +inf so they never appear in top-k
            d2_masked = jnp.where(cand, d2, jnp.inf)

            # pick k smallest; pad with -1 if fewer than k candidates
            topk_idx = jnp.argsort(d2_masked)[:k]  # [k]
            valid_k = (d2_masked[topk_idx] < jnp.inf) & y_valid
            esrc = jnp.where(valid_k, topk_idx, -1).astype(jnp.int32)
            edst = jnp.full((k,), dst_idx, dtype=jnp.int32)
            return esrc, edst, valid_k

        def _do_invalid_dst(_):
            return (
                jnp.full((k,), -1, jnp.int32),
                jnp.full((k,), -1, jnp.int32),
                jnp.zeros((k,), jnp.bool_),
            )

        return jax.lax.cond(y_valid, _do_valid_dst, _do_invalid_dst, operand=None)

    # Scan all destinations; keep [N_dst, k] then flatten
    def scan_body(carry, inputs):
        dst_idx, y_pt, y_valid = inputs
        esrc_row, edst_row, v_row = query_one(dst_idx, y_pt, y_valid)
        esrc, edst, vmask = carry
        esrc = esrc.at[dst_idx].set(esrc_row)
        edst = edst.at[dst_idx].set(edst_row)
        vmask = vmask.at[dst_idx].set(v_row)
        return (esrc, edst, vmask), None

    init_src = jnp.full((N_dst, k), -1, jnp.int32)
    init_dst = jnp.full_like(init_src, -1)
    init_msk = jnp.zeros_like(init_src, jnp.bool_)

    (edge_src_2d, edge_dst_2d, valid_2d), _ = jax.lax.scan(
        scan_body,
        (init_src, init_dst, init_msk),
        (jnp.arange(N_dst, dtype=jnp.int32), y_sanit, valid_mask_y),
    )

    return edge_src_2d.reshape(-1), edge_dst_2d.reshape(-1), valid_2d.reshape(-1)


class KNNConnect(nnx.Module):
    def __init__(
        self,
        k: int,
        eps: float = 1e-6,
        r_cap: float | None = None,
        disallow_self_loops: bool = True,
    ):
        self.k = int(k)
        self.eps = float(eps)
        self.r_cap = None if r_cap is None else float(r_cap)
        self.disallow_self_loops = bool(disallow_self_loops)

    def __call__(
        self,
        src: jnp.ndarray,  # [N_src, 3]
        dst: jnp.ndarray,  # [N_dst, 3]
        valid_mask_src: jnp.ndarray | None = None,  # [N_src] bool
        valid_mask_dst: jnp.ndarray | None = None,  # [N_dst] bool
    ):
        edge_src, edge_dst, valid = knn_graph(
            x=jax.lax.stop_gradient(src),
            y=jax.lax.stop_gradient(dst),
            k=self.k,
            eps=self.eps,
            r_cap=(0.0 if self.r_cap is None else self.r_cap),
            use_cap=(self.r_cap is not None),
            disallow_self_loops=self.disallow_self_loops,
            valid_mask_x=valid_mask_src,
            valid_mask_y=valid_mask_dst,
        )
        return edge_src, edge_dst, valid
