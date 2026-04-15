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

import jax
import jax.numpy as jnp


def scatter_attn_weighted_sum_jax(
    features: jnp.ndarray,  # (E, …)
    weights: jnp.ndarray,  # (E,)
    index: jnp.ndarray,  # (E,)  – segment / destination node id
    valid_mask: jnp.ndarray,  # (E,)  – False ⇒ ignore this edge
    dim_size: int,  # number of segments / nodes
    eps: float = 1e-9,  # small value to keep denominators non-zero
) -> jnp.ndarray:
    masked_logits = jnp.where(valid_mask, weights, -jnp.inf)  # (E,)
    max_per_seg = jax.ops.segment_max(
        masked_logits, index, num_segments=dim_size
    )  # (dim_size,)
    recentered = jnp.where(
        valid_mask, masked_logits - max_per_seg[index], -jnp.inf
    )  # (E,)
    exp_logits = jnp.exp(recentered)
    sum_exp_seg = jax.ops.segment_sum(
        exp_logits, index, num_segments=dim_size
    )  # (dim_size,)
    attn = jnp.where(valid_mask, exp_logits / (sum_exp_seg[index] + eps), 0.0)  # (E,)

    # 6) weight features and aggregate
    weighted_feat = jnp.einsum("e,e...->e...", attn, features)  # (E, …)
    out = jax.ops.segment_sum(
        weighted_feat, index, num_segments=dim_size
    )  # (dim_size, …)

    return out


def scatter_sum_jax(
    features: jnp.ndarray,  # (E, …)
    _: jnp.ndarray,  # (E,)
    index: jnp.ndarray,  # (E,)  – segment / destination node id
    valid_mask: jnp.ndarray,  # (E,)  – False ⇒ ignore this edge
    dim_size: int,  # number of segments / nodes
) -> jnp.ndarray:
    # Mask out invalid features
    masked_features = jnp.where(
        valid_mask[..., None], features, 0.0  # broadcast mask to feature dimensions
    )  # (E, …)

    # Sum features by segment
    out = jax.ops.segment_sum(
        masked_features, index, num_segments=dim_size
    )  # (dim_size, …)

    return out
