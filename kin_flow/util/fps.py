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
from jax.lax import fori_loop


def farthest_point_sampling(x, num_samples, key=None):
    x = jax.lax.stop_gradient(x)
    farthest_points_idx = jnp.zeros(num_samples, dtype=jnp.int32)
    if key is not None:
        first_idx = jax.random.choice(key, jnp.arange(len(x)))
    else:
        first_idx = 0
    farthest_points_idx = farthest_points_idx.at[0].set(first_idx)

    distances = jnp.full(x.shape[0], jnp.inf)

    def sampling_fn(i, val):
        farthest_points_idx, distances = val
        # Get the latest point added to the farthest points
        latest_point_idx = farthest_points_idx[i - 1]
        latest_point = x[latest_point_idx]

        # Compute the squared distances from the latest point to all other points
        new_dr = x - latest_point
        new_distances = jnp.sum(new_dr**2, axis=-1)

        # Update the distances to maintain the minimum distance to the farthest points set
        distances = jnp.minimum(distances, new_distances)

        # Select the point that is farthest to all previous selections
        farthest_point_idx = jnp.argmax(distances)
        farthest_points_idx = farthest_points_idx.at[i].set(farthest_point_idx)

        return farthest_points_idx, distances

    # Iterate over the number of samples to select
    farthest_points_idx, _ = fori_loop(
        1, num_samples, sampling_fn, (farthest_points_idx, distances)
    )

    return farthest_points_idx


import numpy as np


def farthest_point_sampling_np(x, num_samples, seed=None):
    """
    Performs Farthest Point Sampling on a set of points.

    Args:
        x (np.ndarray): A numpy array of shape (N, D) representing N points in D dimensions.
        num_samples (int): The number of points to sample.
        seed (int, optional): A random seed for reproducibility. Defaults to None.

    Returns:
        np.ndarray: An array of indices of the sampled points.
    """
    rng = np.random.default_rng(seed)
    num_points = x.shape[0]
    farthest_points_idx = np.zeros(num_samples, dtype=np.int32)

    if seed is not None:
        first_idx = rng.choice(np.arange(num_points))
    else:
        first_idx = 0
    farthest_points_idx[0] = first_idx

    distances = np.full(num_points, np.inf)

    for i in range(1, num_samples):
        # Get the latest point added to the farthest points
        latest_point_idx = farthest_points_idx[i - 1]
        latest_point = x[latest_point_idx]

        # Compute the squared distances from the latest point to all other points
        new_dr = x - latest_point
        new_distances = np.sum(new_dr**2, axis=-1)

        # Update the distances to maintain the minimum distance to the farthest points set
        distances = np.minimum(distances, new_distances)

        # Select the point that is farthest to all previous selections
        farthest_point_idx = np.argmax(distances)
        farthest_points_idx[i] = farthest_point_idx

    return farthest_points_idx
