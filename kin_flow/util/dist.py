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

import numpy as np
from scipy.spatial.transform import Rotation


def SO3_uniform_R3_normal_np(num_samples):
    R = Rotation.random(num_samples).as_matrix()

    p = np.random.normal(size=(num_samples, 3))
    p = np.clip(p, a_min=-4.5, a_max=4.5)  # against rare edge cases

    T = np.eye(4)[None, ...].repeat(num_samples, axis=0)
    T[:, :3, :3] = R
    T[:, :3, 3] = p

    return T


if __name__ == "__main__":
    samples = SO3_uniform_R3_normal_np(2)
    print(samples)
