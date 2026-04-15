# Copyright (c) 2025 Robert Bosch GmbH
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


def quat_wxyz_to_xyzw(quat: np.ndarray):
    res = np.zeros_like(quat)
    res[..., 0:3] = quat[..., 1:4]
    res[..., 3] = quat[..., 0]
    return res


def quat_xyzw_to_wxyz(quat: np.ndarray):
    res = np.zeros_like(quat)
    res[..., 0] = quat[..., 3]
    res[..., 1:4] = quat[..., 0:3]
    return res
