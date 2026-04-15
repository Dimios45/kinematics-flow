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

import os
from functools import reduce

current_file = os.path.abspath(__file__)


def get_nth_parent(n):
    return reduce(lambda dir, _: os.path.dirname(dir), range(n), current_file)


PACKAGE_PATH = get_nth_parent(2)
GIT_PATH = get_nth_parent(3)
ASSET_PATH = os.path.join(GIT_PATH, "asset")

if __name__ == "__main__":
    print(PACKAGE_PATH)
    print(GIT_PATH)
    print(ASSET_PATH)
