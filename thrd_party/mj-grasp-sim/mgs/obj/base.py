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

from abc import abstractmethod

from mgs.core.mj_xml import MjXml


class CollisionMeshObject(MjXml):
    name: str
    object_id: str
    """
    Objects that have an assosiated collision mesh file (.obj).
    The corresponding .obj file will be used in grasp generation.
    Note, the collision file does not have to be convex and should not be
    partitioned.
    """

    @property
    @abstractmethod
    def obj_file_path(self) -> str: ...
