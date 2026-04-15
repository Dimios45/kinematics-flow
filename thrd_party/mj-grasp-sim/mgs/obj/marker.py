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

from typing import Any, Dict, Tuple

from mgs.core.mj_xml import MjXml
from mgs.util.geo.transforms import SE3Pose

XML = r"""
    <worldbody>
        <body name="{name}" pos="{position}" quat="{quaternion}">
            <freejoint name="joint:{name}"/>
            <inertial pos="0 0 0.17" quat="0.707107 0.707107 0 0" mass="0.3" diaginertia="0.09 0.07 0.05" />
            <site name="site:{name}" pos="0 0 0" size="0.01 0.01 0.01" rgba="0 1 0 1" type="sphere" group="1"/>
        </body>
    </worldbody>
"""


class Marker(MjXml):
    def __init__(self, pose: SE3Pose, name: str):
        pose_vec = pose.to_vec(layout="pq", type="wxyz")
        pos, quat = pose_vec[:3], pose_vec[3:]

        self.pos = pos
        self.quat = quat
        self.name = name

    def to_xml(self) -> Tuple[str, Dict[str, Any]]:
        pos = "{} {} {}".format(self.pos[0], self.pos[1], self.pos[2])
        quat = "{} {} {} {}".format(
            self.quat[0], self.quat[1], self.quat[2], self.quat[3]
        )
        xml_formatted = XML.format(
            **{
                "position": pos,
                "quaternion": quat,
                "name": self.name,
            }
        )
        return xml_formatted, {}
