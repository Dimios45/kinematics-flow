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
import xml.etree.ElementTree as Et
from typing import Any, Dict, Tuple

import yaml
from mgs.obj.base import CollisionMeshObject
from mgs.util.const import ASSET_PATH
from mgs.util.geo.transforms import SE3Pose


class ObjectGSO(CollisionMeshObject):
    def __init__(self, pose: SE3Pose, object_id, name=None):
        pose_vec = pose.to_vec(layout="pq", type="wxyz")
        pos, quat = pose_vec[:3], pose_vec[3:]

        self.pos = pos
        self.quat = quat
        self.name = object_id if name is None else name
        self.object_id = object_id
        self.file_name = "{}.xml".format(object_id)

    @property
    def gso_directory(self):
        return os.path.join(ASSET_PATH, "mj-objects/GoogleScannedObjects")

    @property
    def asset_dir(self):
        return os.path.join(self.gso_directory, self.object_id)

    @property
    def obj_file_path(self):
        return os.path.join(self.asset_dir, "model.obj")

    @classmethod
    def all_object_ids(cls):
        all_object_ids = []
        gso_directory = os.path.join(ASSET_PATH, "mj-objects/GoogleScannedObjects")
        gso_obj_directories = os.listdir(gso_directory)
        for x in gso_obj_directories:
            all_object_ids.append(x)
        return all_object_ids

    def to_xml(self) -> Tuple[str, Dict[str, Any]]:
        et_include = Et.Element("include")
        et_include.set("file", self.file_name)

        xml_string = self.generate_xml()
        return (
            '<include file="{}_{}" />'.format(self.name, self.file_name),
            {"{}_{}".format(self.name, self.file_name): xml_string},
        )

    def generate_xml(self) -> bytes:
        info_file = os.path.join(self.asset_dir, "info.yml")
        assert os.path.isfile(
            info_file
        ), f"The file {info_file} was not found. Did you specify the path to the object folder correctly?"

        with open(info_file, "r") as f:
            info_dict = yaml.safe_load(f)

        original_file = info_dict["original_file"]
        submesh_files = info_dict["submesh_files"]
        submesh_props = info_dict["submesh_props"]
        weight = info_dict["weight"]
        material_map = info_dict["material_map"]

        root = Et.Element("mujoco", attrib={"model": self.name})

        # Assets and Worldbody
        assets = Et.SubElement(root, "asset")
        worldbody = Et.SubElement(root, "worldbody")
        body_attributes = {
            "name": f"{self.name}",
            "pos": " ".join(map(str, self.pos)),
            "quat": " ".join(map(str, self.quat)),
        }
        body = Et.SubElement(worldbody, "body", attrib=body_attributes)

        ## Texture and Material
        texture_attributes = {
            "type": "2d",
            "name": f"{self.name}_tex",
            "file": os.path.join(self.asset_dir, material_map),
        }
        Et.SubElement(assets, "texture", attrib=texture_attributes)

        material_attributes = {
            "name": f"{self.name}_mat",
            "texture": texture_attributes["name"],
            "specular": "0.5",
            "shininess": "0.5",
        }
        Et.SubElement(assets, "material", attrib=material_attributes)

        # Meshes
        orig_mesh_attributes = {
            "name": f"{self.name}_orig",
            "file": os.path.join(self.asset_dir, original_file),
        }
        Et.SubElement(assets, "mesh", attrib=orig_mesh_attributes)

        orig_geom_attributes = {
            "material": material_attributes["name"],
            "mesh": orig_mesh_attributes["name"],
            "group": "2",
            "type": "mesh",
            "contype": "0",
            "conaffinity": "0",
        }
        Et.SubElement(body, "geom", attrib=orig_geom_attributes)

        for i, (submesh_file, submesh_prop) in enumerate(
            zip(submesh_files, submesh_props)
        ):
            collision_mesh_attributes = {
                "name": f"{self.name}_coll_{i}",
                "file": os.path.join(self.asset_dir, submesh_file),
            }
            Et.SubElement(assets, "mesh", attrib=collision_mesh_attributes)
            collision_geom_attributes = {
                "mesh": collision_mesh_attributes["name"],
                "mass": str(weight * submesh_prop),
                "group": "3",
                "type": "mesh",
                "conaffinity": "1",
                "contype": "1",
                "condim": "4",
                "rgba": "1 1 1 1",
                "friction": "1.0 0.3 0.1",
                "solimp": "0.998 0.998 0.001",
                "solref": "0.001 1",
            }
            Et.SubElement(body, "geom", attrib=collision_geom_attributes)

        joint_attributes = {
            "damping": "0.0001",
            "name": f"{self.name}:joint",
            "type": "free",
        }
        Et.SubElement(body, "joint", attrib=joint_attributes)

        return Et.tostring(root)
