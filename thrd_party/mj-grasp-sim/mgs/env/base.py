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
from typing import Any, Protocol, Self

import cv2
import mujoco
import numpy as np
from mgs.core.simualtion import MjSimulation
from scipy.spatial.transform import Rotation


class MjScanEnv(MjSimulation):
    renderer: mujoco.Renderer
    fx: float
    fy: float
    cx: float
    cy: float

    def __init__(self, camera_name: str, image_width: int, image_height):
        self.renderer = mujoco.Renderer(self.model, width=480, height=480)
        self.width = image_width
        self.height = image_height
        cam_id = mujoco.mj_name2id(  # type: ignore
            m=self.model,
            name=camera_name,
            type=mujoco.mjtObj.mjOBJ_CAMERA,  # type: ignore
        )
        self.fovy = self.model.cam_fovy[cam_id]
        self.fovx = (
            2
            * np.arctan(
                self.width
                * 0.5
                / (self.height * 0.5 / np.tan(self.fovy * np.pi / 360 / 2))
            )
            / np.pi
            * 360
        )

        self.fx = (self.width / 2) / (np.tan(self.fovx * np.pi / 180 / 2))
        self.fy = (self.height / 2) / (np.tan(self.fovy * np.pi / 180 / 2))
        self.cx = self.width / 2
        self.cy = self.height / 2

    @abstractmethod
    def update_camera_settings(self, num_images, i):
        pass

    def get_camera_intrinsics(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def get_camera_extrinsics(self):
        pos = np.copy(self.data.cam_xpos)
        rot_mat = np.reshape(np.copy(self.data.cam_xmat), (3, 3))

        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rot_mat
        extrinsics[:3, 3] = pos
        return extrinsics

    def scan(self, num_images=1):
        imgs = []
        depths = []
        segmentations = []
        extrinsics = []
        for i in range(num_images):
            self.update_camera_settings(num_images, i)
            extrinsics.append(self.get_camera_extrinsics())

            img = self.renderer.render()
            img = np.copy(img)
            img = np.expand_dims(img, axis=0)
            imgs.append(img)

            self.renderer.enable_depth_rendering()
            depth = self.renderer.render()
            self.renderer.disable_depth_rendering()

            depth = np.copy(depth)
            depth = np.expand_dims(depth, axis=0)
            depths.append(depth)

            # segmentation
            self.renderer.enable_segmentation_rendering()
            segmentation = self.renderer.render()
            self.renderer.disable_segmentation_rendering()

            segmentation = np.copy(segmentation[..., 0])
            segmentation = np.expand_dims(segmentation, axis=0)
            segmentations.append(segmentation)

        imgs = np.concatenate(imgs, axis=0)
        depth = np.expand_dims(np.concatenate(depths, axis=0), axis=-1)
        rgbd = np.concatenate([imgs, depth], axis=-1)
        segmentation = np.concatenate(segmentations, axis=0)
        extrinsics = np.stack(extrinsics, axis=0)

        mj_transform = np.eye(4)
        mj_transform[:3, :3] = Rotation.from_quat(np.array([1.0, 0, 0, 0])).as_matrix()
        extrinsics = np.einsum("nij,jk->nik", extrinsics, mj_transform)

        # masks
        zero_one_img = (~(segmentation == -1)).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        image_masks = np.stack(
            [
                cv2.erode(zero_one_img[k], kernel, iterations=5)
                for k in range(zero_one_img.shape[0])
            ],
            axis=0,
        ).astype(bool)

        rgbd[..., :-1] = rgbd[..., :-1] / 255.0

        return rgbd, extrinsics, image_masks, segmentation


class Loadable(Protocol):
    @abstractmethod
    def to_dict(self) -> Any:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, state_dict: Any) -> Self:
        pass
