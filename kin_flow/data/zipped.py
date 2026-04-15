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

import kin_flow.util.const as CONST


class ZippedGripperLoader:
    def __init__(self, loaders, batch_size=None, all_samples=False, *, rngs):
        """
        Initialize the zipped gripper loader with arbitrary loaders.

        Args:
            loaders: List of loader objects that implement the iterator protocol
        """
        self.loaders = loaders
        self.all_samples = all_samples
        if not self.all_samples:
            self.batch_size = batch_size
        self.rngs = rngs

    def __iter__(self):
        # Get iterators for each loader
        iterators = [iter(loader) for loader in self.loaders]

        # Iterate as long as all loaders have data
        for data_batch in zip(*iterators):
            # Unpack data from each loader
            # Each returns (scene, grasp, gripper, dir)
            scenes = []
            rotations = []
            positions = []
            joints = []
            dirs = []
            pcd_points = []
            pcd_colors = []
            segmentations = []

            for data in data_batch:
                scene, grasp, dir_path = data

                # Extract scene data
                scenes.append(scene)

                r, p, j = (
                    np.asarray(grasp[0]),
                    np.asarray(grasp[1]),
                    np.asarray(grasp[2]),
                )
                assert (
                    r.shape[0] == p.shape[0] == j.shape[0]
                ), "Grasp data shapes do not match."
                if self.all_samples:
                    random_idx = np.arange(r.shape[0])
                else:
                    random_idx = np.random.randint(
                        low=0,
                        high=r.shape[0],
                        size=(self.batch_size,),
                        dtype=np.int32,
                    )

                # Extract grasp data
                rotations.append(r[random_idx])
                positions.append(p[random_idx])
                joints.append(j[random_idx])
                dirs.append(dir_path)

            # Stack scene data
            scene_points = np.stack([scene[0] for scene in scenes], axis=0)
            scene_colors = np.stack([scene[1] for scene in scenes], axis=0)

            # Create gripper type index for each dataset
            gripper_indices = [
                np.full(len(rot), i, dtype=np.int32) for i, rot in enumerate(rotations)
            ]

            yield {
                # Stack data from all grippers
                "path": dirs,
                "scene_points": scene_points,
                "scene_colors": scene_colors,
                "rotations": rotations,
                "positions": positions,
                "joints": joints,
                "gripper_indices": gripper_indices,
            }
