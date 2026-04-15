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

import numpy as np
import trimesh
from mgs.obj.base import CollisionMeshObject
from mgs.sampler.base import GraspGenerator
from mgs.util.geo.transforms import SE3Pose  # type hint only


class AntipodalGraspGenerator(GraspGenerator):
    def __init__(self, obj: CollisionMeshObject):
        super().__init__(obj)
        self._obj = obj
        self._mesh_orig: trimesh.Trimesh | None = None
        self._mesh_norm: trimesh.Trimesh | None = None
        self._centroid: np.ndarray | None = None
        self._scale_diag: float | None = None
        self._ray_eps: float = 1e-6

    # --- normalization helpers ---

    def _load_mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.load_mesh(self._obj.obj_file_path)
        if isinstance(mesh, trimesh.Scene):
            geoms = [
                g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)
            ]
            if not geoms:
                raise TypeError(
                    f"No Trimesh geometry found in scene: {self._obj.obj_file_path}"
                )
            mesh = geoms[0]
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Unsupported mesh type: {type(mesh)}")
        return mesh

    def _ensure_normalized(self):
        if self._mesh_norm is not None:
            return
        mesh = self._load_mesh()
        centroid = mesh.centroid
        scale_diag = (
            float(mesh.scale)
            if hasattr(mesh, "scale")
            else float(np.linalg.norm(mesh.extents))
        )
        if scale_diag == 0.0:
            raise ValueError("Degenerate mesh: zero extent")

        T = np.eye(4)
        T[:3, 3] = -centroid
        S = np.diag([1.0 / scale_diag, 1.0 / scale_diag, 1.0 / scale_diag, 1.0])
        N = S @ T

        mesh_norm = mesh.copy()
        mesh_norm.apply_transform(N)

        self._mesh_orig = mesh
        self._mesh_norm = mesh_norm
        self._centroid = centroid
        self._scale_diag = scale_diag

    def _denorm_points(self, pts_norm: np.ndarray) -> np.ndarray:
        return pts_norm * self._scale_diag + self._centroid

    def _denorm_poses(self, Hs_norm: np.ndarray) -> np.ndarray:
        Hs = Hs_norm.copy()
        Hs[..., :3, 3] = self._denorm_points(Hs[..., :3, 3])
        return Hs

    # --- core sampler ---

    def generate_grasps(
        self, num: int, eps: float = 1e-5
    ) -> Tuple[SE3Pose, Dict[str, Any]]:
        """
        Return up to `num` grasps from a single pass; if fewer are found,
        the caller should call again.
        """
        self._ensure_normalized()
        mesh = self._mesh_norm

        starts, fidx = trimesh.sample.sample_surface(mesh, num)
        if starts.size == 0:
            return np.zeros((0, 4, 4)), {"width": np.zeros((0,), dtype=float)}

        n_out = mesh.face_normals[fidx]
        n_out /= np.linalg.norm(n_out, axis=1, keepdims=True)
        ray_dirs = -n_out
        ray_origins = starts + ray_dirs * self._ray_eps  # nudge inward

        locs, idx_ray, _ = mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_dirs
        )

        if locs.size == 0:
            return np.zeros((0, 4, 4)), {"width": np.zeros((0,), dtype=float)}

        H_list, widths = [], []

        for i in range(len(ray_origins)):
            hits_i = locs[idx_ray == i]
            if hits_i.size == 0:
                continue
            origin = starts[i]
            d = np.linalg.norm(hits_i - origin, axis=1)
            good = d >= max(eps, self._ray_eps * 10)
            if not np.any(good):
                continue
            hits_i = hits_i[good]
            d = d[good]

            for hit, width in zip(hits_i, d):
                x = hit - origin
                nx = np.linalg.norm(x)
                if nx <= eps:
                    continue
                x = x / nx  # antipodal axis (deterministic)

                # build an orthonormal basis in the plane ⟂ x
                aux = (
                    np.array([1.0, 0.0, 0.0])
                    if abs(x[0]) < 0.9
                    else np.array([0.0, 1.0, 0.0])
                )
                z0 = np.cross(x, aux)
                nz = np.linalg.norm(z0)
                if nz <= 1e-12:
                    aux = np.array([0.0, 0.0, 1.0])
                    z0 = np.cross(x, aux)
                    nz = np.linalg.norm(z0)
                    if nz <= 1e-12:
                        continue
                z0 = z0 / nz
                y0 = np.cross(z0, x)

                thetas = np.random.uniform(0.0, 2.0 * np.pi, size=10)
                cos_t = np.cos(thetas)[:, None]
                sin_t = np.sin(thetas)[:, None]
                z_stack = cos_t * z0[None, :] + sin_t * y0[None, :]
                y_stack = np.cross(z_stack, x[None, :])

                center = (origin + hit) * 0.5

                for z, y in zip(z_stack, y_stack):
                    H = np.eye(4)
                    H[:3, 0] = x
                    H[:3, 1] = y
                    H[:3, 2] = z
                    H[:3, 3] = center
                    H_list.append(H)
                    widths.append(width)
                    if len(H_list) >= num:
                        Hs = self._denorm_poses(np.stack(H_list, axis=0))
                        w = np.asarray(widths, dtype=float) * self._scale_diag
                        return Hs, {"width": w}

        if not H_list:
            return np.zeros((0, 4, 4)), {"width": np.zeros((0,), dtype=float)}
        Hs = self._denorm_poses(np.stack(H_list, axis=0))
        w = np.asarray(widths, dtype=float) * self._scale_diag
        return Hs, {"width": w}
