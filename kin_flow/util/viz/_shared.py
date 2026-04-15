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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import trimesh

# ---------------------------------------------------------------------------
# Asset root  (assets/ lives next to this file)
# ---------------------------------------------------------------------------

_ASSET_ROOT = Path(__file__).resolve().parent / "assets"

# ---------------------------------------------------------------------------
# Geometry kind constants
# ---------------------------------------------------------------------------

GEOM_MESH: int = 0
GEOM_PRIM: int = 1

# ---------------------------------------------------------------------------
# Limb / collision-def dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LimbSpec:
    """Describes one collision element attached to a joint."""

    joint_idx: int  # 0 = base, 1..num_dofs = DOF joints
    geom_kind: int  # GEOM_MESH or GEOM_PRIM
    # mesh
    mesh_name: Optional[str] = None
    # primitive
    prim_type: Optional[str] = None  # "sphere" | "box" | "capsule" | "cylinder"
    prim_params: Optional[Tuple[float, ...]] = None
    # local offset in joint frame
    pos_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class GripperCollisionDef:
    """Full collision model for one gripper."""

    limbs: List[_LimbSpec]
    mesh_files: Dict[str, str]  # mesh_name -> filename
    mesh_scale: float = 1.0
    asset_subdir: str = ""
    # For meshes that need a refquat baked into vertices at load time
    mesh_refquats: Optional[Dict[str, Tuple[float, float, float, float]]] = None


# ===================================================================
# Quaternion / transform helpers
# ===================================================================


def _quat_wxyz_to_R33(q_wxyz) -> np.ndarray:
    """Convert (w, x, y, z) quaternion to 3x3 rotation matrix.

    Handles unnormalized quaternions via ``s = 2/n`` (MuJoCo convention).
    """
    w, x, y, z = (float(v) for v in q_wxyz)
    n = w * w + x * x + y * y + z * z
    if n <= 1e-20:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def _R33_to_quat_wxyz(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to (w, x, y, z) quaternion."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (float(w), float(x), float(y), float(z))


def _T44_from_pos_quat(
    pos_xyz=(0.0, 0.0, 0.0),
    quat_wxyz=(1.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """Build a 4x4 homogeneous transform from position + quaternion."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = _quat_wxyz_to_R33(quat_wxyz)
    T[:3, 3] = np.asarray(pos_xyz, dtype=np.float32)
    return T


def _euler_xyz_to_R33(rx: float, ry: float, rz: float) -> np.ndarray:
    """Extrinsic XYZ Euler angles to 3x3 rotation matrix (Rz @ Ry @ Rx)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def _T44_from_pos_euler(
    pos_xyz=(0.0, 0.0, 0.0),
    euler_xyz=(0.0, 0.0, 0.0),
) -> np.ndarray:
    """Build a 4x4 from position + extrinsic XYZ Euler angles."""
    rx, ry, rz = (float(v) for v in euler_xyz)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = _euler_xyz_to_R33(rx, ry, rz)
    T[:3, 3] = np.asarray(pos_xyz, dtype=np.float32)
    return T


def _rot_axis_angle(axis, angle: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from an axis vector and angle (radians).

    Uses Rodrigues' formula.  ``axis`` need not be unit-length — it will be
    normalised internally.
    """
    a = np.asarray(axis, dtype=np.float64).ravel()
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    a = a / n
    c = np.cos(angle)
    s = np.sin(angle)
    K = np.array(
        [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]],
        dtype=np.float64,
    )
    R = np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)
    return R.astype(np.float32)


def _insert_revolute(T_parent: np.ndarray, axis, angle: float) -> np.ndarray:
    """Return ``T_parent`` with an additional revolute rotation inserted.

    The joint rotation is applied **in the parent body frame** (right-multiply
    of the 3x3 block), which is the MuJoCo convention.
    """
    T = T_parent.copy().astype(np.float64)
    R_joint = _rot_axis_angle(axis, angle).astype(np.float64)
    T[:3, :3] = T[:3, :3] @ R_joint
    return T.astype(np.float32)


# ===================================================================
# Primitive mesh generation via trimesh
# ===================================================================

_PRIM_MESH_CACHE: Dict[Tuple[str, Tuple[float, ...]], trimesh.Trimesh] = {}


def _make_prim_base_mesh(ptype: str, params: Tuple[float, ...]) -> trimesh.Trimesh:
    key = (ptype, params)
    if key in _PRIM_MESH_CACHE:
        return _PRIM_MESH_CACHE[key]

    if ptype == "box":
        hx, hy, hz = params
        m = trimesh.creation.box(extents=(2 * hx, 2 * hy, 2 * hz))
    elif ptype == "sphere":
        (r,) = params
        m = trimesh.creation.icosphere(radius=r, subdivisions=2)
    elif ptype == "capsule":
        r, half_len_z = params
        m = trimesh.creation.capsule(radius=r, height=2 * half_len_z, count=[12, 12])
    elif ptype == "cylinder":
        r, half_len_z = params
        m = trimesh.creation.cylinder(radius=r, height=2 * half_len_z, sections=24)
    else:
        raise ValueError(f"Unknown primitive type: {ptype}")

    _PRIM_MESH_CACHE[key] = m
    return m


# ===================================================================
# Asset mesh loading (with refquat + scale)
# ===================================================================

_ASSET_MESH_CACHE: Dict[
    Tuple[str, str, float, Tuple[float, float, float, float]], trimesh.Trimesh
] = {}


def _load_collision_mesh(
    asset_subdir: str,
    filename: str,
    scale: float,
    refquat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> trimesh.Trimesh:
    key = (asset_subdir, filename, scale, refquat_wxyz)
    if key in _ASSET_MESH_CACHE:
        return _ASSET_MESH_CACHE[key]

    path = _ASSET_ROOT / asset_subdir / filename
    m = trimesh.load(str(path), force="mesh")
    if isinstance(m, trimesh.Scene):
        geoms = list(m.geometry.values())
        m = trimesh.util.concatenate(geoms) if geoms else trimesh.Trimesh()
    if not isinstance(m, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(m)} for {path}")

    # Apply MJCF refquat by rotating vertices in the payload frame
    T_ref = _T44_from_pos_quat((0.0, 0.0, 0.0), refquat_wxyz)
    if not np.allclose(T_ref, np.eye(4, dtype=np.float32), atol=1e-7):
        m = m.copy()
        m.apply_transform(T_ref)

    if abs(scale - 1.0) > 1e-12:
        m = m.copy()
        m.apply_scale(scale)

    _ASSET_MESH_CACHE[key] = m
    return m


# ===================================================================
# Build merged mesh for a single grasp
# ===================================================================


def _build_merged_gripper_mesh(
    T_world_joints: np.ndarray,
    coll_def: GripperCollisionDef,
    color_hex: str = "#aaaaaa",
    opacity: float = 0.35,
) -> go.Mesh3d:
    """Merge all collision limbs into a single Plotly ``Mesh3d`` trace."""
    all_V: List[np.ndarray] = []
    all_F: List[np.ndarray] = []
    v_offset = 0

    for limb in coll_def.limbs:
        T_joint = T_world_joints[limb.joint_idx]
        T_local = _T44_from_pos_quat(limb.pos_xyz, limb.quat_wxyz)
        T_limb = T_joint @ T_local

        if limb.geom_kind == GEOM_MESH:
            assert limb.mesh_name is not None
            refquat = (1.0, 0.0, 0.0, 0.0)
            if coll_def.mesh_refquats and limb.mesh_name in coll_def.mesh_refquats:
                refquat = coll_def.mesh_refquats[limb.mesh_name]
            m = _load_collision_mesh(
                coll_def.asset_subdir,
                coll_def.mesh_files[limb.mesh_name],
                coll_def.mesh_scale,
                refquat,
            )
        else:
            assert limb.prim_type is not None and limb.prim_params is not None
            m = _make_prim_base_mesh(limb.prim_type, limb.prim_params)

        V0 = np.asarray(m.vertices, dtype=np.float32)
        F0 = np.asarray(m.faces, dtype=np.int32)

        V_world = (T_limb[:3, :3] @ V0.T).T + T_limb[:3, 3]

        all_V.append(V_world)
        all_F.append(F0 + v_offset)
        v_offset += V0.shape[0]

    if not all_V:
        V = np.zeros((0, 3), dtype=np.float32)
        F = np.zeros((0, 3), dtype=np.int32)
    else:
        V = np.concatenate(all_V, axis=0)
        F = np.concatenate(all_F, axis=0)

    return go.Mesh3d(
        x=V[:, 0],
        y=V[:, 1],
        z=V[:, 2],
        i=F[:, 0],
        j=F[:, 1],
        k=F[:, 2],
        color=color_hex,
        opacity=opacity,
        flatshading=True,
        showlegend=False,
        lighting=dict(ambient=0.7, diffuse=0.4, specular=0.05),
        name="gripper",
    )


# ===================================================================
# Viz helpers
# ===================================================================


def _anchor_cube_corners(
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
    pad: float = 1.10,
) -> np.ndarray:
    """Return two corner points that force equal-aspect scaling in Plotly."""
    min_xyz = np.asarray(min_xyz, dtype=np.float32)
    max_xyz = np.asarray(max_xyz, dtype=np.float32)
    center = 0.5 * (min_xyz + max_xyz)
    max_dim = float(np.max(max_xyz - min_xyz))
    half = 0.5 * max_dim * pad
    return np.stack([center - half, center + half], axis=0)


def _colors_to_plotly_rgb(c: np.ndarray) -> list:
    """Convert (N, 3) float colors in [0, 1] to Plotly ``rgb(...)`` strings."""
    c = np.asarray(c, dtype=float)
    c = np.clip(c, 0.0, 1.0)
    c255 = (c * 255.0 + 0.5).astype(np.uint8)
    return [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in c255]
