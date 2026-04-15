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

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from kin_flow.util.viz import _allegro as _allegro_mod
from kin_flow.util.viz import _dexee as _dexee_mod
from kin_flow.util.viz import _panda as _panda_mod
from kin_flow.util.viz import _shadow as _shadow_mod
from kin_flow.util.viz import _vx300 as _vx300_mod
from kin_flow.util.viz._shared import (GripperCollisionDef,
                                       _anchor_cube_corners,
                                       _build_merged_gripper_mesh,
                                       _colors_to_plotly_rgb)

# ---------------------------------------------------------------------------
# Per-gripper module imports (lazy-ish — the modules are small)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Registry: gripper_key -> (collision_def_fn, fk_fn)
# ---------------------------------------------------------------------------

_CollisionDefFn = Callable[[], GripperCollisionDef]
_FKFn = Callable[[np.ndarray, np.ndarray, object], np.ndarray]

_REGISTRY: Dict[str, Tuple[_CollisionDefFn, _FKFn]] = {
    "panda": (_panda_mod.collision_def, _panda_mod.joint_world_transforms),
    "shadow": (_shadow_mod.collision_def, _shadow_mod.joint_world_transforms),
    "allegro": (_allegro_mod.collision_def, _allegro_mod.joint_world_transforms),
    "dexee": (_dexee_mod.collision_def, _dexee_mod.joint_world_transforms),
    "vx300": (_vx300_mod.collision_def, _vx300_mod.joint_world_transforms),
}

# Cached collision defs (built on first use)
_COLLISION_DEFS: Dict[str, GripperCollisionDef] = {}


def _get_collision_def(gripper_key: str) -> GripperCollisionDef:
    if gripper_key not in _COLLISION_DEFS:
        builder, _ = _REGISTRY[gripper_key]
        _COLLISION_DEFS[gripper_key] = builder()
    return _COLLISION_DEFS[gripper_key]


def _get_fk_fn(gripper_key: str) -> _FKFn:
    _, fk_fn = _REGISTRY[gripper_key]
    return fk_fn


# ---------------------------------------------------------------------------
# Color constant
# ---------------------------------------------------------------------------

_GRIPPER_COLOR = "#5a8fba"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def viz_grasps(
    gripper_name: str,
    scene_pcd: np.ndarray,
    se3: np.ndarray,
    dof: np.ndarray,
    kin,
    *,
    scene_colors: Optional[np.ndarray] = None,
    max_scene_points: int = 8000,
    gripper_opacity: float = 0.35,
) -> go.Figure:
    """
    Visualize generated grasps using collision model meshes with a sample slider.

    Args:
        gripper_name: e.g. "panda", "shadow", "allegro", "dexee", "vx300".
        scene_pcd: (N, 3) scene point cloud.
        se3: (B, 4, 4) grasp SE3 transforms.
        dof: (B, num_dofs) joint values.
        kin: kinematics model (NP version).  Passed through to the FK function
             but **not used** for the actual FK computation — all grippers use
             their own MJCF body-chain FK.
        scene_colors: optional (N, 3) colors in [0,1].
        max_scene_points: subsample scene to this many points.
        gripper_opacity: opacity for gripper meshes.

    Returns:
        A plotly Figure with a slider to select the grasp sample.
    """
    B = se3.shape[0]
    se3 = np.asarray(se3, dtype=np.float64)
    dof = np.asarray(dof, dtype=np.float64)

    gripper_key = gripper_name.lower().split("_")[0]
    if gripper_key not in _REGISTRY:
        raise ValueError(
            f"Gripper '{gripper_name}' is not supported for collision "
            f"visualization. Supported: {list(_REGISTRY.keys())}."
        )

    coll_def = _get_collision_def(gripper_key)
    fk_fn = _get_fk_fn(gripper_key)

    # --- Subsample scene ---
    scene_pcd = np.asarray(scene_pcd, dtype=np.float32)
    if scene_colors is not None:
        scene_colors = np.asarray(scene_colors, dtype=np.float32)

    if scene_pcd.shape[0] > max_scene_points:
        idx = np.random.choice(scene_pcd.shape[0], max_scene_points, replace=False)
        scene_pcd = scene_pcd[idx]
        if scene_colors is not None:
            scene_colors = scene_colors[idx]

    # Color the ground plane grey
    if scene_colors is not None:
        ground_mask = np.abs(scene_pcd[:, 2]) < 0.001
        scene_colors[ground_mask] = 0.75
        scene_rgb = _colors_to_plotly_rgb(scene_colors)
    else:
        scene_rgb = "rgb(160,160,160)"

    # --- Compute FK for all samples ---
    print(f"viz_grasps: Computing FK for {B} grasps ({gripper_name})...")
    all_joint_transforms: List[np.ndarray] = []
    for b in range(B):
        T_world = fk_fn(se3[b], dof[b], kin)
        all_joint_transforms.append(T_world)

    # --- Build all gripper meshes and compute global vertex bounds ---
    all_meshes: List[go.Mesh3d] = []
    global_mn = scene_pcd.min(axis=0).copy()
    global_mx = scene_pcd.max(axis=0).copy()

    for b in range(B):
        mesh = _build_merged_gripper_mesh(
            all_joint_transforms[b],
            coll_def,
            color_hex=_GRIPPER_COLOR,
            opacity=gripper_opacity,
        )
        mesh.name = f"Grasp {b}"
        all_meshes.append(mesh)
        vx = np.asarray(mesh.x)
        vy = np.asarray(mesh.y)
        vz = np.asarray(mesh.z)
        if len(vx) > 0:
            global_mn = np.minimum(global_mn, [vx.min(), vy.min(), vz.min()])
            global_mx = np.maximum(global_mx, [vx.max(), vy.max(), vz.max()])

    anchor = _anchor_cube_corners(global_mn, global_mx, pad=1.15)

    # --- Build Plotly figure ---
    fig = go.Figure()

    # Trace 0: Anchor cube (invisible, forces equal scaling)
    fig.add_trace(
        go.Scatter3d(
            x=anchor[:, 0],
            y=anchor[:, 1],
            z=anchor[:, 2],
            mode="markers",
            marker=dict(size=0.1, opacity=0.0, color="white"),
            hoverinfo="skip",
            showlegend=False,
            name="anchor",
        )
    )

    # Trace 1: Scene point cloud (static)
    fig.add_trace(
        go.Scatter3d(
            x=scene_pcd[:, 0],
            y=scene_pcd[:, 1],
            z=scene_pcd[:, 2],
            mode="markers",
            marker=dict(size=2, color=scene_rgb, opacity=0.8),
            showlegend=False,
            name="scene",
        )
    )

    # Trace 2: Gripper mesh (initial = sample 0)
    fig.add_trace(all_meshes[0])

    # --- Build animation frames (one per sample) ---
    gripper_trace_idx = [2]
    frames = []
    for b in range(B):
        frames.append(
            go.Frame(
                name=str(b),
                data=[all_meshes[b]],
                traces=gripper_trace_idx,
            )
        )
    fig.frames = frames

    # --- Layout with slider ---
    fig.update_layout(
        title=f"Generated Grasps — {gripper_name} (B={B})",
        scene=dict(
            aspectmode="cube",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        sliders=[
            dict(
                active=0,
                pad={"t": 30},
                currentvalue={"prefix": "Grasp: "},
                steps=[
                    dict(
                        method="animate",
                        label=str(b),
                        args=[
                            [str(b)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    )
                    for b in range(B)
                ],
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.05,
                y=0.02,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 200, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
    )

    return fig
