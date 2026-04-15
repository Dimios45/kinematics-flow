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

import jax.numpy as jnp
from flax import nnx
from mgs.sampler.kin.base import KinematicsModel
from mgs.sampler.kin.seg_op import (kinematic_frames, kinematic_transform,
                                    point_transform)


def _quat_mul(a, b):
    """Hamilton product, (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return jnp.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=jnp.float32,
    )


def _quat_rotate(q, v):
    """Rotate vector v by unit quaternion q=(w,x,y,z)."""
    w, x, y, z = q
    qvec = jnp.array([x, y, z], dtype=jnp.float32)
    uv = jnp.cross(qvec, v)
    uuv = jnp.cross(qvec, uv)
    return v + 2.0 * (w * uv + uuv)


def _normalize(q):
    return q / jnp.linalg.norm(q)


class DexeeKinematicsModel(nnx.Module, KinematicsModel):
    """
    Kinematics for the 3‑finger Dexee hand matching your MuJoCo XML.

    Joint order (num_dofs = 12):
      [F0/J0, F0/J1, F0/J2, F0/J3,
       F1/J0, F1/J1, F1/J2, F1/J3,
       F2/J0, F2/J1, F2/J2, F2/J3]
    """

    def __init__(self):
        # DoFs and graph (three chains of length 4).
        self.num_dofs = 12
        self.num_extra_dofs = 0
        self.kinematics_graph = [
            [0, 1, 2, 3],  # F0
            [4, 5, 6, 7],  # F1
            [8, 9, 10, 11],  # F2
        ]

        # As requested: identity. TCP / base offsets are handled in the XML, not here.
        self.base_to_contact = nnx.Variable(
            jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        )

        # A reasonable initial alignment for approach frames. This rotates the model
        # so that the local -Y finger directions tend to align with the object normal.
        # Feel free to tweak R or t to taste for your seeding strategy.
        Rx90 = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        self.align_to_approach = nnx.Variable(
            (Rx90, jnp.array([0.0, 0.0, 0.07], dtype=jnp.float32))
        )

        # ----------------------------------------------------------------------
        # Static body-to-body transforms from XML, folded per joint as [qw, qx, qy, qz, tx, ty, tz].
        # For J0, we compose the finger base (F{0,1,2}/) with the knuckle body pose.
        # For J1..J3, we use the per-segment offsets as-is (including distal's fixed quaternion).
        # ----------------------------------------------------------------------

        # Common knuckle pose relative to finger_base: pos=(0, 0.015, 0.17902), euler X = -1.0472 rad.
        q_knuckle = jnp.array(
            [jnp.cos(-1.0472 / 2.0), jnp.sin(-1.0472 / 2.0), 0.0, 0.0],
            dtype=jnp.float32,
        )
        t_knuckle = jnp.array([0.0, 0.015, 0.17902], dtype=jnp.float32)

        # Distal fixed frame quaternion on J3: "quat='0 0 -1 1'" (normalized).
        q_distal_raw = jnp.array([0.0, 0.0, -1.0, 1.0], dtype=jnp.float32)
        q_distal = _normalize(q_distal_raw)

        # Finger bases from XML (relative to dexee_gripper).
        # F0: pos="0 0.05 0.017", quat="1 0 0 0"
        q_F0 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        t_F0 = jnp.array([0.0, 0.05, 0.017], dtype=jnp.float32)

        # F1: pos="0.039 -0.029 0.017", quat="-0.16212752892551119 0 0 0.98676981326168844"
        # Sign does not matter; keep as-is.
        q_F1 = jnp.array([-0.16212753, 0.0, 0.0, 0.9867698], dtype=jnp.float32)
        t_F1 = jnp.array([0.039, -0.029, 0.017], dtype=jnp.float32)

        # F2: pos="-0.039 -0.029 0.017", quat="0.16212752892551119 0 0 0.98676981326168844"
        q_F2 = jnp.array([0.16212753, 0.0, 0.0, 0.9867698], dtype=jnp.float32)
        t_F2 = jnp.array([-0.039, -0.029, 0.017], dtype=jnp.float32)

        def j0_static(q_base, t_base):
            q = _quat_mul(q_base, q_knuckle)
            t = t_base + _quat_rotate(q_base, t_knuckle)
            return jnp.concatenate([q, t], axis=0)

        # Segment offsets down the chain (shared across fingers).
        T_J1 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, -0.03, 0.0], dtype=jnp.float32)
        T_J2 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, -0.05, 0.0], dtype=jnp.float32)
        T_J3 = jnp.concatenate(
            [q_distal, jnp.array([0.0, -0.035, 0.0], dtype=jnp.float32)], axis=0
        )

        # Assemble per-finger chains.
        T_F0 = [j0_static(q_F0, t_F0), T_J1, T_J2, T_J3]
        T_F1 = [j0_static(q_F1, t_F1), T_J1, T_J2, T_J3]
        T_F2 = [j0_static(q_F2, t_F2), T_J1, T_J2, T_J3]

        self.kinematics_transforms = nnx.Variable(jnp.stack(T_F0 + T_F1 + T_F2, axis=0))

        # ----------------------------------------------------------------------
        # Joint motions: [tx_per_rad, ty_per_rad, tz_per_rad, ax_x, ax_y, ax_z].
        # All joints are revolute; no translation per angle, axes copied from XML.
        # ----------------------------------------------------------------------
        JT = []
        # F0
        JT += [[0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]  # F0/J0  axis="0 0 -1"
        JT += [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]  # F0/J1  axis="1 0 0"
        JT += [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]  # F0/J2  axis="1 0 0"
        JT += [[0.0, 0.0, 0.0, -1.0, 0.0, 0.0]]  # F0/J3  axis="-1 0 0"
        # F1
        JT += [[0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]
        JT += [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
        JT += [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
        JT += [[0.0, 0.0, 0.0, -1.0, 0.0, 0.0]]
        # F2
        JT += [[0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]
        JT += [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
        JT += [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
        JT += [[0.0, 0.0, 0.0, -1.0, 0.0, 0.0]]
        self.joint_transforms = nnx.Variable(jnp.array(JT, dtype=jnp.float32))

        # Joint ranges (copied from XML for all three fingers).
        JR = []
        # F0
        JR += [[-0.8727, 0.8727]]  # J0
        JR += [[-1.3963, 0.7854]]  # J1
        JR += [[0.0, 1.3963]]  # J2
        JR += [[-0.5236, 1.4835]]  # J3
        # F1
        JR += [[-0.8727, 0.8727]]
        JR += [[-1.3963, 0.7854]]
        JR += [[0.0, 1.3963]]
        JR += [[-0.5236, 1.4835]]
        # F2
        JR += [[-0.8727, 0.8727]]
        JR += [[-1.3963, 0.7854]]
        JR += [[0.0, 1.3963]]
        JR += [[-0.5236, 1.4835]]
        self.joint_ranges = nnx.Variable(jnp.array(JR, dtype=jnp.float32))

        # Fingertip outward directions in each distal joint's local frame.
        # The finger link chain advances along local -Y; these normals follow that.
        self.fingertip_normals = nnx.Variable(
            jnp.array(
                [
                    [0.0, -1.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, -1.0, 0.0],
                ],
                dtype=jnp.float32,
            )
        )

        # J indices of distal links for F0, F1, F2.
        self.fingertip_idx = nnx.Variable(jnp.array([3, 7, 11], dtype=jnp.int32))

        # Local fingertip contact positions (in the distal joint frames).
        # First point sits near your XML 'fingertip_site'; two small neighbors help assignment.
        self.local_fingertip_contact_positions = nnx.Variable(
            jnp.array(
                [
                    [
                        [0, 0.007, 0.03],
                        [0.005, 0.007, 0.03],
                        [-0.005, 0.007, 0.03],
                        [0.0, 0.007, 0.025],
                        [0.005, 0.007, 0.025],
                        [-0.005, 0.007, 0.025],
                    ],
                    [
                        [0, 0.007, 0.03],
                        [0.005, 0.007, 0.03],
                        [-0.005, 0.007, 0.03],
                        [0.0, 0.007, 0.025],
                        [0.005, 0.007, 0.025],
                        [-0.005, 0.007, 0.025],
                    ],
                    [
                        [0, 0.007, 0.03],
                        [0.005, 0.007, 0.03],
                        [-0.005, 0.007, 0.03],
                        [0.0, 0.007, 0.025],
                        [0.005, 0.007, 0.025],
                        [-0.005, 0.007, 0.025],
                    ],
                ],
                dtype=jnp.float32,
            )
        )

        # A gentle pre‑grasp. Tune as you like; ranges enforce safety during optimization.
        self.init_pregrasp_joint = nnx.Variable(
            # jnp.zeros_like(
            jnp.array(
                [
                    0,
                    -1.0,
                    0,
                    0,
                    0,
                    -1.0,
                    0,
                    0,
                    0,
                    -1.0,
                    0,
                    0,
                ],
                dtype=jnp.float32,
            )
            # )
        )
