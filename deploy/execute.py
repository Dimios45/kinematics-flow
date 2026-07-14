"""Execute a selected grasp on the real Franka via panda-py.

Frame chain (see deploy/common.py): the model's raw output pose IS the
Franka Hand base pose (origin at the hand base, z = approach, fingers along
+-y, grasp point at +0.102 z). Here:

    T_base_hand   = T_base_world @ T_world_hand
    T_base_flange = T_base_hand @ inv(T_FLANGE_HAND)
    T_base_EE     = T_base_flange @ F_T_EE   (F_T_EE read from the robot, so
                                              the Desk end-effector config is
                                              honored whatever it is)

Sequence: home -> open -> pre-grasp (retreat along approach) -> approach ->
close -> lift 0.3 m -> home. `--hover` stops 10 cm above the grasp with the
fingers open — run it first on every new setup to validate the frame chain.
"""

import argparse

import numpy as np

from deploy.common import (HAND_TCP_OFFSET, PANDA_MAX_WIDTH, T_FLANGE_HAND,
                           load_extrinsics, se3_inv)

APPROACH_DIST = 0.10  # m, pre-grasp retreat along the approach axis
LIFT_DIST = 0.30  # m, same as the sim success criterion
HOVER_DIST = 0.10  # m above the grasp in --hover mode
SPEED_FACTOR = 0.15
GRASP_FORCE = 40.0  # N


class FrankaExecutor:
    def __init__(self, host: str, speed_factor: float = SPEED_FACTOR):
        import panda_py
        from panda_py import libfranka

        self._panda_py = panda_py
        self.panda = panda_py.Panda(host)
        self.gripper = libfranka.Gripper(host)
        self.speed_factor = speed_factor
        # column-major libfranka array -> row-major 4x4
        self.F_T_EE = np.asarray(self.panda.get_state().F_T_EE).reshape(
            4, 4, order="F"
        )
        self._hand_to_ee = se3_inv(T_FLANGE_HAND) @ self.F_T_EE

    def hand_to_ee(self, T_base_hand: np.ndarray) -> np.ndarray:
        return T_base_hand @ self._hand_to_ee

    def ik(self, T_base_EE: np.ndarray, q_init=None):
        """Joint solution or None if unreachable."""
        if q_init is None:
            q_init = self.panda.get_state().q
        q = self._panda_py.ik(T_base_EE, q_init=np.asarray(q_init))
        return None if np.any(np.isnan(q)) else q

    def move_joints(self, q):
        self.panda.move_to_joint_position(q, speed_factor=self.speed_factor)

    def move_pose(self, T_base_EE):
        self.panda.move_to_pose(T_base_EE, speed_factor=self.speed_factor)

    def open_gripper(self, width: float):
        self.gripper.move(width=min(width, PANDA_MAX_WIDTH), speed=0.1)

    def close_gripper(self, width: float) -> bool:
        return self.gripper.grasp(
            width=max(width - 0.01, 0.0),
            speed=0.05,
            force=GRASP_FORCE,
            epsilon_inner=0.02,
            epsilon_outer=0.08,
        )


def _confirm(step_mode: bool, what: str):
    if step_mode and input(f"  next: {what} — Enter to run, 'q' to abort: ") == "q":
        raise SystemExit("aborted by user")


def execute_grasp(
    executor: FrankaExecutor,
    T_world_hand: np.ndarray,
    width: float,
    T_base_world: np.ndarray,
    hover: bool = False,
    step: bool = False,
) -> bool:
    T_base_hand = T_base_world @ T_world_hand
    approach = T_base_hand[:3, 2]  # grasp approach direction, in base frame
    up = T_base_world[:3, 2]  # world +z in base frame

    def offset(T, vec):
        T2 = T.copy()
        T2[:3, 3] = T2[:3, 3] + vec
        return T2

    grasp_ee = executor.hand_to_ee(T_base_hand)
    pregrasp_ee = executor.hand_to_ee(offset(T_base_hand, -APPROACH_DIST * approach))
    lift_ee = offset(grasp_ee, LIFT_DIST * up)
    hover_ee = executor.hand_to_ee(offset(T_base_hand, HOVER_DIST * up))

    q_pre = executor.ik(pregrasp_ee)
    if q_pre is None or executor.ik(grasp_ee, q_init=q_pre) is None:
        print("grasp unreachable (IK failed)")
        return False

    # open exactly to the predicted width: the model's width already includes
    # the clearance added at data generation, and the collision filter in
    # deploy.select validated the fingers at this opening
    open_width = min(width, PANDA_MAX_WIDTH)
    _confirm(step, f"open gripper to {open_width * 1000:.0f} mm and go home")
    executor.open_gripper(open_width)
    executor.panda.move_to_start(speed_factor=executor.speed_factor)

    if hover:
        q_hover = executor.ik(hover_ee)
        if q_hover is None:
            print("hover pose unreachable")
            return False
        _confirm(step, "move to HOVER pose (10 cm above grasp)")
        executor.move_joints(q_hover)
        tcp = T_base_hand[:3, 3] + HAND_TCP_OFFSET * approach + HOVER_DIST * up
        print(f"Hovering. TCP should be 10 cm above the grasp point "
              f"(base frame target {np.round(tcp, 3)}). Inspect alignment, "
              "then move the arm away manually or re-run without --hover.")
        return True

    _confirm(step, "move to pre-grasp")
    executor.move_joints(q_pre)
    _confirm(step, "approach grasp pose")
    executor.move_pose(grasp_ee)
    _confirm(step, f"close gripper (target {width * 1000:.0f} mm)")
    grasped = executor.close_gripper(width)
    print(f"gripper reports grasp {'SUCCESS' if grasped else 'FAILURE'}")
    _confirm(step, "lift 0.3 m")
    executor.move_pose(lift_ee)
    still_holding = executor.gripper.read_once().width > 0.001
    print(f"after lift: {'still holding' if still_holding else 'object lost'}")
    _confirm(step, "return home")
    executor.panda.move_to_start(speed_factor=executor.speed_factor)
    return bool(grasped and still_holding)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Franka controller IP")
    parser.add_argument("--extrinsics", default="deploy/config/extrinsics.yaml")
    parser.add_argument("--grasps", required=True,
                        help="grasps_ranked.npz from deploy.select (best first)")
    parser.add_argument("--index", type=int, default=0, help="which ranked grasp")
    parser.add_argument("--hover", action="store_true")
    parser.add_argument("--step", action="store_true",
                        help="confirm every motion on the keyboard")
    parser.add_argument("--speed", type=float, default=SPEED_FACTOR)
    args = parser.parse_args()

    T_base_world = load_extrinsics(args.extrinsics)["T_base_world"]
    grasps = np.load(args.grasps)
    T_world_hand = grasps["hand_pose"][args.index]
    width = float(grasps["width"][args.index])

    executor = FrankaExecutor(args.host, speed_factor=args.speed)
    ok = execute_grasp(
        executor, T_world_hand, width, T_base_world,
        hover=args.hover, step=args.step,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
