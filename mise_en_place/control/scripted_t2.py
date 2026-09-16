"""T2 handoff: right picks spoon, transfers to left, then left places."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config as C
from ..sim.env import DinnerTableEnv
from .scripted_t1 import ScriptedConfig, ScriptedT1Expert


@dataclass
class HandoffConfig:
    mid_xyz: tuple[float, float, float] = (0.08, -0.015, 0.14)
    receiver_local_offset: tuple[float, float, float] = (0.060, 0.0, 0.010)
    approach_world_offset: tuple[float, float, float] = (0.0, 0.070, 0.035)
    steps: int = 50


class ScriptedT2Handoff(ScriptedT1Expert):
    """Guarded right-to-left spoon handoff and left-side placement."""

    def __init__(
        self,
        env: DinnerTableEnv,
        cfg: ScriptedConfig | None = None,
        handoff: HandoffConfig | None = None,
    ) -> None:
        super().__init__(env, cfg)
        self.handoff = handoff or HandoffConfig()

    def run_t2(self) -> dict:
        cfg = self.cfg
        mid = np.array(self.handoff.mid_xyz, dtype=float)
        hsteps = self.handoff.steps
        arm, obj = "right", "spoon"
        grasp_z = C.OBJECT_BY_NAME[obj].grasp_z
        grasped = False
        for attempt in range(cfg.grasp_retries):
            if attempt:
                # A failed close can knock a light utensil out of reach.
                self.env.reset(seed=self.env.seed_used or 0)
            xy = self._obj_xy(obj)
            attempt_z = max(0.018, grasp_z - 0.004 * attempt)
            self._go(
                arm,
                np.array([xy[0], xy[1], cfg.hover_z]),
                C.GRIPPER_OPEN,
                cfg.steps_move,
            )
            self._go(
                arm,
                np.array([xy[0], xy[1], attempt_z]),
                C.GRIPPER_OPEN,
                cfg.steps_move,
            )
            # Recenter on the live object after the approach disturbed it.
            xy = self._obj_xy(obj)
            self._go(
                arm,
                np.array([xy[0], xy[1], attempt_z]),
                C.GRIPPER_OPEN,
                20,
            )
            self._go(
                arm,
                np.array([xy[0], xy[1], attempt_z]),
                C.GRIPPER_CLOSED,
                cfg.steps_grip,
            )
            if self.env.holding[arm] == obj:
                grasped = True
                break
        if not grasped:
            return {
                "task": "T2_handoff",
                "success": False,
                "transferred": False,
                "placed_left": False,
                "place_error": float("inf"),
                "steps": self.env.step_count,
                "sim_time": float(self.env.data.time),
                "events": self.env.grasp_events,
                "failure": "right arm could not acquire spoon",
            }
        xy = self._obj_xy(obj)
        self._go(
            arm,
            np.array([xy[0], xy[1], cfg.hover_z]),
            C.GRIPPER_CLOSED,
            cfg.steps_lift,
        )

        self._go("right", mid, C.GRIPPER_CLOSED, hsteps)
        # Receive at the opposite end of the utensil, not at the right TCP.
        # This keeps the two physical grippers separated during transfer.
        spoon_rotation = self.env.data.xmat[
            self.env.obj_body["spoon"]
        ].reshape(3, 3)
        receive = self.env.obj_pos("spoon") + spoon_rotation @ np.array(
            self.handoff.receiver_local_offset
        )
        approach = receive + np.array(self.handoff.approach_world_offset)
        self._go("left", approach, C.GRIPPER_OPEN, hsteps)
        self._go("left", receive, C.GRIPPER_OPEN, cfg.steps_grip)
        self._go("left", receive, C.GRIPPER_CLOSED, cfg.steps_grip)
        self._go(
            "right",
            mid + np.array([0.0, -0.090, 0.045]),
            C.GRIPPER_OPEN,
            cfg.steps_grip,
        )

        target = self._place_xy("fork")  # left-side place after handoff
        # The receiver grasps the far end, so compensate the measured
        # TCP-to-object offset rather than placing the TCP itself at the slot.
        def tcp_target_for_object(desired_object_xyz: np.ndarray) -> np.ndarray:
            offset = self.env.tcp_pos("left") - self.env.obj_pos("spoon")
            return desired_object_xyz + offset

        desired_hover = np.array([target[0], target[1], cfg.place_hover_z])
        self._go(
            "left",
            tcp_target_for_object(desired_hover),
            C.GRIPPER_CLOSED,
            cfg.steps_move,
        )
        desired_place = np.array([target[0], target[1], 0.008])
        # Recompute after each motion because position-only IK may rotate the wrist.
        self._go(
            "left",
            tcp_target_for_object(desired_place),
            C.GRIPPER_CLOSED,
            cfg.steps_move,
        )
        self._go(
            "left",
            tcp_target_for_object(desired_place),
            C.GRIPPER_CLOSED,
            max(20, cfg.steps_grip),
        )
        release_tcp = self.env.tcp_pos("left").copy()
        self._go("left", release_tcp, C.GRIPPER_OPEN, cfg.steps_grip)
        self._go(
            "left",
            release_tcp + np.array([0.0, 0.0, 0.07]),
            C.GRIPPER_OPEN,
            cfg.steps_lift,
        )
        self.env.settle()
        error = float(np.linalg.norm(self.env.obj_pos("spoon")[:2] - target))
        transferred = any(
            arm == "left" and obj == "spoon" and event == "handoff_receive"
            for _, arm, obj, event in self.env.grasp_events
        )
        placed = (
            error <= C.PLACE_TOLERANCE
            and not any(self.env.holding[arm] == "spoon" for arm in C.ARMS)
            and -0.01 < float(self.env.obj_pos("spoon")[2]) < 0.05
        )
        return {
            "task": "T2_handoff",
            "success": transferred and placed,
            "transferred": transferred,
            "placed_left": placed,
            "place_error": error,
            "steps": self.env.step_count,
            "sim_time": float(self.env.data.time),
            "events": self.env.grasp_events,
        }
