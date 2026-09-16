"""Scripted T1 expert: left places fork, right places spoon (parallel)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config as C
from ..sim.env import DinnerTableEnv
from .ik import full_arm_action, solve_ik


@dataclass
class ScriptedConfig:
    hover_z: float = 0.10
    place_hover_z: float = 0.09
    place_z: float = 0.028
    steps_move: int = 55
    steps_grip: int = 28
    steps_lift: int = 35
    grasp_retries: int = 3


class ScriptedT1Expert:
    """Waypoint expert for T1 pick→place (sequential or true parallel)."""

    def __init__(self, env: DinnerTableEnv, cfg: ScriptedConfig | None = None) -> None:
        self.env = env
        self.cfg = cfg or ScriptedConfig()

    def _hold_other(self, arm: str, action: np.ndarray) -> None:
        for i, a in enumerate(C.ARMS):
            if a == arm:
                continue
            for k, aid in enumerate(self.env.act_id[a]):
                action[i * 6 + k] = self.env.data.ctrl[aid]

    def _arm_ctrl(self, arm: str, q6: np.ndarray, steps: int) -> None:
        for _ in range(steps):
            action = np.zeros(C.N_ACTION, dtype=float)
            self._hold_other(arm, action)
            if arm == "left":
                action[:6] = q6
            else:
                action[6:] = q6
            self.env.step(action)

    def _both_ctrl(self, q_left: np.ndarray, q_right: np.ndarray, steps: int) -> None:
        for _ in range(steps):
            action = np.zeros(C.N_ACTION, dtype=float)
            action[:6] = q_left
            action[6:] = q_right
            self.env.step(action)

    def _ik_to(self, arm: str, xyz: np.ndarray, gripper: float) -> np.ndarray:
        env = self.env
        q5 = solve_ik(
            env.model,
            env.data,
            env.tcp_site[arm],
            env.jnt_qadr[arm],
            env.act_id[arm],
            xyz,
            max_iters=80,
            tol=0.004,
            damping=1e-3,
            step_scale=0.65,
        )
        return full_arm_action(q5, gripper)

    def _go(self, arm: str, xyz: np.ndarray, gripper: float, steps: int) -> float:
        q6 = self._ik_to(arm, xyz, gripper)
        self._arm_ctrl(arm, q6, steps)
        q6 = self._ik_to(arm, xyz, gripper)
        self._arm_ctrl(arm, q6, max(8, steps // 4))
        return float(np.linalg.norm(self.env.tcp_pos(arm) - xyz))

    def _go_both(
        self,
        xyz_l: np.ndarray,
        g_l: float,
        xyz_r: np.ndarray,
        g_r: float,
        steps: int,
    ) -> None:
        q_l = self._ik_to("left", xyz_l, g_l)
        q_r = self._ik_to("right", xyz_r, g_r)
        self._both_ctrl(q_l, q_r, steps)
        q_l = self._ik_to("left", xyz_l, g_l)
        q_r = self._ik_to("right", xyz_r, g_r)
        self._both_ctrl(q_l, q_r, max(8, steps // 4))

    def _obj_xy(self, obj: str) -> np.ndarray:
        xy = self.env.obj_pos(obj)[:2].copy()
        if xy[0] < 0.03:
            xy[0] = 0.03
        return xy

    def _place_xy(self, obj: str) -> np.ndarray:
        plate = self.env.obj_pos("plate")[:2]
        nominal = np.array(C.PLACE_TARGET[obj])
        return nominal - np.array(C.PLATE_POS) + plate

    def _pick_place(self, arm: str, obj: str) -> None:
        cfg = self.cfg
        env = self.env
        grasp_z = C.OBJECT_BY_NAME[obj].grasp_z

        grasped = False
        for attempt in range(cfg.grasp_retries):
            obj_xy = self._obj_xy(obj)
            self._go(arm, np.array([obj_xy[0], obj_xy[1], cfg.hover_z]), C.GRIPPER_OPEN, cfg.steps_move)
            self._go(arm, np.array([obj_xy[0], obj_xy[1], grasp_z]), C.GRIPPER_OPEN, cfg.steps_move)
            obj_xy = self._obj_xy(obj)
            self._go(arm, np.array([obj_xy[0], obj_xy[1], grasp_z]), C.GRIPPER_OPEN, 20)
            self._go(arm, np.array([obj_xy[0], obj_xy[1], grasp_z]), C.GRIPPER_CLOSED, cfg.steps_grip)
            if env.holding[arm] == obj:
                grasped = True
                break
            self._go(
                arm,
                np.array([obj_xy[0], obj_xy[1], grasp_z + 0.01 * (attempt + 1)]),
                C.GRIPPER_OPEN,
                cfg.steps_grip,
            )
            grasp_z = max(0.018, grasp_z - 0.004)

        if not grasped:
            obj_xy = self._obj_xy(obj)
            self._go(arm, np.array([obj_xy[0], obj_xy[1], 0.018]), C.GRIPPER_CLOSED, cfg.steps_grip)

        grip = C.GRIPPER_CLOSED if env.holding[arm] == obj else C.GRIPPER_OPEN
        self._go(arm, np.array([*self._obj_xy(obj), cfg.hover_z]), grip, cfg.steps_lift)

        target = self._place_xy(obj)
        self._go(arm, np.array([target[0], target[1], cfg.place_hover_z]), C.GRIPPER_CLOSED, cfg.steps_move)
        self._go(arm, np.array([target[0], target[1], cfg.place_z]), C.GRIPPER_CLOSED, cfg.steps_move)
        self._go(arm, np.array([target[0], target[1], cfg.place_z]), C.GRIPPER_OPEN, cfg.steps_grip)
        self._go(arm, np.array([target[0], target[1], cfg.place_hover_z]), C.GRIPPER_OPEN, cfg.steps_lift)
        self._arm_ctrl(arm, np.array(C.HOME_QPOS, dtype=float), cfg.steps_move)

    def run_true_parallel_t1(self) -> dict:
        """Left fork + right spoon synchronized (true bimanual)."""
        cfg = self.cfg
        env = self.env
        gz_f = C.OBJECT_BY_NAME["fork"].grasp_z
        gz_s = C.OBJECT_BY_NAME["spoon"].grasp_z

        for attempt in range(cfg.grasp_retries):
            fxy, sxy = self._obj_xy("fork"), self._obj_xy("spoon")
            self._go_both(
                np.array([fxy[0], fxy[1], cfg.hover_z]), C.GRIPPER_OPEN,
                np.array([sxy[0], sxy[1], cfg.hover_z]), C.GRIPPER_OPEN,
                cfg.steps_move,
            )
            self._go_both(
                np.array([fxy[0], fxy[1], gz_f]), C.GRIPPER_OPEN,
                np.array([sxy[0], sxy[1], gz_s]), C.GRIPPER_OPEN,
                cfg.steps_move,
            )
            fxy, sxy = self._obj_xy("fork"), self._obj_xy("spoon")
            self._go_both(
                np.array([fxy[0], fxy[1], gz_f]), C.GRIPPER_OPEN,
                np.array([sxy[0], sxy[1], gz_s]), C.GRIPPER_OPEN,
                20,
            )
            self._go_both(
                np.array([fxy[0], fxy[1], gz_f]), C.GRIPPER_CLOSED,
                np.array([sxy[0], sxy[1], gz_s]), C.GRIPPER_CLOSED,
                cfg.steps_grip,
            )
            if env.holding["left"] == "fork" and env.holding["right"] == "spoon":
                break
            self._go_both(
                np.array([fxy[0], fxy[1], gz_f + 0.012]), C.GRIPPER_OPEN,
                np.array([sxy[0], sxy[1], gz_s + 0.012]), C.GRIPPER_OPEN,
                cfg.steps_grip,
            )
            gz_f = max(0.018, gz_f - 0.004)
            gz_s = max(0.018, gz_s - 0.004)

        fxy, sxy = self._obj_xy("fork"), self._obj_xy("spoon")
        g_l = C.GRIPPER_CLOSED if env.holding["left"] == "fork" else C.GRIPPER_OPEN
        g_r = C.GRIPPER_CLOSED if env.holding["right"] == "spoon" else C.GRIPPER_OPEN
        self._go_both(
            np.array([fxy[0], fxy[1], cfg.hover_z]), g_l,
            np.array([sxy[0], sxy[1], cfg.hover_z]), g_r,
            cfg.steps_lift,
        )

        ft, st = self._place_xy("fork"), self._place_xy("spoon")
        self._go_both(
            np.array([ft[0], ft[1], cfg.place_hover_z]), C.GRIPPER_CLOSED,
            np.array([st[0], st[1], cfg.place_hover_z]), C.GRIPPER_CLOSED,
            cfg.steps_move,
        )
        self._go_both(
            np.array([ft[0], ft[1], cfg.place_z]), C.GRIPPER_CLOSED,
            np.array([st[0], st[1], cfg.place_z]), C.GRIPPER_CLOSED,
            cfg.steps_move,
        )
        self._go_both(
            np.array([ft[0], ft[1], cfg.place_z]), C.GRIPPER_OPEN,
            np.array([st[0], st[1], cfg.place_z]), C.GRIPPER_OPEN,
            cfg.steps_grip,
        )
        self._go_both(
            np.array([ft[0], ft[1], cfg.place_hover_z]), C.GRIPPER_OPEN,
            np.array([st[0], st[1], cfg.place_hover_z]), C.GRIPPER_OPEN,
            cfg.steps_lift,
        )
        home = np.array(C.HOME_QPOS, dtype=float)
        self._both_ctrl(home, home, cfg.steps_move)
        env.settle()
        return env.score(("fork", "spoon"))

    def run_parallel_t1(self, *, true_parallel: bool = True) -> dict:
        """Default: true parallel bimanual. Sequential fallback if parallel fails."""
        if true_parallel:
            score = self.run_true_parallel_t1()
            if score["success"]:
                return score
            seed = self.env.seed_used or 0
            self.env.reset(seed=seed)
        self._pick_place("left", "fork")
        self._pick_place("right", "spoon")
        self.env.settle()
        return self.env.score(("fork", "spoon"))
