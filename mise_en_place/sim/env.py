"""Dual SO-101 dinner-table environment.

Action space is 12 joint-position targets (``left`` 6 then ``right`` 6, in
:data:`mise_en_place.config.JOINTS` order), which is exactly what the SO-101
position actuators consume -- so a scripted expert, an ACT checkpoint and a
SmolVLA checkpoint are all interchangeable.

Grasping uses an **attach-on-close abstraction**: each (arm, object) pair has an
inactive weld equality in the MJCF, and the env activates it only when the arm's
TCP is physically within :data:`GRASP_RADIUS` of the object *and* the gripper is
commanded closed.  Objects remain fully dynamic free bodies -- they fall, they
collide, they can be knocked over or dropped -- and success is always measured
on the settled physics state.  See README "Honest scope" for why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .. import config as C
from . import scene_builder

#: TCP must be this close to the object's grasp frame for a close to latch
GRASP_RADIUS = 0.070
#: gripper ctrl below this counts as "commanded closed"
GRASP_CLOSE_CTRL = 0.45
#: gripper ctrl above this counts as "commanded open" and releases
GRASP_OPEN_CTRL = 0.75


@dataclass
class Obs:
    """One control-step observation."""

    qpos: np.ndarray                 # (12,) arm joint positions
    qvel: np.ndarray                 # (12,) arm joint velocities
    tcp: dict[str, np.ndarray]       # arm -> (3,) world TCP position
    objects: dict[str, np.ndarray]   # object -> (3,) world position
    holding: dict[str, str | None]   # arm -> held object name
    t: float
    step: int
    images: dict[str, np.ndarray] = field(default_factory=dict)

    def flat(self) -> np.ndarray:
        """Flat proprio+object vector, the input a learned policy would see."""
        parts = [self.qpos, self.qvel]
        parts += [self.tcp[a] for a in C.ARMS]
        parts += [self.objects[o.name] for o in C.OBJECTS]
        parts.append(np.array([1.0 if self.holding[a] else 0.0 for a in C.ARMS]))
        # Phase disambiguates repeated poses before/after grasp and release.
        parts.append(np.array([self.step / C.MAX_EPISODE_STEPS], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)


class DinnerTableEnv:
    def __init__(
        self,
        scene_xml: Path | None = None,
        render_size: tuple[int, int] = (C.RENDER_H, C.RENDER_W),
        rebuild: bool = False,
    ) -> None:
        path = Path(scene_xml or C.SCENE_XML)
        if rebuild or not path.exists():
            scene_builder.build_scene(out_xml=path)
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self._scratch = mujoco.MjData(self.model)
        self._render_size = render_size
        self._renderer: mujoco.Renderer | None = None

        self._index()
        self.rng = np.random.default_rng(0)
        self.seed_used: int | None = None
        self.holding: dict[str, str | None] = {a: None for a in C.ARMS}
        self.step_count = 0
        self.grasp_events: list[tuple[int, str, str, str]] = []

    # ------------------------------------------------------------------
    # indexing
    # ------------------------------------------------------------------
    def _index(self) -> None:
        m = self.model
        nid = mujoco.mj_name2id

        self.act_id: dict[str, list[int]] = {}
        self.jnt_qadr: dict[str, list[int]] = {}
        self.jnt_dadr: dict[str, list[int]] = {}
        self.tcp_site: dict[str, int] = {}
        self.gripper_body: dict[str, int] = {}
        for arm in C.ARMS:
            self.act_id[arm] = [
                nid(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_{j}") for j in C.JOINTS
            ]
            jids = [nid(m, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_{j}") for j in C.JOINTS]
            self.jnt_qadr[arm] = [int(m.jnt_qposadr[i]) for i in jids]
            self.jnt_dadr[arm] = [int(m.jnt_dofadr[i]) for i in jids]
            self.tcp_site[arm] = nid(m, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_tcp")
            self.gripper_body[arm] = nid(m, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_gripper")

        self.obj_body = {
            n: nid(m, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in [o.name for o in C.OBJECTS] + ["plate"]
        }
        self.obj_qadr = {}
        for n in list(self.obj_body):
            jid = nid(m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_free")
            self.obj_qadr[n] = int(m.jnt_qposadr[jid])

        self.eq_id = {
            (a, o.name): nid(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"{a}_grasp_{o.name}")
            for a in C.ARMS
            for o in C.OBJECTS
        }
        self.light_ids = [
            nid(m, mujoco.mjtObj.mjOBJ_LIGHT, n) for n in ("keylight", "filllight")
        ]
        self._light_diffuse0 = self.model.light_diffuse.copy()

    # ------------------------------------------------------------------
    # reset / randomization
    # ------------------------------------------------------------------
    def reset(self, seed: int = 0, randomize: C.SceneRandomization | None = None) -> Obs:
        rand = randomize if randomize is not None else C.SceneRandomization()
        self.rng = np.random.default_rng(seed)
        self.seed_used = seed
        self.step_count = 0
        self.holding = {a: None for a in C.ARMS}
        self.grasp_events = []

        mujoco.mj_resetData(self.model, self.data)
        self.model.light_diffuse[:] = self._light_diffuse0

        for arm in C.ARMS:
            for k, adr in enumerate(self.jnt_qadr[arm]):
                self.data.qpos[adr] = C.HOME_QPOS[k]

        for spec in C.OBJECTS:
            adr = self.obj_qadr[spec.name]
            x, y = spec.spawn
            if rand.object_xy:
                x += self.rng.uniform(-spec.jitter[0], spec.jitter[0])
                y += self.rng.uniform(-spec.jitter[1], spec.jitter[1])
            yaw = self.rng.uniform(-0.35, 0.35) if rand.object_xy else 0.0
            self.data.qpos[adr : adr + 3] = (x, y, spec.z)
            self.data.qpos[adr + 3 : adr + 7] = (
                np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2),
            )

        padr = self.obj_qadr["plate"]
        px, py = C.PLATE_POS
        if rand.plate_xy:
            px += self.rng.uniform(-rand.plate_jitter[0], rand.plate_jitter[0])
            py += self.rng.uniform(-rand.plate_jitter[1], rand.plate_jitter[1])
        self.data.qpos[padr : padr + 3] = (px, py, 0.006)
        self.data.qpos[padr + 3 : padr + 7] = (1.0, 0.0, 0.0, 0.0)

        if rand.lighting:
            scale = self.rng.uniform(*rand.light_range)
            self.model.light_diffuse[:] = np.clip(self._light_diffuse0 * scale, 0, 1)

        for arm in C.ARMS:
            for k, aid in enumerate(self.act_id[arm]):
                self.data.ctrl[aid] = C.HOME_QPOS[k]
        for eid in self.eq_id.values():
            self.data.eq_active[eid] = 0

        mujoco.mj_forward(self.model, self.data)
        # let objects settle onto the table before the episode starts
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        return self.observe()

    # ------------------------------------------------------------------
    # stepping
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray, cameras: tuple[str, ...] = ()) -> Obs:
        """Apply a 12-vector of joint position targets for one control step."""
        action = np.asarray(action, dtype=float).reshape(C.N_ACTION)
        for i, arm in enumerate(C.ARMS):
            seg = action[i * C.N_JOINTS : (i + 1) * C.N_JOINTS]
            for k, aid in enumerate(self.act_id[arm]):
                lo, hi = self.model.actuator_ctrlrange[aid]
                self.data.ctrl[aid] = float(np.clip(seg[k], lo, hi))

        self._update_grasps()
        for _ in range(C.SIM_STEPS_PER_CONTROL):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        return self.observe(cameras)

    def _update_grasps(self) -> None:
        """Latch, release, or hand off welds from jaw state and proximity."""
        for arm in C.ARMS:
            grip_ctrl = float(self.data.ctrl[self.act_id[arm][5]])
            held = self.holding[arm]

            if held is not None:
                if grip_ctrl >= GRASP_OPEN_CTRL:
                    self.data.eq_active[self.eq_id[(arm, held)]] = 0
                    self.holding[arm] = None
                    self.grasp_events.append((self.step_count, arm, held, "release"))
                continue

            if grip_ctrl > GRASP_CLOSE_CTRL:
                continue
            tcp = self.tcp_pos(arm)
            best, best_d = None, GRASP_RADIUS
            for spec in C.OBJECTS:
                d = float(np.linalg.norm(self.obj_pos(spec.name) - tcp))
                owner = next(
                    (candidate for candidate in C.ARMS if self.holding[candidate] == spec.name),
                    None,
                )
                if owner is not None:
                    # Guarded handoff: the receiver is closed and near while
                    # the current owner explicitly commands release.
                    owner_grip = float(self.data.ctrl[self.act_id[owner][5]])
                    if owner_grip < GRASP_OPEN_CTRL:
                        continue
                if d < best_d:
                    best, best_d = spec.name, d
            if best is not None:
                owner = next(
                    (candidate for candidate in C.ARMS if self.holding[candidate] == best),
                    None,
                )
                if owner is not None:
                    self.data.eq_active[self.eq_id[(owner, best)]] = 0
                    self.holding[owner] = None
                    self.grasp_events.append((self.step_count, owner, best, "handoff_release"))
                self._attach(arm, best)
                if owner is not None:
                    self.grasp_events.append((self.step_count, arm, best, "handoff_receive"))

    def _attach(self, arm: str, obj: str) -> None:
        m, d = self.model, self.data
        b1, b2 = self.gripper_body[arm], self.obj_body[obj]
        R1 = d.xmat[b1].reshape(3, 3)
        rel_pos = R1.T @ (d.xpos[b2] - d.xpos[b1])
        rel_R = R1.T @ d.xmat[b2].reshape(3, 3)
        rel_q = np.empty(4)
        mujoco.mju_mat2Quat(rel_q, rel_R.flatten())

        eid = self.eq_id[(arm, obj)]
        m.eq_data[eid, 0:3] = 0.0          # anchor at body2 origin
        m.eq_data[eid, 3:6] = rel_pos
        m.eq_data[eid, 6:10] = rel_q
        m.eq_data[eid, 10] = 1.0           # torquescale
        d.eq_active[eid] = 1
        self.holding[arm] = obj
        self.grasp_events.append((self.step_count, arm, obj, "grasp"))

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------
    def tcp_pos(self, arm: str) -> np.ndarray:
        return self.data.site_xpos[self.tcp_site[arm]].copy()

    def tcp_mat(self, arm: str) -> np.ndarray:
        return self.data.site_xmat[self.tcp_site[arm]].reshape(3, 3).copy()

    def obj_pos(self, name: str) -> np.ndarray:
        return self.data.xpos[self.obj_body[name]].copy()

    def arm_qpos(self, arm: str) -> np.ndarray:
        return np.array([self.data.qpos[a] for a in self.jnt_qadr[arm]])

    def arm_qvel(self, arm: str) -> np.ndarray:
        return np.array([self.data.qvel[a] for a in self.jnt_dadr[arm]])

    def observe(self, cameras: tuple[str, ...] = ()) -> Obs:
        obs = Obs(
            qpos=np.concatenate([self.arm_qpos(a) for a in C.ARMS]),
            qvel=np.concatenate([self.arm_qvel(a) for a in C.ARMS]),
            tcp={a: self.tcp_pos(a) for a in C.ARMS},
            objects={n: self.obj_pos(n) for n in self.obj_body},
            holding=dict(self.holding),
            t=float(self.data.time),
            step=self.step_count,
        )
        for cam in cameras:
            obs.images[cam] = self.render(cam)
        return obs

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def render(self, camera: str = "demo") -> np.ndarray:
        if self._renderer is None:
            h, w = self._render_size
            self._renderer = mujoco.Renderer(self.model, h, w)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # task evaluation
    # ------------------------------------------------------------------
    def settle(self, steps: int = 120) -> None:
        """Let physics come to rest before scoring."""
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def place_error(self, obj: str) -> float:
        """Planar distance from the object to its target slot, in metres."""
        plate = self.obj_pos("plate")[:2]
        nominal = np.array(C.PLACE_TARGET[obj])
        target = nominal - np.array(C.PLATE_POS) + plate   # follow the plate
        return float(np.linalg.norm(self.obj_pos(obj)[:2] - target))

    def object_placed(self, obj: str) -> bool:
        """Placed = within tolerance of the slot, on the table, not still held."""
        if any(self.holding[a] == obj for a in C.ARMS):
            return False
        z = float(self.obj_pos(obj)[2])
        return self.place_error(obj) <= C.PLACE_TOLERANCE and -0.01 < z < 0.05

    def score(self, objects: tuple[str, ...] = ("fork", "spoon")) -> dict:
        per_obj = {o: self.object_placed(o) for o in objects}
        return {
            "success": all(per_obj.values()),
            "per_object": per_obj,
            "errors": {o: self.place_error(o) for o in objects},
            "steps": self.step_count,
            "sim_time": float(self.data.time),
        }
