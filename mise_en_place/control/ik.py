"""Damped least-squares IK for SO-101 TCP positioning (5 arm DoF)."""

from __future__ import annotations

import mujoco
import numpy as np

from .. import config as C

# Reuse one scratch MjData per model id to avoid OOM across many IK calls.
_SCRATCH: dict[int, mujoco.MjData] = {}


def _scratch_for(model: mujoco.MjModel) -> mujoco.MjData:
    key = id(model)
    d = _SCRATCH.get(key)
    if d is None:
        d = mujoco.MjData(model)
        _SCRATCH[key] = d
    return d


def _dof_ids_for_qadr(model: mujoco.MjModel, qadr: list[int]) -> list[int]:
    out: list[int] = []
    for adr in qadr[:5]:
        for j in range(model.njnt):
            if int(model.jnt_qposadr[j]) == adr:
                out.append(int(model.jnt_dofadr[j]))
                break
    if len(out) != 5:
        raise RuntimeError(f"expected 5 dof ids, got {out}")
    return out


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    qadr: list[int],
    act_ids: list[int],
    target_pos: np.ndarray,
    *,
    max_iters: int = 50,
    tol: float = 0.005,
    damping: float = 2e-3,
    step_scale: float = 0.55,
) -> np.ndarray:
    """Return length-5 joint vector for arm joints (no gripper)."""
    q = np.array([data.qpos[a] for a in qadr[:5]], dtype=float)
    scratch = _scratch_for(model)
    scratch.qpos[:] = data.qpos
    jacp = np.zeros((3, model.nv))
    dof_ids = _dof_ids_for_qadr(model, qadr)

    for _ in range(max_iters):
        for k, adr in enumerate(qadr[:5]):
            scratch.qpos[adr] = q[k]
        mujoco.mj_forward(model, scratch)
        err = np.asarray(target_pos, dtype=float) - scratch.site_xpos[site_id]
        if float(np.linalg.norm(err)) < tol:
            break
        mujoco.mj_jacSite(model, scratch, jacp, None, site_id)
        J = jacp[:, dof_ids]
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
        q = q + step_scale * dq
        for k, aid in enumerate(act_ids[:5]):
            lo, hi = model.actuator_ctrlrange[aid]
            q[k] = float(np.clip(q[k], lo, hi))
    return q


def full_arm_action(q5: np.ndarray, gripper: float) -> np.ndarray:
    """Pack 5 arm joints + gripper into a 6-vector."""
    return np.array([*q5, gripper], dtype=float)
