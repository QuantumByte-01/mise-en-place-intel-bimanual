"""Audit inter-arm contacts, drops, and plate displacement for T1/T2."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from mise_en_place import config as C
from mise_en_place.control.scripted_t1 import ScriptedT1Expert
from mise_en_place.control.scripted_t2 import ScriptedT2Handoff
from mise_en_place.sim.env import DinnerTableEnv


def body_name(env: DinnerTableEnv, geom_id: int) -> str:
    body_id = int(env.model.geom_bodyid[geom_id])
    return mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "world"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--tasks", default="t1,t2")
    parser.add_argument(
        "--output",
        type=Path,
        default=C.ARTIFACTS / "contact_audit.json",
    )
    args = parser.parse_args()
    report: dict[str, dict] = {}

    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    for task in tasks:
        if task not in ("t1", "t2"):
            parser.error(f"unknown task: {task}")
        env = DinnerTableEnv()
        rows = []
        aggregate_pairs: Counter[str] = Counter()
        for index in range(args.episodes):
            seed = args.start_seed + index
            env.reset(seed=seed)
            plate_start = env.obj_pos("plate").copy()
            collision_steps = 0
            penetration_steps = 0
            pair_counts: Counter[str] = Counter()
            original_step = env.step

            def audited_step(action, cameras=()):
                nonlocal collision_steps, penetration_steps
                observation = original_step(action, cameras)
                pairs_this_step: set[str] = set()
                penetrating = False
                for index in range(env.data.ncon):
                    contact = env.data.contact[index]
                    first = body_name(env, int(contact.geom1))
                    second = body_name(env, int(contact.geom2))
                    is_inter_arm = (
                        first.startswith("left_") and second.startswith("right_")
                    ) or (
                        first.startswith("right_") and second.startswith("left_")
                    )
                    if is_inter_arm:
                        pair = " <-> ".join(sorted((first, second)))
                        pairs_this_step.add(pair)
                        penetrating = penetrating or float(contact.dist) < -0.001
                if pairs_this_step:
                    collision_steps += 1
                    pair_counts.update(pairs_this_step)
                if penetrating:
                    penetration_steps += 1
                return observation

            env.step = audited_step  # type: ignore[method-assign]
            if task == "t1":
                score = ScriptedT1Expert(env).run_true_parallel_t1()
            else:
                score = ScriptedT2Handoff(env).run_t2()
            env.step = original_step  # type: ignore[method-assign]
            aggregate_pairs.update(pair_counts)
            drops = [
                spec.name
                for spec in C.OBJECTS
                if float(env.obj_pos(spec.name)[2]) < -0.01
            ]
            row = {
                "seed": seed,
                "success": bool(score["success"]),
                "inter_arm_contact_steps": collision_steps,
                "penetration_steps": penetration_steps,
                "plate_displacement_m": float(
                    np.linalg.norm(env.obj_pos("plate")[:2] - plate_start[:2])
                ),
                "drops": drops,
                "contact_pairs": dict(pair_counts),
            }
            rows.append(row)
            print(
                f"{task} seed={seed} success={int(row['success'])} "
                f"contacts={collision_steps} penetrations={penetration_steps} "
                f"plate_shift={row['plate_displacement_m']:.4f} drops={drops}"
            )
        env.close()
        report[task] = {
            "episodes": args.episodes,
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "collision_free_rate": float(
                np.mean([row["inter_arm_contact_steps"] == 0 for row in rows])
            ),
            "penetration_free_rate": float(
                np.mean([row["penetration_steps"] == 0 for row in rows])
            ),
            "max_plate_displacement_m": max(row["plate_displacement_m"] for row in rows),
            "contact_pairs": dict(aggregate_pairs),
            "rows": rows,
        }

    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({task: {k: v for k, v in result.items() if k not in ("rows", "contact_pairs")} for task, result in report.items()}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
