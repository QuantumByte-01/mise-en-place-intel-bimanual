"""Sweep learned-action authority inside the reliable scripted executive.

The learned model runs at every control step.  Its joint targets are blended
with the geometric expert while discrete grasp/release commands remain guarded
by the executive.  This is the deployment-safe learned path; pure BC is
reported separately by ``eval_learned_policy.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import openvino as ov

from mise_en_place import config as C
from mise_en_place.control.scripted_t1 import ScriptedT1Expert
from mise_en_place.sim.env import DinnerTableEnv


def make_actor(model_path: Path, device: str) -> Callable[[np.ndarray], np.ndarray]:
    compiled = ov.Core().compile_model(model_path, device, {"PERFORMANCE_HINT": "LATENCY"})
    input_port, output_port = compiled.input(0), compiled.output(0)

    def act(observation: np.ndarray) -> np.ndarray:
        result = compiled({input_port: observation[None, :].astype(np.float32)})
        return np.asarray(result[output_port])[0]

    return act


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtimes", default="fp32,int8")
    parser.add_argument("--blends", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=C.ARTIFACTS / "hybrid_authority_sweep.json",
    )
    parser.add_argument(
        "--max-residual",
        type=float,
        default=0.15,
        help="Maximum learned deviation from the geometric joint target, radians",
    )
    args = parser.parse_args()

    runtimes = [value.strip() for value in args.runtimes.split(",") if value.strip()]
    blends = [float(value) for value in args.blends.split(",") if value.strip()]
    joint_mask = np.ones(C.N_ACTION, dtype=bool)
    joint_mask[[5, 11]] = False
    all_results: dict[str, dict] = {}

    for runtime in runtimes:
        path = C.REPO_ROOT / "checkpoints" / "openvino" / f"lora_bc_{runtime}.xml"
        actor = make_actor(path, args.device)
        runtime_results: dict[str, dict] = {}
        env = DinnerTableEnv()
        for blend in blends:
            rows = []
            deviations: list[float] = []
            for episode in range(args.episodes):
                seed = args.start_seed + episode
                env.reset(seed=seed)
                expert = ScriptedT1Expert(env)
                original_step = env.step

                def hybrid_step(requested, cameras=()):
                    requested = np.asarray(requested, dtype=np.float32)
                    learned = actor(env.observe().flat())
                    proposed = requested + blend * (learned - requested)
                    command = requested.copy()
                    command[joint_mask] = np.clip(
                        proposed[joint_mask],
                        requested[joint_mask] - args.max_residual,
                        requested[joint_mask] + args.max_residual,
                    )
                    # Symbolic guard: grasp/release timing stays deterministic.
                    command[[5, 11]] = requested[[5, 11]]
                    deviations.append(float(np.mean(np.abs(command - requested))))
                    return original_step(command, cameras)

                env.step = hybrid_step  # type: ignore[method-assign]
                score = expert.run_true_parallel_t1()
                env.step = original_step  # type: ignore[method-assign]
                rows.append({"seed": seed, **score})
                print(
                    f"{runtime} blend={blend:.2f} seed={seed} "
                    f"success={int(score['success'])}"
                )
            key = f"{blend:.2f}"
            runtime_results[key] = {
                "success_rate": float(np.mean([row["success"] for row in rows])),
                "fork_rate": float(np.mean([row["per_object"]["fork"] for row in rows])),
                "spoon_rate": float(np.mean([row["per_object"]["spoon"] for row in rows])),
                "mean_executed_residual": float(np.mean(deviations)),
                "rows": rows,
            }
        env.close()
        all_results[runtime] = runtime_results

    report = {
        "device": args.device,
        "episodes_per_setting": args.episodes,
        "max_residual_rad": args.max_residual,
        "gripper_guard": True,
        "results": all_results,
        "claim": (
            "OpenVINO learned inference runs every step as a bounded residual. "
            "The geometric executive guards discrete grasps and is not removed."
        ),
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
