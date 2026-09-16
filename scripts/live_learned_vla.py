"""Full learned end-to-end VLA with MuJoCo live viewer.

Optional SmolVLM image+language planning, then closed-loop LoRA BC control
(pure or hybrid) rendered in the interactive viewer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import mujoco.viewer
import numpy as np

from mise_en_place import config as C
from mise_en_place.control.scripted_t1 import ScriptedT1Expert
from mise_en_place.learning import load_policy
from mise_en_place.planner import plan_from_instruction
from mise_en_place.planner.smolvlm import SmolVLMPlanner, validate_t1_proposal
from mise_en_place.sim.env import DinnerTableEnv


class ViewerClosed(RuntimeError):
    """Raised when the user closes the live viewer during an episode."""


def make_torch_actor(checkpoint: Path) -> Callable[[np.ndarray], np.ndarray]:
    import torch

    model = load_policy(checkpoint)

    def act(observation: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            output = model(torch.from_numpy(observation[None, :]))
        return output[0].cpu().numpy()

    return act


def make_openvino_actor(model_path: Path, device: str) -> Callable[[np.ndarray], np.ndarray]:
    import openvino as ov

    compiled = ov.Core().compile_model(model_path, device, {"PERFORMANCE_HINT": "LATENCY"})
    input_port, output_port = compiled.input(0), compiled.output(0)

    def act(observation: np.ndarray) -> np.ndarray:
        result = compiled({input_port: observation[None, :].astype(np.float32)})
        return np.asarray(result[output_port])[0]

    return act


def resolve_actor(args: argparse.Namespace) -> Callable[[np.ndarray], np.ndarray]:
    if args.runtime == "torch":
        return make_torch_actor(args.checkpoint)
    suffix = "fp32" if args.runtime == "fp32" else "int8"
    model_path = C.REPO_ROOT / "checkpoints" / "openvino" / f"lora_bc_{suffix}.xml"
    return make_openvino_actor(model_path, args.device)


def run_vlm_planning(
    env: DinnerTableEnv,
    instruction: str,
    checkpoint: Path,
) -> dict:
    start_image = C.ARTIFACTS / "live_vla_start.png"
    imageio.imwrite(start_image, env.render("overhead"))
    planner = SmolVLMPlanner(checkpoint)
    proposal = planner.propose(instruction, start_image)
    present = {
        name
        for name, position in env.observe().objects.items()
        if float(position[2]) >= -0.01
    }
    validation_errors = validate_t1_proposal(proposal["parsed"], present_objects=present)
    accepted_plan = (
        json.loads(plan_from_instruction(instruction).to_json())
        if validation_errors
        else proposal["parsed"]
    )
    return {
        "instruction": instruction,
        "start_image": str(start_image),
        "vlm": {
            "model": planner.model_id,
            "raw": proposal["raw"],
            "parsed": proposal["parsed"],
            "inference_seconds": proposal["inference_seconds"],
        },
        "validation_errors": validation_errors,
        "fallback_used": bool(validation_errors),
        "accepted_plan": accepted_plan,
    }


def run_pure_learned(
    env: DinnerTableEnv,
    actor: Callable[[np.ndarray], np.ndarray],
    *,
    max_steps: int,
    ema: float,
    wall_step: float,
    viewer,
    original_step,
) -> dict:
    observation = env.observe()
    previous = observation.qpos.copy()
    max_delta = np.asarray([0.10] * 5 + [0.25] + [0.10] * 5 + [0.25])

    for step in range(max_steps):
        if not viewer.is_running():
            raise ViewerClosed
        started = time.perf_counter()
        proposed = actor(observation.flat())
        command = (1.0 - ema) * previous + ema * proposed
        command = previous + np.clip(command - previous, -max_delta, max_delta)
        observation = original_step(command)
        previous = command
        viewer.sync()
        remaining = wall_step - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)
        if step > 100 and env.score()["success"]:
            break

    env.settle()
    return env.score()


def run_hybrid_learned(
    env: DinnerTableEnv,
    actor: Callable[[np.ndarray], np.ndarray],
    *,
    blend: float,
    max_residual: float,
    wall_step: float,
    viewer,
    original_step,
) -> dict:
    joint_mask = np.ones(C.N_ACTION, dtype=bool)
    joint_mask[[5, 11]] = False
    expert = ScriptedT1Expert(env)

    def hybrid_step(requested, cameras=()):
        if not viewer.is_running():
            raise ViewerClosed
        started = time.perf_counter()
        requested = np.asarray(requested, dtype=np.float32)
        learned = actor(env.observe().flat())
        proposed = requested + blend * (learned - requested)
        command = requested.copy()
        command[joint_mask] = np.clip(
            proposed[joint_mask],
            requested[joint_mask] - max_residual,
            requested[joint_mask] + max_residual,
        )
        command[[5, 11]] = requested[[5, 11]]
        obs = original_step(command, cameras)
        viewer.sync()
        remaining = wall_step - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)
        return obs

    env.step = hybrid_step  # type: ignore[method-assign]
    try:
        score = expert.run_true_parallel_t1()
    finally:
        env.step = original_step  # type: ignore[method-assign]
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Live learned end-to-end VLA viewer")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--mode", choices=("pure", "hybrid"), default="pure")
    parser.add_argument("--runtime", choices=("torch", "fp32", "int8"), default="int8")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--blend", type=float, default=0.5, help="Hybrid learned authority")
    parser.add_argument("--max-residual", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--ema", type=float, default=0.45, help="Pure-mode action smoothing")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--instruction",
        default="Set the table: put the fork left of the plate and spoon right of the plate.",
    )
    parser.add_argument("--no-vlm", action="store_true", help="Skip SmolVLM planning stage")
    parser.add_argument(
        "--vlm-checkpoint",
        type=Path,
        default=C.REPO_ROOT / "checkpoints" / "smolvlm2-256m",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=C.REPO_ROOT / "checkpoints" / "lora_bc.pt",
    )
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")

    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    actor = resolve_actor(args)
    env = DinnerTableEnv()
    env.reset(seed=args.seed)
    original_step = env.step
    wall_step = (1.0 / C.CONTROL_HZ) / args.speed

    plan_report: dict | None = None
    if not args.no_vlm:
        print("Running SmolVLM image+language planning (this may take ~30-60s on CPU)...")
        plan_report = run_vlm_planning(env, args.instruction, args.vlm_checkpoint)
        print(json.dumps(plan_report, indent=2))

    print(
        f"Starting live viewer: mode={args.mode} runtime={args.runtime} "
        f"seed={args.seed} device={args.device}"
    )
    if args.mode == "pure":
        print("WARNING: pure learned BC typically fails closed-loop; watch the arms closely.")

    with mujoco.viewer.launch_passive(
        env.model,
        env.data,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        viewer.sync()
        try:
            if args.mode == "pure":
                score = run_pure_learned(
                    env,
                    actor,
                    max_steps=args.max_steps,
                    ema=args.ema,
                    wall_step=wall_step,
                    viewer=viewer,
                    original_step=original_step,
                )
            else:
                score = run_hybrid_learned(
                    env,
                    actor,
                    blend=args.blend,
                    max_residual=args.max_residual,
                    wall_step=wall_step,
                    viewer=viewer,
                    original_step=original_step,
                )
            final_image = C.ARTIFACTS / "live_vla_final.png"
            imageio.imwrite(final_image, env.render("overhead"))
            report = {
                "seed": args.seed,
                "mode": args.mode,
                "runtime": args.runtime,
                "device": args.device,
                "plan": plan_report,
                "score": score,
                "final_image": str(final_image),
            }
            output = C.ARTIFACTS / "live_vla_report.json"
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
            print(f"wrote {output}")
            print("Episode complete. Holding viewer open...")
            completed_at = time.perf_counter()
            while viewer.is_running() and time.perf_counter() - completed_at < args.hold_seconds:
                viewer.sync()
                time.sleep(0.05)
        except ViewerClosed:
            print("Viewer closed; episode stopped.")
        finally:
            env.close()


if __name__ == "__main__":
    main()
