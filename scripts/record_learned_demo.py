"""Record MP4 + keyframes for hybrid or pure learned policy execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import numpy as np

from mise_en_place import config as C
from mise_en_place.control.scripted_t1 import ScriptedT1Expert
from mise_en_place.learning import load_policy
from mise_en_place.sim.env import DinnerTableEnv


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--mode", choices=("hybrid", "pure"), default="hybrid")
    parser.add_argument("--runtime", choices=("torch", "fp32", "int8"), default="int8")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--blend", type=float, default=0.5)
    parser.add_argument("--max-residual", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--ema", type=float, default=0.45)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--every", type=int, default=4)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=C.REPO_ROOT / "checkpoints" / "lora_bc.pt",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.runtime == "torch":
        actor = make_torch_actor(args.checkpoint)
    else:
        suffix = "fp32" if args.runtime == "fp32" else "int8"
        model_path = C.REPO_ROOT / "checkpoints" / "openvino" / f"lora_bc_{suffix}.xml"
        actor = make_openvino_actor(model_path, args.device)

    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tag = f"learned_{args.mode}_{args.runtime}_seed{args.seed}"
    out = args.output or C.ARTIFACTS / f"{tag}.mp4"
    keyframe_dir = C.ARTIFACTS / f"{tag}_keyframes"

    env = DinnerTableEnv()
    env.reset(seed=args.seed)
    frames: list[np.ndarray] = [env.render("demo")]
    start_overhead = env.render("overhead")
    original_step = env.step

    def capture_step(action, cameras=()):
        obs = original_step(action, cameras)
        if env.step_count % args.every == 0:
            frames.append(env.render("demo"))
        return obs

    env.step = capture_step  # type: ignore[method-assign]

    if args.mode == "hybrid":
        joint_mask = np.ones(C.N_ACTION, dtype=bool)
        joint_mask[[5, 11]] = False
        expert = ScriptedT1Expert(env)

        def hybrid_step(requested, cameras=()):
            requested = np.asarray(requested, dtype=np.float32)
            learned = actor(env.observe().flat())
            proposed = requested + args.blend * (learned - requested)
            command = requested.copy()
            command[joint_mask] = np.clip(
                proposed[joint_mask],
                requested[joint_mask] - args.max_residual,
                requested[joint_mask] + args.max_residual,
            )
            command[[5, 11]] = requested[[5, 11]]
            return capture_step(command, cameras)

        env.step = hybrid_step  # type: ignore[method-assign]
        score = expert.run_true_parallel_t1()
        env.step = capture_step  # type: ignore[method-assign]
    else:
        observation = env.observe()
        previous = observation.qpos.copy()
        max_delta = np.asarray([0.10] * 5 + [0.25] + [0.10] * 5 + [0.25])
        for step in range(args.max_steps):
            proposed = actor(observation.flat())
            command = (1.0 - args.ema) * previous + args.ema * proposed
            command = previous + np.clip(command - previous, -max_delta, max_delta)
            observation = capture_step(command)
            previous = command
            if step > 100 and env.score()["success"]:
                break
        env.settle()
        score = env.score()

    frames.append(env.render("demo"))
    final_overhead = env.render("overhead")
    env.close()

    keyframe_dir.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(frames) - 1, 7, dtype=int)
    for number, index in enumerate(indices):
        imageio.imwrite(keyframe_dir / f"{number:02d}_frame_{index:04d}.png", frames[index])
    imageio.imwrite(keyframe_dir / "overhead_start.png", start_overhead)
    imageio.imwrite(keyframe_dir / "overhead_final.png", final_overhead)

    try:
        imageio.mimsave(out, frames, fps=args.fps)
        print(f"wrote {out} ({len(frames)} frames)")
    except Exception as exc:  # noqa: BLE001
        strip = C.ARTIFACTS / f"{tag}_frames"
        strip.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames[:: max(1, len(frames) // 12)]):
            imageio.imwrite(strip / f"f_{index:03d}.png", frame)
        print(f"mp4 failed ({type(exc).__name__}: {exc}); wrote PNG strip to {strip}")

    report = {
        "seed": args.seed,
        "mode": args.mode,
        "runtime": args.runtime,
        "blend": args.blend,
        "score": score,
        "video": str(out),
        "keyframes": str(keyframe_dir),
    }
    report_path = out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
