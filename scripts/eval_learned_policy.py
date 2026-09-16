"""Closed-loop MuJoCo evaluation for Torch/OpenVINO learned policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from mise_en_place import config as C
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
    input_port = compiled.input(0)
    output_port = compiled.output(0)

    def act(observation: np.ndarray) -> np.ndarray:
        result = compiled({input_port: observation[None, :].astype(np.float32)})
        return np.asarray(result[output_port])[0]

    return act


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=("torch", "fp32", "int8"), default="int8")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--ema", type=float, default=0.45)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=C.REPO_ROOT / "checkpoints" / "lora_bc.pt",
    )
    args = parser.parse_args()

    if args.runtime == "torch":
        actor = make_torch_actor(args.checkpoint)
    else:
        suffix = "fp32" if args.runtime == "fp32" else "int8"
        model_path = C.REPO_ROOT / "checkpoints" / "openvino" / f"lora_bc_{suffix}.xml"
        actor = make_openvino_actor(model_path, args.device)

    env = DinnerTableEnv()
    rows = []
    for episode in range(args.episodes):
        seed = args.start_seed + episode
        observation = env.reset(seed=seed)
        previous = observation.qpos.copy()
        for step in range(args.max_steps):
            proposed = actor(observation.flat())
            command = (1.0 - args.ema) * previous + args.ema * proposed
            # A policy error cannot command a single-step full-range jump.
            max_delta = np.asarray([0.10] * 5 + [0.25] + [0.10] * 5 + [0.25])
            command = previous + np.clip(command - previous, -max_delta, max_delta)
            observation = env.step(command)
            previous = command
            if step > 100 and env.score()["success"]:
                break
        env.settle()
        score = env.score()
        row = {"seed": seed, **score}
        rows.append(row)
        print(
            f"runtime={args.runtime} seed={seed} success={int(score['success'])} "
            f"fork={int(score['per_object']['fork'])} "
            f"spoon={int(score['per_object']['spoon'])} steps={score['steps']}"
        )
    env.close()

    summary = {
        "runtime": args.runtime,
        "device": args.device if args.runtime != "torch" else "PyTorch CPU",
        "episodes": args.episodes,
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "fork_rate": float(np.mean([row["per_object"]["fork"] for row in rows])),
        "spoon_rate": float(np.mean([row["per_object"]["spoon"] for row in rows])),
        "rows": rows,
        "warning": (
            "Pure behavior cloning is evaluated without scripted rescue. "
            "The 10/10 scripted controller remains the deployment fallback."
        ),
    }
    output = C.ARTIFACTS / f"learned_eval_{args.runtime}_{args.device.lower()}.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
