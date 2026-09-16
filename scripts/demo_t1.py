"""Run scripted T1 expert for one seed; save frames + score JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio

from mise_en_place import config as C
from mise_en_place.control.scripted_t1 import ScriptedT1Expert
from mise_en_place.planner import plan_from_instruction
from mise_en_place.sim.env import DinnerTableEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instruction", type=str, default="Set the table: place the fork and spoon beside the plate.")
    args = ap.parse_args()

    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    plan = plan_from_instruction(args.instruction)
    print(plan.to_json())

    env = DinnerTableEnv()
    env.reset(seed=args.seed)
    frames = [env.render("demo")]
    expert = ScriptedT1Expert(env)
    score = expert.run_parallel_t1()
    frames.append(env.render("demo"))
    env.close()

    out_dir = C.ARTIFACTS / f"t1_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames):
        imageio.imwrite(out_dir / f"frame_{i:02d}.png", fr)
    (out_dir / "score.json").write_text(json.dumps(score, indent=2), encoding="utf-8")
    (out_dir / "plan.json").write_text(plan.to_json(), encoding="utf-8")
    print(json.dumps(score, indent=2))
    print(f"artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
