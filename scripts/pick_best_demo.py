"""Record multiple demo candidates and rank by task success + placement error."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from mise_en_place import config as C
from mise_en_place.control.scripted_t1 import ScriptedT1Expert
from mise_en_place.control.scripted_t2 import ScriptedT2Handoff
from mise_en_place.sim.env import DinnerTableEnv


def record_candidate(
    task: str,
    seed: int,
    *,
    fps: int = 24,
    every: int = 3,
) -> dict:
    tag = f"candidate_{task}_seed{seed}"
    out_dir = C.ARTIFACTS / "demo_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / f"{tag}.mp4"

    env = DinnerTableEnv()
    env.reset(seed=seed)
    frames: list[np.ndarray] = [env.render("demo")]
    start_overhead = env.render("overhead")
    original_step = env.step

    def stepped(action, cameras=()):
        obs = original_step(action, cameras)
        if env.step_count % every == 0:
            frames.append(env.render("demo"))
        return obs

    env.step = stepped  # type: ignore[method-assign]
    expert = ScriptedT1Expert(env) if task == "t1" else ScriptedT2Handoff(env)
    score = expert.run_true_parallel_t1() if task == "t1" else expert.run_t2()
    frames.append(env.render("demo"))
    final_overhead = env.render("overhead")
    env.close()

    imageio.imwrite(out_dir / f"{tag}_start.png", start_overhead)
    imageio.imwrite(out_dir / f"{tag}_final.png", final_overhead)
    imageio.imwrite(out_dir / f"{tag}_mid.png", frames[len(frames) // 2])
    imageio.mimsave(mp4, frames, fps=fps, macro_block_size=1)

    if task == "t1":
        err = float(score["errors"]["fork"] + score["errors"]["spoon"])
        ok = bool(score["success"])
    else:
        err = float(score.get("place_error", 1.0))
        ok = bool(score["success"])

    return {
        "task": task,
        "seed": seed,
        "success": ok,
        "error_sum": err,
        "frames": len(frames),
        "fps": fps,
        "mp4": str(mp4),
        "start_png": str(out_dir / f"{tag}_start.png"),
        "final_png": str(out_dir / f"{tag}_final.png"),
        "mid_png": str(out_dir / f"{tag}_mid.png"),
        "score": score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("t1", "t2"), default="t1")
    parser.add_argument("--seeds", default="0,3,7,12,42,100")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--every", type=int, default=3)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    rows = []
    for seed in seeds:
        row = record_candidate(args.task, seed, fps=args.fps, every=args.every)
        rows.append(row)
        print(
            f"{args.task} seed={seed} success={int(row['success'])} "
            f"error={row['error_sum']:.4f} frames={row['frames']}"
        )

    successes = [r for r in rows if r["success"]]
    pool = successes if successes else rows
    best = min(pool, key=lambda r: r["error_sum"])
    best_dir = C.ARTIFACTS / "demo_candidates"
    dest_mp4 = C.ARTIFACTS / f"demo_{args.task}_best.mp4"
    shutil.copy2(best["mp4"], dest_mp4)
    shutil.copy2(best["final_png"], C.ARTIFACTS / f"demo_{args.task}_best_final.png")

    report = {
        "task": args.task,
        "best_seed": best["seed"],
        "best": best,
        "all": rows,
        "promoted_to": str(dest_mp4),
    }
    out = best_dir / f"best_{args.task}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best_seed": best["seed"], "promoted_to": str(dest_mp4)}, indent=2))


if __name__ == "__main__":
    main()
