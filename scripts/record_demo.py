"""Record one T1 or T2 episode to MP4 and submission keyframes."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from mise_en_place import config as C
from mise_en_place.control.scripted_t1 import ScriptedT1Expert
from mise_en_place.control.scripted_t2 import ScriptedT2Handoff
from mise_en_place.sim.env import DinnerTableEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("t1", "t2"), default="t1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--every", type=int, default=4, help="Capture every N control steps")
    args = ap.parse_args()

    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = C.ARTIFACTS / f"demo_{args.task}.mp4"

    env = DinnerTableEnv()
    env.reset(seed=args.seed)
    frames: list[np.ndarray] = [env.render("demo")]
    start_overhead = env.render("overhead")

    expert = ScriptedT1Expert(env) if args.task == "t1" else ScriptedT2Handoff(env)
    # monkey-patch step to sample frames
    orig_step = env.step

    def stepped(action, cameras=()):
        obs = orig_step(action, cameras)
        if env.step_count % args.every == 0:
            frames.append(env.render("demo"))
        return obs

    env.step = stepped  # type: ignore[method-assign]
    score = expert.run_true_parallel_t1() if args.task == "t1" else expert.run_t2()
    frames.append(env.render("demo"))
    final_overhead = env.render("overhead")
    env.close()

    keyframe_dir = C.ARTIFACTS / f"demo_{args.task}_keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(frames) - 1, 7, dtype=int)
    for number, index in enumerate(indices):
        imageio.imwrite(keyframe_dir / f"{number:02d}_frame_{index:04d}.png", frames[index])
    imageio.imwrite(keyframe_dir / "overhead_start.png", start_overhead)
    imageio.imwrite(keyframe_dir / "overhead_final.png", final_overhead)

    try:
        imageio.mimsave(out, frames, fps=args.fps)
        print(f"wrote {out} ({len(frames)} frames)")
    except Exception as e:  # noqa: BLE001
        # fallback: PNG strip
        strip = C.ARTIFACTS / f"demo_{args.task}_frames"
        strip.mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(frames[:: max(1, len(frames) // 12)]):
            imageio.imwrite(strip / f"f_{i:03d}.png", fr)
        print(f"mp4 failed ({type(e).__name__}: {e}); wrote PNG strip to {strip}")
    print(f"keyframes -> {keyframe_dir}")
    print(score)


if __name__ == "__main__":
    main()
