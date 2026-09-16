"""Record randomized scripted-expert trajectories for behavior cloning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mise_en_place import config as C
from mise_en_place.control.scripted_t1 import ScriptedT1Expert
from mise_en_place.sim.env import DinnerTableEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth control sample")
    parser.add_argument(
        "--output",
        type=Path,
        default=C.ARTIFACTS / "expert_demos.npz",
    )
    args = parser.parse_args()
    if args.episodes < 1 or args.stride < 1:
        parser.error("--episodes and --stride must be positive")

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_ids: list[int] = []
    sample_seeds: list[int] = []
    summaries: list[dict] = []

    env = DinnerTableEnv()
    for episode_id in range(args.episodes):
        seed = args.start_seed + episode_id
        env.reset(seed=seed)
        expert = ScriptedT1Expert(env)
        original_step = env.step

        def recording_step(action, cameras=()):
            if env.step_count % args.stride == 0:
                observations.append(env.observe().flat())
                actions.append(np.asarray(action, dtype=np.float32).copy())
                episode_ids.append(episode_id)
                sample_seeds.append(seed)
            return original_step(action, cameras)

        env.step = recording_step  # type: ignore[method-assign]
        score = expert.run_true_parallel_t1()
        env.step = original_step  # type: ignore[method-assign]
        summary = {"episode": episode_id, "seed": seed, **score}
        summaries.append(summary)
        print(
            f"episode={episode_id:03d} seed={seed:03d} "
            f"success={int(score['success'])} samples={len(observations)}"
        )
    env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
        seeds=np.asarray(sample_seeds, dtype=np.int32),
    )
    success_rate = float(np.mean([row["success"] for row in summaries]))
    metadata = {
        "episodes": args.episodes,
        "samples": len(observations),
        "observation_dim": int(observations[0].shape[0]),
        "action_dim": int(actions[0].shape[0]),
        "stride": args.stride,
        "expert_success_rate": success_rate,
        "episode_scores": summaries,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"expert_success_rate={success_rate:.3f}")


if __name__ == "__main__":
    main()
