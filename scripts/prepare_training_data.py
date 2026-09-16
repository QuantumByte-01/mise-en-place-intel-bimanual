"""Build an enlarged BC dataset: local expert demos + augmentation + optional HF priors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mise_en_place import config as C


def augment_batch(
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    copies: int,
    obs_noise: float,
    action_noise: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if copies <= 0:
        return observations, actions
    obs_blocks = [observations]
    act_blocks = [actions]
    for _ in range(copies):
        noise_obs = observations + rng.normal(0.0, obs_noise, observations.shape).astype(np.float32)
        noise_act = actions + rng.normal(0.0, action_noise, actions.shape).astype(np.float32)
        obs_blocks.append(noise_obs)
        act_blocks.append(noise_act)
    return np.concatenate(obs_blocks, axis=0), np.concatenate(act_blocks, axis=0)


def try_hf_supplementary(
    target_obs_dim: int,
    target_action_dim: int,
    max_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pull open-source robot BC rows from Hugging Face when dimensions match."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed; skipping HF supplementary data")
        return None

    candidates = [
        ("lerobot/aloha_static_coffee", "observation.state", "action"),
        ("lerobot/pusht", "observation.state", "action"),
    ]
    for repo, obs_key, act_key in candidates:
        try:
            print(f"trying HF dataset {repo} ...")
            rows = load_dataset(repo, split="train", streaming=True)
            obs_list: list[np.ndarray] = []
            act_list: list[np.ndarray] = []
            for row in rows:
                obs = np.asarray(row[obs_key], dtype=np.float32).reshape(-1)
                act = np.asarray(row[act_key], dtype=np.float32).reshape(-1)
                if obs.shape[0] != target_obs_dim or act.shape[0] != target_action_dim:
                    continue
                obs_list.append(obs)
                act_list.append(act)
                if len(obs_list) >= max_samples:
                    break
            if obs_list:
                print(f"loaded {len(obs_list)} HF rows from {repo}")
                return np.stack(obs_list), np.stack(act_list)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {repo}: {type(exc).__name__}: {exc}")
    print("no compatible HF dataset found; using local demos + augmentation only")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=C.ARTIFACTS / "expert_demos.npz")
    parser.add_argument("--output", type=Path, default=C.ARTIFACTS / "training_data.npz")
    parser.add_argument("--augment-copies", type=int, default=2)
    parser.add_argument("--obs-noise", type=float, default=0.015)
    parser.add_argument("--action-noise", type=float, default=0.002)
    parser.add_argument("--hf-samples", type=int, default=0, help="Max HF rows if compatible")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"missing {args.input}; run record_demos.py first")

    data = np.load(args.input)
    observations = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    episode_ids = data["episode_ids"]
    seeds = data["seeds"] if "seeds" in data else np.zeros(len(observations), dtype=np.int32)

    metadata_path = args.input.with_suffix(".json")
    failed: list[int] = []
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        failed = [
            int(row["episode"])
            for row in metadata.get("episode_scores", [])
            if not row.get("success", False)
        ]
        if failed:
            keep = ~np.isin(episode_ids, np.asarray(failed))
            observations = observations[keep]
            actions = actions[keep]
            episode_ids = episode_ids[keep]
            seeds = seeds[keep]
            print(f"excluded failed expert episodes: {failed}")

    rng = np.random.default_rng(args.seed)
    observations, actions = augment_batch(
        observations,
        actions,
        copies=args.augment_copies,
        obs_noise=args.obs_noise,
        action_noise=args.action_noise,
        rng=rng,
    )
    episode_ids = np.tile(episode_ids, args.augment_copies + 1)
    seeds = np.tile(seeds, args.augment_copies + 1)

    hf_obs, hf_act = None, None
    if args.hf_samples > 0:
        hf_pair = try_hf_supplementary(
            observations.shape[1],
            actions.shape[1],
            args.hf_samples,
            rng,
        )
        if hf_pair is not None:
            hf_obs, hf_act = hf_pair
            observations = np.concatenate([observations, hf_obs], axis=0)
            actions = np.concatenate([actions, hf_act], axis=0)
            hf_ids = -np.ones(len(hf_obs), dtype=np.int32)
            episode_ids = np.concatenate([episode_ids, hf_ids])
            seeds = np.concatenate([seeds, np.full(len(hf_obs), -1, dtype=np.int32)])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        observations=observations,
        actions=actions,
        episode_ids=episode_ids,
        seeds=seeds,
    )
    report = {
        "source": str(args.input),
        "output": str(args.output),
        "samples": int(len(observations)),
        "observation_dim": int(observations.shape[1]),
        "action_dim": int(actions.shape[1]),
        "augment_copies": args.augment_copies,
        "hf_rows": int(len(hf_obs)) if hf_obs is not None else 0,
        "excluded_failed_episodes": failed,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
