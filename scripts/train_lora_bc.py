"""Train a dense BC base, then freeze it and adapt only LoRA parameters."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from mise_en_place import config as C
from mise_en_place.learning import LoRABCPolicy, PolicyConfig, save_policy


def select_episodes(ids: np.ndarray, selected: np.ndarray) -> np.ndarray:
    return np.isin(ids, selected)


def train_stage(
    model: LoRABCPolicy,
    observations: torch.Tensor,
    actions: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    label: str,
    obs_noise: float = 0.0,
) -> list[float]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    history: list[float] = []
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(observations))
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            batch_obs = observations[batch]
            if obs_noise > 0:
                batch_obs = batch_obs + torch.randn_like(batch_obs) * obs_noise
            prediction = model(batch_obs)
            normalized_error = (prediction - actions[batch]) / model.action_std
            loss = torch.mean(normalized_error.square())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total += float(loss) * len(batch)
        scheduler.step()
        value = total / len(order)
        history.append(value)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(f"{label} epoch={epoch + 1:03d}/{epochs} normalized_mse={value:.6f}")
    return history


def joint_training_mode(model: LoRABCPolicy) -> None:
    """Unfreeze dense base and LoRA for end-to-end fine-tuning."""
    for layer in model.lora_layers:
        layer.set_lora_enabled(True)
        layer.set_base_trainable(True)
        layer.set_lora_trainable(True)


@torch.inference_mode()
def evaluate(
    model: LoRABCPolicy,
    observations: torch.Tensor,
    actions: torch.Tensor,
    *,
    use_lora: bool,
) -> dict[str, float]:
    for layer in model.lora_layers:
        layer.set_lora_enabled(use_lora)
    model.eval()
    prediction = model(observations)
    error = prediction - actions
    normalized = error / model.action_std
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "normalized_rmse": float(normalized.square().mean().sqrt()),
        "max_abs": float(error.abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=C.ARTIFACTS / "training_data.npz")
    parser.add_argument(
        "--fallback-data",
        type=Path,
        default=C.ARTIFACTS / "expert_demos.npz",
        help="Used when --data is missing",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=C.REPO_ROOT / "checkpoints" / "lora_bc.pt",
    )
    parser.add_argument(
        "--mode",
        choices=("lora", "full", "competition"),
        default="competition",
        help="lora=base+LoRA only; full=train all weights once; competition=base+LoRA+joint",
    )
    parser.add_argument("--base-epochs", type=int, default=60)
    parser.add_argument("--lora-epochs", type=int, default=40)
    parser.add_argument("--joint-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--base-lr", type=float, default=3e-3)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--joint-lr", type=float, default=5e-5)
    parser.add_argument("--obs-noise", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    data_path = args.data if args.data.exists() else args.fallback_data
    if not data_path.exists():
        raise FileNotFoundError(f"missing training data: {args.data} and {args.fallback_data}")
    print(f"loading {data_path}")
    data = np.load(data_path)
    observations_np = data["observations"].astype(np.float32)
    actions_np = data["actions"].astype(np.float32)
    episode_ids = data["episode_ids"]
    metadata_path = args.data.with_suffix(".json")
    failed_episodes: list[int] = []
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        failed_episodes = [
            int(row["episode"])
            for row in metadata.get("episode_scores", [])
            if not row.get("success", False)
        ]
        if failed_episodes:
            keep = ~np.isin(episode_ids, np.asarray(failed_episodes))
            observations_np = observations_np[keep]
            actions_np = actions_np[keep]
            episode_ids = episode_ids[keep]
            print(f"excluded failed expert episodes: {failed_episodes}")
    local_episodes = np.unique(episode_ids[episode_ids >= 0])
    if len(local_episodes) < 5:
        raise ValueError("Need at least 5 recorded episodes for base/adapt/holdout splits")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(local_episodes)
    n_holdout = max(1, round(0.15 * len(local_episodes)))
    n_adapt = max(1, round(0.15 * len(local_episodes)))
    holdout_episodes = local_episodes[:n_holdout]
    adapt_episodes = local_episodes[n_holdout : n_holdout + n_adapt]
    base_episodes = local_episodes[n_holdout + n_adapt :]

    base_mask = select_episodes(episode_ids, base_episodes)
    adapt_mask = select_episodes(episode_ids, adapt_episodes)
    holdout_mask = select_episodes(episode_ids, holdout_episodes)
    train_mask = base_mask | adapt_mask
    all_mask = np.ones(len(episode_ids), dtype=bool)

    config = PolicyConfig(
        obs_dim=observations_np.shape[1],
        action_dim=actions_np.shape[1],
        hidden_dim=args.hidden_dim,
        lora_rank=args.rank,
    )
    model = LoRABCPolicy(config)
    norm_mask = base_mask if np.any(base_mask) else all_mask
    model.set_normalization(observations_np[norm_mask], actions_np[norm_mask])

    observations = torch.from_numpy(observations_np)
    actions = torch.from_numpy(actions_np)
    base_history: list[float] = []
    lora_history: list[float] = []
    joint_history: list[float] = []
    base_holdout: dict[str, float] = {}
    lora_holdout: dict[str, float] = {}
    joint_holdout: dict[str, float] = {}

    if args.mode == "full":
        model.base_training_mode()
        for layer in model.lora_layers:
            layer.set_lora_enabled(True)
            layer.set_lora_trainable(True)
        base_history = train_stage(
            model,
            observations[all_mask],
            actions[all_mask],
            epochs=args.base_epochs,
            batch_size=args.batch_size,
            learning_rate=args.base_lr,
            label="full",
            obs_noise=args.obs_noise,
        )
        base_holdout = evaluate(model, observations[holdout_mask], actions[holdout_mask], use_lora=True)
        lora_holdout = base_holdout
    else:
        model.base_training_mode()
        base_history = train_stage(
            model,
            observations[base_mask],
            actions[base_mask],
            epochs=args.base_epochs,
            batch_size=args.batch_size,
            learning_rate=args.base_lr,
            label="base",
            obs_noise=args.obs_noise,
        )
        base_holdout = evaluate(
            model,
            observations[holdout_mask],
            actions[holdout_mask],
            use_lora=False,
        )

        model.lora_training_mode()
        lora_history = train_stage(
            model,
            observations[adapt_mask],
            actions[adapt_mask],
            epochs=args.lora_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lora_lr,
            label="lora",
            obs_noise=args.obs_noise,
        )
        lora_holdout = evaluate(
            model,
            observations[holdout_mask],
            actions[holdout_mask],
            use_lora=True,
        )

        if args.mode == "competition":
            joint_training_mode(model)
            joint_history = train_stage(
                model,
                observations[train_mask],
                actions[train_mask],
                epochs=args.joint_epochs,
                batch_size=args.batch_size,
                learning_rate=args.joint_lr,
                label="joint",
                obs_noise=args.obs_noise,
            )
            joint_holdout = evaluate(
                model,
                observations[holdout_mask],
                actions[holdout_mask],
                use_lora=True,
            )

    model.inference_mode(use_lora=True)

    metrics = {
        "mode": args.mode,
        "data_path": str(data_path),
        "split": {
            "base_episodes": base_episodes.tolist(),
            "adapt_episodes": adapt_episodes.tolist(),
            "holdout_episodes": holdout_episodes.tolist(),
        },
        "base_holdout": base_holdout,
        "lora_holdout": lora_holdout,
        "joint_holdout": joint_holdout,
        "base_final_normalized_mse": base_history[-1] if base_history else None,
        "lora_final_normalized_mse": lora_history[-1] if lora_history else None,
        "joint_final_normalized_mse": joint_history[-1] if joint_history else None,
        "parameters": model.parameter_summary(),
        "excluded_failed_expert_episodes": failed_episodes,
        "claim": (
            "Competition-tuned proprioceptive BC student with base + LoRA + joint fine-tuning. "
            "Hybrid deployment with geometric executive remains the reliable closed-loop path."
        ),
    }
    save_policy(args.output, model, metrics=metrics)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
