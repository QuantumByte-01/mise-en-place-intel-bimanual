"""Compact behavior-cloning policy with trainable low-rank adapters.

This is an inexpensive local student policy for validating the complete
demonstrations -> LoRA -> OpenVINO INT8 pipeline.  It is deliberately not
presented as a replacement for a vision-language-action foundation model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PolicyConfig:
    obs_dim: int = 42
    action_dim: int = 12
    hidden_dim: int = 128
    lora_rank: int = 8
    lora_alpha: float = 16.0


class LoRALinear(nn.Module):
    """Linear layer plus a switchable low-rank residual update."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = float(alpha / rank)
        self.lora_enabled = True
        nn.init.kaiming_uniform_(self.lora_a, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        if self.lora_enabled:
            output = output + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale
        return output

    def set_lora_enabled(self, enabled: bool) -> None:
        self.lora_enabled = enabled

    def set_base_trainable(self, trainable: bool) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad = trainable

    def set_lora_trainable(self, trainable: bool) -> None:
        self.lora_a.requires_grad = trainable
        self.lora_b.requires_grad = trainable


class LoRABCPolicy(nn.Module):
    """Observation-to-12-DoF action student with LoRA on every projection."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolicyConfig()
        cfg = self.config
        self.fc1 = LoRALinear(cfg.obs_dim, cfg.hidden_dim, cfg.lora_rank, cfg.lora_alpha)
        self.fc2 = LoRALinear(cfg.hidden_dim, cfg.hidden_dim, cfg.lora_rank, cfg.lora_alpha)
        self.head = LoRALinear(cfg.hidden_dim, cfg.action_dim, cfg.lora_rank, cfg.lora_alpha)
        self.register_buffer("obs_mean", torch.zeros(cfg.obs_dim))
        self.register_buffer("obs_std", torch.ones(cfg.obs_dim))
        self.register_buffer("action_mean", torch.zeros(cfg.action_dim))
        self.register_buffer("action_std", torch.ones(cfg.action_dim))

    @property
    def lora_layers(self) -> tuple[LoRALinear, ...]:
        return (self.fc1, self.fc2, self.head)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.obs_mean) / self.obs_std
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        normalized_action = self.head(x)
        return normalized_action * self.action_std + self.action_mean

    def set_normalization(self, observations: np.ndarray, actions: np.ndarray) -> None:
        obs_mean = observations.mean(axis=0)
        obs_std = observations.std(axis=0)
        action_mean = actions.mean(axis=0)
        action_std = actions.std(axis=0)
        self.obs_mean.copy_(torch.as_tensor(obs_mean, dtype=torch.float32))
        self.obs_std.copy_(torch.as_tensor(np.maximum(obs_std, 1e-5), dtype=torch.float32))
        self.action_mean.copy_(torch.as_tensor(action_mean, dtype=torch.float32))
        self.action_std.copy_(torch.as_tensor(np.maximum(action_std, 1e-5), dtype=torch.float32))

    def base_training_mode(self) -> None:
        """Train the dense base while keeping the zero LoRA branch disabled."""
        for layer in self.lora_layers:
            layer.set_lora_enabled(False)
            layer.set_base_trainable(True)
            layer.set_lora_trainable(False)

    def lora_training_mode(self) -> None:
        """Freeze dense weights and train only low-rank adapters."""
        for layer in self.lora_layers:
            layer.set_lora_enabled(True)
            layer.set_base_trainable(False)
            layer.set_lora_trainable(True)

    def inference_mode(self, use_lora: bool = True) -> None:
        for layer in self.lora_layers:
            layer.set_lora_enabled(use_lora)
        self.requires_grad_(False)
        self.eval()

    def parameter_summary(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        lora = sum(
            layer.lora_a.numel() + layer.lora_b.numel() for layer in self.lora_layers
        )
        return {"total": total, "lora": lora, "lora_percent_x100": round(10000 * lora / total)}


def save_policy(
    path: Path,
    model: LoRABCPolicy,
    *,
    metrics: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "metrics": metrics or {},
        },
        path,
    )


def load_policy(path: Path, device: str = "cpu", use_lora: bool = True) -> LoRABCPolicy:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = LoRABCPolicy(PolicyConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.inference_mode(use_lora=use_lora)
    return model
