"""Policy interface + stubs for ACT / SmolVLA (load checkpoint if present)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..sim.env import Obs


class Policy(ABC):
    @abstractmethod
    def act(self, obs: Obs) -> np.ndarray:
        """Return 12-D joint position targets."""


class ScriptedExpertPolicy(Policy):
    """Thin wrapper: not step-wise; use ScriptedT1Expert.run_parallel_t1 instead."""

    def __init__(self, expert) -> None:
        self.expert = expert

    def act(self, obs: Obs) -> np.ndarray:
        raise NotImplementedError("Use ScriptedT1Expert.run_parallel_t1() for full episode.")


class ACTPolicy(Policy):
    def __init__(self, checkpoint: Path | None = None) -> None:
        self.checkpoint = checkpoint
        self.ready = checkpoint is not None and Path(checkpoint).exists()

    def act(self, obs: Obs) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("ACT checkpoint missing — train or place weights under checkpoints/")
        raise NotImplementedError("ACT forward pass wired after training export")


class SmolVLAPolicy(Policy):
    def __init__(self, checkpoint: Path | None = None) -> None:
        self.checkpoint = checkpoint
        self.ready = checkpoint is not None and Path(checkpoint).exists()

    def act(self, obs: Obs) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("SmolVLA checkpoint missing")
        raise NotImplementedError("SmolVLA inference after OpenVINO / torch export")
