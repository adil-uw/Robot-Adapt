"""Teacher policies and action-distribution extraction (Phase 7).

A teacher exposes two things the distillation pipeline needs:

1. ``predict(obs)`` -- an action used to *roll out* and collect states.
2. ``action_distribution(obs)`` -- the teacher's action distribution at a
   state, returned as ``(mean, std)`` tensors describing a diagonal Gaussian.
   This is the "Action Distribution Extraction" box in the proposal.

Two implementations are provided:

* :class:`SB3TeacherPolicy` wraps a trained Stable-Baselines3 PPO/SAC model
  (the real teachers produced in Phase 5).
* :class:`MockTeacher` is a fixed, randomly-initialised network that produces a
  valid Gaussian, letting the whole pipeline run and be tested before any real
  teacher exists.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

# Numerical floor for the standard deviation to keep KL/log-prob finite.
MIN_STD = 1e-4


class TeacherPolicy(ABC):
    """Common interface for anything that can teach the student."""

    action_dim: int

    @abstractmethod
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return an action for environment rollouts (used to collect states)."""

    @abstractmethod
    def action_distribution(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, std)`` of the teacher's diagonal Gaussian for ``obs``.

        ``obs`` has shape ``(batch, obs_dim)``; outputs have shape
        ``(batch, action_dim)``.
        """


class SB3TeacherPolicy(TeacherPolicy):
    """Wrap a trained Stable-Baselines3 model as a distillation teacher."""

    def __init__(self, model_path: str, algo: str = "PPO", device: str = "cpu"):
        algo = algo.upper()
        if algo == "PPO":
            from stable_baselines3 import PPO as Algo
        elif algo == "SAC":
            from stable_baselines3 import SAC as Algo
        else:
            raise ValueError(f"Unsupported algo '{algo}'. Use 'PPO' or 'SAC'.")

        self.algo = algo
        self.model = Algo.load(model_path, device=device)
        self.device = device
        self.action_dim = int(np.prod(self.model.action_space.shape))

    def predict(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def action_distribution(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = obs.to(self.device)
        with torch.no_grad():
            if self.algo == "PPO":
                # PPO uses a diagonal Gaussian policy we can read directly.
                dist = self.model.policy.get_distribution(obs).distribution
                mean = dist.mean
                std = dist.stddev
            else:
                # SAC: read the pre-squash Gaussian parameters from the actor.
                mean, log_std, _ = self.model.actor.get_action_dist_params(obs)
                std = log_std.exp()
        return mean, std.clamp_min(MIN_STD)


class MockTeacher(TeacherPolicy):
    """A fixed random network producing a valid Gaussian over actions.

    Useful for developing and testing the distillation pipeline before the real
    teachers from Phase 5 are available. Outputs are deterministic given the
    seed, so distillation runs are reproducible.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: int = 64,
        std: float = 0.3,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.device = device
        self._fixed_std = std

        gen = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        # Initialise deterministically from the seeded generator, then freeze.
        for param in self.net.parameters():
            param.data = torch.empty_like(param).uniform_(-0.5, 0.5, generator=gen)
            param.requires_grad_(False)
        self.net.to(device)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            mean = torch.tanh(self.net(obs_t))  # keep actions in [-1, 1]
        return mean.squeeze(0).cpu().numpy()

    def action_distribution(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = obs.to(self.device)
        with torch.no_grad():
            mean = self.net(obs)
        std = torch.full_like(mean, self._fixed_std).clamp_min(MIN_STD)
        return mean, std
