"""Student policy network for distillation (Phase 7).

The student is a small MLP that maps observations to the parameters of a
diagonal Gaussian over actions: a state-dependent mean and a state-independent
(learned) log-std, mirroring the structure of an on-policy actor. Distillation
trains this Gaussian to match the teacher's via a KL-divergence loss.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

MIN_LOG_STD = -5.0
MAX_LOG_STD = 2.0


class StudentPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        # State-independent log-std, initialised to a moderate exploration level.
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, std)`` of the student's diagonal Gaussian."""
        features = self.body(obs)
        mean = self.mean_head(features)
        log_std = self.log_std.clamp(MIN_LOG_STD, MAX_LOG_STD)
        std = log_std.exp().expand_as(mean)
        return mean, std

    @torch.no_grad()
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return an action for environment interaction / evaluation.

        Actions are squashed with ``tanh`` into ``[-1, 1]`` to match the Ant
        action bounds. Signature matches the evaluation framework's policy_fn.
        """
        was_1d = np.ndim(obs) == 1
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        if was_1d:
            obs_t = obs_t.unsqueeze(0)

        mean, std = self.forward(obs_t)
        raw = mean if deterministic else torch.normal(mean, std)
        action = torch.tanh(raw)

        action_np = action.squeeze(0).cpu().numpy() if was_1d else action.cpu().numpy()
        return action_np.astype(np.float32)
