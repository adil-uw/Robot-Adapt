"""State collection for policy distillation (Phase 7).

We roll out each teacher in its own environment to gather the *states* the
student must imitate, then pre-compute each teacher's action distribution at
those states. Pre-computing the targets decouples the (expensive) teacher from
the student training loop and lets distillation run as straightforward
supervised learning on ``(obs, target_mean, target_std)`` tuples.
"""

from __future__ import annotations

from typing import Callable, List, Sequence

import gymnasium as gym
import numpy as np
import torch

from training.distillation.teachers import TeacherPolicy

EnvFactory = Callable[[], gym.Env]


def _rollout_states(env: gym.Env, teacher: TeacherPolicy, num_steps: int, seed: int) -> np.ndarray:
    """Collect ``num_steps`` observations by running ``teacher`` in ``env``."""

    observations: List[np.ndarray] = []
    obs, _ = env.reset(seed=seed)
    for _ in range(num_steps):
        observations.append(np.asarray(obs, dtype=np.float32))
        action = teacher.predict(obs)
        obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    return np.asarray(observations, dtype=np.float32)


def collect_dataset(
    teachers: Sequence[TeacherPolicy],
    env_factories: Sequence[EnvFactory],
    steps_per_teacher: int = 5_000,
    batch_size: int = 512,
    seed: int = 0,
    device: str = "cpu",
):
    """Build the distillation dataset.

    Returns three tensors on ``device``: observations ``(N, obs_dim)`` and the
    teacher target Gaussian parameters ``mean`` / ``std`` ``(N, action_dim)``.
    """

    if len(teachers) != len(env_factories):
        raise ValueError("Each teacher needs a matching environment factory.")

    all_obs: List[torch.Tensor] = []
    all_mean: List[torch.Tensor] = []
    all_std: List[torch.Tensor] = []

    for idx, (teacher, make_env) in enumerate(zip(teachers, env_factories)):
        env = make_env()
        obs_np = _rollout_states(env, teacher, steps_per_teacher, seed=seed + idx)
        env.close()

        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

        # Extract teacher action distributions in mini-batches.
        means, stds = [], []
        for start in range(0, len(obs_t), batch_size):
            chunk = obs_t[start : start + batch_size]
            mean, std = teacher.action_distribution(chunk)
            means.append(mean.detach())
            stds.append(std.detach())

        all_obs.append(obs_t)
        all_mean.append(torch.cat(means, dim=0))
        all_std.append(torch.cat(stds, dim=0))

    obs = torch.cat(all_obs, dim=0)
    target_mean = torch.cat(all_mean, dim=0)
    target_std = torch.cat(all_std, dim=0)
    return obs, target_mean, target_std
