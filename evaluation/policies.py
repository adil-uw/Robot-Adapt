"""Factories that turn models/checkpoints into ``policy_fn`` callables.

Everything the evaluator needs is a function ``policy_fn(obs) -> action``.
These helpers build such functions for the policy types used in the project.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np

PolicyFn = Callable[[np.ndarray], np.ndarray]


def random_policy(env: gym.Env) -> PolicyFn:
    """A policy that samples uniformly random actions (Phase 1 baseline)."""

    action_space = env.action_space

    def policy_fn(_obs: np.ndarray) -> np.ndarray:
        return action_space.sample()

    return policy_fn


def sb3_policy(model_path: str, algo: str = "PPO") -> PolicyFn:
    """Load a Stable-Baselines3 model and wrap it as a deterministic policy.

    ``algo`` selects the SB3 class (``PPO`` or ``SAC``) used to load the
    checkpoint. Prediction is deterministic so evaluation is repeatable.
    """

    algo = algo.upper()
    if algo == "PPO":
        from stable_baselines3 import PPO as Algo
    elif algo == "SAC":
        from stable_baselines3 import SAC as Algo
    else:
        raise ValueError(f"Unsupported algo '{algo}'. Use 'PPO' or 'SAC'.")

    model = Algo.load(model_path)

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return action

    return policy_fn


def make_policy(spec: dict, env: gym.Env) -> PolicyFn:
    """Build a policy from a small spec dict.

    Supported specs::

        {"type": "random"}
        {"type": "sb3", "path": "training/models/ppo_ant_100k_steps", "algo": "PPO"}
    """

    kind = spec.get("type")
    if kind == "random":
        return random_policy(env)
    if kind == "sb3":
        return sb3_policy(spec["path"], spec.get("algo", "PPO"))
    raise ValueError(f"Unknown policy spec type: {kind!r}")
