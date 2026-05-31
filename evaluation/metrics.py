"""Core metric collection for policy evaluation (Phase 2).

The functions here run a policy in a *separate* evaluation environment for a
number of episodes across several random seeds and aggregate the per-episode
metrics into mean / standard-deviation statistics.

A "policy" is simply any callable ``policy_fn(observation) -> action``. This
keeps the evaluator decoupled from how the action is produced (random sampling,
a Stable-Baselines3 model, a distilled student network, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Callable, List, Sequence

import gymnasium as gym
import numpy as np

# A policy is any function mapping an observation to an action.
PolicyFn = Callable[[np.ndarray], np.ndarray]
EnvFactory = Callable[[], gym.Env]


@dataclass
class EpisodeResult:
    """Metrics collected from a single evaluation episode."""

    episode_return: float
    length: int
    forward_distance: float
    average_velocity: float
    fell: bool


@dataclass
class PolicyEvaluation:
    """Aggregated statistics for one policy over many episodes/seeds."""

    name: str
    episodes: List[EpisodeResult] = field(default_factory=list)

    # ---- aggregate helpers -------------------------------------------------
    def _values(self, attr: str) -> List[float]:
        return [float(getattr(ep, attr)) for ep in self.episodes]

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        return float(mean(values)) if values else 0.0

    @staticmethod
    def _std(values: Sequence[float]) -> float:
        # Population std is fine for reporting; avoids NaN with a single sample.
        return float(pstdev(values)) if len(values) > 1 else 0.0

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def avg_return(self) -> float:
        return self._mean(self._values("episode_return"))

    @property
    def std_return(self) -> float:
        return self._std(self._values("episode_return"))

    @property
    def avg_length(self) -> float:
        return self._mean(self._values("length"))

    @property
    def std_length(self) -> float:
        return self._std(self._values("length"))

    @property
    def avg_forward_distance(self) -> float:
        return self._mean(self._values("forward_distance"))

    @property
    def std_forward_distance(self) -> float:
        return self._std(self._values("forward_distance"))

    @property
    def avg_velocity(self) -> float:
        return self._mean(self._values("average_velocity"))

    @property
    def std_velocity(self) -> float:
        return self._std(self._values("average_velocity"))

    @property
    def fall_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for ep in self.episodes if ep.fell) / len(self.episodes)

    def as_row(self) -> dict:
        """Flat dictionary suitable for CSV / table rendering."""
        return {
            "policy": self.name,
            "episodes": self.num_episodes,
            "avg_return": round(self.avg_return, 3),
            "std_return": round(self.std_return, 3),
            "avg_length": round(self.avg_length, 1),
            "std_length": round(self.std_length, 1),
            "avg_forward_distance": round(self.avg_forward_distance, 3),
            "std_forward_distance": round(self.std_forward_distance, 3),
            "avg_velocity": round(self.avg_velocity, 4),
            "std_velocity": round(self.std_velocity, 4),
            "fall_rate": round(self.fall_rate, 3),
        }


def run_episode(env: gym.Env, policy_fn: PolicyFn, seed: int | None = None) -> EpisodeResult:
    """Run a single episode and return its metrics.

    Forward distance is the net displacement along the x-axis, and average
    velocity is the mean forward (x) velocity reported by the MuJoCo env. An
    episode is counted as a *fall* when it ``terminated`` (the Ant became
    unhealthy / flipped) as opposed to being ``truncated`` by the time limit.
    """

    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0
    start_x: float | None = None
    last_x: float = 0.0
    velocities: List[float] = []
    terminated = False
    truncated = False

    while True:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1

        # MuJoCo locomotion envs expose x position/velocity in ``info``.
        x_pos = info.get("x_position")
        if x_pos is not None:
            if start_x is None:
                start_x = float(x_pos)
            last_x = float(x_pos)
        if "x_velocity" in info:
            velocities.append(float(info["x_velocity"]))

        if terminated or truncated:
            break

    forward_distance = (last_x - start_x) if start_x is not None else 0.0
    average_velocity = float(mean(velocities)) if velocities else 0.0

    return EpisodeResult(
        episode_return=total_reward,
        length=steps,
        forward_distance=forward_distance,
        average_velocity=average_velocity,
        # Falling = terminated early rather than hitting the time limit.
        fell=bool(terminated and not truncated),
    )


def evaluate_policy(
    env_factory: EnvFactory,
    policy_fn: PolicyFn,
    name: str,
    num_episodes: int = 10,
    seeds: Sequence[int] = (0, 1, 2),
) -> PolicyEvaluation:
    """Evaluate ``policy_fn`` over ``num_episodes`` per seed in ``seeds``.

    A fresh environment is created from ``env_factory`` so evaluation never
    shares state with training. Each (seed, episode) pair uses a distinct,
    reproducible seed.
    """

    evaluation = PolicyEvaluation(name=name)

    for seed in seeds:
        env = env_factory()
        # Seed the action space too so random policies are reproducible.
        try:
            env.action_space.seed(seed)
        except Exception:
            pass

        for ep in range(num_episodes):
            episode_seed = seed * 10_000 + ep
            result = run_episode(env, policy_fn, seed=episode_seed)
            evaluation.episodes.append(result)

        env.close()

    return evaluation
