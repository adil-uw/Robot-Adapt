"""Phase 6: Student trained from scratch on mixed terrain.

This is the key *baseline* for the distillation study: a policy that learns to
walk on both normal (``Ant-v5``) and slippery (``AntSlippery-v5``) terrain using
reinforcement learning ONLY -- no teacher, no distillation. Phase 9 compares it
against the distilled student to test whether distillation helps.

"Mixed terrain" is implemented by :class:`MixedTerrainEnv`, which randomly picks
a terrain at every episode reset, so a single agent experiences both terrains
during training. Multiple such envs are run in parallel for PPO.

Checkpoints are saved at milestones so the best one can be selected with the
Phase 2 evaluation framework rather than assuming the last step is best.

Example
-------
    python -m training.train_student_scratch \
        --algo SAC --milestones 100000 250000 500000 \
        --seed 0
"""

from __future__ import annotations

import argparse
import os
from typing import List, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import environments  # noqa: F401 -- registers AntSlippery-v5
from environments.terrains import make_terrain_env

# SAC was chosen over PPO after the Phase 3 baseline comparison. We cap training
# at 500k for now (1M takes too long); intermediate checkpoints let the Phase 2
# evaluator pick the best one rather than assuming the last step is best.
DEFAULT_MILESTONES = [100_000, 250_000, 500_000]


class MixedTerrainEnv(gym.Env):
    """Ant env that randomly switches terrain (normal/slippery) each episode.

    Holds one underlying env per terrain (they share identical observation and
    action spaces) and routes ``step`` to whichever was selected on the last
    ``reset``. This exposes the agent to both terrains within a single training
    run, which is what "trained on mixed terrain" means in the proposal.
    """

    metadata = {"render_modes": []}

    def __init__(self, terrains: Sequence[str] = ("normal", "slippery"), seed: int | None = None):
        super().__init__()
        if not terrains:
            raise ValueError("Need at least one terrain.")
        self._terrains = list(terrains)
        self._envs = {t: make_terrain_env(t) for t in self._terrains}

        self._current = self._terrains[0]
        self._active = self._envs[self._current]
        self.observation_space = self._active.observation_space
        self.action_space = self._active.action_space
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._current = self._terrains[int(self._rng.integers(len(self._terrains)))]
        self._active = self._envs[self._current]
        obs, info = self._active.reset(seed=seed, options=options)
        info["terrain"] = self._current
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._active.step(action)
        info["terrain"] = self._current
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._active.render()

    def close(self):
        for env in self._envs.values():
            env.close()


def make_mixed_vec_env(n_envs: int, terrains: Sequence[str], seed: int) -> DummyVecEnv:
    """Build a vectorized stack of MixedTerrainEnv instances."""

    def factory(rank: int):
        def _init():
            env = MixedTerrainEnv(terrains=terrains, seed=seed + rank)
            return Monitor(env)

        return _init

    return DummyVecEnv([factory(i) for i in range(n_envs)])


def _resolve_device(device: str) -> str:
    """Map 'auto' to cuda if available, otherwise cpu (small MLPs run fine on CPU)."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _build_model(algo: str, env, args, log_dir: str):
    algo = algo.upper()
    device = _resolve_device(args.device)
    net_arch = list(args.net_arch)

    if algo == "PPO":
        from stable_baselines3 import PPO

        return PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            device=device,
            tensorboard_log=log_dir,
            policy_kwargs=dict(net_arch=net_arch),
        )
    if algo == "SAC":
        from stable_baselines3 import SAC

        # gradient_steps/train_freq are the main wall-clock levers for SAC:
        # fewer gradient updates per env step => much faster (slightly less
        # sample-efficient, but env steps are cheap).
        return SAC(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            device=device,
            tensorboard_log=log_dir,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            policy_kwargs=dict(net_arch=net_arch),
        )
    raise ValueError(f"Unsupported algo '{algo}'. Use 'PPO' or 'SAC'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: student-from-scratch on mixed terrain")
    parser.add_argument("--algo", choices=["PPO", "SAC"], default="SAC")
    parser.add_argument(
        "--terrains",
        nargs="+",
        default=["normal", "slippery"],
        help="Terrains to mix during training",
    )
    parser.add_argument("--milestones", type=int, nargs="+", default=DEFAULT_MILESTONES)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Parallel envs (forced to 1 for SAC).",
    )
    parser.add_argument("--out-dir", default=os.path.join("training", "models"))
    parser.add_argument("--log-dir", default=os.path.join("training", "logs"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix", default="student_scratch")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden layer sizes. Smaller (e.g. 128 128) trains faster.",
    )
    # --- SAC wall-clock levers -------------------------------------------------
    parser.add_argument(
        "--train-freq",
        type=int,
        default=4,
        help="SAC: env steps between training calls. Higher = fewer updates = faster.",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="SAC: gradient updates per training call. Lower = faster.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="SAC: replay batch size")
    parser.add_argument("--learning-starts", type=int, default=5000, help="SAC: warmup steps")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    n_envs = 1 if args.algo == "SAC" else args.n_envs
    env = make_mixed_vec_env(n_envs, args.terrains, args.seed)

    model = _build_model(args.algo, env, args, args.log_dir)

    print(
        f"Training {args.algo} student from scratch on terrains={args.terrains} "
        f"(n_envs={n_envs}, device={_resolve_device(args.device)})"
    )

    milestones: List[int] = sorted(set(args.milestones))
    previous = 0
    for milestone in milestones:
        increment = milestone - previous
        if increment <= 0:
            continue
        model.learn(total_timesteps=increment, reset_num_timesteps=(previous == 0))
        previous = milestone

        checkpoint = os.path.join(args.out_dir, f"{args.prefix}_{args.algo.lower()}_{milestone}_steps")
        model.save(checkpoint)
        print(f"[checkpoint] saved {checkpoint}.zip at {milestone} steps")

    env.close()
    print("Student-from-scratch training finished.")
    print("Evaluate per terrain and measure transfer drop, e.g.:")
    print(
        f"  python -m evaluation.transfer --source-env Ant-v5 --target-env AntSlippery-v5 "
        f"--{args.algo.lower()} {args.out_dir}/{args.prefix}_{args.algo.lower()}_{milestones[-1]}_steps"
    )


if __name__ == "__main__":
    main()
