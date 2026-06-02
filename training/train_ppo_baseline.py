"""Phase 2: PPO baseline training with periodic checkpoints.

Trains PPO on a standard environment (Ant-v5 by default) and saves a checkpoint
at each requested milestone so the *best* policy can later be selected via the
evaluation framework rather than blindly taking the final model. This addresses
the professor's note that RL performance is highly variable.

Example
-------
    python -m training.train_ppo_baseline \
        --env Ant-v5 \
        --milestones 10000 25000 50000 100000 150000 200000 \
        --out-dir training/models --seed 0
"""

from __future__ import annotations

import argparse
import os

import gymnasium as gym
from stable_baselines3 import PPO


DEFAULT_MILESTONES = [
    10_000,
    25_000,
    50_000,
    100_000,
    150_000,
    200_000,
    250_000,
    500_000,
    1_000_000,
    2_000_000,
]


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO baseline training with checkpoints")
    parser.add_argument("--env", default="Ant-v5", help="Gymnasium environment id")
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=DEFAULT_MILESTONES,
        help="Cumulative timestep checkpoints to save",
    )
    parser.add_argument("--out-dir", default=os.path.join("training", "models"))
    parser.add_argument("--log-dir", default=os.path.join("training", "logs"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix", default="ppo_ant", help="Checkpoint filename prefix")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    env = gym.make(args.env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=args.log_dir,
    )

    milestones = sorted(set(args.milestones))
    previous = 0
    for milestone in milestones:
        increment = milestone - previous
        if increment <= 0:
            continue
        # reset_num_timesteps=False keeps the TensorBoard timeline continuous
        # across the incremental training segments.
        model.learn(total_timesteps=increment, reset_num_timesteps=(previous == 0))
        previous = milestone

        checkpoint = os.path.join(args.out_dir, f"{args.prefix}_{milestone}_steps")
        model.save(checkpoint)
        print(f"[checkpoint] saved {checkpoint}.zip at {milestone} steps")

    env.close()
    print("PPO baseline training finished.")
    print(f"Checkpoints in: {args.out_dir}")
    print("Next: select the best checkpoint with `python -m evaluation.evaluate --auto-discover`")


if __name__ == "__main__":
    main()
