"""Phase 7/8 entry point: run the policy distillation pipeline.
"""

from __future__ import annotations

import argparse
import os
from typing import Callable, List

import gymnasium as gym

import environments  

from training.distillation.data import collect_dataset
from training.distillation.distill import DistillConfig, DistillationTrainer
from training.distillation.student import StudentPolicy
from training.distillation.teachers import MockTeacher, SB3TeacherPolicy, TeacherPolicy


def _parse_teacher(spec: str, obs_dim: int, action_dim: int, seed: int, device: str) -> TeacherPolicy:
    """Turn a CLI teacher spec into a TeacherPolicy.

    Specs: ``mock`` | ``ppo:<path>`` | ``sac:<path>``.
    """

    if spec == "mock":
        return MockTeacher(obs_dim, action_dim, seed=seed, device=device)

    if ":" in spec:
        algo, path = spec.split(":", 1)
        algo = algo.upper()
        if algo in ("PPO", "SAC"):
            return SB3TeacherPolicy(path, algo=algo, device=device)

    raise ValueError(f"Unknown teacher spec {spec!r}. Use 'mock', 'ppo:<path>', or 'sac:<path>'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy distillation pipeline")
    parser.add_argument(
        "--teacher",
        action="append",
        default=None,
        help="Teacher spec (repeatable): mock | ppo:<path> | sac:<path>",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=None,
        help="Env id per teacher (repeatable). Defaults to Ant-v5 for every teacher.",
    )
    parser.add_argument("--steps-per-teacher", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--kl-direction", choices=["forward", "reverse"], default="forward")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=os.path.join("training", "models", "distilled_student.pt"))
    args = parser.parse_args()

    teacher_specs = args.teacher or ["mock", "mock"]
    env_ids = args.env or ["Ant-v5"] * len(teacher_specs)
    if len(env_ids) != len(teacher_specs):
        parser.error("Number of --env entries must match number of --teacher entries.")

    # Probe one env to learn the observation/action dimensions.
    probe = gym.make(env_ids[0])
    obs_dim = int(probe.observation_space.shape[0])
    action_dim = int(probe.action_space.shape[0])
    probe.close()

    print(f"obs_dim={obs_dim}  action_dim={action_dim}")
    print(f"teachers={teacher_specs}  envs={env_ids}")

    teachers: List[TeacherPolicy] = [
        _parse_teacher(spec, obs_dim, action_dim, seed=args.seed + i, device=args.device)
        for i, spec in enumerate(teacher_specs)
    ]
    env_factories: List[Callable[[], gym.Env]] = [
        (lambda env_id=env_id: gym.make(env_id)) for env_id in env_ids
    ]

    print("Collecting states and extracting teacher action distributions ...")
    obs, target_mean, target_std = collect_dataset(
        teachers,
        env_factories,
        steps_per_teacher=args.steps_per_teacher,
        seed=args.seed,
        device=args.device,
    )
    print(f"Collected {len(obs)} states.")

    student = StudentPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden=args.hidden)
    config = DistillConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        kl_direction=args.kl_direction,
        device=args.device,
    )
    trainer = DistillationTrainer(student, config)

    print("Training student via KL distillation ...")
    trainer.fit(obs, target_mean, target_std)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    trainer.save(args.out)
    print("Distillation complete.")


if __name__ == "__main__":
    main()
