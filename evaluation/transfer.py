"""Transfer-drop evaluation (Phase 9 support).

Measures how much a single policy degrades when moved from a *source* terrain
(e.g. normal Ant-v5) to a *target* terrain (e.g. slippery / ice). This is the
"transfer drop" metric the professor asked for.

The module is terrain-agnostic: it works today with any two Gymnasium env ids
and will measure real normal->slippery transfer once Person 2's slippery env
(Phase 4) is registered -- no code changes required, just pass its id as
``--target-env``.

For the teacher-student comparison ("student on terrain B after distillation
from teacher A vs. a student trained from scratch"), evaluate each policy here
on the same target terrain and compare their transfer drops.
"""

from __future__ import annotations

import argparse
from typing import Dict

import gymnasium as gym

from evaluation.metrics import PolicyEvaluation, evaluate_policy
from evaluation.policies import random_policy, sb3_policy


def _relative_drop(source: float, target: float) -> float:
    """Relative drop from source to target, as a fraction of |source|."""
    if source == 0:
        return 0.0
    return (source - target) / abs(source)


def compute_transfer_drop(
    source: PolicyEvaluation, target: PolicyEvaluation
) -> Dict[str, float]:
    """Compare a policy's source-terrain vs target-terrain performance."""

    return {
        "return_source": source.avg_return,
        "return_target": target.avg_return,
        "return_drop_pct": 100.0 * _relative_drop(source.avg_return, target.avg_return),
        "forward_distance_source": source.avg_forward_distance,
        "forward_distance_target": target.avg_forward_distance,
        "forward_distance_drop_pct": 100.0
        * _relative_drop(source.avg_forward_distance, target.avg_forward_distance),
        "velocity_source": source.avg_velocity,
        "velocity_target": target.avg_velocity,
        "velocity_drop_pct": 100.0 * _relative_drop(source.avg_velocity, target.avg_velocity),
        "fall_rate_source": source.fall_rate,
        "fall_rate_target": target.fall_rate,
        "fall_rate_increase": target.fall_rate - source.fall_rate,
    }


def print_transfer_drop(name: str, drop: Dict[str, float]) -> None:
    print(f"\nTransfer Drop: {name}")
    print("=" * 52)
    print(f"{'Metric':<20}{'Source':>10}{'Target':>10}{'Drop':>12}")
    print("-" * 52)
    print(
        f"{'Return':<20}{drop['return_source']:>10.2f}{drop['return_target']:>10.2f}"
        f"{drop['return_drop_pct']:>11.1f}%"
    )
    print(
        f"{'Forward Distance':<20}{drop['forward_distance_source']:>10.2f}"
        f"{drop['forward_distance_target']:>10.2f}{drop['forward_distance_drop_pct']:>11.1f}%"
    )
    print(
        f"{'Velocity':<20}{drop['velocity_source']:>10.3f}{drop['velocity_target']:>10.3f}"
        f"{drop['velocity_drop_pct']:>11.1f}%"
    )
    print(
        f"{'Fall Rate':<20}{drop['fall_rate_source']:>10.2f}{drop['fall_rate_target']:>10.2f}"
        f"{drop['fall_rate_increase']:>+11.2f}"
    )
    print("=" * 52)
    print("Lower return/distance/velocity drop and smaller fall-rate increase = better transfer\n")


def _build_single_policy(args: argparse.Namespace):
    """Resolve exactly one policy from the CLI args."""

    if args.ppo:
        return sb3_policy(args.ppo, "PPO"), f"PPO:{args.ppo}"
    if args.sac:
        return sb3_policy(args.sac, "SAC"), f"SAC:{args.sac}"
    if args.student:
        from training.distillation.distill import load_student

        student = load_student(args.student)
        return student.predict, f"Student:{args.student}"
    # Fallback: random policy (useful as a sanity floor).
    probe = gym.make(args.source_env)
    fn = random_policy(probe)
    probe.close()
    return fn, "Random"


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer-drop evaluation (source -> target)")
    parser.add_argument("--source-env", default="Ant-v5", help="Source terrain env id")
    parser.add_argument(
        "--target-env",
        required=True,
        help="Target terrain env id (e.g. Person 2's slippery env)",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ppo", help="PPO checkpoint path (no .zip)")
    group.add_argument("--sac", help="SAC checkpoint path (no .zip)")
    group.add_argument("--student", help="Distilled student .pt path")
    args = parser.parse_args()

    policy_fn, name = _build_single_policy(args)
    seeds = tuple(args.seeds)

    print(f"Evaluating {name} on source terrain ({args.source_env}) ...")
    source_eval = evaluate_policy(
        lambda: gym.make(args.source_env), policy_fn, name, args.episodes, seeds
    )

    print(f"Evaluating {name} on target terrain ({args.target_env}) ...")
    target_eval = evaluate_policy(
        lambda: gym.make(args.target_env), policy_fn, name, args.episodes, seeds
    )

    drop = compute_transfer_drop(source_eval, target_eval)
    print_transfer_drop(name, drop)


if __name__ == "__main__":
    main()
