"""Phase 2 evaluation runner.

Evaluates a set of policies on an environment over multiple episodes and seeds,
then prints a comparison table, writes a CSV, and saves performance plots.

Examples
--------
Evaluate the random baseline plus a trained PPO checkpoint::

    python -m evaluation.evaluate \
        --env Ant-v5 \
        --episodes 10 --seeds 0 1 2 \
        --ppo training/models/ppo_ant_100k_steps

Auto-discover every ``*.zip`` in ``training/models`` and evaluate them as PPO::

    python -m evaluation.evaluate --auto-discover
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple

import gymnasium as gym

from evaluation.metrics import evaluate_policy
from evaluation.policies import random_policy, sb3_policy
from evaluation.reporting import plot_metrics, print_table, select_best, write_csv

DEFAULT_RESULTS_DIR = os.path.join("evaluation", "results")


def _build_policy_specs(args: argparse.Namespace) -> List[Tuple[str, str, str]]:
    """Return a list of (name, path, algo) checkpoint specs from CLI args."""

    specs: List[Tuple[str, str, str]] = []

    for path in args.ppo or []:
        specs.append((f"PPO:{os.path.basename(path)}", path, "PPO"))
    for path in args.sac or []:
        specs.append((f"SAC:{os.path.basename(path)}", path, "SAC"))

    if args.auto_discover:
        for zip_path in sorted(glob.glob(os.path.join(args.models_dir, "*.zip"))):
            path = zip_path[: -len(".zip")]
            name = f"PPO:{os.path.basename(path)}"
            if all(name != existing[0] for existing in specs):
                specs.append((name, path, "PPO"))

    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 policy evaluation framework")
    parser.add_argument("--env", default="Ant-v5", help="Gymnasium environment id")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds")
    parser.add_argument("--ppo", nargs="*", default=[], help="PPO checkpoint paths (no .zip)")
    parser.add_argument("--sac", nargs="*", default=[], help="SAC checkpoint paths (no .zip)")
    parser.add_argument(
        "--student",
        nargs="*",
        default=[],
        help="Distilled student .pt paths (from training.run_distillation)",
    )
    parser.add_argument(
        "--models-dir",
        default=os.path.join("training", "models"),
        help="Directory scanned when --auto-discover is set",
    )
    parser.add_argument(
        "--auto-discover",
        action="store_true",
        help="Evaluate every *.zip checkpoint found in --models-dir as PPO",
    )
    parser.add_argument("--no-random", action="store_true", help="Skip the random baseline")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Where to write outputs")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    parser.add_argument(
        "--select-best",
        action="store_true",
        help="Print the best policy/checkpoint by average return on the eval env",
    )
    args = parser.parse_args()

    def env_factory() -> gym.Env:
        return gym.make(args.env)

    evaluations = []

    if not args.no_random:
        probe = env_factory()
        rand = random_policy(probe)
        probe.close()
        print("Evaluating: Random ...")
        evaluations.append(
            evaluate_policy(env_factory, rand, "Random", args.episodes, tuple(args.seeds))
        )

    for name, path, algo in _build_policy_specs(args):
        if not os.path.exists(path + ".zip"):
            print(f"  [skip] {name}: checkpoint not found at {path}.zip")
            continue
        print(f"Evaluating: {name} ...")
        policy = sb3_policy(path, algo)
        evaluations.append(
            evaluate_policy(env_factory, policy, name, args.episodes, tuple(args.seeds))
        )

    for path in args.student or []:
        if not os.path.exists(path):
            print(f"  [skip] student not found at {path}")
            continue
        from training.distillation.distill import load_student

        name = f"Student:{os.path.basename(path)}"
        print(f"Evaluating: {name} ...")
        student = load_student(path)
        evaluations.append(
            evaluate_policy(env_factory, student.predict, name, args.episodes, tuple(args.seeds))
        )

    if not evaluations:
        print("No policies were evaluated. Pass --ppo/--sac/--student or use --auto-discover.")
        return

    print_table(evaluations)

    if args.select_best:
        best = select_best(evaluations, metric="avg_return", maximize=True)
        if best is not None:
            print(
                f"Best by average return: {best.name} "
                f"(return={best.avg_return:.2f} +/- {best.std_return:.2f})\n"
            )

    csv_path = write_csv(evaluations, os.path.join(args.results_dir, "results.csv"))
    print(f"CSV written to: {csv_path}")

    if not args.no_plots:
        images = plot_metrics(evaluations, os.path.join(args.results_dir, "plots"))
        for img in images:
            print(f"Plot written to: {img}")


if __name__ == "__main__":
    main()
