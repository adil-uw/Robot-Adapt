"""Render static training-curve graphs from TensorBoard logs.

The training scripts already stream metrics to ``tensorboard_log``:

* student-from-scratch / baselines (Stable-Baselines3): ``rollout/ep_rew_mean``,
  ``rollout/ep_len_mean``, ``train/*`` ...
* policy distillation: ``distill/kl_loss_epoch`` and ``distill/kl_loss_step``.

This tool reads those event files and saves PNG graphs (one per metric, with
all selected runs overlaid) so they can go straight into the report. It does
not replace ``tensorboard --logdir training/logs``; it's for static figures.

Examples
--------
Plot the from-scratch student's reward and the distillation KL loss::

    python -m evaluation.plot_training_curves --runs SAC_4 distill_20260602_150943

Plot specific metrics for specific runs::

    python -m evaluation.plot_training_curves --runs SAC_4 \
        --tags rollout/ep_rew_mean rollout/ep_len_mean
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List

DEFAULT_TAGS = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "distill/kl_loss_epoch",
]


def _resolve_run_dirs(runs: List[str], logdir: str) -> List[str]:
    """Accept either bare run names (joined with logdir) or full paths/globs."""
    resolved: List[str] = []
    for run in runs:
        if os.path.isdir(run):
            resolved.append(run)
            continue
        candidate = os.path.join(logdir, run)
        if os.path.isdir(candidate):
            resolved.append(candidate)
            continue
        matches = [p for p in glob.glob(candidate) if os.path.isdir(p)]
        if matches:
            resolved.extend(sorted(matches))
        else:
            print(f"  [warn] run not found: {run}")
    return resolved


def _read_scalar(run_dir: str, tag: str):
    """Return (steps, values) for a scalar tag in a run, or None if absent."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(run_dir)
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return None
    events = accumulator.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from TensorBoard logs")
    parser.add_argument("--runs", nargs="+", required=True, help="Run names (under --logdir) or paths")
    parser.add_argument("--logdir", default=os.path.join("training", "logs"))
    parser.add_argument("--tags", nargs="+", default=DEFAULT_TAGS, help="Scalar tags to plot")
    parser.add_argument(
        "--out-dir",
        default=os.path.join("evaluation", "results", "training_curves"),
        help="Where PNGs are written",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dirs = _resolve_run_dirs(args.runs, args.logdir)
    if not run_dirs:
        print("No valid runs found.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    # For each tag, overlay every run that contains it.
    for tag in args.tags:
        series: Dict[str, tuple] = {}
        for run_dir in run_dirs:
            data = _read_scalar(run_dir, tag)
            if data is not None and data[0]:
                series[os.path.basename(run_dir)] = data

        if not series:
            print(f"  [skip] no run contains tag '{tag}'")
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, (steps, values) in series.items():
            ax.plot(steps, values, label=name, linewidth=1.6)
        ax.set_title(tag)
        ax.set_xlabel("step")
        ax.set_ylabel(tag.split("/")[-1])
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        fname = tag.replace("/", "_") + ".png"
        path = os.path.join(args.out_dir, fname)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"Saved {path}  ({', '.join(series.keys())})")


if __name__ == "__main__":
    main()
