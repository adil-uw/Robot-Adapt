"""Output helpers: pretty tables, CSV export, and performance plots (Phase 2)."""

from __future__ import annotations

import csv
import os
from typing import List, Sequence

from evaluation.metrics import PolicyEvaluation

# Columns shown in the console table (key, header, format spec).
_TABLE_COLUMNS = [
    ("policy", "Policy", "{:<24}"),
    ("episodes", "N", "{:>4}"),
    ("avg_return", "Return", "{:>10}"),
    ("avg_length", "Length", "{:>8}"),
    ("avg_forward_distance", "FwdDist", "{:>9}"),
    ("avg_velocity", "Vel", "{:>8}"),
    ("avg_energy_cost", "Energy", "{:>8}"),
    ("fall_rate", "FallRate", "{:>9}"),
]


def print_table(evaluations: Sequence[PolicyEvaluation]) -> None:
    """Print a human-readable comparison table to stdout."""

    rows = [e.as_row() for e in evaluations]

    header = "".join(fmt.replace("d", "").format(head) for _, head, fmt in _TABLE_COLUMNS)
    print("\nEvaluation Results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        line = ""
        for key, _head, fmt in _TABLE_COLUMNS:
            value = row[key]
            line += fmt.format(value)
        print(line)
    print("=" * len(header))
    print("Return/Length/FwdDist/Vel are means; FallRate is fraction of episodes\n")


def select_best(
    evaluations: Sequence[PolicyEvaluation],
    metric: str = "avg_return",
    maximize: bool = True,
) -> PolicyEvaluation | None:
    """Return the policy with the best value of ``metric``.

    Used to choose the strongest checkpoint based on *evaluation* performance
    rather than assuming the latest checkpoint is best (RL training is noisy and
    can degrade with more steps).
    """

    if not evaluations:
        return None
    key = lambda e: getattr(e, metric)  # noqa: E731
    return max(evaluations, key=key) if maximize else min(evaluations, key=key)


def write_csv(evaluations: Sequence[PolicyEvaluation], path: str) -> str:
    """Write full statistics (including std-devs) to ``path`` as CSV."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = [e.as_row() for e in evaluations]
    fieldnames = list(rows[0].keys()) if rows else []

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def plot_metrics(evaluations: Sequence[PolicyEvaluation], out_dir: str) -> List[str]:
    """Generate one bar chart per metric (with std-dev error bars).

    Returns the list of written image paths. Plotting is optional: if
    matplotlib is unavailable the function returns an empty list.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")  # headless-safe backend
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[plot] matplotlib unavailable, skipping plots: {exc}")
        return []

    os.makedirs(out_dir, exist_ok=True)
    names = [e.name for e in evaluations]

    # (filename, title, value-getter, error-getter)
    metrics = [
        ("return", "Average Return", lambda e: e.avg_return, lambda e: e.std_return),
        ("length", "Episode Length", lambda e: e.avg_length, lambda e: e.std_length),
        (
            "forward_distance",
            "Forward Distance",
            lambda e: e.avg_forward_distance,
            lambda e: e.std_forward_distance,
        ),
        ("velocity", "Average Velocity", lambda e: e.avg_velocity, lambda e: e.std_velocity),
        (
            "energy_cost",
            "Energy / Torque Cost",
            lambda e: e.avg_energy_cost,
            lambda e: e.std_energy_cost,
        ),
        ("fall_rate", "Fall Rate", lambda e: e.fall_rate, lambda _e: 0.0),
    ]

    written: List[str] = []
    for key, title, value_of, error_of in metrics:
        values = [value_of(e) for e in evaluations]
        errors = [error_of(e) for e in evaluations]

        fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(names)), 4))
        ax.bar(names, values, yerr=errors, capsize=4, color="#4C72B0")
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()

        path = os.path.join(out_dir, f"{key}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    return written
