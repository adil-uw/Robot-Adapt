"""Phase 2 evaluation framework for the Cross-Environment Policy Distillation project.

This package provides a scientifically valid evaluation system used to compare
policies (random, PPO, SAC, teachers, students) across multiple episodes and
random seeds, reporting the metrics required by the project proposal:

    * Average Return
    * Episode Length
    * Forward Distance
    * Average Velocity
    * Fall Rate

It also produces result tables, CSV files, and performance plots.
"""

from evaluation.metrics import (
    EpisodeResult,
    PolicyEvaluation,
    evaluate_policy,
    run_episode,
)
from evaluation.policies import make_policy, random_policy, sb3_policy

__all__ = [
    "EpisodeResult",
    "PolicyEvaluation",
    "evaluate_policy",
    "run_episode",
    "make_policy",
    "random_policy",
    "sb3_policy",
]
