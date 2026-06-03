"""Phase 7: Policy Distillation Framework.

Implements the pipeline described in the proposal::

    Teacher Policy
        -> Action Distribution Extraction
        -> KL-Divergence Loss
        -> Student Policy Update

The framework is designed to run *before* real teacher models exist (Phase 5)
by using :class:`MockTeacher`, and to swap in real Stable-Baselines3 teachers
later with no other code changes.
"""

from training.distillation.losses import gaussian_kl_loss
from training.distillation.student import StudentPolicy
from training.distillation.teachers import (
    MockTeacher,
    SB3TeacherPolicy,
    TeacherPolicy,
)

__all__ = [
    "TeacherPolicy",
    "SB3TeacherPolicy",
    "MockTeacher",
    "StudentPolicy",
    "gaussian_kl_loss",
]
