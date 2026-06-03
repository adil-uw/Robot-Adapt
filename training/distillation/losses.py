"""KL-divergence distillation loss (Phase 7).

The student is trained so that its action distribution matches the teacher's.
Both are diagonal Gaussians, so we use the analytic KL divergence summed over
action dimensions and averaged over the batch.

Two directions are supported:

* ``"forward"``  -> KL(teacher || student): mode-covering, the conventional
  choice for policy distillation (the student is pulled to cover the teacher).
* ``"reverse"``  -> KL(student || teacher): mode-seeking.
"""

from __future__ import annotations

import torch
import torch.distributions as D


def gaussian_kl_loss(
    teacher_mean: torch.Tensor,
    teacher_std: torch.Tensor,
    student_mean: torch.Tensor,
    student_std: torch.Tensor,
    direction: str = "forward",
) -> torch.Tensor:
    """Return the scalar mean KL divergence between the two diagonal Gaussians."""

    teacher = D.Independent(D.Normal(teacher_mean, teacher_std), 1)
    student = D.Independent(D.Normal(student_mean, student_std), 1)

    if direction == "forward":
        kl = D.kl_divergence(teacher, student)
    elif direction == "reverse":
        kl = D.kl_divergence(student, teacher)
    else:
        raise ValueError(f"direction must be 'forward' or 'reverse', got {direction!r}")

    return kl.mean()
