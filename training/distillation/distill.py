"""Distillation trainer: the full Teacher -> KL -> Student pipeline (Phase 7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import DataLoader, TensorDataset

from training.distillation.losses import gaussian_kl_loss
from training.distillation.student import StudentPolicy


@dataclass
class DistillConfig:
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 3e-4
    kl_direction: str = "forward"
    device: str = "cpu"


class DistillationTrainer:
    """Trains a :class:`StudentPolicy` to match teacher action distributions."""

    def __init__(self, student: StudentPolicy, config: DistillConfig | None = None):
        self.config = config or DistillConfig()
        self.student = student.to(self.config.device)
        self.optimizer = torch.optim.Adam(
            self.student.parameters(), lr=self.config.learning_rate
        )

    def fit(
        self,
        obs: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> List[float]:
        """Run supervised KL distillation; returns the per-epoch mean loss."""

        dataset = TensorDataset(
            obs.to(self.config.device),
            target_mean.to(self.config.device),
            target_std.to(self.config.device),
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        history: List[float] = []
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            num_batches = 0
            for batch_obs, batch_mean, batch_std in loader:
                student_mean, student_std = self.student(batch_obs)
                loss = gaussian_kl_loss(
                    batch_mean,
                    batch_std,
                    student_mean,
                    student_std,
                    direction=self.config.kl_direction,
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += float(loss.item())
                num_batches += 1

            mean_loss = epoch_loss / max(num_batches, 1)
            history.append(mean_loss)
            print(f"[distill] epoch {epoch + 1:>3}/{self.config.epochs}  KL loss = {mean_loss:.5f}")

        return history

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.student.state_dict(),
                "obs_dim": self.student.obs_dim,
                "action_dim": self.student.action_dim,
            },
            path,
        )
        print(f"[distill] student saved to {path}")


def load_student(path: str, hidden: int = 256, device: str = "cpu") -> StudentPolicy:
    """Reload a distilled student saved with :meth:`DistillationTrainer.save`."""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    student = StudentPolicy(
        obs_dim=checkpoint["obs_dim"],
        action_dim=checkpoint["action_dim"],
        hidden=hidden,
    )
    student.load_state_dict(checkpoint["state_dict"])
    student.to(device)
    student.eval()
    return student
