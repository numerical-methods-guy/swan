"""Training strategy protocol for PyTorch Lightning."""

from abc import ABC, abstractmethod
from typing import Any

import pytorch_lightning as pl
import torch


class TrainingStrategy(ABC):
    """Encapsulates optimizer setup and training-step optimization behavior."""

    automatic_optimization: bool = True

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def configure_optimizers(self, module: pl.LightningModule) -> Any:
        """Return Lightning-compatible optimizer / scheduler configuration."""

    def training_step(self, module: pl.LightningModule, loss: torch.Tensor) -> torch.Tensor:
        """Apply optimization for one training step; return loss for logging."""
        return loss

    def on_train_epoch_end(self, module: pl.LightningModule) -> None:
        """Optional hook for epoch-based schedulers."""

    def on_validation_epoch_end(self, module: pl.LightningModule) -> None:
        """Optional hook for validation-driven schedulers."""

    def optimizer_zero_grad(
        self, module: pl.LightningModule, epoch: int, batch_idx: int, optimizer
    ) -> None:
        """Optional Lightning hook; default is a no-op (Muon zeros grads in training_step)."""

    def _optim_cfg(self, name) -> dict:
        """Return optimizer-specific config dict, or empty dict if missing."""
        return self.config.get("training", {}).get(name, {})
    
    def _training_cfg(self) -> dict:
        """Return optimizer-specific config dict, or empty dict if missing."""
        return self.config.get("training", {})

    def _resolve_lr(self, module: pl.LightningModule, name) -> float:
        train_cfg = self._optim_cfg(name)
        lr = train_cfg["finetune_learning_rate"]
        if module.nfuture > 0:
            lr = train_cfg["finetune_learning_rate"]
        return lr
