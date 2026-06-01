"""Adam training strategy with MultiStepLR or ReduceLROnPlateau."""

import torch

import pytorch_lightning as pl

from training.strategies.base import TrainingStrategy


class AdamStrategy(TrainingStrategy):
    automatic_optimization = True

    def configure_optimizers(self, module: pl.LightningModule):
        lr = self._resolve_lr(module)
        optimizer = torch.optim.Adam(module.parameters(), lr=lr, foreach=True)

        train_cfg = self.config["training"]
        milestones = train_cfg.get("lr_milestones", None)
        gamma = train_cfg.get("lr_gamma", 0.5)

        if milestones is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=gamma
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def optimizer_zero_grad(
        self, module: pl.LightningModule, epoch: int, batch_idx: int, optimizer
    ) -> None:
        optimizer.zero_grad(set_to_none=True)
