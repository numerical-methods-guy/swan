"""Adam training strategy with MultiStepLR or ReduceLROnPlateau."""

import torch

import pytorch_lightning as pl

from training.strategies.base import TrainingStrategy


class AdamWStrategy(TrainingStrategy):
    automatic_optimization = True

    def configure_optimizers(self, module: pl.LightningModule):
        lr = self._resolve_lr(module)

        train_cfg = self.config["training"]
        milestones = train_cfg.get("lr_milestones", None)
        gamma = train_cfg.get("lr_gamma", 0.5)

        # Initialize optimizer, grab AdamW specific hyperparameters from config -> training
        
        # Default to pytorch default values
        beta1 = train_cfg.get("beta1", 0.9)
        beta2 = train_cfg.get("beta2", 0.999)
        decay_lambda = train_cfg.get("adamw_weight_decay", 0.01)
        cosine_eta_min = train_cfg.get("cosine_eta_min", None)

        optimizer = torch.optim.AdamW(module.parameters(), lr=lr, betas = (beta1,beta2), weight_decay=decay_lambda, foreach=True)

        if cosine_eta_min is not None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=train_cfg.get("pretrain_epochs"),
                eta_min=cosine_eta_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
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
