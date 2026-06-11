"""Adam training strategy with MultiStepLR or ReduceLROnPlateau."""

import torch

import pytorch_lightning as pl

from training.strategies.base import TrainingStrategy


class AdamWStrategy(TrainingStrategy):
    automatic_optimization = True
    optimizer_name = "adamw"

    def configure_optimizers(self, module: pl.LightningModule):
        lr = self._resolve_lr(module, self.optimizer_name)
        train_adamw_cfg = self._optim_cfg(self.optimizer_name)
        train_common_cfg = self._training_cfg()


        # Initialize optimizer, grab AdamW specific hyperparameters from config -> training
        
        # Default to pytorch default values
        beta1 = train_adamw_cfg.get("beta1", 0.9)
        beta2 = train_adamw_cfg.get("beta2", 0.999)
        epsilon = train_adamw_cfg.get("epsilon", 1.0e-8)
        weight_decay = train_adamw_cfg.get("weight_decay", 0.01)

        milestones = train_common_cfg.get("lr_milestones", None)
        gamma = train_common_cfg.get("lr_gamma", 0.5)
        cosine_eta_min = train_common_cfg.get("cosine_eta_min", None)

        optimizer = torch.optim.AdamW(module.parameters(), lr=lr, betas = (beta1,beta2), eps=epsilon, weight_decay=weight_decay, foreach=True)

        if cosine_eta_min is not None:
            num_epochs = train_common_cfg.get(
                "finetune_epochs" if module.nfuture > 0 else "pretrain_epochs"
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(num_epochs)),
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
