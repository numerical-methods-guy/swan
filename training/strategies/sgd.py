"""SGD training strategy with MultiStepLR or ReduceLROnPlateau."""

import torch

import pytorch_lightning as pl

from training.strategies.base import TrainingStrategy


class SGDStrategy(TrainingStrategy):
    automatic_optimization = True

    def configure_optimizers(self, module: pl.LightningModule):
        lr = self._resolve_lr(module, "sgd")
        train_common_cfg = self._training_cfg()
        train_sgd_cfg = self._optim_cfg("sgd")

        milestones = train_common_cfg.get("lr_milestones", None)
        gamma = train_common_cfg.get("lr_gamma", 0.5)
        cosine_eta_min = train_common_cfg.get("cosine_eta_min", None)
        num_epochs = train_common_cfg.get("pretrain_epochs")

        momentum = train_sgd_cfg.get("momentum", 0.9)
        weight_decay = train_sgd_cfg.get("weight_decay", 0.0)


        print(
            f"Using SGD optimizer with lr={lr}, "
            f"momentum={momentum}, "
            f"weight_decay={weight_decay}"
        )

        optimizer = torch.optim.SGD(
            module.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        if cosine_eta_min is not None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
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