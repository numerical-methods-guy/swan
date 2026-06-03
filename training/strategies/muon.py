"""Muon + AdamW dual-optimizer training strategy with manual optimization."""

import torch

import pytorch_lightning as pl

from training.strategies.base import TrainingStrategy


def split_params_for_muon(model: torch.nn.Module) -> tuple[list, list]:
    """Split parameters into Muon (2-D) and AdamW (everything else) groups.

    ``torch.optim.Muon`` requires exactly 2-D parameters (matrices).
    Conv filters (4-D), biases (1-D), and other tensors go to AdamW.
    """
    muon_params = []
    other_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2:
            muon_params.append(param)
        else:
            other_params.append(param)
    return muon_params, other_params


class MuonStrategy(TrainingStrategy):
    automatic_optimization = False

    def __init__(self, config: dict):
        super().__init__(config)
        if not hasattr(torch.optim, "Muon"):
            raise ImportError(
                "torch.optim.Muon is not available in this PyTorch build. "
                "Install PyTorch >= 2.9 or use --optimizer adam."
            )

    def configure_optimizers(self, module: pl.LightningModule):
        train_cfg = self.config["training"]
        lr = self._resolve_lr(module)

        muon_lr = train_cfg.get("muon_lr", lr)
        adamw_lr = train_cfg.get("adamw_lr", lr)
        muon_momentum = train_cfg.get("muon_momentum", 0.95)
        muon_wd = train_cfg.get("muon_weight_decay", 0.0)
        adamw_wd = train_cfg.get("adamw_weight_decay", 1e-4)

        muon_params, other_params = split_params_for_muon(module.model)

        muon_optimizer = torch.optim.Muon(
            muon_params,
            lr=muon_lr,
            momentum=muon_momentum,
            nesterov=True,
            ns_steps=5,
            weight_decay=muon_wd,
        )
        adamw_optimizer = torch.optim.AdamW(
            other_params,
            lr=adamw_lr,
            weight_decay=adamw_wd,
            foreach=True,
        )

        optimizers = [muon_optimizer, adamw_optimizer]

        milestones = train_cfg.get("lr_milestones", None)
        gamma = train_cfg.get("lr_gamma", 0.5)
        cosine_eta_min = train_cfg.get("cosine_eta_min", None)
        num_epochs = train_cfg.get("pretrain_epochs")

        if cosine_eta_min is not None:
            schedulers = [
                {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                        muon_optimizer, T_max=num_epochs, eta_min=cosine_eta_min
                    ),
                    "interval": "epoch",
                },
                {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                        adamw_optimizer, T_max=num_epochs, eta_min=cosine_eta_min
                    ),
                    "interval": "epoch",
                }
            ]
        elif milestones is not None:
            schedulers = [
                {
                    "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                        muon_optimizer, milestones=milestones, gamma=gamma
                    ),
                    "interval": "epoch",
                },
                {
                    "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                        adamw_optimizer, milestones=milestones, gamma=gamma
                    ),
                    "interval": "epoch",
                },
            ]
        else:
            schedulers = [
                {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        adamw_optimizer, mode="min", factor=0.5, patience=5
                    ),
                    "monitor": "val_loss",
                },
            ]

        return optimizers, schedulers

    def training_step(self, module: pl.LightningModule, loss: torch.Tensor) -> torch.Tensor:
        opt_muon, opt_adamw = module.optimizers()

        opt_muon.zero_grad(set_to_none=True)
        opt_adamw.zero_grad(set_to_none=True)

        module.manual_backward(loss)

        opt_muon.step()
        opt_adamw.step()

        return loss

    def _get_schedulers(self, module: pl.LightningModule) -> list:
        schedulers = module.lr_schedulers()
        if not schedulers:
            return []
        return schedulers if isinstance(schedulers, list) else [schedulers]

    def on_train_epoch_end(self, module: pl.LightningModule) -> None:
        for scheduler in self._get_schedulers(module):
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

    def on_validation_epoch_end(self, module: pl.LightningModule) -> None:
        for scheduler in self._get_schedulers(module):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                val_loss = module.trainer.callback_metrics.get("val_loss")
                if val_loss is not None:
                    scheduler.step(val_loss)
