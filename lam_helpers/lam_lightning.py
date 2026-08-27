# lam_lightning.py
"""
PyTorch-Lightning wrapper for LAMParadis.

Mirrors the optimizer/scheduler logic in train.py exactly:
  - Optimizer : torch.optim.Adam(foreach=True), no weight decay
  - Scheduler : MultiStepLR if training.lr_milestones is set in config,
                otherwise ReduceLROnPlateau(factor=0.5, patience=5)
  - optimizer_zero_grad: set_to_none=True (carried from train.py)
  - Loss: plain MSE over the normalised HR patch
    (ParadisLoss is for the global sphere; its latitude/pressure weighting
     is meaningless on a local patch, and its SWE construction in train.py
     already degenerates to near-uniform weighted MSE anyway)

Batch dict expected from LAMPatchDataset:
    {
      "lr_halo"     : [B, 5, lr_win_nlat, lr_win_nlon],
      "hr_patch_t0" : [B, 3, patch_nlat_hr, patch_nlon_hr],
      "hr_patch_t1" : [B, 3, patch_nlat_hr, patch_nlon_hr],
    }
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from lam_model import build_lam_model


class LAMLightningModule(pl.LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.model  = build_lam_model(config)

        tc = config["training"]
        self.lr         = float(tc["learning_rate"])
        self.milestones = tc.get("lr_milestones", None)
        self.lr_gamma   = float(tc.get("lr_gamma", 0.5))

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, lr_halo: torch.Tensor, hr_patch_t0: torch.Tensor):
        return self.model(lr_halo, hr_patch_t0)

    # -----------------------------------------------------------------------
    # Steps
    # -----------------------------------------------------------------------

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        pred = self.model(batch["lr_halo"], batch["hr_patch_t0"])
        loss = F.mse_loss(pred, batch["hr_patch_t1"])
        self.log(
            f"{stage}_loss", loss,
            on_step  = (stage == "train"),
            on_epoch = True,
            prog_bar = True,
            sync_dist= False,
        )
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int):
        self._shared_step(batch, "val")

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        """Zero gradients by setting to None — mirrors train.py."""
        optimizer.zero_grad(set_to_none=True)

    # -----------------------------------------------------------------------
    # Optimiser + scheduler  (mirrors configure_optimizers in train.py)
    # -----------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr      = self.lr,
            foreach = True,
        )

        if self.milestones is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones = self.milestones,
                gamma      = self.lr_gamma,
            )
            return {
                "optimizer":    optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode    = "min",
                factor  = 0.5,
                patience= 5,
            )
            return {
                "optimizer":    optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }
