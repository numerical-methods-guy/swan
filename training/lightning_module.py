"""PyTorch Lightning module for PARADIS shallow-water training."""

import pytorch_lightning as pl
import torch

from torch_harmonics.examples.losses import (
    L1LossS2,
    L2LossS2,
    SquaredL2LossS2,
    W11LossS2,
)

from model.paradis import Paradis
from training.loss import build_paradis_loss
from training.strategies import build_strategy


class SWELightningModule(pl.LightningModule):
    """Lightning module for the PARADIS shallow water equation model."""

    def __init__(self, config: dict, optimizer: str = "adam"):
        super().__init__()
        self.save_hyperparameters({"config": config, "optimizer": optimizer})
        self.config = config
        self.optimizer_name = optimizer

        self.nlat = config["data"]["nlat"]
        self.nlon = config["data"]["nlon"]
        self.grid = config["data"]["grid"]

        if "paradis" not in config["model"]:
            raise ValueError(
                "PARADIS model config not found. Add a 'model.paradis' section to your config."
            )
        self.model = Paradis(config)

        self.loss_fn = build_paradis_loss(config)
        self.metric_sq_l2 = SquaredL2LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_l1 = L1LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_l2 = L2LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_w11 = W11LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)

        self.nfuture = 0
        self.strategy = build_strategy(optimizer, config)
        self.automatic_optimization = self.strategy.automatic_optimization

    def forward(self, fields, winds):
        return self.model(fields, winds)

    def _compute_loss(self, batch):
        inp_fields, inp_winds, tar_fields, _tar_winds = batch
        prd = self.model(inp_fields, inp_winds)
        for _ in range(self.nfuture):
            prd = self.model(prd, inp_winds)
        return prd, tar_fields, self.loss_fn(prd, tar_fields)

    def training_step(self, batch, batch_idx):
        if hasattr(self.strategy, "training_step_batch"):
            return self.strategy.training_step_batch(self, batch, batch_idx)

        prd, tar_fields, loss = self._compute_loss(batch)
        del prd, tar_fields
        result = self.strategy.training_step(self, loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        prd, tar_fields, loss = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_sq_l2", self.metric_sq_l2(prd, tar_fields), sync_dist=True)
        self.log("val_l1", self.metric_l1(prd, tar_fields), sync_dist=True)
        self.log("val_l2", self.metric_l2(prd, tar_fields), sync_dist=True)
        self.log("val_w11", self.metric_w11(prd, tar_fields), sync_dist=True)
        return loss

    def configure_optimizers(self):
        return self.strategy.configure_optimizers(self)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        self.strategy.optimizer_zero_grad(self, epoch, batch_idx, optimizer)

    def on_train_epoch_end(self):
        self.strategy.on_train_epoch_end(self)

    def on_validation_epoch_end(self):
        self.strategy.on_validation_epoch_end(self)

    def on_load_checkpoint(self, checkpoint):
        """Filter out W11 mesh buffers that are recomputed on instantiation."""
        state_dict = checkpoint["state_dict"]
        for k in ["metric_w11.k_phi_mesh", "metric_w11.k_theta_mesh"]:
            if k in state_dict:
                del state_dict[k]
