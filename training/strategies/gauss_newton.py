"""Gauss-Newton training strategy with manual batch updates."""

from typing import Any

import pytorch_lightning as pl
import torch

from training.optimizers import build_gauss_newton
from training.strategies.base import TrainingStrategy


class GaussNewtonStrategy(TrainingStrategy):
    """Run a Gauss-Newton update using the full training batch."""

    automatic_optimization = False

    def __init__(self, config: dict):
        super().__init__(config)
        self.gauss_newton = build_gauss_newton(config)

    def configure_optimizers(self, module: pl.LightningModule) -> Any:
        return None

    def training_step_batch(
        self,
        module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> torch.Tensor:
        inp_fields, inp_winds, tar_fields, _tar_winds = batch
        stats = self.gauss_newton.step(
            module.model,
            inp_fields,
            inp_winds,
            tar_fields,
            nfuture=module.nfuture,
        )

        module.log("train_loss", stats.loss, on_step=True, on_epoch=True, prog_bar=True)
        module.log("gn_residual_norm", stats.residual_norm, on_step=True)
        module.log("gn_gradient_norm", stats.gradient_norm, on_step=True)
        module.log("gn_step_norm", stats.step_norm, on_step=True, prog_bar=True)
        module.log("gn_cg_iterations", float(stats.cg_iterations), on_step=True)
        module.log("gn_cg_residual_norm", stats.cg_residual_norm, on_step=True)
        return stats.loss
