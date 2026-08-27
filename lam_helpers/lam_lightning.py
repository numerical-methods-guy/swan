"""PyTorch-Lightning wrapper for LAMParadis.

Expected one-step batch keys:
- lr_halo: [B, 5, lr_win_nlat, lr_win_nlon]
- hr_patch_t0: [B, 5, patch_nlat_hr, patch_nlon_hr]
- lr_patch_t0_hrref: [B, 5, patch_nlat_hr, patch_nlon_hr]
- hr_patch_t1: [B, 5, patch_nlat_hr, patch_nlon_hr]
- lr_patch_t1_hrref: [B, 5, patch_nlat_hr, patch_nlon_hr]

Expected rollout-finetuning batch keys:
- lr_halo_seq: [B, T, 5, lr_win_nlat, lr_win_nlon]
- hr_patch_t0: [B, 5, patch_nlat_hr, patch_nlon_hr]
- lr_input_seq: [B, T, 5, patch_nlat_hr, patch_nlon_hr]
- hr_target_seq: [B, T, 5, patch_nlat_hr, patch_nlon_hr]
- lr_ref_seq: [B, T, 5, patch_nlat_hr, patch_nlon_hr]

lr_patch_t0_hrref / lr_input_seq must be constructed from the LR state at the
same current time as the HR encoder input, interpolated to HR patch resolution
and normalized using HR normalization statistics.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from lam_helpers.lam_blending import build_lr_blend_weight_hr
from lam_helpers.lam_model import build_lam_model


class LAMLightningModule(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(config)
        self.config = config
        self.model = build_lam_model(config)

        tc = config["training"]
        ft_cfg = config.get("finetuning", {})
        self.finetuning_enabled = bool(ft_cfg.get("enabled", False))
        self.initial_horizon = int(ft_cfg.get("initial_horizon", 1))
        self.max_horizon = int(ft_cfg.get("max_horizon", 1))
        self.epochs_per_horizon = int(ft_cfg.get("epochs_per_horizon", 1))
        self.lead_loss_decay = float(ft_cfg.get("lead_loss_decay", 1.0))
        self.detach_rollout = bool(ft_cfg.get("detach_rollout", True))
        self.validation_uses_curriculum = bool(
            ft_cfg.get("validation_uses_curriculum", True)
        )

        if self.initial_horizon < 1:
            raise ValueError("finetuning.initial_horizon must be >= 1")
        if self.max_horizon < self.initial_horizon:
            raise ValueError(
                "finetuning.max_horizon must be >= finetuning.initial_horizon"
            )
        if self.epochs_per_horizon < 1:
            raise ValueError("finetuning.epochs_per_horizon must be >= 1")
        if self.lead_loss_decay <= 0.0:
            raise ValueError("finetuning.lead_loss_decay must be > 0")

        self.lr = float(
            tc["finetune_learning_rate"]
            if self.finetuning_enabled
            else tc["learning_rate"]
        )

        self.milestones = tc.get("lr_milestones", None)
        self.lr_gamma = float(tc.get("lr_gamma", 0.5))

        lam_cfg = config["lam"]
        s_lat = int(lam_cfg["refinement_factor_lat"])
        s_lon = int(lam_cfg["refinement_factor_lon"])
        if s_lat != s_lon:
            raise ValueError("Only isotropic refinement is supported for blending.")

        patch_nlat_hr = int(lam_cfg["patch_nlat_lr"]) * s_lat
        patch_nlon_hr = int(lam_cfg["patch_nlon_lr"]) * s_lon

        blend_cfg = lam_cfg.get("blending", {})
        self.blending_enabled = bool(blend_cfg.get("enabled", False))
        self.apply_to_hr_input = bool(blend_cfg.get("apply_to_hr_input", False))
        self.apply_in_training_loss = bool(
            blend_cfg.get("apply_in_training_loss", True)
        )

        blend_width_hr = int(blend_cfg.get("width_hr", 0))
        self.training_edge_hr_weight = float(
            blend_cfg.get("training_edge_hr_weight", 0.0)
        )

        if not 0.0 <= self.training_edge_hr_weight <= 1.0:
            raise ValueError(
                "training_edge_hr_weight must be in [0, 1], "
                f"got {self.training_edge_hr_weight}"
            )

        w_lr = build_lr_blend_weight_hr(
            patch_nlat_hr=patch_nlat_hr,
            patch_nlon_hr=patch_nlon_hr,
            blend_width_hr=blend_width_hr,
            dtype=torch.float32,
        )

        if self.training_edge_hr_weight == 1.0:
            target_hr_mix = torch.ones_like(w_lr)
        elif self.training_edge_hr_weight == 0.0:
            target_hr_mix = 1.0 - w_lr
        else:
            target_hr_mix = (
                self.training_edge_hr_weight
                + (1.0 - self.training_edge_hr_weight) * (1.0 - w_lr)
            )

        self.register_buffer("blend_weight_lr_hr", w_lr, persistent=False)
        self.register_buffer(
            "target_hr_mix_weight", target_hr_mix, persistent=False
        )

    def forward(self, lr_halo: torch.Tensor, hr_patch_t0: torch.Tensor):
        """Raw model forward pass; pre-encoder input blending is handled in training."""
        return self.model(lr_halo, hr_patch_t0)

    def _build_hr_encoder_input(
        self,
        hr_state: torch.Tensor,
        lr_input_ref: torch.Tensor,
    ) -> torch.Tensor:
        """Blend current HR and same-time interpolated LR before HR encoding.

        The LR blend weight is one at the patch edge and zero in the free
        interior, so the encoder receives LR values at the outer edge and HR
        values in the unblended interior.
        """
        if not (self.blending_enabled and self.apply_to_hr_input):
            return hr_state

        if hr_state.shape != lr_input_ref.shape:
            raise ValueError(
                "hr_state and lr_input_ref must have identical shapes; got "
                f"{tuple(hr_state.shape)} and {tuple(lr_input_ref.shape)}"
            )

        w_lr = self.blend_weight_lr_hr.to(
            device=hr_state.device,
            dtype=hr_state.dtype,
        )

        return (1.0 - w_lr) * hr_state + w_lr * lr_input_ref

    def _build_target(
        self,
        hr_target: torch.Tensor,
        lr_ref: torch.Tensor,
    ) -> torch.Tensor:
        """Optionally create the blend-aware target used by the MSE loss."""
        if self.blending_enabled and self.apply_in_training_loss:
            w_hr = self.target_hr_mix_weight.to(
                device=hr_target.device,
                dtype=hr_target.dtype,
            )
            return w_hr * hr_target + (1.0 - w_hr) * lr_ref
        return hr_target

    @staticmethod
    def _compute_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)

    def _active_horizon(self, stage: str) -> int:
        if not self.finetuning_enabled:
            return 1
        if stage == "val" and not self.validation_uses_curriculum:
            return self.max_horizon

        epoch_index = max(int(self.current_epoch), 0)
        increments = epoch_index // self.epochs_per_horizon
        return min(self.initial_horizon + increments, self.max_horizon)

    def _shared_step(
        self,
        batch: dict,
        stage: str,
        backward_fn: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """Shared train/validation step.

        If ``backward_fn`` is provided (training only), the loss is
        backpropagated immediately:

        - one-step mode: backward on the single loss;
        - rollout mode: backward on each lead step's scaled loss right
          after its forward pass, freeing that lead step's activations
          before the next lead step runs.

        The returned tensor is detached whenever ``backward_fn`` was used.
        """
        if not self.finetuning_enabled:
            hr_encoder_input = self._build_hr_encoder_input(
                hr_state=batch["hr_patch_t0"],
                lr_input_ref=batch["lr_patch_t0_hrref"],
            )
            pred = self.model(batch["lr_halo"], hr_encoder_input)
            target = self._build_target(
                hr_target=batch["hr_patch_t1"],
                lr_ref=batch["lr_patch_t1_hrref"],
            )
            loss = self._compute_loss(pred, target)
            self.log(
                f"{stage}_loss",
                loss,
                on_step=(stage == "train"),
                on_epoch=True,
                prog_bar=True,
                sync_dist=False,
            )
            if backward_fn is not None:
                backward_fn(loss)
                return loss.detach()
            return loss

        horizon = self._active_horizon(stage)
        lr_halo_seq = batch["lr_halo_seq"]
        lr_input_seq = batch["lr_input_seq"]
        hr_target_seq = batch["hr_target_seq"]
        lr_ref_seq = batch["lr_ref_seq"]

        if lr_halo_seq.shape[1] < horizon:
            raise RuntimeError(
                f"Batch provides {lr_halo_seq.shape[1]} rollout steps, "
                f"but active horizon is {horizon}."
            )
        if lr_input_seq.shape[1] < horizon:
            raise RuntimeError(
                f"Batch provides {lr_input_seq.shape[1]} LR input steps, "
                f"but active horizon is {horizon}."
            )

        per_lead_backward = backward_fn is not None
        if per_lead_backward and not self.detach_rollout:
            raise RuntimeError(
                "Per-lead-step backward requires finetuning.detach_rollout=true. "
                "With detach_rollout=false each lead step's graph depends on "
                "the previous lead step, so the graphs cannot be freed "
                "independently."
            )

        hr_state = batch["hr_patch_t0"]

        # Precompute the normalization constant (identical to the old
        # running total_weight) so each lead loss can be scaled BEFORE its
        # individual backward pass.
        total_weight = sum(self.lead_loss_decay ** i for i in range(horizon))

        weighted_loss = hr_state.new_zeros(())

        for lead_idx in range(horizon):
            hr_encoder_input = self._build_hr_encoder_input(
                hr_state=hr_state,
                lr_input_ref=lr_input_seq[:, lead_idx],
            )
            pred = self.model(lr_halo_seq[:, lead_idx], hr_encoder_input)

            target = self._build_target(
                hr_target=hr_target_seq[:, lead_idx],
                lr_ref=lr_ref_seq[:, lead_idx],
            )
            lead_loss = self._compute_loss(pred, target)
            lead_weight = self.lead_loss_decay ** lead_idx
            scaled_lead_loss = (lead_weight / total_weight) * lead_loss

            self.log(
                f"{stage}_loss_lead_{lead_idx + 1}",
                lead_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
            )

            if per_lead_backward:
                # Backpropagate THIS lead step now. Its autograd graph (and
                # all of its saved activations) is freed before the next
                # lead step's forward pass, so peak GPU memory stays at
                # roughly one lead step instead of growing with the horizon.
                backward_fn(scaled_lead_loss)
                weighted_loss = weighted_loss + scaled_lead_loss.detach()
            else:
                weighted_loss = weighted_loss + scaled_lead_loss

            hr_state = pred.detach() if self.detach_rollout else pred

        loss = weighted_loss
        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )
        self.log(
            f"{stage}_active_horizon",
            float(horizon),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        optimizer = self.optimizers()
        optimizer.zero_grad(set_to_none=True)

        # Per-lead-step backward happens inside _shared_step via backward_fn.
        loss = self._shared_step(batch, "train", backward_fn=self.manual_backward)

        optimizer.step()
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            foreach=True,
        )

        if self.milestones is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.milestones,
                gamma=self.lr_gamma,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
