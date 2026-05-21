import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from torch_harmonics.examples.losses import (
    SquaredL2LossS2,
    L1LossS2,
    L2LossS2,
    W11LossS2,
)

from model.paradis import Paradis
from pde_dataset_with_winds import PdeDatasetWithWinds
from utils.loss import ParadisLoss


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def update_config_from_args(config, unknown_args):
    """Update config with command-line arguments in dot notation."""
    for i in range(0, len(unknown_args), 2):
        if i + 1 >= len(unknown_args):
            break

        key = unknown_args[i].lstrip("-")
        val = unknown_args[i + 1]

        keys = key.split(".")
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        try:
            if "." in val:
                val = float(val)
            else:
                val = int(val)
        except ValueError:
            pass

        current[keys[-1]] = val

    return config


def build_paradis_loss(config):
    """Construct a ParadisLoss for the shallow water equation setting.

    The SWE model has three output channels (geopotential, vorticity, divergence)
    with no pressure-level structure, so all variables are treated as surface
    variables and pressure weighting is effectively bypassed.
    """
    nlat = config["data"]["nlat"]
    loss_cfg = config.get("loss", {})

    loss_function = loss_cfg.get("loss_function", "reversed_huber")
    delta_loss = loss_cfg.get("delta_loss", 1.0)

    lat_grid = torch.linspace(-90.0, 90.0, nlat, dtype=torch.float32)

    num_features = 3
    num_surface_vars = 3
    output_name_order = ["h", "vorticity", "divergence"]

    # Dummy single pressure level so the atmospheric loop is a no-op.
    pressure_levels = torch.tensor([1000.0], dtype=torch.float32)

    var_loss_weights = torch.ones(num_features, dtype=torch.float32)

    return ParadisLoss(
        loss_function=loss_function,
        lat_grid=lat_grid,
        pressure_levels=pressure_levels,
        num_features=num_features,
        num_surface_vars=num_surface_vars,
        var_loss_weights=var_loss_weights,
        output_name_order=output_name_order,
        delta_loss=delta_loss,
    )


def split_params_for_muon(model):
    """Split model parameters into two groups for the dual-optimizer setup.

    ``torch.optim.Muon`` requires **exactly** 2-D parameters (matrices).
    Conv filters (4-D), biases (1-D), and any other non-matrix tensors are
    routed to AdamW instead.

    Args:
        model: The nn.Module whose parameters will be split.

    Returns:
        tuple(list, list): (muon_params, other_params) where muon_params
        contains all exactly-2D parameter tensors and other_params contains
        everything else.
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


class SWELightningModule(pl.LightningModule):
    """Lightning module for the PARADIS shallow water equation model."""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

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

        self.automatic_optimization = False

    def forward(self, fields, winds):
        return self.model(fields, winds)

    def training_step(self, batch, batch_idx):
        opt_muon, opt_adamw = self.optimizers()

        inp_fields, inp_winds, tar_fields, tar_winds = batch
        prd = self.model(inp_fields, inp_winds)
        for _ in range(self.nfuture):
            prd = self.model(prd, inp_winds)

        loss = self.loss_fn(prd, tar_fields)

        opt_muon.zero_grad(set_to_none=True)
        opt_adamw.zero_grad(set_to_none=True)

        self.manual_backward(loss)

        opt_muon.step()
        opt_adamw.step()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def _get_schedulers(self):
        """Return all active LR schedulers as a flat list."""
        schedulers = self.lr_schedulers()
        if not schedulers:
            return []
        return schedulers if isinstance(schedulers, list) else [schedulers]

    def on_train_epoch_end(self):
        """Step epoch-based schedulers (e.g. MultiStepLR) at the end of each training epoch."""
        for scheduler in self._get_schedulers():
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

    def on_validation_epoch_end(self):
        """Step ReduceLROnPlateau after validation so it sees the current val_loss."""
        for scheduler in self._get_schedulers():
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                val_loss = self.trainer.callback_metrics.get("val_loss")
                if val_loss is not None:
                    scheduler.step(val_loss)

    def validation_step(self, batch, batch_idx):
        inp_fields, inp_winds, tar_fields, tar_winds = batch
        prd = self.model(inp_fields, inp_winds)
        for _ in range(self.nfuture):
            prd = self.model(prd, inp_winds)
        loss = self.loss_fn(prd, tar_fields)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_sq_l2", self.metric_sq_l2(prd, tar_fields), sync_dist=True)
        self.log("val_l1", self.metric_l1(prd, tar_fields), sync_dist=True)
        self.log("val_l2", self.metric_l2(prd, tar_fields), sync_dist=True)
        self.log("val_w11", self.metric_w11(prd, tar_fields), sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Configure Muon (for exactly 2-D weight matrices) + AdamW (for everything else).

        ``torch.optim.Muon`` only accepts exactly 2-D parameters, so conv
        weights (4-D), biases (1-D), and any other non-matrix tensors go to
        AdamW.  The AdamW LR defaults to ``lr * 0.1`` since Muon's effective
        step size is larger.  All LR values are independently configurable via
        the config keys ``muon_lr``, ``adamw_lr``, ``muon_momentum``,
        ``muon_weight_decay``, and ``adamw_weight_decay``.

        MultiStepLR is applied to both optimizers when
        ``training.lr_milestones`` and ``training.lr_gamma`` are present.
        Otherwise ReduceLROnPlateau monitors ``val_loss`` for AdamW only
        (Muon with a fixed LR is the standard recommendation).
        """
        train_cfg = self.config["training"]
        lr = train_cfg["learning_rate"]
        if self.nfuture > 0:
            lr = train_cfg["finetune_learning_rate"]

        muon_lr = train_cfg.get("muon_lr", lr)
        adamw_lr = train_cfg.get("adamw_lr", lr)
        muon_momentum = train_cfg.get("muon_momentum", 0.95)
        muon_wd = train_cfg.get("muon_weight_decay", 0.0)
        adamw_wd = train_cfg.get("adamw_weight_decay", 1e-4)

        muon_params, other_params = split_params_for_muon(self.model)

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

        if milestones is not None:
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

    def on_load_checkpoint(self, checkpoint):
        """Filter out W11 mesh buffers that are recomputed on instantiation."""
        state_dict = checkpoint["state_dict"]
        for k in ["metric_w11.k_phi_mesh", "metric_w11.k_theta_mesh"]:
            if k in state_dict:
                del state_dict[k]


def create_datasets(config, device):
    """Create training and validation datasets."""
    dt = config["data"]["dt"]
    nsteps = dt // config["data"]["dt_solver"]
    nlat = config["data"]["nlat"]
    nlon = config["data"]["nlon"]

    train_dataset = PdeDatasetWithWinds(
        dt=dt,
        nsteps=nsteps,
        dims=(nlat, nlon),
        normalize=True,
        device=device,
    )
    train_dataset.sht = train_dataset.solver.sht
    train_dataset.set_initial_condition("random")
    train_dataset.set_num_examples(config["data"]["num_train_examples"])

    val_dataset = PdeDatasetWithWinds(
        dt=dt,
        nsteps=nsteps,
        dims=(nlat, nlon),
        normalize=True,
        device=device,
    )
    val_dataset.sht = val_dataset.solver.sht
    val_dataset.set_initial_condition("random")
    val_dataset.set_num_examples(config["data"]["num_val_examples"])

    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Checkpoint to resume from (for finetuning only)",
    )

    known_args, unknown_args = parser.parse_known_args()

    mp.set_start_method("spawn", force=True)

    config = load_config(known_args.config)
    config = update_config_from_args(config, unknown_args)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    pl.seed_everything(config["experiment"]["seed"], workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, val_dataset = create_datasets(config, device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        persistent_workers=(config["data"]["num_workers"] > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        persistent_workers=(config["data"]["num_workers"] > 0),
    )

    model = SWELightningModule(config)

    precision = 32
    if config["training"]["amp_mode"] == "fp16":
        precision = 16
    elif config["training"]["amp_mode"] == "bf16":
        precision = "bf16"

    if config["training"]["pretrain_epochs"] > 0 and known_args.resume_from is None:
        print("\n" + "=" * 70)
        print(
            f"STARTING PRETRAINING FOR {config['training']['pretrain_epochs']} EPOCHS"
        )
        print("=" * 70 + "\n")

        logger = TensorBoardLogger(
            config["training"]["save_dir"], name=config["experiment"]["name"]
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            filename="pretrain-{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
            mode="min",
            save_last=True,
        )

        trainer = pl.Trainer(
            max_epochs=config["training"]["pretrain_epochs"],
            logger=logger,
            callbacks=[
                checkpoint_callback,
                LearningRateMonitor(logging_interval="epoch"),
            ],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision=precision,
            log_every_n_steps=config["training"]["log_every_n_steps"],
            check_val_every_n_epoch=1,
            enable_progress_bar=True,
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print(f"\nBest pretrain checkpoint: {checkpoint_callback.best_model_path}")

    elif known_args.resume_from is not None:
        print("\n" + "=" * 70)
        print(f"SKIPPING PRETRAINING - Loading checkpoint: {known_args.resume_from}")
        print("=" * 70 + "\n")

        checkpoint = torch.load(known_args.resume_from, map_location=device)
        state_dict = checkpoint["state_dict"]
        for key in ["metric_w11.k_phi_mesh", "metric_w11.k_theta_mesh"]:
            if key in state_dict:
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully.\n")

    if config["training"]["finetune_epochs"] > 0:
        print("\n" + "=" * 70)
        print(f"STARTING FINETUNING FOR {config['training']['finetune_epochs']} EPOCHS")
        print("=" * 70 + "\n")

        dt = config["data"]["dt"]
        new_nsteps = 2 * dt // config["data"]["dt_solver"]
        train_dataset.nsteps = new_nsteps
        val_dataset.nsteps = new_nsteps
        model.nfuture = config["training"]["nfuture"]

        finetune_logger = TensorBoardLogger(
            config["training"]["save_dir"],
            name=f"{config['experiment']['name']}_finetune",
        )
        finetune_checkpoint = ModelCheckpoint(
            monitor="val_loss",
            filename="finetune-{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
            mode="min",
            save_last=True,
        )

        finetune_trainer = pl.Trainer(
            max_epochs=config["training"]["finetune_epochs"],
            logger=finetune_logger,
            callbacks=[
                finetune_checkpoint,
                LearningRateMonitor(logging_interval="epoch"),
            ],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision=precision,
            log_every_n_steps=config["training"]["log_every_n_steps"],
            check_val_every_n_epoch=1,
            enable_progress_bar=True,
        )
        finetune_trainer.fit(
            model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
        print(f"\nBest finetune checkpoint: {finetune_checkpoint.best_model_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
