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

from model import ParadisModel
from pde_dataset_with_winds import PdeDatasetWithWinds


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
        self.model = ParadisModel(config)

        self.loss_fn = SquaredL2LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_l1 = L1LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_l2 = L2LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_w11 = W11LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)

        self.nfuture = 0

    def forward(self, fields, winds):
        return self.model(fields, winds)

    def training_step(self, batch, batch_idx):
        inp_fields, inp_winds, tar_fields, tar_winds = batch
        prd = self.model(inp_fields, inp_winds)
        for _ in range(self.nfuture):
            prd = self.model(prd, inp_winds)
        loss = self.loss_fn(prd, tar_fields)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inp_fields, inp_winds, tar_fields, tar_winds = batch
        prd = self.model(inp_fields, inp_winds)
        for _ in range(self.nfuture):
            prd = self.model(prd, inp_winds)
        loss = self.loss_fn(prd, tar_fields)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_l1", self.metric_l1(prd, tar_fields), sync_dist=True)
        self.log("val_l2", self.metric_l2(prd, tar_fields), sync_dist=True)
        self.log("val_w11", self.metric_w11(prd, tar_fields), sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Configure Adam with either MultiStepLR or ReduceLROnPlateau.

        MultiStepLR is used when ``training.lr_milestones`` and ``training.lr_gamma``
        are present in the config, providing a deterministic decay schedule that is
        more reproducible than plateau-based decay on a regenerated dataset.
        If those keys are absent, ReduceLROnPlateau is used as a fallback.
        """
        lr = self.config["training"]["learning_rate"]
        if self.nfuture > 0:
            lr = self.config["training"]["finetune_learning_rate"]

        # foreach=True enables the fused multi-tensor Adam kernel, avoiding
        # per-parameter Python overhead on the optimizer step.
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, foreach=True)

        milestones = self.config["training"].get("lr_milestones", None)
        gamma = self.config["training"].get("lr_gamma", 0.5)

        if milestones is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=gamma
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        """Zero gradients by setting them to None rather than filling with zeros.

        This avoids a full memset over all parameter gradient buffers, saving
        one memory write per parameter per step.
        """
        optimizer.zero_grad(set_to_none=True)

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
        dt=dt, nsteps=nsteps, dims=(nlat, nlon), normalize=True, device=device,
    )
    train_dataset.sht = train_dataset.solver.sht
    train_dataset.set_initial_condition("random")
    train_dataset.set_num_examples(config["data"]["num_train_examples"])

    val_dataset = PdeDatasetWithWinds(
        dt=dt, nsteps=nsteps, dims=(nlat, nlon), normalize=True, device=device,
    )
    val_dataset.sht = val_dataset.solver.sht
    val_dataset.set_initial_condition("random")
    val_dataset.set_num_examples(config["data"]["num_val_examples"])

    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Checkpoint to resume from (for finetuning only)",
    )

    known_args, unknown_args = parser.parse_known_args()

    mp.set_start_method("spawn", force=True)

    config = load_config(known_args.config)
    config = update_config_from_args(config, unknown_args)

    # TF32 on both the matmul and cuDNN paths. The cuDNN flag covers the
    # depthwise/pointwise convolutions in PARADIS's SepConv blocks, which
    # set_float32_matmul_precision alone does not affect.
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
        print(f"STARTING PRETRAINING FOR {config['training']['pretrain_epochs']} EPOCHS")
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
            callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="epoch")],
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
            callbacks=[finetune_checkpoint, LearningRateMonitor(logging_interval="epoch")],
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
