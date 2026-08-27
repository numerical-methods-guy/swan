#!/usr/bin/env python3
"""
train_paradis_hr.py

Train the global PARADIS model at HR resolution using the pre-generated
HDF5 dataset (swe_paired.h5) from generate_dataset.py.

This script is identical to train.py except:
  1. create_datasets() uses HRGlobalDataset instead of PdeDatasetWithWinds.
  2. config["data"]["nlat"] and ["nlon"] are overridden to the HR grid
     dimensions (lr_nlat * refinement_factor_lat, lr_nlon * refinement_factor_lon)
     so that Paradis.__init__ builds the correct mesh_size.
  3. h5_path is read from config["dataset"]["h5_path"].

All optimizer, scheduler, loss, and Lightning module logic is unchanged
from train.py to ensure a fair comparison.

Usage (Colab):
    !python train_paradis_hr.py --config config_paradis_lam.yaml

    # Override individual keys:
    !python train_paradis_hr.py --config config_paradis_lam.yaml \
        --training.pretrain_epochs 30 \
        --dataset.h5_path data/swe_paired.h5
"""

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
from swe_solver.hr_global_dataset import HRGlobalDataset
from utils.loss import ParadisLoss


# ---------------------------------------------------------------------------
# Config helpers  (identical to train.py)
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def update_config_from_args(config: dict, unknown_args: list) -> dict:
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
            val = float(val) if "." in val else int(val)
        except ValueError:
            pass
        current[keys[-1]] = val
    return config


def _override_hr_dims(config: dict) -> dict:
    """
    Override data.nlat / data.nlon to the HR grid so Paradis and ParadisLoss
    build their mesh at HR resolution.
    """
    lc = config["lam"]
    s_lat = int(lc["refinement_factor_lat"])
    s_lon = int(lc["refinement_factor_lon"])
    lr_nlat = int(config["data"]["nlat"])
    lr_nlon = int(config["data"]["nlon"])
    config["data"]["nlat"] = lr_nlat * s_lat
    config["data"]["nlon"] = lr_nlon * s_lon
    return config


# ---------------------------------------------------------------------------
# Loss  (identical to train.py)
# ---------------------------------------------------------------------------

def build_paradis_loss(config: dict) -> ParadisLoss:
    nlat = config["data"]["nlat"]
    loss_cfg = config.get("loss", {})
    loss_function = loss_cfg.get("loss_function", "reversed_huber")
    delta_loss    = loss_cfg.get("delta_loss", 1.0)
    lat_grid = torch.linspace(-90.0, 90.0, nlat, dtype=torch.float32)
    pressure_levels = torch.tensor([1000.0], dtype=torch.float32)
    var_loss_weights = torch.ones(3, dtype=torch.float32)
    return ParadisLoss(
        loss_function    = loss_function,
        lat_grid         = lat_grid,
        pressure_levels  = pressure_levels,
        num_features     = 3,
        num_surface_vars = 3,
        var_loss_weights = var_loss_weights,
        output_name_order= ["h", "vorticity", "divergence"],
        delta_loss       = delta_loss,
    )


# ---------------------------------------------------------------------------
# Lightning module  (identical to SWELightningModule in train.py)
# ---------------------------------------------------------------------------

class SWELightningModule(pl.LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.nlat   = config["data"]["nlat"]
        self.nlon   = config["data"]["nlon"]
        self.grid   = config["data"]["grid"]

        self.model   = Paradis(config)
        self.loss_fn = build_paradis_loss(config)

        self.metric_sq_l2 = SquaredL2LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_l1    = L1LossS2       (nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_l2    = L2LossS2       (nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_w11   = W11LossS2      (nlat=self.nlat, nlon=self.nlon, grid=self.grid)
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
        self.log("val_loss",    loss,                              prog_bar=True, sync_dist=True)
        self.log("val_sq_l2",   self.metric_sq_l2(prd, tar_fields), sync_dist=True)
        self.log("val_l1",      self.metric_l1   (prd, tar_fields), sync_dist=True)
        self.log("val_l2",      self.metric_l2   (prd, tar_fields), sync_dist=True)
        self.log("val_w11",     self.metric_w11  (prd, tar_fields), sync_dist=True)
        return loss

    def configure_optimizers(self):
        lr = self.config["training"]["learning_rate"]
        if self.nfuture > 0:
            lr = self.config["training"]["finetune_learning_rate"]
        optimizer  = torch.optim.Adam(self.parameters(), lr=lr, foreach=True)
        milestones = self.config["training"].get("lr_milestones", None)
        gamma      = self.config["training"].get("lr_gamma", 0.5)
        if milestones is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=gamma)
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5)
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        for k in ["metric_w11.k_phi_mesh", "metric_w11.k_theta_mesh"]:
            if k in state_dict:
                del state_dict[k]


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def create_datasets(config: dict):
    dsc          = config.get("dataset", {})
    h5_path      = dsc.get("h5_path", "data/swe_paired.h5")
    preload      = dsc.get("preload", False)
    num_train    = int(config["data"]["num_train_examples"])
    num_val      = int(config["data"]["num_val_examples"])

    common = dict(
        h5_path       = h5_path,
        num_train_ics = num_train,
        num_val_ics   = num_val,
        normalize     = config["data"].get("normalize", True),
        preload       = preload,
    )
    train_ds = HRGlobalDataset(split="train", **common)
    val_ds   = HRGlobalDataset(split="val",   **common)

    print(f"  HR PARADIS dataset loaded from: {h5_path}")
    print(f"  HR grid : {config['data']['nlat']} x {config['data']['nlon']}")
    print(f"  Train ICs : {len(train_ds)}  |  Val ICs : {len(val_ds)}")
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train HR PARADIS on HDF5 data")
    parser.add_argument("--config",      default="config_paradis_lam.yaml")
    parser.add_argument("--resume_from", default=None,
                        help="Checkpoint to resume from (for finetuning only)")
    known_args, unknown_args = parser.parse_known_args()

    mp.set_start_method("spawn", force=True)

    config = load_config(known_args.config)
    config = update_config_from_args(config, unknown_args)
    config = _override_hr_dims(config)   # nlat/nlon → HR dims before building model

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision("high")

    pl.seed_everything(config["experiment"]["seed"], workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds = create_datasets(config)

    bsz          = int(config["data"]["batch_size"])
    num_workers  = int(config["data"].get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True,
                              num_workers=num_workers,
                              persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=bsz, shuffle=False,
                              num_workers=num_workers,
                              persistent_workers=(num_workers > 0))

    model = SWELightningModule(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    precision = {"fp16": 16, "bf16": "bf16"}.get(
        config["training"].get("amp_mode", "none"), 32)

    # --- Pretraining ---
    if config["training"]["pretrain_epochs"] > 0 and known_args.resume_from is None:
        print("\n" + "=" * 70)
        print(f"STARTING PRETRAINING FOR {config['training']['pretrain_epochs']} EPOCHS")
        print("=" * 70 + "\n")

        logger = TensorBoardLogger(
            config["training"]["save_dir"],
            name=config["experiment"]["name"] + "_HR_PARADIS")

        ckpt_cb = ModelCheckpoint(
            monitor="val_loss",
            filename="hr_paradis-pretrain-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3, mode="min", save_last=True)

        trainer = pl.Trainer(
            max_epochs          = config["training"]["pretrain_epochs"],
            logger              = logger,
            callbacks           = [ckpt_cb, LearningRateMonitor("epoch")],
            accelerator         = "gpu" if torch.cuda.is_available() else "cpu",
            devices             = 1,
            precision           = precision,
            log_every_n_steps   = config["training"]["log_every_n_steps"],
            check_val_every_n_epoch = 1,
            enable_progress_bar = True,
        )
        trainer.fit(model, train_loader, val_loader)
        print(f"\nBest pretrain checkpoint: {ckpt_cb.best_model_path}")

    elif known_args.resume_from is not None:
        print(f"\nLoading checkpoint: {known_args.resume_from}")
        ckpt = torch.load(known_args.resume_from, map_location=device)
        sd   = ckpt["state_dict"]
        for k in ["metric_w11.k_phi_mesh", "metric_w11.k_theta_mesh"]:
            sd.pop(k, None)
        model.load_state_dict(sd, strict=False)
        print("Checkpoint loaded.\n")

    # --- Finetuning (optional) ---
    if config["training"]["finetune_epochs"] > 0:
        print("\n" + "=" * 70)
        print(f"STARTING FINETUNING FOR {config['training']['finetune_epochs']} EPOCHS")
        print("=" * 70 + "\n")

        model.nfuture = config["training"]["nfuture"]

        ft_logger = TensorBoardLogger(
            config["training"]["save_dir"],
            name=config["experiment"]["name"] + "_HR_PARADIS_finetune")

        ft_ckpt_cb = ModelCheckpoint(
            monitor="val_loss",
            filename="hr_paradis-finetune-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3, mode="min", save_last=True)

        ft_trainer = pl.Trainer(
            max_epochs          = config["training"]["finetune_epochs"],
            logger              = ft_logger,
            callbacks           = [ft_ckpt_cb, LearningRateMonitor("epoch")],
            accelerator         = "gpu" if torch.cuda.is_available() else "cpu",
            devices             = 1,
            precision           = precision,
            log_every_n_steps   = config["training"]["log_every_n_steps"],
            check_val_every_n_epoch = 1,
            enable_progress_bar = True,
        )
        ft_trainer.fit(model, train_loader, val_loader)
        print(f"\nBest finetune checkpoint: {ft_ckpt_cb.best_model_path}")

    print("\n" + "=" * 70)
    print("HR PARADIS TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
