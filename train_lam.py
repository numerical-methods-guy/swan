# train_lam.py
"""
Training script for LAMParadis.

Replaces the old on-the-fly PdeDatasetWithWinds approach with the
pre-generated HDF5 dataset (generate_dataset.py output) read through
LAMPatchDataset.  The old create_lr_hr_datasets, recompute_all_stats,
and builtins hacks are all removed.

Usage (Colab):
    !python train_lam.py --config config_paradis_lam.yaml

    # Override any config key with dot notation:
    !python train_lam.py --config config_paradis_lam.yaml \
        --lam.halo_radius 6 \
        --training.lam_epochs 50
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from lam_patch_dataset import LAMPatchDataset
from lam_lightning import LAMLightningModule


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def update_config_from_args(config: dict, unknown_args: list) -> dict:
    """Dot-notation config overrides, e.g. --lam.halo_radius 6"""
    for i in range(0, len(unknown_args) - 1, 2):
        key = unknown_args[i].lstrip("-")
        val = unknown_args[i + 1]
        keys = key.split(".")
        node = config
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        try:
            val = int(val) if "." not in val else float(val)
        except ValueError:
            pass
        node[keys[-1]] = val
    return config


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def create_datasets(config: dict):
    """Build train and val LAMPatchDataset from the pre-generated HDF5 file."""
    dc   = config["data"]
    lc   = config["lam"]
    dsc  = config.get("dataset", {})

    h5_path          = dsc.get("h5_path", "data/swe_paired.h5")
    preload          = dsc.get("preload", False)
    patch_nlat_lr    = int(lc["patch_nlat_lr"])
    patch_nlon_lr    = int(lc["patch_nlon_lr"])
    halo_radius      = int(lc["halo_radius"])
    exclude_pole_rows= int(lc.get("exclude_pole_rows", 4))
    num_train_ics    = int(dc["num_train_examples"])

    common = dict(
        h5_path           = h5_path,
        patch_nlat_lr     = patch_nlat_lr,
        patch_nlon_lr     = patch_nlon_lr,
        halo_radius       = halo_radius,
        exclude_pole_rows = exclude_pole_rows,
        num_train_ics     = num_train_ics,
        normalize         = dc.get("normalize", True),
        preload           = preload,
    )

    train_ds = LAMPatchDataset(split="train", **common)
    val_ds   = LAMPatchDataset(split="val",   **common)

    print(train_ds.geometry_summary())
    print(f"  Train patches : {len(train_ds)}")
    print(f"  Val   patches : {len(val_ds)}")
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train LAMParadis")
    parser.add_argument("--config", default="config_paradis_lam.yaml")
    known, unknown = parser.parse_known_args()

    config = load_config(known.config)
    config = update_config_from_args(config, unknown)

    pl.seed_everything(config["experiment"]["seed"], workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- datasets & loaders --------------------------------------------------
    train_ds, val_ds = create_datasets(config)

    batch_size  = int(config["data"]["batch_size"])
    num_workers = int(config["data"].get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        persistent_workers = num_workers > 0,
        pin_memory  = device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        persistent_workers = num_workers > 0,
        pin_memory  = device.type == "cuda",
    )

    # --- model ---------------------------------------------------------------
    model = LAMLightningModule(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- precision -----------------------------------------------------------
    amp_mode  = config["training"].get("amp_mode", "none")
    precision = {"fp16": 16, "bf16": "bf16"}.get(amp_mode, 32)

    # --- callbacks & logger --------------------------------------------------
    logger = TensorBoardLogger(
        save_dir = config["training"]["save_dir"],
        name     = config["experiment"]["name"] + "_LAM",
    )

    checkpoint_cb = ModelCheckpoint(
        monitor   = "val_loss",
        filename  = "lam-epoch{epoch:02d}-val{val_loss:.4f}",
        save_top_k= 3,
        mode      = "min",
        save_last = True,
    )

    max_epochs = config["training"].get(
        "lam_epochs", config["training"]["pretrain_epochs"]
    )

    # --- trainer -------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs          = max_epochs,
        logger              = logger,
        callbacks           = [checkpoint_cb, LearningRateMonitor("epoch")],
        accelerator         = "gpu" if torch.cuda.is_available() else "cpu",
        devices             = 1,
        precision           = precision,
        log_every_n_steps   = config["training"]["log_every_n_steps"],
        check_val_every_n_epoch = 1,
        enable_progress_bar = True,
    )

    trainer.fit(model, train_loader, val_loader)

    print("\nLAM TRAINING COMPLETE")
    print(f"Best checkpoint : {checkpoint_cb.best_model_path}")
    print(f"Best val_loss   : {checkpoint_cb.best_model_score:.6f}")


if __name__ == "__main__":
    main()
