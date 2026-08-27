# train_lam.py
"""
Training script for LAMParadis.


Replaces the old on-the-fly PdeDatasetWithWinds approach with the
pre-generated HDF5 dataset (generate_dataset.py output) read through
LAMPatchDataset. The old create_lr_hr_datasets, recompute_all_stats,
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
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lam_helpers.lam_lightning import LAMLightningModule
from lam_helpers.lam_patch_dataset import LAMPatchDataset


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
    dc = config["data"]
    lc = config["lam"]
    dsc = config.get("dataset", {})

    h5_path = dsc.get("h5_path", "data/swe_paired.h5")
    preload = dsc.get("preload", False)
    patch_nlat_lr = int(lc["patch_nlat_lr"])
    patch_nlon_lr = int(lc["patch_nlon_lr"])
    halo_radius = int(lc["halo_radius"])
    exclude_pole_rows = int(lc.get("exclude_pole_rows", 4))
    num_train_ics = int(dc["num_train_examples"])
    ft_cfg = config.get("finetuning", {})
    finetuning_enabled = bool(ft_cfg.get("enabled", False))

    max_rollout_steps = (
        int(ft_cfg.get("max_horizon", 1))
        if finetuning_enabled
        else 1
    )

    common = dict(
        h5_path=h5_path,
        patch_nlat_lr=patch_nlat_lr,
        patch_nlon_lr=patch_nlon_lr,
        halo_radius=halo_radius,
        exclude_pole_rows=exclude_pole_rows,
        num_train_ics=num_train_ics,
        normalize=dc.get("normalize", True),
        preload=preload,
        max_rollout_steps=max_rollout_steps,
        finetuning_enabled=finetuning_enabled
    )

    train_ds = LAMPatchDataset(split="train", **common)
    val_ds = LAMPatchDataset(split="val", **common)

    print(train_ds.geometry_summary())
    print(f"  Train patches : {len(train_ds)}")
    print(f"  Val   patches : {len(val_ds)}")
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Loss plotting callback
# ---------------------------------------------------------------------------


class LossCurveCallback(Callback):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss")
        if train_loss is not None:
            self.train_losses.append(float(train_loss.detach().cpu()))

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_loss")
        if val_loss is not None:
            self.val_losses.append(float(val_loss.detach().cpu()))

    def on_fit_end(self, trainer, pl_module):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 5))
        if self.train_losses:
            plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Train Loss")
        if self.val_losses:
            plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "loss_curves.png", dpi=150)
        plt.close()

def load_model_weights_only(
    module: LAMLightningModule,
    checkpoint_path: str,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint.get("state_dict", checkpoint)

    model_state = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }

    if not model_state:
        raise ValueError(
            f"No 'model.*' weights found in checkpoint: {checkpoint_path}"
        )

    missing, unexpected = module.model.load_state_dict(
        model_state,
        strict=True,
    )

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}"
        )

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

    train_ds, val_ds = create_datasets(config)

    batch_size = int(config["data"]["batch_size"])
    num_workers = int(config["data"].get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
    )

    model = LAMLightningModule(config)

    ft_cfg = config.get("finetuning", {})
    finetuning_enabled = bool(ft_cfg.get("enabled", False))

    if finetuning_enabled:
        checkpoint_path = ft_cfg.get("checkpoint_path")

        if not checkpoint_path:
            raise ValueError(
                "finetuning.enabled is true, but "
                "finetuning.checkpoint_path is not set."
            )

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Fine-tuning checkpoint not found: {checkpoint_path}"
            )

        load_model_weights_only(model, checkpoint_path)

        print(f"Fine-tuning from : {checkpoint_path}")
        print(f"Fine-tune LR     : {model.lr:.3e}")
        print(
            "Horizon schedule : "
            f"{model.initial_horizon} -> {model.max_horizon}, "
            f"every {model.epochs_per_horizon} epochs"
        )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    amp_mode = config["training"].get("amp_mode", "none")
    precision = {"fp16": 16, "bf16": "bf16"}.get(amp_mode, 32)

    logger = TensorBoardLogger(
        save_dir=config["training"]["save_dir"],
        name=config["experiment"]["name"] + "_LAM",
    )

    checkpoint_dir = Path(logger.log_dir) / "checkpoints"

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor="val_loss",
        filename="lam-epoch{epoch:02d}-val{val_loss:.4f}",
        save_top_k=3,
        mode="min",
        save_last=True,
    )

    loss_curve_cb = LossCurveCallback(checkpoint_dir)

    ft_cfg = config.get("finetuning", {})

    max_epochs = (
        int(ft_cfg["epochs"])
        if bool(ft_cfg.get("enabled", False))
        else int(
            config["training"].get(
                "lam_epochs",
                config["training"]["pretrain_epochs"],
            )
        )
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_cb, LearningRateMonitor("epoch"), loss_curve_cb],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=precision,
        log_every_n_steps=config["training"]["log_every_n_steps"],
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)

    print("\nLAM TRAINING COMPLETE")
    print(f"Best checkpoint : {checkpoint_cb.best_model_path}")
    print(f"Best val_loss   : {checkpoint_cb.best_model_score:.6f}")
    print(f"Loss curve plot : {checkpoint_dir / 'loss_curves.png'}")


if __name__ == "__main__":
    main()
