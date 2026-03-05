import os
import argparse
import yaml
import torch


import types
import math


import torch.nn as nn
from torch.utils.data import DataLoader

import torch.multiprocessing as mp

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from torch_harmonics import RealSHT
from torch_harmonics.examples import PdeDataset
from torch_harmonics.examples.losses import (
    SquaredL2LossS2,
    L1LossS2,
    L2LossS2,
    W11LossS2,
)
from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperator
from torch_harmonics.examples.models.s2transformer import SphericalTransformer

from paradis import ParadisModel
from pde_dataset_with_winds import PdeDatasetWithWinds



def _make_spectral_taper(nlat, nmodes, device, k_cut, power):
    k_lat = torch.arange(nlat, device=device, dtype=torch.float32)
    k_lon = torch.arange(nmodes, device=device, dtype=torch.float32)
    k_lat_grid, k_lon_grid = torch.meshgrid(k_lat, k_lon, indexing="ij")
    k = torch.sqrt(k_lat_grid**2 + k_lon_grid**2)

    k_cut = max(float(k_cut), 1.0)
    power = max(float(power), 1.0)

    return torch.exp(-((k / k_cut) ** power))


def _smooth_random_initial_condition_wrapper(
    solver,
    *,
    k_cut=12.0,
    power=6.0,
    clip_sigma=None,
    height_clip_sigma=None,
    height_extra_smoothing=1.0,
    mach=0.2,
):
    orig_fn = solver.random_initial_condition

    def new_random_initial_condition(self, mach=mach):
        ic = orig_fn(mach=mach)  # spectral coeffs (3, nlat, nmodes)

        nlat = ic.shape[-2]
        nmodes = ic.shape[-1]

        # global taper for all channels
        taper = _make_spectral_taper(nlat, nmodes, ic.device, k_cut, power)
        ic = ic * taper

        # extra smoothing just for height/geopotential channel
        if height_extra_smoothing is not None and float(height_extra_smoothing) > 1.0:
            taper_h = _make_spectral_taper(
                nlat, nmodes, ic.device, k_cut / float(height_extra_smoothing), power
            )
            ic0 = ic[0] * taper_h
            ic = torch.stack([ic0, ic[1], ic[2]], dim=0)

        # optional soft clipping in grid space
        if (clip_sigma is not None) or (height_clip_sigma is not None):
            grid = self.spec2grid(ic)  # (3, nlat, nlon)

            cs_all = float(clip_sigma) if clip_sigma is not None else None
            cs_h = float(height_clip_sigma) if height_clip_sigma is not None else cs_all

            if cs_h is not None and cs_h > 0:
                grid0 = cs_h * torch.tanh(grid[0] / cs_h)
                grid = torch.stack([grid0, grid[1], grid[2]], dim=0)

            if cs_all is not None and cs_all > 0:
                grid1 = cs_all * torch.tanh(grid[1] / cs_all)
                grid2 = cs_all * torch.tanh(grid[2] / cs_all)
                grid = torch.stack([grid[0], grid1, grid2], dim=0)

            ic = self.sht(grid)

        return ic

    return types.MethodType(new_random_initial_condition, solver)



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
    """Unified Lightning Module for SFNO, Transformer, and Paradis models."""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        self.nlat = config["data"]["nlat"]
        self.nlon = config["data"]["nlon"]
        self.grid = config["data"]["grid"]
        self.model_type = config["experiment"]["model_type"]

        # PARADIS always uses winds
        self.use_winds = self.model_type == "paradis"

        if self.model_type == "sfno":
            self.model = self._create_sfno_model()
        elif self.model_type == "transformer":
            self.model = self._create_transformer_model()
        elif self.model_type == "paradis":
            self.model = self._create_paradis_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.loss_fn = SquaredL2LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_l1 = L1LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_l2 = L2LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_w11 = W11LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)

        self.nfuture = 0

    def _create_sfno_model(self):
        """Create SFNO model."""
        if "sfno" not in self.config["model"]:
            raise ValueError(
                "SFNO model config not found. Use config_sfno.yaml or add 'model.sfno' section."
            )

        model_config = self.config["model"]["sfno"]
        return SphericalFourierNeuralOperator(
            img_size=(self.nlat, self.nlon),
            grid=self.grid,
            grid_internal=self.grid,
            scale_factor=model_config["scale_factor"],
            in_chans=3,
            out_chans=3,
            embed_dim=model_config["embed_dim"],
            num_layers=model_config["num_layers"],
            normalization_layer=model_config["normalization_layer"],
            use_mlp=model_config["use_mlp"],
            mlp_ratio=model_config["mlp_ratio"],
            drop_rate=model_config["dropout"],
            hard_thresholding_fraction=model_config["hard_thresholding_fraction"],
            residual_prediction=True,
        )

    def _create_transformer_model(self):
        """Create Spherical Transformer model."""
        if "transformer" not in self.config["model"]:
            raise ValueError(
                "Transformer model config not found. Use config_transformer.yaml or add 'model.transformer' section."
            )

        model_config = self.config["model"]["transformer"]
        return SphericalTransformer(
            img_size=(self.nlat, self.nlon),
            grid=self.grid,
            scale_factor=model_config["scale_factor"],
            in_chans=3,
            out_chans=3,
            embed_dim=model_config["embed_dim"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            use_mlp=model_config["use_mlp"],
            mlp_ratio=model_config["mlp_ratio"],
            drop_rate=model_config["dropout"],
            drop_path_rate=model_config["drop_path"],
            pos_embed=model_config["pos_embed"],
        )

    def _create_paradis_model(self):
        """Create PARADIS model."""
        if "paradis" not in self.config["model"]:
            raise ValueError(
                "PARADIS model config not found. Use config_paradis.yaml or add 'model.paradis' section."
            )

        return ParadisModel(self.config)

    def forward(self, *args):
        if self.use_winds:
            return self.model(args[0], args[1])
        else:
            return self.model(args[0])

    def training_step(self, batch, batch_idx):
        if self.use_winds:
            inp_fields, inp_winds, tar_fields, tar_winds = batch
            prd = self.model(inp_fields, inp_winds)

            for _ in range(self.nfuture):
                prd = self.model(prd, inp_winds)

            loss = self.loss_fn(prd, tar_fields)
        else:
            inp, tar = batch
            prd = self.model(inp)
            for _ in range(self.nfuture):
                prd = self.model(prd)
            loss = self.loss_fn(prd, tar)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.use_winds:
            inp_fields, inp_winds, tar_fields, tar_winds = batch
            prd = self.model(inp_fields, inp_winds)

            for _ in range(self.nfuture):
                prd = self.model(prd, inp_winds)

            loss = self.loss_fn(prd, tar_fields)
            l1 = self.metric_l1(prd, tar_fields)
            l2 = self.metric_l2(prd, tar_fields)
            w11 = self.metric_w11(prd, tar_fields)
        else:
            inp, tar = batch
            prd = self.model(inp)
            for _ in range(self.nfuture):
                prd = self.model(prd)
            loss = self.loss_fn(prd, tar)
            l1 = self.metric_l1(prd, tar)
            l2 = self.metric_l2(prd, tar)
            w11 = self.metric_w11(prd, tar)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_l1", l1, sync_dist=True)
        self.log("val_l2", l2, sync_dist=True)
        self.log("val_w11", w11, sync_dist=True)

        return loss

    def configure_optimizers(self):
        lr = self.config["training"]["learning_rate"]

        if self.nfuture > 0:
            lr = self.config["training"]["finetune_learning_rate"]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def on_load_checkpoint(self, checkpoint):
        """Filter out problematic keys during checkpoint loading."""
        state_dict = checkpoint["state_dict"]
        keys_to_remove = ["metric_w11.k_phi_mesh", "metric_w11.k_theta_mesh"]
        for k in keys_to_remove:
            if k in state_dict:
                del state_dict[k]


def create_datasets(config, device):
    """Create training and validation datasets."""
    dt = config["data"]["dt"]
    dt_solver = config["data"]["dt_solver"]
    nsteps = dt // dt_solver
    nlat = config["data"]["nlat"]
    nlon = config["data"]["nlon"]
    grid = config["data"]["grid"]

    # PARADIS always uses winds
    model_type = config["experiment"]["model_type"]
    use_winds = model_type == "paradis"

    if use_winds:
        train_dataset = PdeDatasetWithWinds(
            dt=dt,
            nsteps=nsteps,
            dims=(nlat, nlon),
            grid=grid,
            normalize=True,
            device=device,
        )
        train_dataset.sht = RealSHT(nlat=nlat, nlon=nlon, grid=grid).to(device)
        train_dataset.set_initial_condition("random")
        train_dataset.set_num_examples(config["data"]["num_train_examples"])

        val_dataset = PdeDatasetWithWinds(
            dt=dt,
            nsteps=nsteps,
            dims=(nlat, nlon),
            grid=grid,
            normalize=True,
            device=device,
        )
        val_dataset.sht = RealSHT(nlat=nlat, nlon=nlon, grid=grid).to(device)
        val_dataset.set_initial_condition("random")
        val_dataset.set_num_examples(config["data"]["num_val_examples"])
    else:
        train_dataset = PdeDataset(
            dt=dt,
            nsteps=nsteps,
            dims=(nlat, nlon),
            grid=grid,
            normalize=True,
            device=device,
        )
        train_dataset.sht = RealSHT(nlat=nlat, nlon=nlon, grid=grid).to(device)
        train_dataset.set_initial_condition("random")
        train_dataset.set_num_examples(config["data"]["num_train_examples"])

        val_dataset = PdeDataset(
            dt=dt,
            nsteps=nsteps,
            dims=(nlat, nlon),
            grid=grid,
            normalize=True,
            device=device,
        )
        val_dataset.sht = RealSHT(nlat=nlat, nlon=nlon, grid=grid).to(device)
        val_dataset.set_initial_condition("random")
        val_dataset.set_num_examples(config["data"]["num_val_examples"])
    # -------------------------------------------------------
    # Optional: smoother random ICs to avoid jerkiness/spikes
    # -------------------------------------------------------
    ic_cfg = config.get("data", {}).get("initial_condition", {})
    ic_mode = ic_cfg.get("mode", "random")

    if ic_mode == "smooth_random":
        k_cut = ic_cfg.get("k_cut", 12.0)
        power = ic_cfg.get("power", 6.0)
        clip_sigma = ic_cfg.get("clip_sigma", None)
        height_clip_sigma = ic_cfg.get("height_clip_sigma", None)
        height_extra_smoothing = ic_cfg.get("height_extra_smoothing", 1.0)
        mach = ic_cfg.get("mach", 0.2)

        train_dataset.solver.random_initial_condition = _smooth_random_initial_condition_wrapper(
            train_dataset.solver,
            k_cut=k_cut,
            power=power,
            clip_sigma=clip_sigma,
            height_clip_sigma=height_clip_sigma,
            height_extra_smoothing=height_extra_smoothing,
            mach=mach,
        )
        val_dataset.solver.random_initial_condition = _smooth_random_initial_condition_wrapper(
            val_dataset.solver,
            k_cut=k_cut,
            power=power,
            clip_sigma=clip_sigma,
            height_clip_sigma=height_clip_sigma,
            height_extra_smoothing=height_extra_smoothing,
            mach=mach,
        )

        print(
            f"[IC] smooth_random enabled: k_cut={k_cut}, power={power}, "
            f"clip_sigma={clip_sigma}, height_clip_sigma={height_clip_sigma}, "
            f"height_extra_smoothing={height_extra_smoothing}, mach={mach}"
        )
    else:
        print("[IC] default random initial conditions (no smoothing)")


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

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(config["experiment"]["seed"], workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model type: {config['experiment']['model_type']}")

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

        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        trainer = pl.Trainer(
            max_epochs=config["training"]["pretrain_epochs"],
            logger=logger,
            callbacks=[checkpoint_callback, lr_monitor],
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

        keys_to_ignore = ["metric_w11.k_phi_mesh", "metric_w11.k_theta_mesh"]
        for key in keys_to_ignore:
            if key in state_dict:
                del state_dict[key]

        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully.\n")

    if config["training"]["finetune_epochs"] > 0:
        print("\n" + "=" * 70)
        print(f"STARTING FINETUNING FOR {config['training']['finetune_epochs']} EPOCHS")
        print("=" * 70 + "\n")

        dt = config["data"]["dt"]
        dt_solver = config["data"]["dt_solver"]
        new_nsteps = 2 * dt // dt_solver
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

        finetune_lr_monitor = LearningRateMonitor(logging_interval="epoch")

        finetune_trainer = pl.Trainer(
            max_epochs=config["training"]["finetune_epochs"],
            logger=finetune_logger,
            callbacks=[finetune_checkpoint, finetune_lr_monitor],
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
