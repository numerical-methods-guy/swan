import os
import json
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
from dataset.pde_dataset_with_winds import PdeDatasetWithWinds
from utils.dataset_utils import build_mixed_dataset
from utils.loss import ParadisLoss
from utils.amse_loss import AMSELoss


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
    """Construct a loss function for the shallow water equation setting.

    The SWE model has three output channels (geopotential, vorticity, divergence)
    with no pressure-level structure, so all variables are treated as surface
    variables and pressure weighting is effectively bypassed.
    """
    nlat = config["data"]["nlat"]
    nlon = config["data"]["nlon"]
    grid = config["data"]["grid"]
    loss_cfg = config.get("loss", {})

    loss_function = loss_cfg.get("loss_function", "reversed_huber")

    if loss_function == "amse":
        return AMSELoss(nlat=nlat, nlon=nlon, grid=grid)
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


def parse_ic_dict(s):
    """Parse a JSON string into an ic_dict, converting list values to tuples for precomputed."""
    d = json.loads(s)
    return {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}


class SWELightningModule(pl.LightningModule):
    """Lightning module for the PARADIS shallow water equation model."""

    def __init__(self, config, solver, inp_mean, inp_var, wind_mean, wind_var, should_detach=False):
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

        self.solver = solver
        self.inp_mean = inp_mean
        self.inp_var = inp_var
        self.wind_mean = wind_mean
        self.wind_var = wind_var
        self.should_detach = should_detach

    def forward(self, fields, winds):
        return self.model(fields, winds)

    def _fields_to_winds(self, prd):
        """Unnormalize predicted fields, extract winds via solver, renormalize."""
        device = prd.device
        fields = prd * self.inp_var.sqrt().to(device) + self.inp_mean.to(device)
        spec = self.solver.grid2spec(fields)
        winds = self.solver.getuv(spec[:, 1:])
        winds = (winds - self.wind_mean.to(device)) / self.wind_var.sqrt().to(device)
        if self.should_detach:
            winds = winds.detach()
        return winds

    def training_step(self, batch, batch_idx):
        inp_fields, inp_winds, tar_fields, tar_winds = batch
        # tar_fields: (batch, n_rollout_steps, 3, nlat, nlon)
        n_steps = tar_fields.shape[1]
        prd = self.model(inp_fields, inp_winds)
        total_loss = self.loss_fn(prd, tar_fields[:, 0])
        for k in range(1, n_steps):
            cur_winds = self._fields_to_winds(prd)
            prd = self.model(prd, cur_winds)
            total_loss = total_loss + self.loss_fn(prd, tar_fields[:, k])
        loss = total_loss / n_steps
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inp_fields, inp_winds, tar_fields, tar_winds = batch
        n_steps = tar_fields.shape[1]
        prd = self.model(inp_fields, inp_winds)
        total_loss = self.loss_fn(prd, tar_fields[:, 0])
        for k in range(1, n_steps):
            cur_winds = self._fields_to_winds(prd)
            prd = self.model(prd, cur_winds)
            total_loss = total_loss + self.loss_fn(prd, tar_fields[:, k])
        loss = total_loss / n_steps
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_sq_l2", self.metric_sq_l2(prd, tar_fields[:, -1]), sync_dist=True)
        self.log("val_l1", self.metric_l1(prd, tar_fields[:, -1]), sync_dist=True)
        self.log("val_l2", self.metric_l2(prd, tar_fields[:, -1]), sync_dist=True)
        self.log("val_w11", self.metric_w11(prd, tar_fields[:, -1]), sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Configure Adam with either MultiStepLR or ReduceLROnPlateau."""
        lr = self.config["training"]["learning_rate"]
        if self.nfuture > 0:
            lr = self.config["training"]["finetune_learning_rate"]

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
        """Zero gradients by setting them to None rather than filling with zeros."""
        optimizer.zero_grad(set_to_none=True)

    def on_load_checkpoint(self, checkpoint):
        """Filter out W11 mesh buffers that are recomputed on instantiation."""
        state_dict = checkpoint["state_dict"]
        for k in ["metric_w11.k_phi_mesh", "metric_w11.k_theta_mesh"]:
            if k in state_dict:
                del state_dict[k]


def create_datasets(config, device, train_ic_dict=None, val_ic_dict=None,
                    n_rollout_steps=1, input_step_idx=0):
    """Create training and validation datasets.

    Args:
        config: config dict
        device: torch device
        train_ic_dict: dict mapping IC type to number of training examples,
                       e.g. {"random": 100, "galewsky": 50}.
                       If None, defaults to {"random": config["data"]["num_train_examples"]}.
        val_ic_dict: dict mapping IC type to number of validation examples.
                     If None, defaults to {"random": config["data"]["num_val_examples"]}.
        n_rollout_steps: number of target steps per sample
        input_step_idx: which solver step to use as NN input
    """
    dt = config["data"]["dt"]
    nsteps = dt // config["data"]["dt_solver"]
    nlat = config["data"]["nlat"]
    nlon = config["data"]["nlon"]

    if train_ic_dict is None:
        train_ic_dict = {"random": config["data"]["num_train_examples"]}

    if val_ic_dict is None:
        val_ic_dict = {"random": config["data"]["num_val_examples"]}

    train_dataset, _ = build_mixed_dataset(
        ic_dict=train_ic_dict,
        dt=dt,
        nsteps=nsteps,
        n_rollout_steps=n_rollout_steps,
        input_step_idx=input_step_idx,
        dims=(nlat, nlon),
        device=device,
        normalize=True,
    )

    val_dataset, _ = build_mixed_dataset(
        ic_dict=val_ic_dict,
        dt=dt,
        nsteps=nsteps,
        n_rollout_steps=n_rollout_steps,
        input_step_idx=input_step_idx,
        dims=(nlat, nlon),
        device=device,
        normalize=True,
    )

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
    parser.add_argument(
        "--n_rollout_steps", type=int, default=1,
        help="Number of autoregressive target steps per sample",
    )
    parser.add_argument(
        "--input_step_idx", type=int, default=0,
        help="Number of solver warm-up steps before the NN input",
    )
    parser.add_argument(
        "--train_ic_dict", type=str, default=None,
        help='JSON ic_dict for training, e.g. \'{"random": 100, "galewsky": 50}\'',
    )
    parser.add_argument(
        "--val_ic_dict", type=str, default=None,
        help='JSON ic_dict for validation, e.g. \'{"random": 20}\'',
    )
    parser.add_argument(
        "--should_detach", action="store_true", default=False,
        help="Detach recomputed winds from the computation graph between rollout steps",
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

    train_ic_dict = parse_ic_dict(known_args.train_ic_dict) if known_args.train_ic_dict else None
    val_ic_dict = parse_ic_dict(known_args.val_ic_dict) if known_args.val_ic_dict else None

    train_dataset, val_dataset = create_datasets(
        config, device,
        train_ic_dict=train_ic_dict,
        val_ic_dict=val_ic_dict,
        n_rollout_steps=known_args.n_rollout_steps,
        input_step_idx=known_args.input_step_idx,
    )

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

    solver = train_dataset.datasets[0].solver
    model = SWELightningModule(
        config,
        solver=solver,
        inp_mean=train_dataset.inp_mean,
        inp_var=train_dataset.inp_var,
        wind_mean=train_dataset.wind_mean,
        wind_var=train_dataset.wind_var,
        should_detach=known_args.should_detach,
    )

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
