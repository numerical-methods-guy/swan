import os
import json
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from paradis_tkdv.tkdv_paradis import ParadisTKdV
from utils.tkdv_dataset_utils import build_tkdv_dataset

try:
    from utils.reverse_huber_loss import ReversedHuberLoss
    _HAS_REVERSED_HUBER = True
except ImportError:
    _HAS_REVERSED_HUBER = False


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


def build_tkdv_loss(config):
    """Construct a loss function for the tKdV setting."""
    loss_cfg = config.get("loss", {})
    loss_function = loss_cfg.get("loss_function", "reversed_huber")
    if loss_function == "reversed_huber" and _HAS_REVERSED_HUBER:
        delta = loss_cfg.get("delta_loss", 1.0)
        return ReversedHuberLoss(delta=delta)
    return nn.MSELoss()


def parse_ic_dict(s):
    """Parse a JSON string into an ic_dict.

    Values can be:
      - int: number of examples, no kwargs
      - [n, kwargs_dict]: number of examples + IC kwargs
      - [n, folder_str]: precomputed IC (folder path)
    """
    d = json.loads(s)
    result = {}
    for k, v in d.items():
        if isinstance(v, list):
            result[k] = (v[0], v[1])  # (n_examples, dict_or_str)
        else:
            result[k] = v
    return result


class TKdVLightningModule(pl.LightningModule):
    """Lightning module for the ParadisTKdV model."""

    def __init__(self, config, inp_mean, inp_var):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        self.J = config["data"]["J"]

        if "paradis" not in config["model"]:
            raise ValueError(
                "PARADIS model config not found. Add a 'model.paradis' section to your config."
            )
        self.model = ParadisTKdV(config)

        self.loss_fn = build_tkdv_loss(config)

        self.nfuture = 0

        self.inp_mean = inp_mean
        self.inp_var = inp_var

    def forward(self, fields):
        return self.model(fields)

    def training_step(self, batch, batch_idx):
        inp_fields, tar_fields = batch
        # tar_fields: (batch, n_rollout_steps, J)
        n_steps = tar_fields.shape[1]
        prd = self.model(inp_fields.unsqueeze(1)).squeeze(1)
        if torch.isnan(prd).any() or torch.isinf(prd).any():
            import logging
            logging.warning(
                f"NaN/Inf in model output at step 0, batch {batch_idx}: "
                f"inp_fields nan={torch.isnan(inp_fields).any().item()} "
                f"inf={torch.isinf(inp_fields).any().item()} "
                f"prd min={prd[~torch.isnan(prd)].min().item() if not torch.isnan(prd).all() else 'all-nan'} "
                f"max={prd[~torch.isnan(prd)].max().item() if not torch.isnan(prd).all() else 'all-nan'}"
            )
        total_loss = self.loss_fn(prd, tar_fields[:, 0])
        for k in range(1, n_steps):
            prd = self.model(prd.unsqueeze(1)).squeeze(1)
            if torch.isnan(prd).any() or torch.isinf(prd).any():
                import logging
                logging.warning(
                    f"NaN/Inf in model output at step {k}, batch {batch_idx}: "
                    f"prd min={prd[~torch.isnan(prd)].min().item() if not torch.isnan(prd).all() else 'all-nan'} "
                    f"max={prd[~torch.isnan(prd)].max().item() if not torch.isnan(prd).all() else 'all-nan'}"
                )
            total_loss = total_loss + self.loss_fn(prd, tar_fields[:, k])
        loss = total_loss / n_steps
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inp_fields, tar_fields = batch
        n_steps = tar_fields.shape[1]
        prd = self.model(inp_fields.unsqueeze(1)).squeeze(1)
        total_loss = self.loss_fn(prd, tar_fields[:, 0])
        for k in range(1, n_steps):
            prd = self.model(prd.unsqueeze(1)).squeeze(1)
            total_loss = total_loss + self.loss_fn(prd, tar_fields[:, k])
        loss = total_loss / n_steps
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_mse", F.mse_loss(prd, tar_fields[:, -1]), sync_dist=True)
        self.log("val_l1", F.l1_loss(prd, tar_fields[:, -1]), sync_dist=True)
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

def create_datasets(config, device, train_ic_dict=None, val_ic_dict=None,
                    n_rollout_steps=1, input_step_idx=0):
    """Create training and validation datasets.

    Args:
        config: config dict
        device: torch device
        train_ic_dict: dict mapping a label to a (n_examples, folder) tuple.
                       Required — raises ValueError if None.
        val_ic_dict: dict mapping a label to a (n_examples, folder) tuple.
                     Required — raises ValueError if None.
        n_rollout_steps: number of target steps per sample
        input_step_idx: which solver step to use as NN input
    """
    if train_ic_dict is None:
        raise ValueError("train_ic_dict is required for tKdV (precomputed folders only)")
    if val_ic_dict is None:
        raise ValueError("val_ic_dict is required for tKdV (precomputed folders only)")

    dt = config["data"]["dt"]
    nsteps = dt // config["data"]["dt_solver"]

    train_dataset, train_stats = build_tkdv_dataset(
        ic_dict=train_ic_dict,
        dt=dt,
        nsteps=nsteps,
        n_rollout_steps=n_rollout_steps,
        input_step_idx=input_step_idx,
        device=device,
        normalize=True,
    )

    val_dataset, _ = build_tkdv_dataset(
        ic_dict=val_ic_dict,
        dt=dt,
        nsteps=nsteps,
        n_rollout_steps=n_rollout_steps,
        input_step_idx=input_step_idx,
        device=device,
        normalize=True,
    )

    return train_dataset, val_dataset, train_stats


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
        "--start_ckpt",
        type=str,
        default=None,
        help="Checkpoint to warm-start pretraining weights from (pretraining still runs from epoch 1)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs to use (uses DDP strategy when >1)",
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

    train_dataset, val_dataset, train_stats = create_datasets(
        config, device,
        train_ic_dict=train_ic_dict,
        val_ic_dict=val_ic_dict,
        n_rollout_steps=known_args.n_rollout_steps,
        input_step_idx=known_args.input_step_idx,
    )

    # Overwrite val stats with train stats so both datasets use identical normalization.
    for attr in ("inp_mean", "inp_var"):
        val = getattr(train_dataset, attr)
        setattr(val_dataset, attr, val)
        for sub in val_dataset.datasets:
            setattr(sub, attr, val)

    # Save normalization stats alongside checkpoints.
    stats_save_dir = config["training"]["save_dir"]
    os.makedirs(stats_save_dir, exist_ok=True)
    torch.save({
        "inp_mean": train_dataset.inp_mean.cpu(),
        "inp_var": train_dataset.inp_var.cpu(),
        "input_step_idx": known_args.input_step_idx,
    }, os.path.join(stats_save_dir, "stats.pt"))
    per_ic_json = [
        {
            "label": s["label"],
            "inp_mean": s["inp_mean"].cpu().tolist(),
            "inp_var": s["inp_var"].cpu().tolist(),
        }
        for s in train_stats["per_ic"]
    ]
    with open(os.path.join(stats_save_dir, "stats.json"), "w") as _f:
        json.dump({
            "inp_mean": train_dataset.inp_mean.cpu().tolist(),
            "inp_var": train_dataset.inp_var.cpu().tolist(),
            "input_step_idx": known_args.input_step_idx,
            "per_ic": per_ic_json,
        }, _f, indent=2)
    print(f"Saved normalization stats to {stats_save_dir}/stats.{{pt,json}}")

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

    model = TKdVLightningModule(
        config,
        inp_mean=train_dataset.inp_mean,
        inp_var=train_dataset.inp_var,
    )

    precision = 32
    if config["training"]["amp_mode"] == "fp16":
        precision = 16
    elif config["training"]["amp_mode"] == "bf16":
        precision = "bf16"

    if known_args.start_ckpt is not None:
        print(f"\nWarm-starting weights from: {known_args.start_ckpt}")
        ckpt = torch.load(known_args.start_ckpt, map_location=device)
        state_dict = ckpt["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        print("Warm-start weights loaded.\n")

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
            devices=known_args.devices,
            strategy="auto" if known_args.devices == 1 else "ddp",
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
            devices=known_args.devices,
            strategy="auto" if known_args.devices == 1 else "ddp",
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
