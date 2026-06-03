"""Generic Optuna tuner for SWAN optimizer strategies.

This script tunes hyperparameters without MLflow integration.
It saves Optuna trial results to CSV and the best parameters to YAML.
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
from typing import Any

import optuna
import torch
import yaml
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

import pytorch_lightning as pl

from training.config import load_config, update_config_from_args
from training.datasets import create_datasets
from training.lightning_module import SWELightningModule
from training.strategies import available_optimizer_names, resolve_optimizer_name


def set_nested_config_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a config value using dot notation, e.g. training.learning_rate."""
    keys = dotted_key.split(".")
    current = config

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def suggest_value(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    """Ask Optuna for one hyperparameter value from a YAML specification."""
    param_type = spec["type"].lower()

    if param_type == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
            step=spec.get("step", None),
        )

    if param_type == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )

    if param_type == "categorical":
        return trial.suggest_categorical(name, spec["choices"])

    if param_type == "fixed":
        return spec["value"]

    raise ValueError(f"Unknown parameter type '{param_type}' for parameter '{name}'.")


def load_search_space(
    search_space_path: str,
    optimizer_name: str,
    include_global: bool = False,
) -> dict[str, Any]:
    """Load optimizer-specific search space, optionally including global params."""
    with open(search_space_path, "r") as f:
        all_spaces = yaml.safe_load(f)

    key = optimizer_name.lower()

    if key not in all_spaces:
        available = ", ".join(sorted(all_spaces))
        raise ValueError(
            f"No search space found for optimizer '{optimizer_name}'. "
            f"Available sections: {available}"
        )

    search_space: dict[str, Any] = {}

    if include_global:
        search_space.update(all_spaces.get("global", {}))

    optimizer_space = all_spaces[key]

    repeated = set(search_space).intersection(optimizer_space)
    if repeated:
        raise ValueError(
            f"Duplicate parameter names between global and {optimizer_name}: "
            f"{sorted(repeated)}"
        )

    search_space.update(optimizer_space)
    return search_space


def apply_trial_to_config(
    trial: optuna.Trial,
    config: dict[str, Any],
    optimizer_name: str,
    search_space: dict[str, Any],
) -> dict[str, Any]:
    """Update config with Optuna-suggested hyperparameters."""
    config["training"]["optimizer"] = optimizer_name

    for param_name, spec in search_space.items():
        value = suggest_value(trial, param_name, spec)
        config_key = spec.get("config_key", f"training.{param_name}")
        set_nested_config_value(config, config_key, value)

    return config


def choose_precision(config: dict[str, Any]):
    """Match train.py precision logic."""
    precision = 32
    if config["training"]["amp_mode"] == "fp16":
        precision = 16
    elif config["training"]["amp_mode"] == "bf16":
        precision = "bf16"
    return precision


def build_dataloaders(config: dict[str, Any], device: torch.device):
    """Create fresh datasets and dataloaders for one Optuna trial."""
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

    return train_dataset, val_dataset, train_loader, val_loader


def objective(
    trial: optuna.Trial,
    base_config: dict[str, Any],
    optimizer_name: str,
    search_space: dict[str, Any],
) -> float:
    """One Optuna trial: choose hyperparameters, train once, return val_loss."""
    config = copy.deepcopy(base_config)
    config = apply_trial_to_config(trial, config, optimizer_name, search_space)

    # First version tunes pretraining only.
    # This avoids mixing pretraining and finetuning in the first Optuna integration.
    config["training"]["finetune_epochs"] = 0

    pl.seed_everything(config["experiment"]["seed"], workers=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        config, device
    )

    model = SWELightningModule(config, optimizer=optimizer_name)

    trainer = pl.Trainer(
        max_epochs=config["training"]["pretrain_epochs"],
        logger=False,
        callbacks=[],
        enable_checkpointing=False,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=choose_precision(config),
        log_every_n_steps=config["training"]["log_every_n_steps"],
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    val_loss = trainer.callback_metrics.get("val_loss")
    if val_loss is None:
        val_results = trainer.validate(model, dataloaders=val_loader, verbose=False)
        val_loss = val_results[0]["val_loss"]

    score = (
        float(val_loss.detach().cpu().item())
        if hasattr(val_loss, "detach")
        else float(val_loss)
    )

    del trainer
    del model
    del train_loader
    del val_loader
    del train_dataset
    del val_dataset
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_paradis.yaml")
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=available_optimizer_names(),
        default=None,
        help="Optimizer to tune.",
    )
    parser.add_argument(
        "--search_space",
        type=str,
        default="optuna_search_spaces.yaml",
        help="Path to Optuna search-space YAML.",
    )
    parser.add_argument(
        "--tune_global",
        action="store_true",
        help="Also include the global search-space section.",
    )
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optional Optuna storage URL, e.g. sqlite:///optuna_sgd.db.",
    )
    parser.add_argument("--output_dir", type=str, default="optuna_results")

    known_args, unknown_args = parser.parse_known_args()

    mp.set_start_method("spawn", force=True)

    base_config = load_config(known_args.config)
    base_config = update_config_from_args(base_config, unknown_args)

    optimizer_name = resolve_optimizer_name(known_args.optimizer, base_config)

    search_space = load_search_space(
        known_args.search_space,
        optimizer_name,
        include_global=known_args.tune_global,
    )

    sampler = optuna.samplers.TPESampler(seed=base_config["experiment"]["seed"])

    study = optuna.create_study(
        direction="minimize",
        study_name=known_args.study_name,
        storage=known_args.storage,
        load_if_exists=(known_args.storage is not None),
        sampler=sampler,
    )

    study.optimize(
        lambda trial: objective(
            trial,
            base_config,
            optimizer_name,
            search_space,
        ),
        n_trials=known_args.n_trials,
    )

    os.makedirs(known_args.output_dir, exist_ok=True)

    trials_csv = os.path.join(
        known_args.output_dir,
        f"{study.study_name}_trials.csv",
    )
    study.trials_dataframe().to_csv(trials_csv, index=False)

    best_yaml = os.path.join(
        known_args.output_dir,
        f"{study.study_name}_best_params.yaml",
    )
    with open(best_yaml, "w") as f:
        yaml.safe_dump(
            {
                "optimizer": optimizer_name,
                "best_value": study.best_value,
                "best_params": study.best_params,
            },
            f,
            sort_keys=False,
        )

    print("\n" + "=" * 70)
    print("OPTUNA TUNING COMPLETE")
    print("=" * 70)
    print(f"Optimizer: {optimizer_name}")
    print(f"Best validation loss: {study.best_value}")
    print(f"Best parameters: {study.best_params}")
    print(f"Saved trials to: {trials_csv}")
    print(f"Saved best params to: {best_yaml}")


if __name__ == "__main__":
    main()