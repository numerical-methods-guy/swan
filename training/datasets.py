"""Dataset factory for SWAN training."""

import torch

from pde_dataset_with_winds import PdeDatasetWithWinds


def create_datasets(config: dict, device: torch.device):
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
