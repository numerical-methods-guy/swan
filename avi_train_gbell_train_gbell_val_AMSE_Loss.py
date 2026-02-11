import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import torch.multiprocessing as mp

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from amse_loss import AMSELoss
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

        #self.loss_fn = SquaredL2LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.loss_fn = AMSELoss(nlat=self.nlat, nlon=self.nlon, grid=self.grid, norm="backward")
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


#----------------------------------------------------------------------------------------------
#++++++++++++++++++++++              NEW                 +++++++++++++++++++++++++++++++++++++++++++++

def _great_circle_distance(lat, lon, lat0, lon0):
    sin1, cos1 = torch.sin(lat), torch.cos(lat)
    sin0, cos0 = torch.sin(lat0), torch.cos(lat0)
    dlon = lon - lon0
    cosgamma = sin1 * sin0 + cos1 * cos0 * torch.cos(dlon)
    cosgamma = torch.clamp(cosgamma, -1.0, 1.0)
    return torch.acos(cosgamma)


class GaussianBellsPhiWrapper(torch.utils.data.Dataset):
    """
    Wrap PdeDataset so that:
      - channel 0 (geopotential) = Gaussian bells (scaled to match original random IC scale)
      - channels 1-2 (velocity-related) = original solver.random_initial_condition(mach)

    Output matches base dataset: (inp_grid, tar_grid), with optional base normalization.
    """

    def __init__(
        self,
        base_dataset,
        mach=0.2,
        k_min=1,
        k_max=8,
        sigma_min_deg=5.0,
        sigma_max_deg=20.0,
        signed=True,
        seed=None,
        use_base_normalization=True,
    ):
        self.base = base_dataset
        self.solver = base_dataset.solver
        self.device = base_dataset.device
        self.mach = mach

        self.k_min = k_min
        self.k_max = k_max
        self.sigma_min = math.radians(sigma_min_deg)
        self.sigma_max = math.radians(sigma_max_deg)
        self.signed = signed
        self.seed = seed

        # lat/lon mesh (assumes solver exposes lats/lons in radians)
        self.lat = self.solver.lats.reshape(-1, 1).to(self.device)
        self.lon = self.solver.lons.reshape(1, -1).to(self.device)

        self.use_base_normalization = use_base_normalization and getattr(self.base, "normalize", False)

        # --- scale-matching: compute reference std of channel-0 fluctuations from ORIGINAL IC ---
        # We do this once so bell amplitudes match "what torch_harmonics would have produced".
        with torch.inference_mode():
            ref_spec = self.solver.random_initial_condition(mach=self.mach)
            ref_grid = self.solver.spec2grid(ref_spec)  # (3,nlat,nlon)
            ref_phi = ref_grid[0]
            ref_phi_fluc = ref_phi - ref_phi.mean()
            self.ref_phi_std = ref_phi_fluc.std().clamp_min(1e-12)

    def __len__(self):
        return len(self.base)

    def _rand(self, shape, idx, offset=0):
        dtype = self.solver.lap.dtype
        if self.seed is None:
            return torch.rand(*shape, device=self.device, dtype=dtype)
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.seed) + int(idx) + int(offset))
        return torch.rand(*shape, device=self.device, dtype=dtype, generator=g)

    def _sample_phi_bells_grid(self, idx):
        dtype = self.solver.lap.dtype

        K = int(torch.randint(self.k_min, self.k_max + 1, (1,), device=self.device).item())

        # uniform centers on sphere: u~U[-1,1], lon~U[0,2pi), lat=asin(u)
        u = 2.0 * self._rand((K,), idx, offset=10) - 1.0
        lat0 = torch.asin(u)
        lon0 = 2.0 * math.pi * self._rand((K,), idx, offset=20)

        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * self._rand((K,), idx, offset=30)

        # amplitudes in *relative* units; we scale final field to match ref_phi_std anyway
        amp = self._rand((K,), idx, offset=40)
        if self.signed:
            signs = torch.where(self._rand((K,), idx, offset=50) < 0.5, -torch.ones_like(amp), torch.ones_like(amp))
            amp = amp * signs

        bump = torch.zeros(self.solver.nlat, self.solver.nlon, device=self.device, dtype=dtype)
        for i in range(K):
            gamma = _great_circle_distance(self.lat, self.lon, lat0[i].view(1, 1), lon0[i].view(1, 1))
            bump = bump + amp[i] * torch.exp(-(gamma * gamma) / (2.0 * sigma[i] * sigma[i]))

        # make bump zero-mean and unit-std before scaling
        bump = bump - bump.mean()
        bump = bump / (bump.std() + 1e-6)

        # geopotential mean in grid: g*havg (constant)
        phi_mean = self.solver.gravity * self.solver.havg
        phi_grid = phi_mean + self.ref_phi_std * bump
        return phi_grid

    def __getitem__(self, idx):
        with torch.inference_mode():
            # 1) original IC in spectral space (this contains original vel channels)
            inp_spec = self.solver.random_initial_condition(mach=self.mach)

            # 2) build bell geopotential in grid space
            phi_grid = self._sample_phi_bells_grid(idx)

            # 3) convert bell phi to spectral coeffs and replace only channel 0
            # grid2spec expects 3 channels; we provide phi + zeros.
            zeros = torch.zeros_like(phi_grid)
            phi_spec0 = self.solver.grid2spec(torch.stack([phi_grid, zeros, zeros], dim=0))[0]

            inp_spec = inp_spec.clone()
            inp_spec[0] = phi_spec0
            inp_spec = torch.tril(inp_spec)

            # 4) timestep in spectral, then back to grid like base dataset
            tar_spec = self.solver.timestep(inp_spec, self.base.nsteps)
            inp_grid = self.solver.spec2grid(inp_spec)
            tar_grid = self.solver.spec2grid(tar_spec)

            # 5) optional: reuse base normalization
            if self.use_base_normalization:
                inp_grid = (inp_grid - self.base.inp_mean) / torch.sqrt(self.base.inp_var)
                tar_grid = (tar_grid - self.base.inp_mean) / torch.sqrt(self.base.inp_var)

            return inp_grid.clone(), tar_grid.clone()


class GaussianBellsPhiWrapperWithWinds(torch.utils.data.Dataset):
    """
    Correct winds wrapper:
      - construct inp_spec from random IC
      - replace phi (channel 0) with gaussian-bells phi (in spectral via grid2spec)
      - timestep inp_spec -> tar_spec (consistent pair!)
      - compute winds from matching specs
      - apply base normalization (fields + winds) to match training inputs
    Returns: (inp_fields, inp_winds, tar_fields, tar_winds)
    """

    def __init__(
        self,
        base_dataset,
        mach=0.2,
        k_min=1,
        k_max=8,
        sigma_min_deg=5.0,
        sigma_max_deg=20.0,
        signed=True,
        seed=None,
    ):
        self.base = base_dataset
        self.solver = base_dataset.solver
        self.device = base_dataset.device
        self.mach = mach

        self.k_min = k_min
        self.k_max = k_max
        self.sigma_min = math.radians(sigma_min_deg)
        self.sigma_max = math.radians(sigma_max_deg)
        self.signed = signed
        self.seed = seed

        self.lat = self.solver.lats.reshape(-1, 1).to(self.device)
        self.lon = self.solver.lons.reshape(1, -1).to(self.device)

        # use same normalization tensors as base dataset (must exist if normalize=True)
        self.use_base_normalization = getattr(self.base, "normalize", False)
        if self.use_base_normalization:
            self.inp_mean = self.base.inp_mean
            self.inp_var  = self.base.inp_var
            self.wind_mean = self.base.wind_mean
            self.wind_var  = self.base.wind_var

        # scale reference from original IC channel-0 fluctuations
        with torch.inference_mode():
            ref_spec = self.solver.random_initial_condition(mach=self.mach)
            ref_grid = self.solver.spec2grid(ref_spec)
            ref_phi = ref_grid[0]
            self.ref_phi_std = (ref_phi - ref_phi.mean()).std().clamp_min(1e-12)

    def __len__(self):
        return len(self.base)

    def _rand(self, shape, idx, offset=0):
        dtype = self.solver.lap.dtype
        if self.seed is None:
            return torch.rand(*shape, device=self.device, dtype=dtype)
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.seed) + int(idx) + int(offset))
        return torch.rand(*shape, device=self.device, dtype=dtype, generator=g)

    def _sample_phi_bells_grid(self, idx):
        dtype = self.solver.lap.dtype

        # IMPORTANT: make K deterministic too (no global RNG)
        uK = self._rand((1,), idx, offset=5).item()
        K = int(self.k_min + math.floor(uK * (self.k_max - self.k_min + 1)))
        K = max(self.k_min, min(self.k_max, K))

        u = 2.0 * self._rand((K,), idx, 10) - 1.0
        lat0 = torch.asin(u)
        lon0 = 2.0 * math.pi * self._rand((K,), idx, 20)

        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * self._rand((K,), idx, 30)

        amp = self._rand((K,), idx, 40)
        if self.signed:
            signs = torch.where(self._rand((K,), idx, 50) < 0.5,
                                -torch.ones_like(amp), torch.ones_like(amp))
            amp = amp * signs

        bump = torch.zeros(self.solver.nlat, self.solver.nlon, device=self.device, dtype=dtype)
        for i in range(K):
            gamma = _great_circle_distance(self.lat, self.lon, lat0[i].view(1, 1), lon0[i].view(1, 1))
            bump = bump + amp[i] * torch.exp(-(gamma * gamma) / (2.0 * sigma[i] * sigma[i]))

        bump = bump - bump.mean()
        bump = bump / (bump.std() + 1e-6)

        phi_mean = self.solver.gravity * self.solver.havg
        return phi_mean + self.ref_phi_std * bump

    def __getitem__(self, idx):
        with torch.inference_mode():
            # 1) random IC in spectral space
            inp_spec = self.solver.random_initial_condition(mach=self.mach)

            # 2) gaussian-bells phi in grid, convert to spec and replace channel 0
            phi_grid = self._sample_phi_bells_grid(idx).to(self.solver.lap.dtype)
            zeros = torch.zeros_like(phi_grid)
            phi_spec0 = self.solver.grid2spec(torch.stack([phi_grid, zeros, zeros], dim=0))[0]

            inp_spec = inp_spec.clone()
            inp_spec[0] = phi_spec0
            inp_spec = torch.tril(inp_spec)

            # 3) timestep THAT modified IC to get target (this was the fatal missing piece)
            tar_spec = self.solver.timestep(inp_spec, self.base.nsteps)

            # 4) convert to grid fields
            inp_fields = self.solver.spec2grid(inp_spec)
            tar_fields = self.solver.spec2grid(tar_spec)

            # 5) winds from matching specs (same convention as forecast)
            inp_winds = self.solver.getuv(inp_spec[1:])
            tar_winds = self.solver.getuv(tar_spec[1:])

            # 6) apply the same normalization as base dataset
            if self.use_base_normalization:
                inp_fields = (inp_fields - self.inp_mean) / torch.sqrt(self.inp_var)
                tar_fields = (tar_fields - self.inp_mean) / torch.sqrt(self.inp_var)
                inp_winds  = (inp_winds  - self.wind_mean) / torch.sqrt(self.wind_var)
                tar_winds  = (tar_winds  - self.wind_mean) / torch.sqrt(self.wind_var)

            return inp_fields.clone(), inp_winds.clone(), tar_fields.clone(), tar_winds.clone()



#----------------------------------------------------------------------------------------------



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
    if use_winds:
        train_dataset = GaussianBellsPhiWrapperWithWinds(
        train_dataset,
        mach=0.2,
        k_min=1, k_max=8,
        sigma_min_deg=5.0, sigma_max_deg=20.0,
        signed=True,
        seed=config["experiment"]["seed"],
    )
        val_dataset = GaussianBellsPhiWrapperWithWinds(
        val_dataset,
        mach=0.2,
        k_min=1, k_max=8,
        sigma_min_deg=5.0, sigma_max_deg=20.0,
        signed=True,
        seed=config["experiment"]["seed"] + 12345,
    )
    else:
        train_dataset = GaussianBellsPhiWrapper(
        train_dataset,
        mach=0.2,
        k_min=1, k_max=8,
        sigma_min_deg=5.0, sigma_max_deg=20.0,
        signed=True,
        seed=config["experiment"]["seed"],
        use_base_normalization=True,
    )
        val_dataset = GaussianBellsPhiWrapper(
        val_dataset,
        mach=0.2,
        k_min=1, k_max=8,
        sigma_min_deg=5.0, sigma_max_deg=20.0,
        signed=True,
        seed=config["experiment"]["seed"] + 12345,
        use_base_normalization=True,
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
