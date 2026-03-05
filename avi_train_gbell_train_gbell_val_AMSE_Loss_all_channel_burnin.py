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


def _online_update_mean_var(sum_, sumsq_, count_, x):
    # x: (C,H,W) or (B,C,H,W) or (K,C,H,W) -> we reduce over last two dims only
    if x.dim() == 3:
        sum_   = sum_   + x.sum(dim=(-1, -2))
        sumsq_ = sumsq_ + (x * x).sum(dim=(-1, -2))
        count_ = count_ + x.shape[-1] * x.shape[-2]
    else:
        # collapse everything except channel and H,W
        # (...,C,H,W) -> treat ... as batch-like and sum them too
        sum_   = sum_   + x.sum(dim=tuple(range(x.dim()-2)))
        sumsq_ = sumsq_ + (x * x).sum(dim=tuple(range(x.dim()-2)))
        count_ = count_ + x.shape[-1] * x.shape[-2] * (x.numel() // (x.shape[-1] * x.shape[-2] * x.shape[-3]))
    return sum_, sumsq_, count_


def _finalize_mean_var(sum_, sumsq_, count_, eps=1e-12):
    mean = (sum_ / count_).reshape(-1, 1, 1)
    var = (sumsq_ / count_ - (sum_ / count_)**2).clamp_min(eps).reshape(-1, 1, 1)
    return mean, var


def compute_stats_for_gbells_allfields_ic(
    solver,
    seed,
    num_samples=200,
    mach=0.2,
    k_min=1,
    k_max=8,
    sigma_min_deg=5.0,
    sigma_max_deg=20.0,
    signed=True,
):
    """
    Compute mean/var for the Gaussian-bells ALL-FIELDS IC distribution (t=0),
    for both fields (3 channels) and winds (2 channels).
    Returns:
      field_mean, field_var: (3,1,1)
      wind_mean, wind_var:   (2,1,1)
    """
    import math
    device = solver.lap.device
    dtype = solver.lap.dtype

    lat = solver.lats.reshape(-1, 1).to(device)
    lon = solver.lons.reshape(1, -1).to(device)

    sigma_min = math.radians(sigma_min_deg)
    sigma_max = math.radians(sigma_max_deg)

    def _great_circle_distance(lat, lon, lat0, lon0):
        sin1, cos1 = torch.sin(lat), torch.cos(lat)
        sin0, cos0 = torch.sin(lat0), torch.cos(lat0)
        dlon = lon - lon0
        cosgamma = sin1 * sin0 + cos1 * cos0 * torch.cos(dlon)
        cosgamma = torch.clamp(cosgamma, -1.0, 1.0)
        return torch.acos(cosgamma)

    def _rand(shape, idx, offset):
        g = torch.Generator(device=device)
        g.manual_seed(int(seed) + int(idx) + int(offset))
        return torch.rand(*shape, device=device, dtype=dtype, generator=g)

    # --- ref stds from ORIGINAL random IC (scale-matching) ---
    with torch.no_grad():
        ref_spec = solver.random_initial_condition(mach=mach)
        ref_grid = solver.spec2grid(ref_spec)  # (3,H,W)

        ref0 = ref_grid[0]
        ref1 = ref_grid[1]
        ref2 = ref_grid[2]
        ref_std0 = (ref0 - ref0.mean()).std().clamp_min(1e-12)
        ref_std1 = (ref1 - ref1.mean()).std().clamp_min(1e-12)
        ref_std2 = (ref2 - ref2.mean()).std().clamp_min(1e-12)

    def _sample_bells_grid(idx, ref_std, mean, offset_base):
        # deterministic K
        uK = _rand((1,), idx, offset_base + 5).item()
        K = int(k_min + math.floor(uK * (k_max - k_min + 1)))
        K = max(k_min, min(k_max, K))

        u = 2.0 * _rand((K,), idx, offset_base + 10) - 1.0
        lat0 = torch.asin(u)
        lon0 = 2.0 * math.pi * _rand((K,), idx, offset_base + 20)

        sigma = sigma_min + (sigma_max - sigma_min) * _rand((K,), idx, offset_base + 30)

        amp = _rand((K,), idx, offset_base + 40)
        if signed:
            signs = torch.where(
                _rand((K,), idx, offset_base + 50) < 0.5,
                -torch.ones_like(amp),
                torch.ones_like(amp),
            )
            amp = amp * signs

        bump = torch.zeros(solver.nlat, solver.nlon, device=device, dtype=dtype)
        for i in range(K):
            gamma = _great_circle_distance(lat, lon, lat0[i].view(1, 1), lon0[i].view(1, 1))
            bump = bump + amp[i] * torch.exp(-(gamma * gamma) / (2.0 * sigma[i] * sigma[i]))

        bump = bump - bump.mean()
        bump = bump / (bump.std() + 1e-6)

        return (mean + ref_std * bump).to(dtype)

    # accumulators in float64 for stability
    sum_f = torch.zeros(3, device=device, dtype=torch.float64)
    sumsq_f = torch.zeros(3, device=device, dtype=torch.float64)
    count_f = torch.tensor(0.0, device=device, dtype=torch.float64)

    sum_w = torch.zeros(2, device=device, dtype=torch.float64)
    sumsq_w = torch.zeros(2, device=device, dtype=torch.float64)
    count_w = torch.tensor(0.0, device=device, dtype=torch.float64)

    phi_mean = solver.gravity * solver.havg

    with torch.no_grad():
        for i in range(int(num_samples)):
            phi_grid  = _sample_bells_grid(i, ref_std0, phi_mean, offset_base=0)
            vort_grid = _sample_bells_grid(i, ref_std1, 0.0,    offset_base=1000)
            div_grid  = _sample_bells_grid(i, ref_std2, 0.0,    offset_base=2000)

            inp_grid = torch.stack([phi_grid, vort_grid, div_grid], dim=0)  # (3,H,W)
            inp_spec = solver.grid2spec(inp_grid)
            inp_spec = torch.tril(inp_spec)

            grid = solver.spec2grid(inp_spec).to(torch.float64)      # (3,H,W)
            uv   = solver.getuv(inp_spec[1:]).to(torch.float64)      # (2,H,W)

            sum_f, sumsq_f, count_f = _online_update_mean_var(sum_f, sumsq_f, count_f, grid)
            sum_w, sumsq_w, count_w = _online_update_mean_var(sum_w, sumsq_w, count_w, uv)

    field_mean, field_var = _finalize_mean_var(sum_f, sumsq_f, count_f)
    wind_mean, wind_var   = _finalize_mean_var(sum_w, sumsq_w, count_w)

    return field_mean.to(dtype), field_var.to(dtype), wind_mean.to(dtype), wind_var.to(dtype)



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


class GaussianBellsAllFieldsWrapper(torch.utils.data.Dataset):
    """
    Gaussian bells IC for ALL THREE fields (no winds dataset):
      - build inp_grid with gaussian bells in channels 0,1,2
      - grid2spec -> inp_spec (triangular)
      - timestep inp_spec -> tar_spec
      - returns (inp_fields, tar_fields)
    IMPORTANT: normalization will be done by a trajectory wrapper using gbells stats.
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

        # to be injected (gbells stats)
        self.inp_mean = None   # (3,1,1)
        self.inp_var  = None   # (3,1,1)

        # scale matching: per-channel std from ORIGINAL random IC (grid)
        with torch.inference_mode():
            ref_spec = self.solver.random_initial_condition(mach=self.mach)
            ref_grid = self.solver.spec2grid(ref_spec)  # (3,H,W)
            self.ref_std0 = (ref_grid[0] - ref_grid[0].mean()).std().clamp_min(1e-12)
            self.ref_std1 = (ref_grid[1] - ref_grid[1].mean()).std().clamp_min(1e-12)
            self.ref_std2 = (ref_grid[2] - ref_grid[2].mean()).std().clamp_min(1e-12)

    def __len__(self):
        return len(self.base)

    def _rand(self, shape, idx, offset=0):
        dtype = self.solver.lap.dtype
        if self.seed is None:
            return torch.rand(*shape, device=self.device, dtype=dtype)
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.seed) + int(idx) + int(offset))
        return torch.rand(*shape, device=self.device, dtype=dtype, generator=g)

    def _sample_bells_grid(self, idx, ref_std, mean=0.0, offset_base=0):
        dtype = self.solver.lap.dtype

        uK = self._rand((1,), idx, offset=offset_base + 5).item()
        K = int(self.k_min + math.floor(uK * (self.k_max - self.k_min + 1)))
        K = max(self.k_min, min(self.k_max, K))

        u = 2.0 * self._rand((K,), idx, offset_base + 10) - 1.0
        lat0 = torch.asin(u)
        lon0 = 2.0 * math.pi * self._rand((K,), idx, offset_base + 20)

        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * self._rand((K,), idx, offset_base + 30)

        amp = self._rand((K,), idx, offset_base + 40)
        if self.signed:
            signs = torch.where(
                self._rand((K,), idx, offset_base + 50) < 0.5,
                -torch.ones_like(amp),
                torch.ones_like(amp),
            )
            amp = amp * signs

        bump = torch.zeros(self.solver.nlat, self.solver.nlon, device=self.device, dtype=dtype)
        for i in range(K):
            gamma = _great_circle_distance(self.lat, self.lon, lat0[i].view(1, 1), lon0[i].view(1, 1))
            bump = bump + amp[i] * torch.exp(-(gamma * gamma) / (2.0 * sigma[i] * sigma[i]))

        bump = bump - bump.mean()
        bump = bump / (bump.std() + 1e-6)
        return (mean + ref_std * bump).to(dtype)

    def build_inp_spec(self, idx):
        phi_mean = self.solver.gravity * self.solver.havg
        phi_grid  = self._sample_bells_grid(idx, self.ref_std0, mean=phi_mean, offset_base=0)
        vort_grid = self._sample_bells_grid(idx, self.ref_std1, mean=0.0,    offset_base=1000)
        div_grid  = self._sample_bells_grid(idx, self.ref_std2, mean=0.0,    offset_base=2000)
        inp_grid = torch.stack([phi_grid, vort_grid, div_grid], dim=0)
        inp_spec = torch.tril(self.solver.grid2spec(inp_grid))
        return inp_spec

    def __getitem__(self, idx):
        with torch.inference_mode():
            inp_spec = self.build_inp_spec(idx)
            tar_spec = self.solver.timestep(inp_spec, self.base.nsteps)
            inp_fields = self.solver.spec2grid(inp_spec)
            tar_fields = self.solver.spec2grid(tar_spec)
            return inp_fields.clone(), tar_fields.clone()


class GaussianBellsAllFieldsWrapperWithWinds(torch.utils.data.Dataset):
    """
    Same as above, but returns winds too:
      (inp_fields, inp_winds, tar_fields, tar_winds)
    IMPORTANT: no normalization here; trajectory wrapper will normalize with gbells stats.
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

        # to be injected (gbells stats)
        self.inp_mean = None   # (3,1,1)
        self.inp_var  = None   # (3,1,1)
        self.wind_mean = None  # (2,1,1)
        self.wind_var  = None  # (2,1,1)

        with torch.inference_mode():
            ref_spec = self.solver.random_initial_condition(mach=self.mach)
            ref_grid = self.solver.spec2grid(ref_spec)
            self.ref_std0 = (ref_grid[0] - ref_grid[0].mean()).std().clamp_min(1e-12)
            self.ref_std1 = (ref_grid[1] - ref_grid[1].mean()).std().clamp_min(1e-12)
            self.ref_std2 = (ref_grid[2] - ref_grid[2].mean()).std().clamp_min(1e-12)

    def __len__(self):
        return len(self.base)

    def _rand(self, shape, idx, offset=0):
        dtype = self.solver.lap.dtype
        if self.seed is None:
            return torch.rand(*shape, device=self.device, dtype=dtype)
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.seed) + int(idx) + int(offset))
        return torch.rand(*shape, device=self.device, dtype=dtype, generator=g)

    def _sample_bells_grid(self, idx, ref_std, mean=0.0, offset_base=0):
        dtype = self.solver.lap.dtype

        uK = self._rand((1,), idx, offset=offset_base + 5).item()
        K = int(self.k_min + math.floor(uK * (self.k_max - self.k_min + 1)))
        K = max(self.k_min, min(self.k_max, K))

        u = 2.0 * self._rand((K,), idx, offset_base + 10) - 1.0
        lat0 = torch.asin(u)
        lon0 = 2.0 * math.pi * self._rand((K,), idx, offset_base + 20)
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * self._rand((K,), idx, offset_base + 30)

        amp = self._rand((K,), idx, offset_base + 40)
        if self.signed:
            signs = torch.where(
                self._rand((K,), idx, offset_base + 50) < 0.5,
                -torch.ones_like(amp),
                torch.ones_like(amp),
            )
            amp = amp * signs

        bump = torch.zeros(self.solver.nlat, self.solver.nlon, device=self.device, dtype=dtype)
        for i in range(K):
            gamma = _great_circle_distance(self.lat, self.lon, lat0[i].view(1, 1), lon0[i].view(1, 1))
            bump = bump + amp[i] * torch.exp(-(gamma * gamma) / (2.0 * sigma[i] * sigma[i]))

        bump = bump - bump.mean()
        bump = bump / (bump.std() + 1e-6)
        return (mean + ref_std * bump).to(dtype)

    def build_inp_spec(self, idx):
        phi_mean = self.solver.gravity * self.solver.havg
        phi_grid  = self._sample_bells_grid(idx, self.ref_std0, mean=phi_mean, offset_base=0)
        vort_grid = self._sample_bells_grid(idx, self.ref_std1, mean=0.0,    offset_base=1000)
        div_grid  = self._sample_bells_grid(idx, self.ref_std2, mean=0.0,    offset_base=2000)
        inp_grid = torch.stack([phi_grid, vort_grid, div_grid], dim=0)
        inp_spec = torch.tril(self.solver.grid2spec(inp_grid))
        return inp_spec

    def __getitem__(self, idx):
        with torch.inference_mode():
            inp_spec = self.build_inp_spec(idx)
            tar_spec = self.solver.timestep(inp_spec, self.base.nsteps)

            inp_fields = self.solver.spec2grid(inp_spec)
            tar_fields = self.solver.spec2grid(tar_spec)

            inp_winds = self.solver.getuv(inp_spec[1:])
            tar_winds = self.solver.getuv(tar_spec[1:])

            return inp_fields.clone(), inp_winds.clone(), tar_fields.clone(), tar_winds.clone()
class TrajectoryFromSolver(torch.utils.data.Dataset):
    """
    No-winds trajectory:
      returns (x0, x_seq) where x_seq has shape (K,3,H,W)
    Normalizes using gbells stats injected into the base wrapper.
    """
    def __init__(self, gbells_wrapper, K, step_nsteps=None):
        self.base = gbells_wrapper
        self.solver = self.base.solver
        self.device = self.base.device
        self.K = int(K)
        assert self.K >= 1
        self.step_nsteps = int(step_nsteps) if step_nsteps is not None else int(self.base.base.nsteps)

        self.inp_mean = self.base.inp_mean
        self.inp_var  = self.base.inp_var

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        with torch.inference_mode():
            inp_spec = self.base.build_inp_spec(idx)
            x0 = self.solver.spec2grid(inp_spec)

            cur_spec = inp_spec
            xs = []
            for _ in range(self.K):
                cur_spec = self.solver.timestep(cur_spec, self.step_nsteps)
                xs.append(self.solver.spec2grid(cur_spec))
            x_seq = torch.stack(xs, dim=0)  # (K,3,H,W)

            assert self.inp_mean is not None and self.inp_var is not None
            x0    = (x0    - self.inp_mean) / torch.sqrt(self.inp_var)
            x_seq = (x_seq - self.inp_mean) / torch.sqrt(self.inp_var)
            return x0.clone(), x_seq.clone()


class TrajectoryFromSolverWithWinds(torch.utils.data.Dataset):
    """
    Winds trajectory:
      returns (x0_fields, x0_winds, x_seq_fields, x_seq_winds)
    where x_seq_fields is (K,3,H,W), x_seq_winds is (K,2,H,W)
    Normalizes using gbells stats injected into the base wrapper.
    """
    def __init__(self, gbells_wrapper, K, step_nsteps=None):
        self.base = gbells_wrapper
        self.solver = self.base.solver
        self.device = self.base.device
        self.K = int(K)
        assert self.K >= 1
        self.step_nsteps = int(step_nsteps) if step_nsteps is not None else int(self.base.base.nsteps)

        self.inp_mean = self.base.inp_mean
        self.inp_var  = self.base.inp_var
        self.wind_mean = self.base.wind_mean
        self.wind_var  = self.base.wind_var

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        with torch.inference_mode():
            inp_spec = self.base.build_inp_spec(idx)
            x0_fields = self.solver.spec2grid(inp_spec)
            x0_winds  = self.solver.getuv(inp_spec[1:])

            cur_spec = inp_spec
            xs_f, xs_w = [], []
            for _ in range(self.K):
                cur_spec = self.solver.timestep(cur_spec, self.step_nsteps)
                xs_f.append(self.solver.spec2grid(cur_spec))
                xs_w.append(self.solver.getuv(cur_spec[1:]))

            x_seq_fields = torch.stack(xs_f, dim=0)  # (K,3,H,W)
            x_seq_winds  = torch.stack(xs_w, dim=0)  # (K,2,H,W)

            assert self.inp_mean is not None and self.inp_var is not None
            assert self.wind_mean is not None and self.wind_var is not None

            x0_fields = (x0_fields - self.inp_mean) / torch.sqrt(self.inp_var)
            x_seq_fields = (x_seq_fields - self.inp_mean) / torch.sqrt(self.inp_var)

            x0_winds = (x0_winds - self.wind_mean) / torch.sqrt(self.wind_var)
            x_seq_winds = (x_seq_winds - self.wind_mean) / torch.sqrt(self.wind_var)

            return x0_fields.clone(), x0_winds.clone(), x_seq_fields.clone(), x_seq_winds.clone()


#----------------------------------------------------------------------------------------------




def create_datasets(config, device, K):
    dt = config["data"]["dt"]
    dt_solver = config["data"]["dt_solver"]
    nsteps = dt // dt_solver
    nlat = config["data"]["nlat"]
    nlon = config["data"]["nlon"]
    grid = config["data"]["grid"]

    model_type = config["experiment"]["model_type"]
    use_winds = model_type == "paradis"

    if use_winds:
        base_train = PdeDatasetWithWinds(dt=dt, nsteps=nsteps, dims=(nlat, nlon), grid=grid,
                                        normalize=True, device=device)
        base_train.sht = RealSHT(nlat=nlat, nlon=nlon, grid=grid).to(device)
        base_train.set_initial_condition("random")
        base_train.set_num_examples(config["data"]["num_train_examples"])

        base_val = PdeDatasetWithWinds(dt=dt, nsteps=nsteps, dims=(nlat, nlon), grid=grid,
                                      normalize=True, device=device)
        base_val.sht = RealSHT(nlat=nlat, nlon=nlon, grid=grid).to(device)
        base_val.set_initial_condition("random")
        base_val.set_num_examples(config["data"]["num_val_examples"])

        bells_train = GaussianBellsAllFieldsWrapperWithWinds(
            base_train, mach=0.2, k_min=1, k_max=8, sigma_min_deg=5.0, sigma_max_deg=20.0,
            signed=True, seed=int(config["experiment"]["seed"])
        )
        bells_val = GaussianBellsAllFieldsWrapperWithWinds(
            base_val, mach=0.2, k_min=1, k_max=8, sigma_min_deg=5.0, sigma_max_deg=20.0,
            signed=True, seed=int(config["experiment"]["seed"]) + 12345
        )

        gb_field_mean, gb_field_var, gb_wind_mean, gb_wind_var = compute_stats_for_gbells_allfields_ic(
            solver=base_train.solver,
            seed=int(config["experiment"]["seed"]),
            num_samples=200,  # hardcoded exactly like your reference script
            mach=0.2,
            k_min=1, k_max=8,
            sigma_min_deg=5.0, sigma_max_deg=20.0,
            signed=True,
        )

        for ds in [bells_train, bells_val]:
            ds.inp_mean  = gb_field_mean
            ds.inp_var   = gb_field_var
            ds.wind_mean = gb_wind_mean
            ds.wind_var  = gb_wind_var

        print("Gaussian-bells normalization set:")
        print("  fields mean:", gb_field_mean.view(-1).tolist())
        print("  fields std :", torch.sqrt(gb_field_var).view(-1).tolist())
        print("  winds  mean:", gb_wind_mean.view(-1).tolist())
        print("  winds  std :", torch.sqrt(gb_wind_var).view(-1).tolist())

        train_ds = TrajectoryFromSolverWithWinds(bells_train, K=K, step_nsteps=nsteps)
        val_ds   = TrajectoryFromSolverWithWinds(bells_val,   K=K, step_nsteps=nsteps)
        return train_ds, val_ds

    else:
        base_train = PdeDataset(dt=dt, nsteps=nsteps, dims=(nlat, nlon), grid=grid,
                                normalize=True, device=device)
        base_train.sht = RealSHT(nlat=nlat, nlon=nlon, grid=grid).to(device)
        base_train.set_initial_condition("random")
        base_train.set_num_examples(config["data"]["num_train_examples"])

        base_val = PdeDataset(dt=dt, nsteps=nsteps, dims=(nlat, nlon), grid=grid,
                              normalize=True, device=device)
        base_val.sht = RealSHT(nlat=nlat, nlon=nlon, grid=grid).to(device)
        base_val.set_initial_condition("random")
        base_val.set_num_examples(config["data"]["num_val_examples"])

        bells_train = GaussianBellsAllFieldsWrapper(
            base_train, mach=0.2, k_min=1, k_max=8, sigma_min_deg=5.0, sigma_max_deg=20.0,
            signed=True, seed=int(config["experiment"]["seed"])
        )
        bells_val = GaussianBellsAllFieldsWrapper(
            base_val, mach=0.2, k_min=1, k_max=8, sigma_min_deg=5.0, sigma_max_deg=20.0,
            signed=True, seed=int(config["experiment"]["seed"]) + 12345
        )

        gb_field_mean, gb_field_var, gb_wind_mean, gb_wind_var = compute_stats_for_gbells_allfields_ic(
            solver=base_train.solver,
            seed=int(config["experiment"]["seed"]),
            num_samples=200,
            mach=0.2,
            k_min=1, k_max=8,
            sigma_min_deg=5.0, sigma_max_deg=20.0,
            signed=True,
        )

        for ds in [bells_train, bells_val]:
            ds.inp_mean  = gb_field_mean
            ds.inp_var   = gb_field_var

        print("Gaussian-bells normalization set:")
        print("  fields mean:", gb_field_mean.view(-1).tolist())
        print("  fields std :", torch.sqrt(gb_field_var).view(-1).tolist())

        train_ds = TrajectoryFromSolver(bells_train, K=K, step_nsteps=nsteps)
        val_ds   = TrajectoryFromSolver(bells_val,   K=K, step_nsteps=nsteps)
        return train_ds, val_ds



#-------------------- OLD CREATE DATASETS SCRIPT (DON'T USE ANYMORE) -------------------------------

def create_datasets_OLD(config, device):
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
class SWERolloutLightningModule(pl.LightningModule):
    def __init__(self, config, n_rollout, burn_in, detach_after_burnin=True):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config

        self.nlat = config["data"]["nlat"]
        self.nlon = config["data"]["nlon"]
        self.grid = config["data"]["grid"]
        self.model_type = config["experiment"]["model_type"]
        self.use_winds = self.model_type == "paradis"

        if self.model_type == "sfno":
            self.model = self._create_sfno_model()
        elif self.model_type == "transformer":
            self.model = self._create_transformer_model()
        elif self.model_type == "paradis":
            self.model = self._create_paradis_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # ---- LOSS ----
        # Keep your AMSELoss (exact same place you currently instantiate it)
        self.loss_fn = AMSELoss(nlat=self.nlat, nlon=self.nlon, grid=self.grid, norm="backward")
        # If you want the other script's loss instead, swap the above to:
        # self.loss_fn = SquaredL2LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)

        self.metric_l1 = L1LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_l2 = L2LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_w11 = W11LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)

        self.n_rollout = int(n_rollout)
        self.burn_in = int(burn_in)
        self.detach_after_burnin = bool(detach_after_burnin)

        assert self.n_rollout >= 1
        assert 0 <= self.burn_in < self.n_rollout, "burn_in must be < n_rollout"

    def _create_sfno_model(self):
        mc = self.config["model"]["sfno"]
        return SphericalFourierNeuralOperator(
            img_size=(self.nlat, self.nlon),
            grid=self.grid,
            grid_internal=self.grid,
            scale_factor=mc["scale_factor"],
            in_chans=3,
            out_chans=3,
            embed_dim=mc["embed_dim"],
            num_layers=mc["num_layers"],
            normalization_layer=mc["normalization_layer"],
            use_mlp=mc["use_mlp"],
            mlp_ratio=mc["mlp_ratio"],
            drop_rate=mc["dropout"],
            hard_thresholding_fraction=mc["hard_thresholding_fraction"],
            residual_prediction=True,
        )

    def _create_transformer_model(self):
        mc = self.config["model"]["transformer"]
        return SphericalTransformer(
            img_size=(self.nlat, self.nlon),
            grid=self.grid,
            scale_factor=mc["scale_factor"],
            in_chans=3,
            out_chans=3,
            embed_dim=mc["embed_dim"],
            num_layers=mc["num_layers"],
            num_heads=mc["num_heads"],
            use_mlp=mc["use_mlp"],
            mlp_ratio=mc["mlp_ratio"],
            drop_rate=mc["dropout"],
            drop_path_rate=mc["drop_path"],
            pos_embed=mc["pos_embed"],
        )

    def _create_paradis_model(self):
        return ParadisModel(self.config)

    def forward(self, *args):
        return self.model(*args) if self.use_winds else self.model(args[0])

    def _rollout_and_loss(self, x0_fields, x0_winds, x_seq_fields):
        """
        x0_fields: (B,3,H,W)
        x_seq_fields: (B,K,3,H,W)
        """
        prd = x0_fields
        loss = 0.0
        denom = 0

        for t in range(1, self.n_rollout + 1):
            if self.use_winds:
                prd = self.model(prd, x0_winds)
            else:
                prd = self.model(prd)

            if t == self.burn_in and self.detach_after_burnin:
                prd = prd.detach()

            if t > self.burn_in:
                loss = loss + self.loss_fn(prd, x_seq_fields[:, t - 1])
                denom += 1

        return loss / max(denom, 1)

    def training_step(self, batch, batch_idx):
        if self.use_winds:
            x0_fields, x0_winds, x_seq_fields, x_seq_winds = batch
            loss = self._rollout_and_loss(x0_fields, x0_winds, x_seq_fields)
        else:
            x0_fields, x_seq_fields = batch
            loss = self._rollout_and_loss(x0_fields, None, x_seq_fields)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.use_winds:
            x0_fields, x0_winds, x_seq_fields, x_seq_winds = batch
            loss = self._rollout_and_loss(x0_fields, x0_winds, x_seq_fields)

            with torch.no_grad():
                prd = x0_fields
                for _ in range(self.n_rollout):
                    prd = self.model(prd, x0_winds)
                tar_final = x_seq_fields[:, self.n_rollout - 1]
                l1 = self.metric_l1(prd, tar_final)
                l2 = self.metric_l2(prd, tar_final)
                w11 = self.metric_w11(prd, tar_final)
        else:
            x0_fields, x_seq_fields = batch
            loss = self._rollout_and_loss(x0_fields, None, x_seq_fields)

            with torch.no_grad():
                prd = x0_fields
                for _ in range(self.n_rollout):
                    prd = self.model(prd)
                tar_final = x_seq_fields[:, self.n_rollout - 1]
                l1 = self.metric_l1(prd, tar_final)
                l2 = self.metric_l2(prd, tar_final)
                w11 = self.metric_w11(prd, tar_final)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_l1", l1, sync_dist=True)
        self.log("val_l2", l2, sync_dist=True)
        self.log("val_w11", w11, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # keep your current behavior: use learning_rate for pretrain, finetune_learning_rate if you want
        lr = self.config["training"]["learning_rate"]
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        for k in ["metric_w11.k_phi_mesh", "metric_w11.k_theta_mesh"]:
            if k in state_dict:
                del state_dict[k]

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
        "--init_from",
        type=str,
        default=None,
        help="Checkpoint to INITIALIZE model weights from (weights only). Pretraining will still run.",
    )
    parser.add_argument("--burn_in", type=int, default=5,
                        help="Ignore loss for steps 1..burn_in. Start loss at burn_in+1.")
    parser.add_argument("--detach_after_burnin", action="store_true", default=True,
                        help="If set, detach the prediction after burn_in to truncate gradients.")
    parser.add_argument("--n_rollout", type=int, default=None,
                        help="Total rollout steps K. If not set, uses 1 + training.nfuture from config.")
    known_args, unknown_args = parser.parse_known_args()

    mp.set_start_method("spawn", force=True)

    config = load_config(known_args.config)
    config = update_config_from_args(config, unknown_args)

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(config["experiment"]["seed"], workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model type: {config['experiment']['model_type']}")

    #train_dataset, val_dataset = create_datasets(config, device)
    if known_args.n_rollout is None:
        K = 1 + int(config["training"]["nfuture"])
    else:
        K = int(known_args.n_rollout)

    if not (0 <= known_args.burn_in < K):
        raise ValueError(f"burn_in must be in [0, K-1]. Got burn_in={known_args.burn_in}, K={K}")

    train_dataset, val_dataset = create_datasets(config, device, K=K)
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

    #model = SWELightningModule(config)
    model = SWERolloutLightningModule(
        config=config,
        n_rollout=K,
        burn_in=known_args.burn_in,
        detach_after_burnin=known_args.detach_after_burnin,
    )
    print(f"Rollout steps K={K}, burn_in={known_args.burn_in}, detach_after_burnin={known_args.detach_after_burnin}")
    # ---- weights-only initialization ----
    if known_args.init_from is not None:
        print("\n" + "=" * 70)  
        print(f"INITIALIZING WEIGHTS FROM: {known_args.init_from} (weights only)")
        print("=" * 70 + "\n")

        ckpt = torch.load(known_args.init_from, map_location=device)
        state_dict = ckpt["state_dict"]

        # same cleanup
        keys_to_ignore = ["metric_w11.k_phi_mesh", "metric_w11.k_theta_mesh"]
        for k in keys_to_ignore:
            if k in state_dict:
                del state_dict[k]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}\n")
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
        #train_dataset.nsteps = new_nsteps
        
        train_dataset.step_nsteps = new_nsteps
        #val_dataset.nsteps = new_nsteps
        
        val_dataset.step_nsteps = new_nsteps
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
