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

def _online_update_mean_var(sum_, sumsq_, count_, x):
    # x: (C,H,W)
    sum_ = sum_ + x.sum(dim=(-1, -2))            # (C,)
    sumsq_ = sumsq_ + (x * x).sum(dim=(-1, -2)) # (C,)
    count_ = count_ + x.shape[-1] * x.shape[-2]  # scalar
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
    Compute mean/var for the *Gaussian-bells ALL-FIELDS IC distribution* (t=0),
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

    # --- ref means/stds from Williamson Case 6 ---
    with torch.no_grad():
        ref_spec = make_williamson_case6_ic_spec_from_winds(
            solver,
            R=4,
            omega=7.848e-6,
            K=None,
            h0=8000.0,
            flip_vort=False,
        )
        ref_grid = solver.spec2grid(ref_spec)  # (3,H,W)

        ref0 = ref_grid[0]
        ref1 = ref_grid[1]
        ref2 = ref_grid[2]

        ref_mean0 = ref0.mean()
        ref_mean1 = ref1.mean()
        ref_mean2 = ref2.mean()

        ref_std0 = (ref0 - ref_mean0).std().clamp_min(1e-12)
        ref_std1 = (ref1 - ref_mean1).std().clamp_min(1e-12)
        ref_std2 = (ref2 - ref_mean2).std().clamp_min(1e-12)

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

    # accumulators in float64
    sum_f = torch.zeros(3, device=device, dtype=torch.float64)
    sumsq_f = torch.zeros(3, device=device, dtype=torch.float64)
    count_f = torch.tensor(0.0, device=device, dtype=torch.float64)

    sum_w = torch.zeros(2, device=device, dtype=torch.float64)
    sumsq_w = torch.zeros(2, device=device, dtype=torch.float64)
    count_w = torch.tensor(0.0, device=device, dtype=torch.float64)

    # means come from Williamson Case 6 above

    with torch.no_grad():
        for i in range(int(num_samples)):
            phi_grid  = _sample_bells_grid(i, ref_std0, ref_mean0, offset_base=0)
            vort_grid = _sample_bells_grid(i, ref_std1, ref_mean1, offset_base=1000)
            div_grid  = _sample_bells_grid(i, ref_std2, ref_mean2, offset_base=2000)

            inp_grid = torch.stack([phi_grid, vort_grid, div_grid], dim=0)  # (3,H,W)
            inp_spec = solver.grid2spec(inp_grid)
            inp_spec = torch.tril(inp_spec)

            grid = solver.spec2grid(inp_spec).to(torch.float64)      # (3,H,W)
            uv   = solver.getuv(inp_spec[1:]).to(torch.float64)      # (2,H,W)

            sum_f, sumsq_f, count_f = _online_update_mean_var(sum_f, sumsq_f, count_f, grid)
            sum_w, sumsq_w, count_w = _online_update_mean_var(sum_w, sumsq_w, count_w, uv)

    field_mean, field_var = _finalize_mean_var(sum_f, sumsq_f, count_f)
    wind_mean, wind_var   = _finalize_mean_var(sum_w, sumsq_w, count_w)

    # cast back to solver dtype
    field_mean = field_mean.to(dtype)
    field_var  = field_var.to(dtype)
    wind_mean  = wind_mean.to(dtype)
    wind_var   = wind_var.to(dtype)

    return field_mean, field_var, wind_mean, wind_var
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





class GaussianBellsAllFieldsWrapperWithWinds(torch.utils.data.Dataset):
    """
    Gaussian bells for ALL THREE fields (phi + vorticity + divergence) while keeping winds code unchanged:
      - build inp_grid with gaussian bells in channels 0,1,2 (channel 0 gets mean g*havg, others mean 0)
      - grid2spec -> inp_spec
      - timestep inp_spec -> tar_spec
      - winds computed as before via getuv(spec[1:])
      - apply base normalization (fields + winds)
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
        channel1_mean_zero=True,
        channel2_mean_zero=True,
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

        self.channel1_mean_zero = channel1_mean_zero
        self.channel2_mean_zero = channel2_mean_zero

        self.lat = self.solver.lats.reshape(-1, 1).to(self.device)
        self.lon = self.solver.lons.reshape(1, -1).to(self.device)

        ## normalization tensors from base dataset (must exist if normalize=True)
        #self.use_base_normalization = getattr(self.base, "normalize", False)
        #if self.use_base_normalization:
        #    self.inp_mean = self.base.inp_mean
        #    self.inp_var  = self.base.inp_var
        #    self.wind_mean = self.base.wind_mean
        #    self.wind_var  = self.base.wind_var

        # We ALWAYS normalize in this wrapper, but stats will be set AFTER construction
        self.use_base_normalization = True
        self.inp_mean = None
        self.inp_var  = None
        self.wind_mean = None
        self.wind_var  = None



        # scale matching: use Williamson Case 6 scales instead of random-IC scales
        with torch.inference_mode():
            ref_spec = make_williamson_case6_ic_spec_from_winds(
                self.solver,
                R=4,
                omega=7.848e-6,
                K=None,
                h0=8000.0,
                flip_vort=False,
            )
            ref_grid = self.solver.spec2grid(ref_spec)  # (3,nlat,nlon)

            ref0 = ref_grid[0]
            ref1 = ref_grid[1]
            ref2 = ref_grid[2]

            self.ref_mean0 = ref0.mean()
            self.ref_mean1 = ref1.mean()
            self.ref_mean2 = ref2.mean()

            self.ref_std0 = (ref0 - self.ref_mean0).std().clamp_min(1e-12)
            self.ref_std1 = (ref1 - self.ref_mean1).std().clamp_min(1e-12)
            self.ref_std2 = (ref2 - self.ref_mean2).std().clamp_min(1e-12)

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
        """
        Make a zero-mean, unit-std bell field bump, then scale by ref_std and add mean.
        offset_base ensures different RNG streams for different channels.
        """
        dtype = self.solver.lap.dtype

        # deterministic K (avoid global RNG)
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

    def __getitem__(self, idx):
        with torch.inference_mode():
            # Build gaussian-bells grid for all 3 channels using Williamson-6 means/stds
            phi_grid = self._sample_bells_grid(
                idx, ref_std=self.ref_std0, mean=self.ref_mean0, offset_base=0
            )

            vort_grid = self._sample_bells_grid(
                idx, ref_std=self.ref_std1, mean=self.ref_mean1, offset_base=1000
            )
            div_grid = self._sample_bells_grid(
                idx, ref_std=self.ref_std2, mean=self.ref_mean2, offset_base=2000
            )

            inp_grid = torch.stack([phi_grid, vort_grid, div_grid], dim=0)  # (3,nlat,nlon)

            # Convert full 3-channel grid to spectral
            inp_spec = self.solver.grid2spec(inp_grid)
            inp_spec = torch.tril(inp_spec)

            # Timestep to target
            tar_spec = self.solver.timestep(inp_spec, self.base.nsteps)

            # Convert to grid fields
            inp_fields = self.solver.spec2grid(inp_spec)
            tar_fields = self.solver.spec2grid(tar_spec)

            # Winds computed EXACTLY as before (but will reflect new vort/div)
            inp_winds = self.solver.getuv(inp_spec[1:])
            tar_winds = self.solver.getuv(tar_spec[1:])

            # Apply same normalization as base dataset
            #if self.use_base_normalization:
            #    inp_fields = (inp_fields - self.inp_mean) / torch.sqrt(self.inp_var)
            #    tar_fields = (tar_fields - self.inp_mean) / torch.sqrt(self.inp_var)
            #    inp_winds  = (inp_winds  - self.wind_mean) / torch.sqrt(self.wind_var)
            #    tar_winds  = (tar_winds  - self.wind_mean) / torch.sqrt(self.wind_var)


            # ALWAYS normalize using gbells stats (must be set in create_datasets)
            assert self.inp_mean is not None and self.inp_var is not None, "Gaussian-bells field stats not set"
            assert self.wind_mean is not None and self.wind_var is not None, "Gaussian-bells wind stats not set"

            inp_fields = (inp_fields - self.inp_mean) / torch.sqrt(self.inp_var)
            tar_fields = (tar_fields - self.inp_mean) / torch.sqrt(self.inp_var)
            inp_winds  = (inp_winds  - self.wind_mean) / torch.sqrt(self.wind_var)
            tar_winds  = (tar_winds  - self.wind_mean) / torch.sqrt(self.wind_var)


            return inp_fields.clone(), inp_winds.clone(), tar_fields.clone(), tar_winds.clone()





#----------------------------------------------------------------------------------------------




def make_williamson_case2_ic_spec_from_winds(
    solver,
    gh0=29400.0,
    u0=None,
    flip_vort=False,
):
    device = solver.lap.device
    dtype  = solver.lap.dtype

    lat = solver.lats.to(device=device, dtype=dtype).reshape(-1, 1)
    lon = solver.lons.to(device=device, dtype=dtype).reshape(1, -1)

    a     = solver.radius.to(dtype=dtype)
    Omega = solver.omega.to(dtype=dtype)

    if u0 is None:
        day = torch.as_tensor(86400.0, device=device, dtype=dtype)
        u0  = 2.0 * torch.pi * a / (12.0 * day)
    else:
        u0 = torch.as_tensor(float(u0), device=device, dtype=dtype)

    sinlat = torch.sin(lat)
    coslat = torch.cos(lat)

    u = u0 * coslat
    v = torch.zeros_like(u)

    u_grid = u + 0.0 * lon
    v_grid = v + 0.0 * lon

    phi_lat = torch.as_tensor(float(gh0), device=device, dtype=dtype) \
              - (a * Omega * u0 + 0.5 * u0 * u0) * (sinlat ** 2)
    phi_grid = phi_lat + 0.0 * lon

    uv_grid = torch.stack([u_grid, v_grid], dim=0)
    vrtdiv_spec = solver.vrtdivspec(uv_grid)
    if flip_vort:
        vrtdiv_spec[0] = -vrtdiv_spec[0]

    phi_spec = solver.grid2spec(torch.stack([phi_grid], dim=0))[0]

    ctype = torch.complex128 if solver.lap.dtype == torch.float64 else torch.complex64
    uspec = torch.zeros(3, solver.lmax, solver.mmax, dtype=ctype, device=device)
    uspec[0] = phi_spec
    uspec[1:] = vrtdiv_spec.to(dtype=ctype)

    return torch.tril(uspec)


def make_williamson_case6_ic_spec_from_winds(
    solver,
    R=4,
    omega=7.848e-6,
    K=None,
    h0=8000.0,
    flip_vort=False,
    eps_cos=1e-6,
):
    device = solver.lap.device
    dtype  = solver.lap.dtype

    lat = solver.lats.to(device=device, dtype=dtype).reshape(-1, 1)
    lon = solver.lons.to(device=device, dtype=dtype).reshape(1, -1)

    a     = solver.radius.to(dtype=dtype)
    Omega = solver.omega.to(dtype=dtype)
    g     = solver.gravity.to(dtype=dtype)

    R_int = int(R)
    omega_t = torch.as_tensor(float(omega), device=device, dtype=dtype)
    if K is None:
        K_t = omega_t
    else:
        K_t = torch.as_tensor(float(K), device=device, dtype=dtype)

    h0_t = torch.as_tensor(float(h0), device=device, dtype=dtype)

    sinlat = torch.sin(lat)
    coslat = torch.cos(lat)
    cos_safe = torch.clamp(torch.abs(coslat), min=eps_cos) * torch.sign(coslat + 0.0)

    A = (omega_t / 2.0) * (2.0 * Omega + omega_t) * (coslat ** 2) \
        + (K_t ** 2) / 4.0 * (coslat ** (2 * R_int)) * (
            (R_int + 1) * (coslat ** 2)
            + (2.0 * (R_int ** 2) - R_int - 2.0)
            - 2.0 * (R_int ** 2) * (cos_safe ** (-2))
        )

    B = 2.0 * (Omega + omega_t) * K_t / ((R_int + 1) * (R_int + 2)) * (coslat ** R_int) * (
        (R_int ** 2 + 2 * R_int + 2)
        - ((R_int + 1) ** 2) * (coslat ** 2)
    )

    C = (K_t ** 2) / 4.0 * (coslat ** (2 * R_int)) * (
        (R_int + 1) * (coslat ** 2) - (R_int + 2.0)
    )

    h = h0_t + (a ** 2) * (A + B * torch.cos(R_int * lon) + C * torch.cos(2.0 * R_int * lon)) / g

    cos_pow = coslat ** (R_int - 1)
    u = a * omega_t * coslat + a * K_t * cos_pow * (R_int * (sinlat ** 2) - (coslat ** 2)) * torch.cos(R_int * lon)
    v = -a * K_t * R_int * cos_pow * sinlat * torch.sin(R_int * lon)

    phi_grid = g * h

    uv_grid = torch.stack([u, v], dim=0)
    vrtdiv_spec = solver.vrtdivspec(uv_grid)
    if flip_vort:
        vrtdiv_spec[0] = -vrtdiv_spec[0]

    phi_spec = solver.grid2spec(torch.stack([phi_grid], dim=0))[0]

    ctype = torch.complex128 if solver.lap.dtype == torch.float64 else torch.complex64
    uspec = torch.zeros(3, solver.lmax, solver.mmax, dtype=ctype, device=device)
    uspec[0] = phi_spec
    uspec[1:] = vrtdiv_spec.to(dtype=ctype)

    return torch.tril(uspec)


def _run_rollout_eval_case(pl_module, case_name, ic_spec, autoreg_steps=5):
    bundle = pl_module.rollout_eval_bundle
    solver = bundle["solver"].to(pl_module.device)
    nsteps = bundle["nsteps"]
    use_winds = bundle["use_winds"]

    inp_mean = bundle["inp_mean"].to(pl_module.device)
    inp_var  = bundle["inp_var"].to(pl_module.device)

    wind_mean = bundle.get("wind_mean", None)
    wind_var  = bundle.get("wind_var", None)
    if wind_mean is not None:
        wind_mean = wind_mean.to(pl_module.device)
    if wind_var is not None:
        wind_var = wind_var.to(pl_module.device)

    sqrt_inp_var = torch.sqrt(inp_var)
    sqrt_wind_var = torch.sqrt(wind_var) if wind_var is not None else None

    metrics_data = {
        "loss": [],
        "L1_error": [],
        "L2_error": [],
        "W11_error": [],
    }

    was_training = pl_module.training
    pl_module.eval()

    with torch.no_grad():
        if use_winds:
            prd_fields = (solver.spec2grid(ic_spec) - inp_mean) / sqrt_inp_var
            prd_fields = prd_fields.unsqueeze(0)

            prd_winds = solver.getuv(ic_spec[1:])
            prd_winds = (prd_winds - wind_mean) / sqrt_wind_var
            prd_winds = prd_winds.unsqueeze(0)
        else:
            prd_fields = (solver.spec2grid(ic_spec) - inp_mean) / sqrt_inp_var
            prd_fields = prd_fields.unsqueeze(0)

        uspec = ic_spec.clone()

        print("-" * 70, flush=True)
        print(f"{case_name}: rollout evaluation for {autoreg_steps} steps", flush=True)

        for step in range(1, autoreg_steps + 1):
            if use_winds:
                prd_fields = pl_module(prd_fields, prd_winds)
                prd_unnorm = prd_fields * sqrt_inp_var + inp_mean
                prd_spec = solver.sht(prd_unnorm.squeeze(0))

                prd_uv_grid = solver.getuv(prd_spec[1:])
                prd_winds = (prd_uv_grid - wind_mean) / sqrt_wind_var
                prd_winds = prd_winds.unsqueeze(0)
            else:
                prd_fields = pl_module(prd_fields)
                prd_unnorm = prd_fields * sqrt_inp_var + inp_mean
                prd_spec = solver.sht(prd_unnorm.squeeze(0))

            uspec = solver.timestep(uspec, nsteps)
            ref_grid = solver.spec2grid(uspec)
            ref_fields = (ref_grid - inp_mean) / sqrt_inp_var
            ref_fields = ref_fields.unsqueeze(0)

            l1 = pl_module.metric_l1(prd_fields, ref_fields).item()
            l2 = pl_module.metric_l2(prd_fields, ref_fields).item()
            w11 = pl_module.metric_w11(prd_fields, ref_fields).item()
            loss = pl_module.loss_fn(prd_fields, ref_fields).item()

            metrics_data["loss"].append(loss)
            metrics_data["L1_error"].append(l1)
            metrics_data["L2_error"].append(l2)
            metrics_data["W11_error"].append(w11)

            print(
                f"{case_name} Step {step}: "
                f"L1_error: {l1:.6f}, "
                f"L2_error: {l2:.6f}, "
                f"W11_error: {w11:.6f}, "
                f"loss: {loss:.6f}",
                flush=True,
            )

    summary = {}
    print(f"{case_name} SUMMARY", flush=True)
    for key, vals in metrics_data.items():
        vals_t = torch.tensor(vals, dtype=torch.float64, device=pl_module.device)
        mean_val = vals_t.mean()
        std_val = vals_t.std(unbiased=False)
        summary[f"{key}_mean"] = mean_val
        summary[f"{key}_std"] = std_val
        print(
            f"{case_name} {key:12s}: {mean_val.item():.6f} ± {std_val.item():.6f}",
            flush=True,
        )

    if was_training:
        pl_module.train()

    return summary


class WilliamsonRolloutCallback(pl.Callback):
    def __init__(self, autoreg_steps=5):
        super().__init__()
        self.autoreg_steps = autoreg_steps

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if not hasattr(pl_module, "rollout_eval_bundle"):
            return

        solver = pl_module.rollout_eval_bundle["solver"].to(pl_module.device)

        print("\n" + "=" * 70, flush=True)
        print(f"WILLIAMSON ROLLOUT EVAL AFTER EPOCH {trainer.current_epoch}", flush=True)
        print("=" * 70, flush=True)

        wc2_ic = make_williamson_case2_ic_spec_from_winds(
            solver,
            gh0=29400.0,
            flip_vort=False,
        )
        wc2_summary = _run_rollout_eval_case(
            pl_module,
            "Williamson Case 2",
            wc2_ic,
            autoreg_steps=self.autoreg_steps,
        )

        wc6_ic = make_williamson_case6_ic_spec_from_winds(
            solver,
            R=4,
            omega=7.848e-6,
            K=None,
            h0=8000.0,
            flip_vort=False,
        )
        wc6_summary = _run_rollout_eval_case(
            pl_module,
            "Williamson Case 6",
            wc6_ic,
            autoreg_steps=self.autoreg_steps,
        )

        trainer.callback_metrics["wc2_rollout_loss"] = wc2_summary["loss_mean"].detach()
        trainer.callback_metrics["wc6_rollout_loss"] = wc6_summary["loss_mean"].detach()

        pl_module.log("wc2_rollout_loss", wc2_summary["loss_mean"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("wc6_rollout_loss", wc6_summary["loss_mean"], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        print("=" * 70 + "\n", flush=True)

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
        train_dataset = GaussianBellsAllFieldsWrapperWithWinds(
        train_dataset,
        mach=0.2,
        k_min=1, k_max=8,
        sigma_min_deg=5.0, sigma_max_deg=20.0,
        signed=True,
        seed=config["experiment"]["seed"],
    )
        val_dataset = GaussianBellsAllFieldsWrapperWithWinds(
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
    # --- NEW: compute normalization stats from Gaussian-bells ALL-FIELDS IC samples ---
    gb_field_mean, gb_field_var, gb_wind_mean, gb_wind_var = compute_stats_for_gbells_allfields_ic(
        solver=train_dataset.solver,
        seed=int(config["experiment"]["seed"]),
        num_samples=200,      # <-- hardcoded, no config changes
        mach=0.2,
        k_min=1, k_max=8,
        sigma_min_deg=5.0, sigma_max_deg=20.0,
        signed=True,
    )

    # Inject stats into BOTH train and val wrappers so they normalize identically
    train_dataset.inp_mean  = gb_field_mean
    train_dataset.inp_var   = gb_field_var
    train_dataset.wind_mean = gb_wind_mean
    train_dataset.wind_var  = gb_wind_var

    val_dataset.inp_mean  = gb_field_mean
    val_dataset.inp_var   = gb_field_var
    val_dataset.wind_mean = gb_wind_mean
    val_dataset.wind_var  = gb_wind_var

    print("Gaussian-bells normalization set:")
    print("  fields mean:", gb_field_mean.view(-1).tolist())
    print("  fields std :", torch.sqrt(gb_field_var).view(-1).tolist())
    print("  winds  mean:", gb_wind_mean.view(-1).tolist())
    print("  winds  std :", torch.sqrt(gb_wind_var).view(-1).tolist())
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

    dt = config["data"]["dt"]
    dt_solver = config["data"]["dt_solver"]
    nsteps = dt // dt_solver

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

    model.rollout_eval_bundle = {
        "solver": train_dataset.solver,
        "nsteps": nsteps,
        "use_winds": (config["experiment"]["model_type"] == "paradis"),
        "inp_mean": train_dataset.inp_mean,
        "inp_var": train_dataset.inp_var,
        "wind_mean": getattr(train_dataset, "wind_mean", None),
        "wind_var": getattr(train_dataset, "wind_var", None),
    }

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

        wc2_checkpoint_callback = ModelCheckpoint(
            monitor="wc2_rollout_loss",
            filename="pretrain-best-wc2-epoch{epoch:02d}-wc2_{wc2_rollout_loss:.6f}",
            save_top_k=1,
            mode="min",
            save_last=False,
            auto_insert_metric_name=False,
        )

        wc6_checkpoint_callback = ModelCheckpoint(
            monitor="wc6_rollout_loss",
            filename="pretrain-best-wc6-epoch{epoch:02d}-wc6_{wc6_rollout_loss:.6f}",
            save_top_k=1,
            mode="min",
            save_last=False,
            auto_insert_metric_name=False,
        )

        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        trainer = pl.Trainer(
            max_epochs=config["training"]["pretrain_epochs"],
            logger=logger,
            callbacks=[
                WilliamsonRolloutCallback(autoreg_steps=5),
                checkpoint_callback,
                wc2_checkpoint_callback,
                wc6_checkpoint_callback,
                lr_monitor,
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

        finetune_wc2_checkpoint = ModelCheckpoint(
            monitor="wc2_rollout_loss",
            filename="finetune-best-wc2-epoch{epoch:02d}-wc2_{wc2_rollout_loss:.6f}",
            save_top_k=1,
            mode="min",
            save_last=False,
            auto_insert_metric_name=False,
        )

        finetune_wc6_checkpoint = ModelCheckpoint(
            monitor="wc6_rollout_loss",
            filename="finetune-best-wc6-epoch{epoch:02d}-wc6_{wc6_rollout_loss:.6f}",
            save_top_k=1,
            mode="min",
            save_last=False,
            auto_insert_metric_name=False,
        )

        finetune_lr_monitor = LearningRateMonitor(logging_interval="epoch")

        finetune_trainer = pl.Trainer(
            max_epochs=config["training"]["finetune_epochs"],
            logger=finetune_logger,
            callbacks=[
                WilliamsonRolloutCallback(autoreg_steps=5),
                finetune_checkpoint,
                finetune_wc2_checkpoint,
                finetune_wc6_checkpoint,
                finetune_lr_monitor,
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
