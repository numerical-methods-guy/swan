import os
import argparse
import yaml
import torch
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


# -----------------------
# config helpers
# -----------------------
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def update_config_from_args(config, unknown_args):
    for i in range(0, len(unknown_args), 2):
        if i + 1 >= len(unknown_args):
            break
        key = unknown_args[i].lstrip("-")
        val = unknown_args[i + 1]
        keys = key.split(".")
        cur = config
        for k in keys[:-1]:
            if k not in cur:
                cur[k] = {}
            cur = cur[k]
        try:
            if "." in val:
                val = float(val)
            else:
                val = int(val)
        except ValueError:
            pass
        cur[keys[-1]] = val
    return config


def _online_update_mean_var(sum_, sumsq_, count_, x):
    # x: (C,H,W)
    sum_ = sum_ + x.sum(dim=(-1, -2))
    sumsq_ = sumsq_ + (x * x).sum(dim=(-1, -2))
    count_ = count_ + x.shape[-1] * x.shape[-2]
    return sum_, sumsq_, count_


def _finalize_mean_var(sum_, sumsq_, count_, eps=1e-12):
    mean = (sum_ / count_).reshape(-1, 1, 1)
    var = (sumsq_ / count_ - (sum_ / count_)**2).clamp_min(eps).reshape(-1, 1, 1)
    return mean, var


def _great_circle_distance(lat, lon, lat0, lon0):
    sin1, cos1 = torch.sin(lat), torch.cos(lat)
    sin0, cos0 = torch.sin(lat0), torch.cos(lat0)
    dlon = lon - lon0
    cosgamma = sin1 * sin0 + cos1 * cos0 * torch.cos(dlon)
    cosgamma = torch.clamp(cosgamma, -1.0, 1.0)
    return torch.acos(cosgamma)


def make_williamson_case2_ic_spec_from_winds(
    solver,
    gh0=29400.0,
    u0=None,
    flip_vort=False,
):
    device = solver.lap.device
    dtype = solver.lap.dtype

    lat = solver.lats.to(device=device, dtype=dtype).reshape(-1, 1)
    lon = solver.lons.to(device=device, dtype=dtype).reshape(1, -1)

    a = solver.radius.to(dtype=dtype)
    Omega = solver.omega.to(dtype=dtype)

    if u0 is None:
        day = torch.as_tensor(86400.0, device=device, dtype=dtype)
        u0 = 2.0 * torch.pi * a / (12.0 * day)
    else:
        u0 = torch.as_tensor(float(u0), device=device, dtype=dtype)

    sinlat = torch.sin(lat)
    coslat = torch.cos(lat)

    u = u0 * coslat
    v = torch.zeros_like(u)

    u_grid = u + 0.0 * lon
    v_grid = v + 0.0 * lon

    phi_lat = torch.as_tensor(float(gh0), device=device, dtype=dtype) - (a * Omega * u0 + 0.5 * u0 * u0) * (sinlat ** 2)
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
    dtype = solver.lap.dtype

    lat = solver.lats.to(device=device, dtype=dtype).reshape(-1, 1)
    lon = solver.lons.to(device=device, dtype=dtype).reshape(1, -1)

    a = solver.radius.to(dtype=dtype)
    Omega = solver.omega.to(dtype=dtype)
    g = solver.gravity.to(dtype=dtype)

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

    A = (omega_t / 2.0) * (2.0 * Omega + omega_t) * (coslat ** 2) + (K_t ** 2) / 4.0 * (coslat ** (2 * R_int)) * (
        (R_int + 1) * (coslat ** 2)
        + (2.0 * (R_int ** 2) - R_int - 2.0)
        - 2.0 * (R_int ** 2) * (cos_safe ** (-2))
    )

    B = 2.0 * (Omega + omega_t) * K_t / ((R_int + 1) * (R_int + 2)) * (coslat ** R_int) * (
        (R_int ** 2 + 2 * R_int + 2) - ((R_int + 1) ** 2) * (coslat ** 2)
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
    for both fields (3 channels) and winds (2 channels), but scale the bells
    using Williamson Case 6 means/stds.
    """
    device = solver.lap.device
    dtype = solver.lap.dtype

    lat = solver.lats.reshape(-1, 1).to(device)
    lon = solver.lons.reshape(1, -1).to(device)

    sigma_min = math.radians(sigma_min_deg)
    sigma_max = math.radians(sigma_max_deg)

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
        ref_grid = solver.spec2grid(ref_spec)

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

    sum_f = torch.zeros(3, device=device, dtype=torch.float64)
    sumsq_f = torch.zeros(3, device=device, dtype=torch.float64)
    count_f = torch.tensor(0.0, device=device, dtype=torch.float64)

    sum_w = torch.zeros(2, device=device, dtype=torch.float64)
    sumsq_w = torch.zeros(2, device=device, dtype=torch.float64)
    count_w = torch.tensor(0.0, device=device, dtype=torch.float64)

    with torch.no_grad():
        for i in range(int(num_samples)):
            phi_grid = _sample_bells_grid(i, ref_std0, ref_mean0, offset_base=0)
            vort_grid = _sample_bells_grid(i, ref_std1, ref_mean1, offset_base=1000)
            div_grid = _sample_bells_grid(i, ref_std2, ref_mean2, offset_base=2000)

            inp_grid = torch.stack([phi_grid, vort_grid, div_grid], dim=0)
            inp_spec = solver.grid2spec(inp_grid)
            inp_spec = torch.tril(inp_spec)

            grid = solver.spec2grid(inp_spec).to(torch.float64)
            uv = solver.getuv(inp_spec[1:]).to(torch.float64)

            sum_f, sumsq_f, count_f = _online_update_mean_var(sum_f, sumsq_f, count_f, grid)
            sum_w, sumsq_w, count_w = _online_update_mean_var(sum_w, sumsq_w, count_w, uv)

    field_mean, field_var = _finalize_mean_var(sum_f, sumsq_f, count_f)
    wind_mean, wind_var = _finalize_mean_var(sum_w, sumsq_w, count_w)

    field_mean = field_mean.to(dtype)
    field_var = field_var.to(dtype)
    wind_mean = wind_mean.to(dtype)
    wind_var = wind_var.to(dtype)

    return field_mean, field_var, wind_mean, wind_var


class GaussianBellsPhiWrapperWithWinds(torch.utils.data.Dataset):
    """
    Unused in the current rollout-loss training path, kept here for completeness.
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

        self.use_base_normalization = getattr(self.base, "normalize", False)
        if self.use_base_normalization:
            self.inp_mean = self.base.inp_mean
            self.inp_var = self.base.inp_var
            self.wind_mean = self.base.wind_mean
            self.wind_var = self.base.wind_var

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
        uK = self._rand((1,), idx, offset=5).item()
        K = int(self.k_min + math.floor(uK * (self.k_max - self.k_min + 1)))
        K = max(self.k_min, min(self.k_max, K))

        u = 2.0 * self._rand((K,), idx, 10) - 1.0
        lat0 = torch.asin(u)
        lon0 = 2.0 * math.pi * self._rand((K,), idx, 20)
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * self._rand((K,), idx, 30)

        amp = self._rand((K,), idx, 40)
        if self.signed:
            signs = torch.where(self._rand((K,), idx, 50) < 0.5, -torch.ones_like(amp), torch.ones_like(amp))
            amp = amp * signs

        bump = torch.zeros(self.solver.nlat, self.solver.nlon, device=self.device, dtype=dtype)
        for i in range(K):
            gamma = _great_circle_distance(self.lat, self.lon, lat0[i].view(1, 1), lon0[i].view(1, 1))
            bump = bump + amp[i] * torch.exp(-(gamma * gamma) / (2.0 * sigma[i] * sigma[i]))

        bump = bump - bump.mean()
        bump = bump / (bump.std() + 1e-6)

        phi_mean = self.solver.gravity * self.solver.havg
        return (phi_mean + self.ref_phi_std * bump).to(dtype)

    def __getitem__(self, idx):
        with torch.inference_mode():
            inp_spec = self.solver.random_initial_condition(mach=self.mach)

            phi_grid = self._sample_phi_bells_grid(idx)
            zeros = torch.zeros_like(phi_grid)
            phi_spec0 = self.solver.grid2spec(torch.stack([phi_grid, zeros, zeros], dim=0))[0]

            inp_spec = inp_spec.clone()
            inp_spec[0] = phi_spec0
            inp_spec = torch.tril(inp_spec)

            tar_spec = self.solver.timestep(inp_spec, self.base.nsteps)

            inp_fields = self.solver.spec2grid(inp_spec)
            tar_fields = self.solver.spec2grid(tar_spec)

            inp_winds = self.solver.getuv(inp_spec[1:])
            tar_winds = self.solver.getuv(tar_spec[1:])

            if self.use_base_normalization:
                inp_fields = (inp_fields - self.inp_mean) / torch.sqrt(self.inp_var)
                tar_fields = (tar_fields - self.inp_mean) / torch.sqrt(self.inp_var)
                inp_winds = (inp_winds - self.wind_mean) / torch.sqrt(self.wind_var)
                tar_winds = (tar_winds - self.wind_mean) / torch.sqrt(self.wind_var)

            return inp_fields.clone(), inp_winds.clone(), tar_fields.clone(), tar_winds.clone()


class GaussianBellsAllFieldsWrapperWithWinds(torch.utils.data.Dataset):
    """
    Gaussian bells for ALL THREE fields (phi + vorticity + divergence),
    scaled using Williamson Case 6 means/stds.
    This wrapper does NOT normalize; trajectory dataset will normalize using gbells stats.
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

        self.inp_mean = None
        self.inp_var = None
        self.wind_mean = None
        self.wind_var = None

        # scale matching: Williamson Case 6 means/stds
        with torch.inference_mode():
            ref_spec = make_williamson_case6_ic_spec_from_winds(
                self.solver,
                R=4,
                omega=7.848e-6,
                K=None,
                h0=8000.0,
                flip_vort=False,
            )
            ref_grid = self.solver.spec2grid(ref_spec)

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
        phi_grid = self._sample_bells_grid(idx, ref_std=self.ref_std0, mean=self.ref_mean0, offset_base=0)
        vort_grid = self._sample_bells_grid(idx, ref_std=self.ref_std1, mean=self.ref_mean1, offset_base=1000)
        div_grid = self._sample_bells_grid(idx, ref_std=self.ref_std2, mean=self.ref_mean2, offset_base=2000)

        inp_grid = torch.stack([phi_grid, vort_grid, div_grid], dim=0)
        inp_spec = self.solver.grid2spec(inp_grid)
        inp_spec = torch.tril(inp_spec)
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


class TrajectoryFromSolverWithWinds(torch.utils.data.Dataset):
    """
    Converts an IC generator (in spectral space) into a trajectory of length K:
      x0 = inp_fields (normalized with gbells stats)
      x1..xK = solver rollout targets (normalized with gbells stats)
    Returns:
      inp_fields, inp_winds, tar_seq_fields, tar_seq_winds
    """
    def __init__(self, gbells_wrapper, K, step_nsteps=None):
        self.base = gbells_wrapper
        self.solver = self.base.solver
        self.device = self.base.device
        self.K = int(K)
        assert self.K >= 1

        self.step_nsteps = int(step_nsteps) if step_nsteps is not None else int(self.base.base.nsteps)

        self.inp_mean = self.base.inp_mean
        self.inp_var = self.base.inp_var
        self.wind_mean = self.base.wind_mean
        self.wind_var = self.base.wind_var

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        with torch.inference_mode():
            inp_spec = self.base.build_inp_spec(idx)

            inp_fields = self.solver.spec2grid(inp_spec)
            inp_winds = self.solver.getuv(inp_spec[1:])

            tar_seq_fields = []
            tar_seq_winds = []

            cur_spec = inp_spec
            for _ in range(self.K):
                cur_spec = self.solver.timestep(cur_spec, self.step_nsteps)
                cur_fields = self.solver.spec2grid(cur_spec)
                cur_winds = self.solver.getuv(cur_spec[1:])
                tar_seq_fields.append(cur_fields)
                tar_seq_winds.append(cur_winds)

            tar_seq_fields = torch.stack(tar_seq_fields, dim=0)
            tar_seq_winds = torch.stack(tar_seq_winds, dim=0)

            assert self.inp_mean is not None and self.inp_var is not None, "Gaussian-bells field stats not set"
            assert self.wind_mean is not None and self.wind_var is not None, "Gaussian-bells wind stats not set"

            inp_fields = (inp_fields - self.inp_mean) / torch.sqrt(self.inp_var)
            tar_seq_fields = (tar_seq_fields - self.inp_mean) / torch.sqrt(self.inp_var)

            inp_winds = (inp_winds - self.wind_mean) / torch.sqrt(self.wind_var)
            tar_seq_winds = (tar_seq_winds - self.wind_mean) / torch.sqrt(self.wind_var)

            return inp_fields.clone(), inp_winds.clone(), tar_seq_fields.clone(), tar_seq_winds.clone()


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

        self.loss_fn = SquaredL2LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_l1 = L1LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_l2 = L2LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_w11 = W11LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)

        self.n_rollout = int(n_rollout)
        self.burn_in = int(burn_in)
        self.detach_after_burnin = bool(detach_after_burnin)

        assert self.n_rollout >= 1
        assert 0 <= self.burn_in < self.n_rollout, "burn_in must be < n_rollout"

        self.rollout_eval_bundle = None

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

    def _bundle_on_device(self):
        if self.rollout_eval_bundle is None:
            raise RuntimeError("rollout_eval_bundle has not been set on the model.")
        bundle = self.rollout_eval_bundle
        solver = bundle["solver"].to(self.device)
        out = {
            "solver": solver,
            "inp_mean": bundle["inp_mean"].to(self.device),
            "inp_var": bundle["inp_var"].to(self.device),
            "wind_mean": bundle["wind_mean"].to(self.device) if bundle.get("wind_mean", None) is not None else None,
            "wind_var": bundle["wind_var"].to(self.device) if bundle.get("wind_var", None) is not None else None,
            "nsteps": bundle["nsteps"],
            "use_winds": bundle["use_winds"],
        }
        return out

    def _recompute_pred_winds(self, prd_fields_norm):
        bundle = self._bundle_on_device()
        solver = bundle["solver"]
        inp_mean = bundle["inp_mean"]
        inp_var = bundle["inp_var"]
        wind_mean = bundle["wind_mean"]
        wind_var = bundle["wind_var"]

        prd_fields_phys = prd_fields_norm * torch.sqrt(inp_var) + inp_mean
        prd_spec = solver.grid2spec(prd_fields_phys)
        prd_spec = torch.tril(prd_spec)
        prd_winds_phys = solver.getuv(prd_spec[:, 1:])
        prd_winds_norm = (prd_winds_phys - wind_mean) / torch.sqrt(wind_var)
        return prd_winds_norm

    def _rollout_and_loss(self, inp_fields, inp_winds, tar_seq_fields):
        """
        inp_fields: (B,3,H,W)
        inp_winds:  (B,2,H,W)
        tar_seq_fields: (B,K,3,H,W)
        returns scalar loss = sum_{t=burn_in+1..K} loss(pred_t, tar_t)
        """
        prd = inp_fields
        cur_winds = inp_winds
        loss = 0.0
        denom = 0

        for t in range(1, self.n_rollout + 1):
            if self.use_winds:
                prd = self.model(prd, cur_winds)
            else:
                prd = self.model(prd)

            if t == self.burn_in and self.detach_after_burnin:
                prd = prd.detach()

            if self.use_winds and t < self.n_rollout:
                cur_winds = self._recompute_pred_winds(prd)
                if t == self.burn_in and self.detach_after_burnin:
                    cur_winds = cur_winds.detach()

            if t > self.burn_in:
                loss = loss + self.loss_fn(prd, tar_seq_fields[:, t - 1])
                denom += 1

        loss = loss / max(denom, 1)
        return loss

    def _rollout_final_prediction(self, inp_fields, inp_winds):
        prd = inp_fields
        cur_winds = inp_winds

        for t in range(1, self.n_rollout + 1):
            if self.use_winds:
                prd = self.model(prd, cur_winds)
                if t < self.n_rollout:
                    cur_winds = self._recompute_pred_winds(prd)
            else:
                prd = self.model(prd)

        return prd

    def training_step(self, batch, batch_idx):
        if self.use_winds:
            inp_fields, inp_winds, tar_seq_fields, tar_seq_winds = batch
            loss = self._rollout_and_loss(inp_fields, inp_winds, tar_seq_fields)
        else:
            raise RuntimeError("This script currently expects winds (PARADIS) trajectory batches.")

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.use_winds:
            inp_fields, inp_winds, tar_seq_fields, tar_seq_winds = batch
            loss = self._rollout_and_loss(inp_fields, inp_winds, tar_seq_fields)

            with torch.no_grad():
                prd = self._rollout_final_prediction(inp_fields, inp_winds)
                tar_final = tar_seq_fields[:, self.n_rollout - 1]
                l1 = self.metric_l1(prd, tar_final)
                l2 = self.metric_l2(prd, tar_final)
                w11 = self.metric_w11(prd, tar_final)
        else:
            raise RuntimeError("This script currently expects winds (PARADIS) trajectory batches.")

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_l1", l1, sync_dist=True)
        self.log("val_l2", l2, sync_dist=True)
        self.log("val_w11", w11, sync_dist=True)
        return loss

    def configure_optimizers(self):
        lr = self.config["training"]["finetune_learning_rate"]
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}


def _run_rollout_eval_case(pl_module, case_name, ic_spec, autoreg_steps=20):
    bundle = pl_module._bundle_on_device()
    solver = bundle["solver"]
    nsteps = bundle["nsteps"]
    use_winds = bundle["use_winds"]

    inp_mean = bundle["inp_mean"]
    inp_var = bundle["inp_var"]
    wind_mean = bundle["wind_mean"]
    wind_var = bundle["wind_var"]

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
        prd_fields = (solver.spec2grid(ic_spec) - inp_mean) / sqrt_inp_var
        prd_fields = prd_fields.unsqueeze(0)

        if use_winds:
            prd_winds = solver.getuv(ic_spec[1:])
            prd_winds = (prd_winds - wind_mean) / sqrt_wind_var
            prd_winds = prd_winds.unsqueeze(0)
        else:
            prd_winds = None

        uspec = ic_spec.clone()

        print("-" * 70, flush=True)
        print(f"{case_name}: rollout evaluation for {autoreg_steps} steps", flush=True)

        for step in range(1, autoreg_steps + 1):
            if use_winds:
                prd_fields = pl_module(prd_fields, prd_winds)
                prd_unnorm = prd_fields * sqrt_inp_var + inp_mean
                prd_spec = solver.grid2spec(prd_unnorm.squeeze(0))
                prd_spec = torch.tril(prd_spec)
                prd_uv_grid = solver.getuv(prd_spec[1:])
                prd_winds = (prd_uv_grid - wind_mean) / sqrt_wind_var
                prd_winds = prd_winds.unsqueeze(0)
            else:
                prd_fields = pl_module(prd_fields)

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
        print(f"{case_name} {key:12s}: {mean_val.item():.6f} ± {std_val.item():.6f}", flush=True)

    if was_training:
        pl_module.train()

    return summary


class WilliamsonRolloutCallback(pl.Callback):
    def __init__(self, autoreg_steps=20):
        super().__init__()
        self.autoreg_steps = autoreg_steps

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if not hasattr(pl_module, "rollout_eval_bundle") or pl_module.rollout_eval_bundle is None:
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


def create_datasets_for_rollout(config, device, K):
    dt = config["data"]["dt"]
    dt_solver = config["data"]["dt_solver"]
    nsteps = dt // dt_solver
    nlat = config["data"]["nlat"]
    nlon = config["data"]["nlon"]
    grid = config["data"]["grid"]

    model_type = config["experiment"]["model_type"]
    use_winds = model_type == "paradis"
    if not use_winds:
        raise RuntimeError("This rollout-loss script is set up for PARADIS (winds) only.")

    base_train = PdeDatasetWithWinds(
        dt=dt, nsteps=nsteps, dims=(nlat, nlon), grid=grid, normalize=True, device=device
    )
    base_train.sht = RealSHT(nlat=nlat, nlon=nlon, grid=grid).to(device)
    base_train.set_initial_condition("random")
    base_train.set_num_examples(config["data"]["num_train_examples"])

    base_val = PdeDatasetWithWinds(
        dt=dt, nsteps=nsteps, dims=(nlat, nlon), grid=grid, normalize=True, device=device
    )
    base_val.sht = RealSHT(nlat=nlat, nlon=nlon, grid=grid).to(device)
    base_val.set_initial_condition("random")
    base_val.set_num_examples(config["data"]["num_val_examples"])

    bells_train = GaussianBellsAllFieldsWrapperWithWinds(
        base_train,
        mach=0.2, k_min=1, k_max=8,
        sigma_min_deg=5.0, sigma_max_deg=20.0,
        signed=True,
        seed=config["experiment"]["seed"],
    )
    bells_val = GaussianBellsAllFieldsWrapperWithWinds(
        base_val,
        mach=0.2, k_min=1, k_max=8,
        sigma_min_deg=5.0, sigma_max_deg=20.0,
        signed=True,
        seed=config["experiment"]["seed"] + 12345,
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
        ds.inp_mean = gb_field_mean
        ds.inp_var = gb_field_var
        ds.wind_mean = gb_wind_mean
        ds.wind_var = gb_wind_var

    print("Gaussian-bells normalization set:")
    print("  fields mean:", gb_field_mean.view(-1).tolist())
    print("  fields std :", torch.sqrt(gb_field_var).view(-1).tolist())
    print("  winds  mean:", gb_wind_mean.view(-1).tolist())
    print("  winds  std :", torch.sqrt(gb_wind_var).view(-1).tolist())

    train_ds = TrajectoryFromSolverWithWinds(bells_train, K=K, step_nsteps=nsteps)
    val_ds = TrajectoryFromSolverWithWinds(bells_val, K=K, step_nsteps=nsteps)

    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--pretrain_ckpt",
        type=str,
        required=True,
        help="Path to BEST pretrain checkpoint (.ckpt) to load weights from.",
    )
    parser.add_argument(
        "--burn_in",
        type=int,
        default=5,
        help="Ignore loss for steps 1..burn_in. Start loss at burn_in+1.",
    )
    parser.add_argument(
        "--detach_after_burnin",
        action="store_true",
        default=True,
        help="If set, detach the prediction after burn_in to truncate gradients.",
    )
    parser.add_argument(
        "--n_rollout",
        type=int,
        default=None,
        help="Total rollout steps K. If not set, uses 1 + training.nfuture from config.",
    )
    known_args, unknown_args = parser.parse_known_args()

    mp.set_start_method("spawn", force=True)

    config = load_config(known_args.config)
    config = update_config_from_args(config, unknown_args)

    pl.seed_everything(config["experiment"]["seed"], workers=True)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if known_args.n_rollout is None:
        K = 1 + int(config["training"]["nfuture"])
    else:
        K = int(known_args.n_rollout)

    if not (0 <= known_args.burn_in < K):
        raise ValueError(f"burn_in must be in [0, K-1]. Got burn_in={known_args.burn_in}, K={K}")

    train_ds, val_ds = create_datasets_for_rollout(config, device, K=K)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        persistent_workers=(config["data"]["num_workers"] > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        persistent_workers=(config["data"]["num_workers"] > 0),
    )

    model = SWERolloutLightningModule(
        config=config,
        n_rollout=K,
        burn_in=known_args.burn_in,
        detach_after_burnin=known_args.detach_after_burnin,
    )

    dt = config["data"]["dt"]
    dt_solver = config["data"]["dt_solver"]
    nsteps = dt // dt_solver

    model.rollout_eval_bundle = {
        "solver": train_ds.solver,
        "nsteps": nsteps,
        "use_winds": (config["experiment"]["model_type"] == "paradis"),
        "inp_mean": train_ds.inp_mean,
        "inp_var": train_ds.inp_var,
        "wind_mean": train_ds.wind_mean,
        "wind_var": train_ds.wind_var,
    }

    print(f"Loading pretrain checkpoint weights: {known_args.pretrain_ckpt}")
    ckpt = torch.load(known_args.pretrain_ckpt, map_location=device)
    state_dict = ckpt["state_dict"]

    for k in ["metric_w11.k_phi_mesh", "metric_w11.k_theta_mesh"]:
        if k in state_dict:
            del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    print("Loaded weights.\n")

    logger = TensorBoardLogger(
        config["training"]["save_dir"],
        name=f"{config['experiment']['name']}_rolloutloss",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="rolloutloss-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    wc2_checkpoint_callback = ModelCheckpoint(
        monitor="wc2_rollout_loss",
        filename="rolloutloss-best-wc2-epoch{epoch:02d}-wc2_{wc2_rollout_loss:.6f}",
        save_top_k=1,
        mode="min",
        save_last=False,
        auto_insert_metric_name=False,
    )

    wc6_checkpoint_callback = ModelCheckpoint(
        monitor="wc6_rollout_loss",
        filename="rolloutloss-best-wc6-epoch{epoch:02d}-wc6_{wc6_rollout_loss:.6f}",
        save_top_k=1,
        mode="min",
        save_last=False,
        auto_insert_metric_name=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    precision = 32
    if config["training"]["amp_mode"] == "fp16":
        precision = 16
    elif config["training"]["amp_mode"] == "bf16":
        precision = "bf16"

    trainer = pl.Trainer(
        max_epochs=config["training"]["finetune_epochs"],
        logger=logger,
        callbacks=[
            WilliamsonRolloutCallback(autoreg_steps=20),
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

    print(f"Rollout steps K={K}, burn_in={known_args.burn_in}, detach_after_burnin={known_args.detach_after_burnin}")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"\nBest rollout-loss checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best WC2 rollout checkpoint: {wc2_checkpoint_callback.best_model_path}")
    print(f"Best WC6 rollout checkpoint: {wc6_checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
