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

    # accumulators in float64
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

    # cast back to solver dtype
    field_mean = field_mean.to(dtype)
    field_var  = field_var.to(dtype)
    wind_mean  = wind_mean.to(dtype)
    wind_var   = wind_var.to(dtype)

    return field_mean, field_var, wind_mean, wind_var

# -----------------------
# gaussian bells helpers (same scaling idea you already use)
# -----------------------
def _great_circle_distance(lat, lon, lat0, lon0):
    sin1, cos1 = torch.sin(lat), torch.cos(lat)
    sin0, cos0 = torch.sin(lat0), torch.cos(lat0)
    dlon = lon - lon0
    cosgamma = sin1 * sin0 + cos1 * cos0 * torch.cos(dlon)
    cosgamma = torch.clamp(cosgamma, -1.0, 1.0)
    return torch.acos(cosgamma)

class GaussianBellsPhiWrapperWithWinds(torch.utils.data.Dataset):
    """
    Produces *ONE-STEP* pair (inp_fields, inp_winds, tar_fields, tar_winds) consistent with your earlier wrapper.
    We'll wrap this again into a trajectory dataset below.
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
            self.inp_mean  = self.base.inp_mean
            self.inp_var   = self.base.inp_var
            self.wind_mean = self.base.wind_mean
            self.wind_var  = self.base.wind_var

        # scale reference for phi fluctuations from original IC
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
        # deterministic K
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
                inp_winds  = (inp_winds  - self.wind_mean) / torch.sqrt(self.wind_var)
                tar_winds  = (tar_winds  - self.wind_mean) / torch.sqrt(self.wind_var)

            return inp_fields.clone(), inp_winds.clone(), tar_fields.clone(), tar_winds.clone()


class GaussianBellsAllFieldsWrapperWithWinds(torch.utils.data.Dataset):
    """
    Gaussian bells for ALL THREE fields (phi + vorticity + divergence):
      - build inp_grid with gaussian bells in channels 0,1,2
        (channel 0 mean = g*havg, channels 1/2 mean = 0)
      - grid2spec -> inp_spec
      - timestep inp_spec -> tar_spec
      - winds computed via getuv(spec[1:])
    IMPORTANT: This wrapper does NOT normalize; trajectory dataset will normalize using gbells stats.
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

        # These will be injected later (create_datasets_for_rollout)
        self.inp_mean = None   # (3,1,1)
        self.inp_var  = None   # (3,1,1)
        self.wind_mean = None  # (2,1,1)
        self.wind_var  = None  # (2,1,1)

        # scale matching: per-channel std from ORIGINAL random IC (grid)
        with torch.inference_mode():
            ref_spec = self.solver.random_initial_condition(mach=self.mach)
            ref_grid = self.solver.spec2grid(ref_spec)  # (3,H,W)

            ref0 = ref_grid[0]
            ref1 = ref_grid[1]
            ref2 = ref_grid[2]
            self.ref_std0 = (ref0 - ref0.mean()).std().clamp_min(1e-12)
            self.ref_std1 = (ref1 - ref1.mean()).std().clamp_min(1e-12)
            self.ref_std2 = (ref2 - ref2.mean()).std().clamp_min(1e-12)

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

        # deterministic K
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
        """Build the Gaussian-bells ALL-FIELDS IC in spectral space (triangular)."""
        phi_mean = self.solver.gravity * self.solver.havg

        phi_grid = self._sample_bells_grid(idx, ref_std=self.ref_std0, mean=phi_mean, offset_base=0)
        vort_grid = self._sample_bells_grid(idx, ref_std=self.ref_std1, mean=0.0, offset_base=1000)
        div_grid  = self._sample_bells_grid(idx, ref_std=self.ref_std2, mean=0.0, offset_base=2000)

        inp_grid = torch.stack([phi_grid, vort_grid, div_grid], dim=0)  # (3,H,W)
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


# -----------------------
# NEW: trajectory wrapper
# -----------------------


class TrajectoryFromSolverWithWinds(torch.utils.data.Dataset):
    """
    Converts an IC generator (in spectral space) into a trajectory of length K:
      x0 = inp_fields (normalized with gbells stats)
      x1..xK = solver rollout targets (normalized with gbells stats)
    Returns:
      inp_fields, inp_winds, tar_seq_fields, tar_seq_winds
    where tar_seq_fields has shape (K, C, H, W)
    """
    def __init__(self, gbells_wrapper, K, step_nsteps=None):
        self.base = gbells_wrapper              # GaussianBellsAllFieldsWrapperWithWinds
        self.solver = self.base.solver
        self.device = self.base.device
        self.K = int(K)
        assert self.K >= 1

        self.step_nsteps = int(step_nsteps) if step_nsteps is not None else int(self.base.base.nsteps)

        # gbells normalization tensors (must be injected before DataLoader iteration)
        self.inp_mean = self.base.inp_mean
        self.inp_var  = self.base.inp_var
        self.wind_mean = self.base.wind_mean
        self.wind_var  = self.base.wind_var

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        with torch.inference_mode():
            inp_spec = self.base.build_inp_spec(idx)

            # x0
            inp_fields = self.solver.spec2grid(inp_spec)
            inp_winds  = self.solver.getuv(inp_spec[1:])

            # rollout targets x1..xK
            tar_seq_fields = []
            tar_seq_winds = []

            cur_spec = inp_spec
            for _ in range(self.K):
                cur_spec = self.solver.timestep(cur_spec, self.step_nsteps)
                cur_fields = self.solver.spec2grid(cur_spec)
                cur_winds  = self.solver.getuv(cur_spec[1:])
                tar_seq_fields.append(cur_fields)
                tar_seq_winds.append(cur_winds)

            tar_seq_fields = torch.stack(tar_seq_fields, dim=0)  # (K,3,H,W)
            tar_seq_winds  = torch.stack(tar_seq_winds,  dim=0)  # (K,2,H,W)

            # normalize using gbells stats (assert they are set)
            assert self.inp_mean is not None and self.inp_var is not None, "Gaussian-bells field stats not set"
            assert self.wind_mean is not None and self.wind_var is not None, "Gaussian-bells wind stats not set"

            inp_fields = (inp_fields - self.inp_mean) / torch.sqrt(self.inp_var)
            tar_seq_fields = (tar_seq_fields - self.inp_mean) / torch.sqrt(self.inp_var)

            inp_winds = (inp_winds - self.wind_mean) / torch.sqrt(self.wind_var)
            tar_seq_winds = (tar_seq_winds - self.wind_mean) / torch.sqrt(self.wind_var)

            return inp_fields.clone(), inp_winds.clone(), tar_seq_fields.clone(), tar_seq_winds.clone()


#---------Not using this old one------------------------------
class TrajectoryFromSolverWithWinds_OLD(torch.utils.data.Dataset):
    """
    Converts an IC generator (in spectral space) into a trajectory of length K:
      x0 = inp_fields (normalized)
      x1..xK = solver rollout targets (normalized) for each step
    Returns:
      inp_fields, inp_winds, tar_seq_fields, tar_seq_winds
    where tar_seq_fields has shape (K, C, H, W)
    """
    def __init__(self, base_dataset_with_bells, K, step_nsteps=None):
        self.base = base_dataset_with_bells  # must have .solver, .base.nsteps, normalization tensors via base.base
        self.solver = self.base.solver
        self.device = self.base.device
        self.K = int(K)
        assert self.K >= 1

        # how many solver substeps correspond to "one dataset step"
        # default: use underlying base dataset nsteps
        self.step_nsteps = int(step_nsteps) if step_nsteps is not None else int(self.base.base.nsteps)

        # normalization tensors
        self.use_base_normalization = getattr(self.base.base, "normalize", False)
        if self.use_base_normalization:
            self.inp_mean  = self.base.base.inp_mean
            self.inp_var   = self.base.base.inp_var
            self.wind_mean = self.base.base.wind_mean
            self.wind_var  = self.base.base.wind_var

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        with torch.inference_mode():
            # We want spectral IC consistent with bells wrapper,
            # so we reconstruct it by reusing the wrapper logic:
            # easiest: call base to get (inp_fields, inp_winds, _, _), but we need SPEC for solver rollout.
            # So we replicate the core of GaussianBellsPhiWrapperWithWinds in a minimal way by calling it,
            # then invert is not possible. Better: modify GaussianBellsPhiWrapperWithWinds to optionally return inp_spec.
            # For simplicity here: we *do* re-generate spec using the underlying solver:
            # We'll rely on the same RNG used in base wrapper by calling base wrapper and ALSO directly computing tar via solver
            # is hard without spec.
            #
            # -> Practical fix: we directly generate inp_spec here by duplicating the base wrapper’s generation.
            #
            # So we assume base is GaussianBellsPhiWrapperWithWinds and we access its internals:
            inp_spec = self.solver.random_initial_condition(mach=self.base.mach)

            phi_grid = self.base._sample_phi_bells_grid(idx).to(self.solver.lap.dtype)
            zeros = torch.zeros_like(phi_grid)
            phi_spec0 = self.solver.grid2spec(torch.stack([phi_grid, zeros, zeros], dim=0))[0]

            inp_spec = inp_spec.clone()
            inp_spec[0] = phi_spec0
            inp_spec = torch.tril(inp_spec)

            # x0 in grid
            inp_fields = self.solver.spec2grid(inp_spec)
            inp_winds  = self.solver.getuv(inp_spec[1:])

            # rollout ground truth targets x1..xK by stepping the solver
            tar_seq_fields = []
            tar_seq_winds = []

            cur_spec = inp_spec
            for _ in range(self.K):
                cur_spec = self.solver.timestep(cur_spec, self.step_nsteps)
                cur_fields = self.solver.spec2grid(cur_spec)
                cur_winds  = self.solver.getuv(cur_spec[1:])
                tar_seq_fields.append(cur_fields)
                tar_seq_winds.append(cur_winds)

            tar_seq_fields = torch.stack(tar_seq_fields, dim=0)  # (K,3,H,W)
            tar_seq_winds  = torch.stack(tar_seq_winds,  dim=0)  # (K,2,H,W)

            # normalize exactly like training
            if self.use_base_normalization:
                inp_fields = (inp_fields - self.inp_mean) / torch.sqrt(self.inp_var)
                tar_seq_fields = (tar_seq_fields - self.inp_mean) / torch.sqrt(self.inp_var)

                inp_winds = (inp_winds - self.wind_mean) / torch.sqrt(self.wind_var)
                tar_seq_winds = (tar_seq_winds - self.wind_mean) / torch.sqrt(self.wind_var)

            return inp_fields.clone(), inp_winds.clone(), tar_seq_fields.clone(), tar_seq_winds.clone()

class ReversedHuberLossNoLat(torch.nn.Module):
    """
    Reverse huber (continuous):
        rho(a)=a                           if a<=delta
        rho(a)=(a^2+delta^2)/(2*delta)     if a>delta
    No latitude weights.
    """
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = float(delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target: (B,C,H,W) or (C,H,W)
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        e = (pred - target).abs()
        d = torch.as_tensor(self.delta, device=pred.device, dtype=pred.dtype)
        loss = torch.where(e <= d, e, (e * e + d * d) / (2.0 * d))
        return loss.mean()
# -----------------------
# Lightning Module with burn-in rollout loss
# -----------------------
class SWERolloutLightningModule(pl.LightningModule):
    #def __init__(self, config, n_rollout, burn_in, detach_after_burnin=True):
    def __init__(self, config, n_rollout, burn_in, detach_after_burnin=True, loss_delta: float = 1.0, lat_grid_deg: torch.Tensor = None):
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

        #self.loss_fn = SquaredL2LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        if lat_grid_deg is None:
            raise ValueError("lat_grid_deg must be provided for latitude-weighted loss.")
        self.loss_fn = ReversedHuberLossLat(lat_grid_deg=lat_grid_deg, delta=float(loss_delta), apply_latitude_weights=True)
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

    def _rollout_and_loss(self, inp_fields, inp_winds, tar_seq_fields):
        """
        inp_fields: (B,3,H,W)
        tar_seq_fields: (B,K,3,H,W)
        returns scalar loss = sum_{t=burn_in+1..K} loss(pred_t, tar_t)
        with optional detach after burn-in.
        """
        prd = inp_fields
        loss = 0.0
        denom = 0

        for t in range(1, self.n_rollout + 1):
            if self.use_winds:
                prd = self.model(prd, inp_winds)
            else:
                prd = self.model(prd)

            if t == self.burn_in and self.detach_after_burnin:
                prd = prd.detach()

            if t > self.burn_in:
                # tar_seq_fields index: t-1
                loss = loss + self.loss_fn(prd, tar_seq_fields[:, t - 1])
                denom += 1

        loss = loss / max(denom, 1)
        return loss

    def training_step(self, batch, batch_idx):
        if self.use_winds:
            inp_fields, inp_winds, tar_seq_fields, tar_seq_winds = batch
            loss = self._rollout_and_loss(inp_fields, inp_winds, tar_seq_fields)
        else:
            # if you later add a non-winds trajectory dataset, adapt similarly
            raise RuntimeError("This script currently expects winds (PARADIS) trajectory batches.")

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.use_winds:
            inp_fields, inp_winds, tar_seq_fields, tar_seq_winds = batch
            loss = self._rollout_and_loss(inp_fields, inp_winds, tar_seq_fields)

            # optional: report metrics only at final step (or average); here final step:
            with torch.no_grad():
                prd = inp_fields
                for _ in range(self.n_rollout):
                    prd = self.model(prd, inp_winds)
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


# -----------------------
# dataset creation
# -----------------------
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

    #bells_train = GaussianBellsPhiWrapperWithWinds(
    #    base_train,
    #    mach=0.2, k_min=1, k_max=8,
    #    sigma_min_deg=5.0, sigma_max_deg=20.0,
    #    signed=True,
    #    seed=config["experiment"]["seed"],
    #)
    #bells_val = GaussianBellsPhiWrapperWithWinds(
    #    base_val,
    #    mach=0.2, k_min=1, k_max=8,
    #    sigma_min_deg=5.0, sigma_max_deg=20.0,
    #    signed=True,
    #    seed=config["experiment"]["seed"] + 12345,
    #)
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
    # --- NEW: compute normalization stats from Gaussian-bells ALL-FIELDS IC samples ---
    gb_field_mean, gb_field_var, gb_wind_mean, gb_wind_var = compute_stats_for_gbells_allfields_ic(
        solver=base_train.solver,  # or bells_train.solver (same)
        seed=int(config["experiment"]["seed"]),
        num_samples=200,      # hardcoded, no config changes
        mach=0.2,
        k_min=1, k_max=8,
        sigma_min_deg=5.0, sigma_max_deg=20.0,
        signed=True,
    )

    # Inject stats into BOTH wrappers so they normalize identically via trajectory dataset
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
    # Trajectory dataset: targets x1..xK using solver, each step uses "nsteps"
    train_ds = TrajectoryFromSolverWithWinds(bells_train, K=K, step_nsteps=nsteps)
    val_ds   = TrajectoryFromSolverWithWinds(bells_val,   K=K, step_nsteps=nsteps)
    lat_grid_deg = (base_train.solver.lats * 180.0 / math.pi).to(torch.float32)
    return train_ds, val_ds, lat_grid_deg


# -----------------------
# main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--pretrain_ckpt", type=str, required=True,
                        help="Path to BEST pretrain checkpoint (.ckpt) to load weights from.")
    parser.add_argument("--burn_in", type=int, default=5,
                        help="Ignore loss for steps 1..burn_in. Start loss at burn_in+1.")
    parser.add_argument("--detach_after_burnin", action="store_true", default=True,
                        help="If set, detach the prediction after burn_in to truncate gradients.")
    parser.add_argument("--n_rollout", type=int, default=None,
                        help="Total rollout steps K. If not set, uses 1 + training.nfuture from config.")
    parser.add_argument(
        "--loss_delta",
        type=float,
        default=1.0,
        help="Delta for reverse huber: linear for |e|<=delta, quadratic for |e|>delta (continuous).",
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

    train_ds, val_ds, lat_grid_deg = create_datasets_for_rollout(config, device, K=K)

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

    #model = SWERolloutLightningModule(
    #    config=config,
    #    n_rollout=K,
    #    burn_in=known_args.burn_in,
    #    detach_after_burnin=known_args.detach_after_burnin,
    #)
    model = SWERolloutLightningModule(
        config=config,
        n_rollout=K,
        burn_in=known_args.burn_in,`
        detach_after_burnin=known_args.detach_after_burnin,
        loss_delta=known_args.loss_delta,
        lat_grid_deg=lat_grid_deg,
    )

    # load best pretrain weights
    print(f"Loading pretrain checkpoint weights: {known_args.pretrain_ckpt}")
    ckpt = torch.load(known_args.pretrain_ckpt, map_location=device)
    state_dict = ckpt["state_dict"]

    # drop problematic buffers if present
    for k in ["metric_w11.k_phi_mesh", "metric_w11.k_theta_mesh"]:
        if k in state_dict:
            del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    print("Loaded weights.\n")

    # trainer
    logger = TensorBoardLogger(config["training"]["save_dir"],
                              name=f"{config['experiment']['name']}_rolloutloss")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="rolloutloss-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
        save_last=True,
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
        callbacks=[checkpoint_callback, lr_monitor],
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


if __name__ == "__main__":
    main()
