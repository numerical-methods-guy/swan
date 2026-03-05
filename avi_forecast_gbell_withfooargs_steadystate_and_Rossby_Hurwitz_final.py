# avi_forecast_gbell.py
#
# Same behavior as forecast.py (saves .pt outputs, saves PNGs, prints metrics, writes metrics.csv),
# PLUS: optional TensorBoard logging of scalars + figures/images.
#
# Change added in this version:
# - Add CLI control of dt_solver via --dt_solver to override config["data"]["dt_solver"].

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import pytorch_lightning as pl

from torch.utils.tensorboard import SummaryWriter

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




def _online_update_mean_var(sum_, sumsq_, count_, x):
    # x: (C,H,W)
    # returns updated (sum, sumsq, count)
    sum_ = sum_ + x.sum(dim=(-1, -2))            # (C,)
    sumsq_ = sumsq_ + (x * x).sum(dim=(-1, -2)) # (C,)
    count_ = count_ + x.shape[-1] * x.shape[-2]  # scalar
    return sum_, sumsq_, count_


def _finalize_mean_var(sum_, sumsq_, count_, eps=1e-12):
    # mean, var as (C,1,1)
    mean = (sum_ / count_).reshape(-1, 1, 1)
    var = (sumsq_ / count_ - (sum_ / count_)**2).clamp_min(eps).reshape(-1, 1, 1)
    return mean, var

def make_gaussian_bells_ic_spec(
    solver,
    seed,
    gbells_cfg,
    mach=0.2,
):
    """
    Training-style gaussian-bells IC:
      1) draw random spectral IC
      2) compute ref_phi_std from its *grid* phi fluctuations
      3) create gaussian-bells phi_grid with SAME ref_phi_std scaling
      4) replace only phi channel in spectral IC
    Returns: ic_spec (3,lmax,mmax) complex
    """
    with torch.no_grad():
        ic = solver.random_initial_condition(mach=mach)  # (3,lmax,mmax)

        ref_grid = solver.spec2grid(ic)   # (3,nlat,nlon)
        ref_phi = ref_grid[0]
        ref_phi_std = (ref_phi - ref_phi.mean()).std().clamp_min(1e-12)

        phi_grid = make_phi_gaussian_bells_grid(
            solver=solver,
            ref_phi_std=ref_phi_std,
            seed=seed,
            **gbells_cfg,
        ).to(ref_grid.dtype)

        zeros = torch.zeros_like(phi_grid)
        phi_spec0 = solver.grid2spec(torch.stack([phi_grid, zeros, zeros], dim=0))[0]

        ic = ic.clone()
        ic[0] = phi_spec0
        ic = torch.tril(ic)
        return ic





def make_gaussian_bells_ic_spec_training_exact(
    solver,
    idx,            # sample index
    seed,           # base seed like config["experiment"]["seed"]
    mach=0.2,
    k_min=1,
    k_max=8,
    sigma_min_deg=5.0,
    sigma_max_deg=20.0,
    signed=True,
):
    """
    Matches GaussianBellsPhiWrapperWithWinds defaults + RNG behavior:
      - mach=0.2
      - k_min=1,k_max=8
      - sigma range 5..20 degrees
      - signed=True
      - RNG: manual_seed(seed + idx + offset)
      - K is deterministic using offset=5, NOT torch.randint
      - ref_phi_std computed once from random IC (we do it here per-call; OK for stats)
    """
    device = solver.lap.device
    dtype  = solver.lap.dtype

    lat = solver.lats.reshape(-1, 1).to(device)
    lon = solver.lons.reshape(1, -1).to(device)

    def _rand(shape, offset=0):
        g = torch.Generator(device=device)
        g.manual_seed(int(seed) + int(idx) + int(offset))
        return torch.rand(*shape, device=device, dtype=dtype, generator=g)

    # 1) random IC in spectral space
    inp_spec = solver.random_initial_condition(mach=mach)

    # 2) reference std from original random IC channel-0 fluctuations
    ref_grid = solver.spec2grid(inp_spec)
    ref_phi_std = (ref_grid[0] - ref_grid[0].mean()).std().clamp_min(1e-12)

    # 3) deterministic K from seeded uniform (offset=5) like training
    uK = _rand((1,), offset=5).item()
    K = int(k_min + math.floor(uK * (k_max - k_min + 1)))
    K = max(k_min, min(k_max, K))

    # 4) sample centers/sigmas/amps like training offsets
    u = 2.0 * _rand((K,), offset=10) - 1.0
    lat0 = torch.asin(u)
    lon0 = 2.0 * math.pi * _rand((K,), offset=20)

    sigma_min = math.radians(sigma_min_deg)
    sigma_max = math.radians(sigma_max_deg)
    sigma = sigma_min + (sigma_max - sigma_min) * _rand((K,), offset=30)

    amp = _rand((K,), offset=40)
    if signed:
        signs = torch.where(_rand((K,), offset=50) < 0.5,
                            -torch.ones_like(amp), torch.ones_like(amp))
        amp = amp * signs

    # 5) build bump, normalize bump to mean 0 std 1 (with +1e-6 exactly like training)
    bump = torch.zeros(solver.nlat, solver.nlon, device=device, dtype=dtype)
    for i in range(K):
        gamma = _great_circle_distance(lat, lon, lat0[i].view(1, 1), lon0[i].view(1, 1))
        bump = bump + amp[i] * torch.exp(-(gamma * gamma) / (2.0 * sigma[i] * sigma[i]))

    bump = bump - bump.mean()
    bump = bump / (bump.std() + 1e-6)

    # 6) phi_mean = g*havg and scale by ref_phi_std
    phi_mean = solver.gravity * solver.havg
    phi_grid = phi_mean + ref_phi_std * bump

    zeros = torch.zeros_like(phi_grid)
    phi_spec0 = solver.grid2spec(torch.stack([phi_grid, zeros, zeros], dim=0))[0]

    inp_spec = inp_spec.clone()
    inp_spec[0] = phi_spec0
    inp_spec = torch.tril(inp_spec)
    return inp_spec






def compute_stats_for_ic_distribution(
    solver,
    make_ic_spec_fn,
    num_samples,
    device,
):
    """
    Returns:
      field_mean, field_var: (3,1,1)
      wind_mean, wind_var:   (2,1,1)
    """
    sum_f = torch.zeros(3, device=device, dtype=torch.float64)
    sumsq_f = torch.zeros(3, device=device, dtype=torch.float64)
    count_f = torch.tensor(0.0, device=device, dtype=torch.float64)

    sum_w = torch.zeros(2, device=device, dtype=torch.float64)
    sumsq_w = torch.zeros(2, device=device, dtype=torch.float64)
    count_w = torch.tensor(0.0, device=device, dtype=torch.float64)

    with torch.no_grad():
        for i in range(num_samples):
            ic_spec = make_ic_spec_fn(i)

            grid = solver.spec2grid(ic_spec).to(torch.float64)     # (3,H,W)
            uv = solver.getuv(ic_spec[1:]).to(torch.float64)       # (2,H,W)

            sum_f, sumsq_f, count_f = _online_update_mean_var(sum_f, sumsq_f, count_f, grid)
            sum_w, sumsq_w, count_w = _online_update_mean_var(sum_w, sumsq_w, count_w, uv)

    field_mean, field_var = _finalize_mean_var(sum_f, sumsq_f, count_f)
    wind_mean, wind_var   = _finalize_mean_var(sum_w, sumsq_w, count_w)

    return field_mean.to(solver.lap.dtype), field_var.to(solver.lap.dtype), \
           wind_mean.to(solver.lap.dtype), wind_var.to(solver.lap.dtype)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def update_config_from_args(config, unknown_args):
    # expects pairs: --a.b.c value
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

        # best-effort cast
        try:
            if "." in val:
                val = float(val)
            else:
                val = int(val)
        except ValueError:
            # allow strings/bools like "bicubic"
            if val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False

        cur[keys[-1]] = val
    return config

class SWELightningModule(pl.LightningModule):
    """Unified Lightning Module for loading SFNO, Transformer, and Paradis models."""

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
        return self.model(args[0])



def _great_circle_distance(lat, lon, lat0, lon0):
    sin1, cos1 = torch.sin(lat), torch.cos(lat)
    sin0, cos0 = torch.sin(lat0), torch.cos(lat0)
    dlon = lon - lon0
    cosgamma = sin1 * sin0 + cos1 * cos0 * torch.cos(dlon)
    cosgamma = torch.clamp(cosgamma, -1.0, 1.0)
    return torch.acos(cosgamma)


def _rand(device, dtype, shape, seed=None):
    if seed is None:
        return torch.rand(*shape, device=device, dtype=dtype)
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return torch.rand(*shape, device=device, dtype=dtype, generator=g)


def make_phi_gaussian_bells_grid(
    solver,
    ref_phi_std,
    seed,
    k_min=1,
    k_max=8,
    sigma_min_deg=5.0,
    sigma_max_deg=20.0,
    signed=True,
):
    device = solver.lap.device
    dtype = solver.lap.dtype

    lat = solver.lats.reshape(-1, 1).to(device)
    lon = solver.lons.reshape(1, -1).to(device)

    #K = int(torch.randint(k_min, k_max + 1, (1,), device=device).item())



    uK = _rand(device, dtype, (1,), seed=seed + 5).item()
    K = int(k_min + math.floor(uK * (k_max - k_min + 1)))
    K = max(k_min, min(k_max, K))



    # uniform centers on sphere
    u = 2.0 * _rand(device, dtype, (K,), seed=seed + 10) - 1.0
    lat0 = torch.asin(u)
    lon0 = 2.0 * math.pi * _rand(device, dtype, (K,), seed=seed + 20)

    sigma_min = math.radians(sigma_min_deg)
    sigma_max = math.radians(sigma_max_deg)
    sigma = sigma_min + (sigma_max - sigma_min) * _rand(device, dtype, (K,), seed=seed + 30)

    amp = _rand(device, dtype, (K,), seed=seed + 40)
    if signed:
        signs = torch.where(_rand(device, dtype, (K,), seed=seed + 50) < 0.5,
                            -torch.ones_like(amp), torch.ones_like(amp))
        amp = amp * signs

    bump = torch.zeros(solver.nlat, solver.nlon, device=device, dtype=dtype)
    for i in range(K):
        gamma = _great_circle_distance(lat, lon, lat0[i].view(1, 1), lon0[i].view(1, 1))
        bump = bump + amp[i] * torch.exp(-(gamma * gamma) / (2.0 * sigma[i] * sigma[i]))

    bump = bump - bump.mean()
    bump = bump / (bump.std() + 1e-6)

    phi_mean = solver.gravity * solver.havg
    return phi_mean + ref_phi_std * bump


def compute_energy_spectra(fields, sht):
    """
    Compute energy spectra for shallow water equation fields.

    Args:
        fields: Tensor of shape (batch, 3, nlat, nlon) containing [h, vorticity, divergence]
        sht: RealSHT object for spherical harmonic transforms

    Returns:
        Dictionary containing power spectra for rotational, divergent, and potential energy
    """
    h = fields[:, 0:1]
    vort = fields[:, 1:2]
    div = fields[:, 2:3]

    h_spec = sht(h)
    vort_spec = sht(vort)
    div_spec = sht(div)

    h_power = torch.abs(h_spec) ** 2
    vort_power = torch.abs(vort_spec) ** 2
    div_power = torch.abs(div_spec) ** 2

    if h_power.dim() == 4:
        h_power = h_power.squeeze(1)
        vort_power = vort_power.squeeze(1)
        div_power = div_power.squeeze(1)

    batch_size = fields.shape[0]
    nlat = h_power.shape[1]
    nmodes = h_power.shape[2]
    max_k = min(nlat // 2, nmodes)

    rot_spectrum = torch.zeros(batch_size, max_k, device=fields.device)
    div_spectrum = torch.zeros(batch_size, max_k, device=fields.device)
    pot_spectrum = torch.zeros(batch_size, max_k, device=fields.device)
    counts = torch.zeros(max_k, device=fields.device)

    k_lat = torch.arange(nlat, device=fields.device, dtype=torch.float32)
    k_lon = torch.arange(nmodes, device=fields.device, dtype=torch.float32)
    k_lat_grid, k_lon_grid = torch.meshgrid(k_lat, k_lon, indexing="ij")
    k_total = torch.sqrt(k_lat_grid**2 + k_lon_grid**2)
    k_bins = torch.clamp(k_total.long(), 0, max_k - 1)

    for k in range(max_k):
        mask = k_bins == k
        if mask.any():
            counts[k] = mask.sum().float()
            for b in range(batch_size):
                rot_spectrum[b, k] = (vort_power[b] * mask).sum()
                div_spectrum[b, k] = (div_power[b] * mask).sum()
                pot_spectrum[b, k] = (h_power[b] * mask).sum()

    counts = torch.maximum(counts, torch.ones_like(counts))
    rot_spectrum = rot_spectrum / counts.unsqueeze(0)
    div_spectrum = div_spectrum / counts.unsqueeze(0)
    pot_spectrum = pot_spectrum / counts.unsqueeze(0)

    for k in range(1, max_k):
        rot_spectrum[:, k] *= 2.0
        div_spectrum[:, k] *= 2.0
        pot_spectrum[:, k] *= 2.0

    total_spectrum = rot_spectrum + div_spectrum + pot_spectrum

    rot_spectrum = rot_spectrum.mean(dim=0)
    div_spectrum = div_spectrum.mean(dim=0)
    pot_spectrum = pot_spectrum.mean(dim=0)
    total_spectrum = total_spectrum.mean(dim=0)

    return {
        "rotational": rot_spectrum.cpu().numpy(),
        "divergent": div_spectrum.cpu().numpy(),
        "potential": pot_spectrum.cpu().numpy(),
        "total": total_spectrum.cpu().numpy(),
        "wavenumbers": np.arange(max_k),
    }


def make_energy_spectra_figure(pred_spectra, truth_spectra, step, model_name="Model"):
    """Create (but do not save) an energy spectra figure."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    k = pred_spectra["wavenumbers"]
    k_plot = k[1:]

    k_ref = np.array([5.0, 50.0])
    ref_minus3 = 1e4 * k_ref ** (-3.0)
    ref_minus5_3 = 1e3 * k_ref ** (-5.0 / 3.0)

    titles = [
        "Rotational Kinetic Energy",
        "Divergent Kinetic Energy",
        "Potential Energy",
        "Total Energy",
    ]
    keys = ["rotational", "divergent", "potential", "total"]

    for idx, (title, key) in enumerate(zip(titles, keys)):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        ax.loglog(
            k_plot,
            pred_spectra[key][1:],
            "b-",
            linewidth=2,
            label=f"{model_name} Prediction",
            alpha=0.8,
        )
        ax.loglog(
            k_plot,
            truth_spectra[key][1:],
            "r--",
            linewidth=2,
            label="Ground Truth",
            alpha=0.8,
        )
        ax.loglog(k_ref, ref_minus3, "k:", linewidth=1.5, alpha=0.5, label=r"$k^{-3}$")
        ax.loglog(
            k_ref, ref_minus5_3, "k-.", linewidth=1.5, alpha=0.5, label=r"$k^{-5/3}$"
        )

        ax.set_xlabel("Wavenumber $l$", fontsize=11)
        ax.set_ylabel("Power Spectrum", fontsize=11)
        ax.set_title(f"{title} (t={step})", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="lower left", fontsize=9)
        ax.set_xlim([1, k_plot[-1]])

    return fig


def plot_energy_spectra(
    pred_spectra, truth_spectra, step, output_path, model_name="Model"
):
    """Save spectra plot to PNG (same behavior as forecast.py)."""
    fig = make_energy_spectra_figure(
        pred_spectra, truth_spectra, step, model_name=model_name
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _log_step_scalars(writer, step, step_metrics):
    """Helper: log scalars to TensorBoard."""
    if writer is None:
        return
    for k, v in step_metrics.items():
        writer.add_scalar(f"metrics/{k}", float(v), step)


def _log_field_triplet(writer, step, pred_data, truth_data, error_data, plot_channel):
    """Helper: log prediction/truth/error as images (HW) to TensorBoard."""
    if writer is None:
        return
    writer.add_image(
        f"fields/pred_ch{plot_channel}", pred_data, step, dataformats="HW"
    )
    writer.add_image(
        f"fields/truth_ch{plot_channel}", truth_data, step, dataformats="HW"
    )
    writer.add_image(
        f"fields/error_ch{plot_channel}", error_data, step, dataformats="HW"
    )


def make_williamson_case6_ic_spec_from_winds(
    solver,
    R=4,
    omega=7.848e-6,
    K=None,          # if None: K = omega (as in your reference code)
    h0=8000.0,
    flip_vort=False,
    eps_cos=1e-6,    # to avoid cos(lat)^(-2) blow-up at poles
):
    """
    Williamson Case 6 (Rossby–Haurwitz wave) IC for state [phi, vorticity, divergence] in SPECTRAL space.

    - Builds physical winds u(lat,lon), v(lat,lon) on the sphere (eastward/northward).
    - Builds height h(lat,lon).
    - Sets phi = g*h.
    - Computes vort/div *spectrally* from winds using solver.vrtdivspec(uv_grid).
    """
    device = solver.lap.device
    dtype  = solver.lap.dtype  # float64 typically

    lat = solver.lats.to(device=device, dtype=dtype).reshape(-1, 1)  # (nlat,1)
    lon = solver.lons.to(device=device, dtype=dtype).reshape(1, -1)  # (1,nlon)

    a     = solver.radius.to(dtype=dtype)   # Earth radius
    Omega = solver.omega.to(dtype=dtype)    # rotation rate
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

    # avoid coslat^{-2} at poles
    cos_safe = torch.clamp(torch.abs(coslat), min=eps_cos) * torch.sign(coslat + 0.0)

    # --- A,B,C as in your reference code (geom.coslat etc.) ---
    # A = omega/2 * (2*Omega + omega) * cos^2
    #   + (K^2)/4 * cos^(2R) * ( (R+1) cos^2 + (2R^2 - R - 2) - 2 R^2 cos^{-2} )
    A = (omega_t / 2.0) * (2.0 * Omega + omega_t) * (coslat ** 2) \
        + (K_t ** 2) / 4.0 * (coslat ** (2 * R_int)) * (
            (R_int + 1) * (coslat ** 2)
            + (2.0 * (R_int ** 2) - R_int - 2.0)
            - 2.0 * (R_int ** 2) * (cos_safe ** (-2))
        )

    # B = 2*(Omega+omega)*K/((R+1)(R+2)) * cos^R * ( (R^2+2R+2) - (R+1)^2 cos^2 )
    B = 2.0 * (Omega + omega_t) * K_t / ((R_int + 1) * (R_int + 2)) * (coslat ** R_int) * (
        (R_int ** 2 + 2 * R_int + 2)
        - ((R_int + 1) ** 2) * (coslat ** 2)
    )

    # C = (K^2)/4 * cos^(2R) * ( (R+1) cos^2 - (R+2) )
    C = (K_t ** 2) / 4.0 * (coslat ** (2 * R_int)) * (
        (R_int + 1) * (coslat ** 2) - (R_int + 2.0)
    )

    # --- Height h(lat,lon) ---
    # h = h0 + (a^2*(A + B cos(R lon) + C cos(2R lon))) / g
    h = h0_t + (a ** 2) * (A + B * torch.cos(R_int * lon) + C * torch.cos(2.0 * R_int * lon)) / g

    # --- Winds u(lat,lon), v(lat,lon) ---
    # u = a*omega*coslat + a*K*coslat^(R-1) * ( R sin^2 - cos^2 ) * cos(R lon)
    # v = -a*K*R*coslat^(R-1) * sinlat * sin(R lon)
    cos_pow = coslat ** (R_int - 1)
    u = a * omega_t * coslat + a * K_t * cos_pow * (R_int * (sinlat ** 2) - (coslat ** 2)) * torch.cos(R_int * lon)
    v = -a * K_t * R_int * cos_pow * sinlat * torch.sin(R_int * lon)

    # phi = g*h (your state uses geopotential)
    phi_grid = g * h

    # pack uv and compute vort/div spectrally
    uv_grid = torch.stack([u, v], dim=0)              # (2,nlat,nlon)
    vrtdiv_spec = solver.vrtdivspec(uv_grid)          # (2,lmax,mmax) = [vort_spec, div_spec]
    if flip_vort:
        vrtdiv_spec[0] = -vrtdiv_spec[0]

    # spectral phi
    phi_spec = solver.grid2spec(torch.stack([phi_grid], dim=0))[0]  # (lmax,mmax)

    # assemble [phi, vort, div]
    ctype = torch.complex128 if solver.lap.dtype == torch.float64 else torch.complex64
    uspec = torch.zeros(3, solver.lmax, solver.mmax, dtype=ctype, device=device)
    uspec[0] = phi_spec
    uspec[1:] = vrtdiv_spec.to(dtype=ctype)

    return torch.tril(uspec)
    
def make_williamson_case2_ic_spec(solver, gh0=29400.0, flip_vort=False):
    """
    Build Williamson Case 2 steady nonlinear zonal geostrophic flow as
    spectral initial condition for state [phi, vorticity, divergence].
    """
    device = solver.lap.device
    dtype = solver.lap.dtype

    lat = solver.lats.to(device=device, dtype=dtype).reshape(-1, 1)  # (nlat,1)
    lon = solver.lons.to(device=device, dtype=dtype).reshape(1, -1)  # (1,nlon)

    # radius and rotation rate: adjust if your solver uses different names
    a = float(solver.radius)#float(getattr(solver, "radius", getattr(solver, "a")))
    print([k for k in dir(solver) if "mega" in k.lower() or "rot" in k.lower() or k.lower() == "omega"])
    Omega = float(solver.omega)#float(getattr(solver, "Omega", getattr(solver, "rotation_speed")))

    day = 86400.0
    u0 = 2.0 * math.pi * a / (12.0 * day)

    sinlat = torch.sin(lat)

    # geopotential phi = g h  (directly from Williamson Case 2)
    phi = gh0 - (a * Omega * u0 + 0.5 * u0 * u0) * (sinlat ** 2)

    # vorticity/divergence for u=u0 cos(lat), v=0
    vort = (u0 / a) * sinlat
    if flip_vort:
        vort = -vort
    div = torch.zeros_like(vort)

    phi_grid  = phi  + 0.0 * lon
    vort_grid = vort + 0.0 * lon
    div_grid  = div  + 0.0 * lon

    grid_fields = torch.stack([phi_grid, vort_grid, div_grid], dim=0)  # (3,nlat,nlon)
    return solver.grid2spec(grid_fields)

def make_williamson_case2_ic_spec_from_winds(
    solver,
    gh0=29400.0,
    u0=None,              # if None, uses 2πa/(12 days)
    flip_vort=False,
):
    """
    Build Williamson Case 2 IC for state [phi, vorticity, divergence] in SPECTRAL space,
    computing vort/div from PHYSICAL winds (u,v) using solver.vrtdivspec (spectral method).

    Assumes:
      - u is zonal wind (eastward) in m/s
      - v is meridional wind (northward) in m/s
      - phi is geopotential = g*h (units m^2/s^2), consistent with this solver's galewsky IC
    """
    device = solver.lap.device
    dtype  = solver.lap.dtype  # float64 typically

    lat = solver.lats.to(device=device, dtype=dtype).reshape(-1, 1)  # (nlat,1)
    lon = solver.lons.to(device=device, dtype=dtype).reshape(1, -1)  # (1,nlon)

    a     = solver.radius.to(dtype=dtype)
    Omega = solver.omega.to(dtype=dtype)

    if u0 is None:
        day = torch.as_tensor(86400.0, device=device, dtype=dtype)
        u0  = 2.0 * torch.pi * a / (12.0 * day)   # Williamson Case 2
    else:
        u0 = torch.as_tensor(float(u0), device=device, dtype=dtype)

    sinlat = torch.sin(lat)
    coslat = torch.cos(lat)

    # --- Winds on the sphere (physical winds, NOT contravariant) ---
    # solid body rotation: u = u0 cos(lat), v = 0
    u = u0 * coslat
    v = torch.zeros_like(u)

    # broadcast to full grid
    u_grid = u + 0.0 * lon
    v_grid = v + 0.0 * lon

    # --- Geopotential phi = g*h (Williamson Case 2) ---
    # phi(lat) = gh0 - (a Ω u0 + 0.5 u0^2) sin^2(lat)
    phi_lat = torch.as_tensor(float(gh0), device=device, dtype=dtype) \
              - (a * Omega * u0 + 0.5 * u0 * u0) * (sinlat ** 2)
    phi_grid = phi_lat + 0.0 * lon

    # --- Spectral vorticity/divergence from winds (spectral method) ---
    uv_grid = torch.stack([u_grid, v_grid], dim=0)          # (2,nlat,nlon)
    vrtdiv_spec = solver.vrtdivspec(uv_grid)                # (2,lmax,mmax)  [vort_spec, div_spec] convention
    if flip_vort:
        vrtdiv_spec[0] = -vrtdiv_spec[0]

    # --- Spectral geopotential ---
    phi_spec = solver.grid2spec(torch.stack([phi_grid], dim=0))[0]  # (lmax,mmax)

    # --- Assemble [phi, vort, div] ---
    ctype = torch.complex128 if solver.lap.dtype == torch.float64 else torch.complex64
    uspec = torch.zeros(3, solver.lmax, solver.mmax, dtype=ctype, device=device)
    uspec[0] = phi_spec
    uspec[1:] = vrtdiv_spec.to(dtype=ctype)

    return torch.tril(uspec)


def autoregressive_inference_with_winds(
    model,
    dataset,
    loss_fn,
    metrics_dict,
    output_dir,
    nsteps,
    model_name="Model",
    autoreg_steps=10,
    plot_channel=0,
    save_plots=True,
    spectral_analysis=True,
    device=torch.device("cpu"),
    writer=None,
    ic_spec=None
    #use_gbells=False,
    #gbells_cfg=None,
    #seed=0
):
    """Perform autoregressive inference and generate forecast plots (with winds)."""
    model.eval()
    model.to(device)

    dataset.solver = dataset.solver.to(device)
    dataset.sht = dataset.sht.to(device)
    dataset.inp_mean = dataset.inp_mean.to(device)
    dataset.inp_var = dataset.inp_var.to(device)
    dataset.wind_mean = dataset.wind_mean.to(device)
    dataset.wind_var = dataset.wind_var.to(device)

    os.makedirs(output_dir, exist_ok=True)
    if spectral_analysis:
        spectral_dir = output_dir
        os.makedirs(spectral_dir, exist_ok=True)

    metrics_data = {key: [] for key in metrics_dict.keys()}
    metrics_data["loss"] = []

    pad_width = len(str(autoreg_steps))

    print(f"Starting Autoregressive Inference for {autoreg_steps} steps...")

    with torch.no_grad():
        #ic = dataset.solver.random_initial_condition(mach=0.2)
        ic = ic_spec if ic_spec is not None else dataset.solver.random_initial_condition(mach=0.2)
        #if use_gbells:
            # compute reference std of phi fluctuations (same idea as training)
            #ref_grid = dataset.solver.spec2grid(ic)
            #ref_phi = ref_grid[0]
            #ref_phi_std = (ref_phi - ref_phi.mean()).std().clamp_min(1e-12)

            #phi_grid = make_phi_gaussian_bells_grid(
            #    solver=dataset.solver,
            #    ref_phi_std=ref_phi_std,
           #     seed=seed,
          #      **gbells_cfg,
         #   ).to(ref_grid.dtype)

        #    # replace only channel-0 in spectral space using grid2spec
        #    zeros = torch.zeros_like(phi_grid)
        #    phi_spec0 = dataset.solver.grid2spec(torch.stack([phi_grid, zeros, zeros], dim=0))[0]
        #    ic = ic.clone()
        #    ic[0] = phi_spec0
        #    ic = torch.tril(ic)
        inp_mean = dataset.inp_mean
        inp_var = dataset.inp_var
        wind_mean = dataset.wind_mean
        wind_var = dataset.wind_var

        prd_fields = (dataset.solver.spec2grid(ic) - inp_mean) / torch.sqrt(inp_var)
        prd_fields = prd_fields.unsqueeze(0)

        prd_winds = dataset.solver.getuv(ic[1:])
        prd_winds = (prd_winds - wind_mean) / torch.sqrt(wind_var)
        prd_winds = prd_winds.unsqueeze(0)

        uspec = ic.clone()

        prd_uv_grid = dataset.solver.getuv(ic[1:])
        outputs = {"fields": prd_fields[0].cpu(), "winds": prd_uv_grid.cpu()}
        torch.save(
            outputs, os.path.join(output_dir, f"prediction_{0:0{pad_width}d}.pt")
        )
        torch.save(outputs, os.path.join(output_dir, f"truth_{0:0{pad_width}d}.pt"))

        if save_plots:
            fig = plt.figure(figsize=(6, 5))
            pred_data = prd_fields[0, plot_channel].cpu().numpy()
            ax = fig.add_subplot(1, 1, 1)
            im = ax.imshow(pred_data, vmin=-10, vmax=10, cmap="twilight_shifted")
            ax.set_title("Initial Condition (t=0)", fontsize=12, fontweight="bold")
            ax.axis("off")
            fig.subplots_adjust(bottom=0.15)
            cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
            fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
            fname = f"comparison_{0:0{pad_width}d}.png"
            plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
            plt.close()

        if writer is not None:
            # Always log initial condition image for TB browsing
            pred_data = prd_fields[0, plot_channel].cpu().numpy()
            writer.add_image(
                f"fields/initial_ch{plot_channel}", pred_data, 0, dataformats="HW"
            )
            writer.flush()

        if spectral_analysis:
            pred_spectra = compute_energy_spectra(prd_fields, dataset.sht)
            truth_spectra = compute_energy_spectra(prd_fields, dataset.sht)
            plot_path = os.path.join(spectral_dir, f"spectra_{0:0{pad_width}d}.png")
            plot_energy_spectra(pred_spectra, truth_spectra, 0, plot_path, model_name)

            if writer is not None:
                fig = make_energy_spectra_figure(
                    pred_spectra, truth_spectra, 0, model_name=model_name
                )
                writer.add_figure("spectra", fig, global_step=0)
                plt.close(fig)
                writer.flush()

        for step in range(1, autoreg_steps + 1):
            prd_fields = model(prd_fields, prd_winds)

            prd_unnorm = prd_fields * torch.sqrt(inp_var) + inp_mean
            prd_spec = dataset.sht(prd_unnorm.squeeze(0))

            prd_uv_grid = dataset.solver.getuv(prd_spec[1:])

            prd_winds = (prd_uv_grid - wind_mean) / torch.sqrt(wind_var)
            prd_winds = prd_winds.unsqueeze(0)

            uspec = dataset.solver.timestep(uspec, nsteps)
            ref_grid = dataset.solver.spec2grid(uspec)
            ref_uv_grid = dataset.solver.getuv(uspec[1:])

            ref_fields = (ref_grid - inp_mean) / torch.sqrt(inp_var)
            ref_fields = ref_fields.unsqueeze(0)

            pred_outputs = {"fields": prd_fields[0].cpu(), "winds": prd_uv_grid.cpu()}
            truth_outputs = {"fields": ref_fields[0].cpu(), "winds": ref_uv_grid.cpu()}
            torch.save(
                pred_outputs,
                os.path.join(output_dir, f"prediction_{step:0{pad_width}d}.pt"),
            )
            torch.save(
                truth_outputs,
                os.path.join(output_dir, f"truth_{step:0{pad_width}d}.pt"),
            )

            step_metrics = {}
            for name, metric_fn in metrics_dict.items():
                val = metric_fn(prd_fields, ref_fields).item()
                metrics_data[name].append(val)
                step_metrics[name] = val

            loss_val = loss_fn(prd_fields, ref_fields).item()
            metrics_data["loss"].append(loss_val)
            step_metrics["loss"] = loss_val

            metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in step_metrics.items()])
            print(f"Step {step}: {metrics_str}")

            _log_step_scalars(writer, step, step_metrics)

            if save_plots:
                fig = plt.figure(figsize=(18, 5))

                pred_data = prd_fields[0, plot_channel].cpu().numpy()
                truth_data = ref_fields[0, plot_channel].cpu().numpy()
                error_data = pred_data - truth_data

                ax1 = fig.add_subplot(1, 3, 1)
                im1 = ax1.imshow(pred_data, vmin=-10, vmax=10, cmap="twilight_shifted")
                ax1.set_title(f"Prediction (t={step})", fontsize=12, fontweight="bold")
                ax1.axis("off")

                ax2 = fig.add_subplot(1, 3, 2)
                im2 = ax2.imshow(truth_data, vmin=-10, vmax=10, cmap="twilight_shifted")
                ax2.set_title(
                    f"Ground Truth (t={step})", fontsize=12, fontweight="bold"
                )
                ax2.axis("off")

                ax3 = fig.add_subplot(1, 3, 3)
                error_max = max(abs(error_data.min()), abs(error_data.max()))
                im3 = ax3.imshow(error_data, vmin=-error_max, vmax=error_max, cmap="RdBu_r")
                ax3.set_title(f"Error (t={step})", fontsize=12, fontweight="bold")
                ax3.axis("off")

                fig.subplots_adjust(bottom=0.15, wspace=0.3)
                cbar_ax1 = fig.add_axes([0.08, 0.08, 0.22, 0.03])
                fig.colorbar(im1, cax=cbar_ax1, orientation="horizontal")
                cbar_ax2 = fig.add_axes([0.39, 0.08, 0.22, 0.03])
                fig.colorbar(im2, cax=cbar_ax2, orientation="horizontal")
                cbar_ax3 = fig.add_axes([0.70, 0.08, 0.22, 0.03])
                fig.colorbar(im3, cax=cbar_ax3, orientation="horizontal")

                fname = f"comparison_{step:0{pad_width}d}.png"
                plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
                plt.close()

            if writer is not None:
                # Log images for TensorBoard
                pred_data = prd_fields[0, plot_channel].cpu().numpy()
                truth_data = ref_fields[0, plot_channel].cpu().numpy()
                error_data = pred_data - truth_data
                _log_field_triplet(writer, step, pred_data, truth_data, error_data, plot_channel)
                writer.flush()

            if spectral_analysis:
                pred_spectra = compute_energy_spectra(prd_fields, dataset.sht)
                truth_spectra = compute_energy_spectra(ref_fields, dataset.sht)
                plot_path = os.path.join(spectral_dir, f"spectra_{step:0{pad_width}d}.png")
                plot_energy_spectra(pred_spectra, truth_spectra, step, plot_path, model_name)

                if writer is not None:
                    fig = make_energy_spectra_figure(
                        pred_spectra, truth_spectra, step, model_name=model_name
                    )
                    writer.add_figure("spectra", fig, global_step=step)
                    plt.close(fig)
                    writer.flush()

    summary = {}
    for key, values in metrics_data.items():
        if len(values) > 0:
            summary[f"{key}_mean"] = np.mean(values)
            summary[f"{key}_std"] = np.std(values)
        else:
            summary[f"{key}_mean"] = float("nan")
            summary[f"{key}_std"] = float("nan")

    return summary


def autoregressive_inference(
    model,
    dataset,
    loss_fn,
    metrics_dict,
    output_dir,
    nsteps,
    model_name="Model",
    autoreg_steps=10,
    plot_channel=0,
    save_plots=True,
    spectral_analysis=True,
    device=torch.device("cpu"),
    writer=None,
    ic_spec=None
):
    """Perform autoregressive inference for standard models (SFNO, Transformer)."""
    model.eval()
    model.to(device)

    dataset.solver = dataset.solver.to(device)
    dataset.sht = dataset.sht.to(device)
    dataset.inp_mean = dataset.inp_mean.to(device)
    dataset.inp_var = dataset.inp_var.to(device)

    os.makedirs(output_dir, exist_ok=True)
    if spectral_analysis:
        spectral_dir = output_dir
        os.makedirs(spectral_dir, exist_ok=True)

    metrics_data = {key: [] for key in metrics_dict.keys()}
    metrics_data["loss"] = []

    pad_width = len(str(autoreg_steps))

    print(f"Starting Autoregressive Inference for {autoreg_steps} steps...")

    with torch.no_grad():
        #ic = dataset.solver.random_initial_condition(mach=0.2)
        ic = ic_spec if ic_spec is not None else dataset.solver.random_initial_condition(mach=0.2)
        inp_mean = dataset.inp_mean
        inp_var = dataset.inp_var

        prd = (dataset.solver.spec2grid(ic) - inp_mean) / torch.sqrt(inp_var)
        prd = prd.unsqueeze(0)

        uspec = ic.clone()

        prd_uv_grid = dataset.solver.getuv(ic[1:])

        outputs = {"fields": prd[0].cpu(), "winds": prd_uv_grid.cpu()}
        torch.save(
            outputs, os.path.join(output_dir, f"prediction_{0:0{pad_width}d}.pt")
        )
        torch.save(outputs, os.path.join(output_dir, f"truth_{0:0{pad_width}d}.pt"))

        if save_plots:
            fig = plt.figure(figsize=(6, 5))
            pred_data = prd[0, plot_channel].cpu().numpy()
            ax = fig.add_subplot(1, 1, 1)
            im = ax.imshow(pred_data, vmin=-10, vmax=10, cmap="twilight_shifted")
            ax.set_title("Initial Condition (t=0)", fontsize=12, fontweight="bold")
            ax.axis("off")
            fig.subplots_adjust(bottom=0.15)
            cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
            fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
            fname = f"comparison_{0:0{pad_width}d}.png"
            plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
            plt.close()

        if writer is not None:
            pred_data = prd[0, plot_channel].cpu().numpy()
            writer.add_image(
                f"fields/initial_ch{plot_channel}", pred_data, 0, dataformats="HW"
            )
            writer.flush()

        if spectral_analysis:
            pred_spectra = compute_energy_spectra(prd, dataset.sht)
            truth_spectra = compute_energy_spectra(prd, dataset.sht)
            plot_path = os.path.join(spectral_dir, f"spectra_{0:0{pad_width}d}.png")
            plot_energy_spectra(pred_spectra, truth_spectra, 0, plot_path, model_name)

            if writer is not None:
                fig = make_energy_spectra_figure(
                    pred_spectra, truth_spectra, 0, model_name=model_name
                )
                writer.add_figure("spectra", fig, global_step=0)
                plt.close(fig)
                writer.flush()

        for step in range(1, autoreg_steps + 1):
            prd = model(prd)

            prd_unnorm = prd * torch.sqrt(inp_var) + inp_mean
            prd_spec = dataset.sht(prd_unnorm.squeeze(0))
            prd_uv_grid = dataset.solver.getuv(prd_spec[1:])

            uspec = dataset.solver.timestep(uspec, nsteps)
            ref_grid = dataset.solver.spec2grid(uspec)

            ref_uv_grid = dataset.solver.getuv(uspec[1:])

            ref = (ref_grid - inp_mean) / torch.sqrt(inp_var)
            ref = ref.unsqueeze(0)

            pred_outputs = {"fields": prd[0].cpu(), "winds": prd_uv_grid.cpu()}
            truth_outputs = {"fields": ref[0].cpu(), "winds": ref_uv_grid.cpu()}
            torch.save(
                pred_outputs,
                os.path.join(output_dir, f"prediction_{step:0{pad_width}d}.pt"),
            )
            torch.save(
                truth_outputs,
                os.path.join(output_dir, f"truth_{step:0{pad_width}d}.pt"),
            )

            step_metrics = {}
            for name, metric_fn in metrics_dict.items():
                val = metric_fn(prd, ref).item()
                metrics_data[name].append(val)
                step_metrics[name] = val

            loss_val = loss_fn(prd, ref).item()
            metrics_data["loss"].append(loss_val)
            step_metrics["loss"] = loss_val

            metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in step_metrics.items()])
            print(f"Step {step}: {metrics_str}")

            _log_step_scalars(writer, step, step_metrics)

            if save_plots:
                fig = plt.figure(figsize=(18, 5))

                pred_data = prd[0, plot_channel].cpu().numpy()
                truth_data = ref[0, plot_channel].cpu().numpy()
                error_data = pred_data - truth_data

                ax1 = fig.add_subplot(1, 3, 1)
                im1 = ax1.imshow(pred_data, vmin=-10, vmax=10, cmap="twilight_shifted")
                ax1.set_title(f"Prediction (t={step})", fontsize=12, fontweight="bold")
                ax1.axis("off")

                ax2 = fig.add_subplot(1, 3, 2)
                im2 = ax2.imshow(truth_data, vmin=-10, vmax=10, cmap="twilight_shifted")
                ax2.set_title(
                    f"Ground Truth (t={step})", fontsize=12, fontweight="bold"
                )
                ax2.axis("off")

                ax3 = fig.add_subplot(1, 3, 3)
                error_max = max(abs(error_data.min()), abs(error_data.max()))
                im3 = ax3.imshow(error_data, vmin=-error_max, vmax=error_max, cmap="RdBu_r")
                ax3.set_title(f"Error (t={step})", fontsize=12, fontweight="bold")
                ax3.axis("off")

                fig.subplots_adjust(bottom=0.15, wspace=0.3)
                cbar_ax1 = fig.add_axes([0.08, 0.08, 0.22, 0.03])
                fig.colorbar(im1, cax=cbar_ax1, orientation="horizontal")
                cbar_ax2 = fig.add_axes([0.39, 0.08, 0.22, 0.03])
                fig.colorbar(im2, cax=cbar_ax2, orientation="horizontal")
                cbar_ax3 = fig.add_axes([0.70, 0.08, 0.22, 0.03])
                fig.colorbar(im3, cax=cbar_ax3, orientation="horizontal")

                fname = f"comparison_{step:0{pad_width}d}.png"
                plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
                plt.close()

            if writer is not None:
                pred_data = prd[0, plot_channel].cpu().numpy()
                truth_data = ref[0, plot_channel].cpu().numpy()
                error_data = pred_data - truth_data
                _log_field_triplet(writer, step, pred_data, truth_data, error_data, plot_channel)
                writer.flush()

            if spectral_analysis:
                pred_spectra = compute_energy_spectra(prd, dataset.sht)
                truth_spectra = compute_energy_spectra(ref, dataset.sht)
                plot_path = os.path.join(spectral_dir, f"spectra_{step:0{pad_width}d}.png")
                plot_energy_spectra(pred_spectra, truth_spectra, step, plot_path, model_name)

                if writer is not None:
                    fig = make_energy_spectra_figure(
                        pred_spectra, truth_spectra, step, model_name=model_name
                    )
                    writer.add_figure("spectra", fig, global_step=step)
                    plt.close(fig)
                    writer.flush()

    summary = {}
    for key, values in metrics_data.items():
        if len(values) > 0:
            summary[f"{key}_mean"] = np.mean(values)
            summary[f"{key}_std"] = np.std(values)
        else:
            summary[f"{key}_mean"] = float("nan")
            summary[f"{key}_std"] = float("nan")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Autoregressive inference for Shallow Water Equations"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--autoreg_steps", type=int, default=10, help="Number of autoregressive steps"
    )
    parser.add_argument(
        "--plot_channel",
        type=int,
        default=0,
        help="Channel to plot (0=Geopotential, 1=Vorticity, 2=Divergence)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda/cpu)"
    )
    parser.add_argument("--no_plots", action="store_true", help="Disable saving plots")
    parser.add_argument(
        "--spectral_analysis",
        action="store_true",
        default=True,
        help="Perform spectral analysis",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--williamson_case2", action="store_true",
                    help="Use Williamson Case 2 steady geostrophic initial condition.")
    parser.add_argument("--williamson_case6", action="store_true",
                    help="Use Williamson Case 6 (Rossby–Haurwitz) initial condition.") 
    parser.add_argument("--flip_vort", action="store_true",
                    help="Flip sign of vorticity in Williamson Case 2 IC (if needed).")
    # stats sampling
    parser.add_argument("--stats_nsamples", type=int, default=200)

    # gaussian-bells defaults (match GaussianBellsPhiWrapperWithWinds)
    parser.add_argument("--gbells_mach", type=float, default=0.2)

    parser.add_argument("--gbells_k_min", type=int, default=1)
    parser.add_argument("--gbells_k_max", type=int, default=8)

    parser.add_argument("--gbells_sigma_min_deg", type=float, default=5.0)
    parser.add_argument("--gbells_sigma_max_deg", type=float, default=20.0)

    # IMPORTANT: training uses signed=True always.
    # Use a flag that can DISABLE signing, but default is signed=True.
    parser.add_argument("--gbells_unsigned", action="store_true",
                        help="If set, do NOT randomize signs (training default is signed).")
    # TensorBoard logdir (optional)
    parser.add_argument(
        "--tb_logdir",
        type=str,
        default=None,
        help="If set, write TensorBoard scalars + figures here (e.g. logs/run1).",
    )

    # Added: override dt_solver from CLI
    parser.add_argument(
        "--dt_solver",
        type=int,
        default=None,
        help="Override config['data']['dt_solver'] for forecast run.",
    )
    #parser.add_argument(
    #"--gbells",
    #action="store_true",
    #help="If set, use Gaussian-bells geopotential initial condition (phi channel) like training.",
#)
    #parser.add_argument("--gbells_k_min", type=int, default=1)
    #parser.add_argument("--gbells_k_max", type=int, default=8)
    #parser.add_argument("--gbells_sigma_min_deg", type=float, default=5.0)
    #parser.add_argument("--gbells_sigma_max_deg", type=float, default=20.0)
    #parser.add_argument("--gbells_signed", action="store_true", default=True)

    #args = parser.parse_args()
    args, unknown_args = parser.parse_known_args()



    pl.seed_everything(args.seed)
    #gbells_cfg = dict(
        #k_min=args.gbells_k_min,
        #k_max=args.gbells_k_max,
        #sigma_min_deg=args.gbells_sigma_min_deg,
        #sigma_max_deg=args.gbells_sigma_max_deg,
        #signed=args.gbells_signed,
    #)

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = load_config(args.config)
    config = update_config_from_args(config, unknown_args)

    # Apply override if provided
    if args.dt_solver is not None:
        config["data"]["dt_solver"] = int(args.dt_solver)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    writer = SummaryWriter(log_dir=args.tb_logdir) if args.tb_logdir else None

    model_type = config["experiment"]["model_type"]
    if model_type == "sfno":
        model_name = "SFNO"
    elif model_type == "transformer":
        model_name = "Spherical Transformer"
    elif model_type == "paradis":
        model_name = "PARADIS"
    else:
        model_name = model_type.upper()

    print("=" * 70)
    print("FORECAST CONFIGURATION")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Autoregressive steps: {args.autoreg_steps}")
    print(f"Output directory: {args.output_dir}")
    print(f"Save plots: {not args.no_plots}")
    print(f"Spectral analysis: {args.spectral_analysis}")
    print(f"TensorBoard logdir: {args.tb_logdir if args.tb_logdir else '(disabled)'}")
    print("=" * 70 + "\n")

    print(f"Loading checkpoint: {args.checkpoint}")
    model_module = SWELightningModule(config)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["state_dict"]

    keys_to_ignore = ["metric_w11.k_phi_mesh", "metric_w11.k_theta_mesh"]
    for key in keys_to_ignore:
        if key in state_dict:
            print(f"Removing buffer: {key}")
            del state_dict[key]

    model_module.load_state_dict(state_dict, strict=False)
    model_module.eval()
    print("Checkpoint loaded successfully.\n")

    print("Setting up Shallow Water Solver...")
    dt = config["data"]["dt"]
    dt_solver = config["data"]["dt_solver"]
    nsteps = dt // dt_solver

    use_winds = model_type == "paradis"

    if use_winds:
        dataset = PdeDatasetWithWinds(
            dt=dt,
            nsteps=nsteps,
            dims=(config["data"]["nlat"], config["data"]["nlon"]),
            grid=config["data"]["grid"],
            normalize=True,
            device=device,
        )
    else:
        dataset = PdeDataset(
            dt=dt,
            nsteps=nsteps,
            dims=(config["data"]["nlat"], config["data"]["nlon"]),
            grid=config["data"]["grid"],
            normalize=True,
            device=device,
        )
    





    # --- build gbells cfg from args ---
    #gbells_cfg = dict(
    #    k_min=args.gbells_k_min,
    #    k_max=args.gbells_k_max,
    #    sigma_min_deg=args.gbells_sigma_min_deg,
    #    sigma_max_deg=args.gbells_sigma_max_deg,
    #    signed=args.gbells_signed,
    #)


    gbells_cfg = dict(
        mach=args.gbells_mach,  # default 0.2
        k_min=args.gbells_k_min,
        k_max=args.gbells_k_max,
        sigma_min_deg=args.gbells_sigma_min_deg,
        sigma_max_deg=args.gbells_sigma_max_deg,
        signed=(not args.gbells_unsigned),  # default True, matches training
    )


    dataset.solver = dataset.solver.to(device)
    solver = dataset.solver

    ns = int(args.stats_nsamples)
    base_seed = int(args.seed)

    # R: random IC stats (fields + winds)
    R_field_mean, R_field_var, R_wind_mean, R_wind_var = compute_stats_for_ic_distribution(
        solver=solver,
        make_ic_spec_fn=lambda i: solver.random_initial_condition(mach=gbells_cfg["mach"]),      #default 0.2
        num_samples=ns,
        device=device,
    )

    # G: gaussian-bells IC stats (fields + winds)
    #G_field_mean, G_field_var, G_wind_mean, G_wind_var = compute_stats_for_ic_distribution(
    #    solver=solver,
    #    make_ic_spec_fn=lambda i: make_gaussian_bells_ic_spec(
    #        solver=solver,
    #        seed=args.seed + 1000 + i,   # different each sample
    #        gbells_cfg=gbells_cfg,
    #        mach=0.2,
    #    ),
    #    num_samples=ns,
    #    device=device,
    #)
    


    G_field_mean, G_field_var, G_wind_mean, G_wind_var = compute_stats_for_ic_distribution(
        solver=solver,
        make_ic_spec_fn=lambda i: make_gaussian_bells_ic_spec_training_exact(
            solver=solver,
            idx=i,
            seed=base_seed,          # matches training train_dataset wrapper
            mach=gbells_cfg["mach"], # default 0.2
            k_min=gbells_cfg["k_min"],
            k_max=gbells_cfg["k_max"],
            sigma_min_deg=gbells_cfg["sigma_min_deg"],
            sigma_max_deg=gbells_cfg["sigma_max_deg"],
            signed=gbells_cfg["signed"],  # default True
        ),
        num_samples=ns,
        device=device,
    )




    # S: steady-state stats (deterministic => sampling doesn't change it, but we keep same API)
    # If you want *exactly* the Williamson IC every time:
    #S_ic = make_williamson_case2_ic_spec_from_winds(
    #    solver,
    #    gh0=29400.0,
    #    flip_vort=args.flip_vort,
    #)
    if args.williamson_case6:
        S_ic = make_williamson_case6_ic_spec_from_winds(
            solver,
            R=4, omega=7.848e-6, K=None, h0=8000.0,
            flip_vort=args.flip_vort,
        )
    else:
        S_ic = make_williamson_case2_ic_spec_from_winds(
            solver,
            gh0=29400.0,
            flip_vort=args.flip_vort,
        )
    S_field_mean, S_field_var, S_wind_mean, S_wind_var = compute_stats_for_ic_distribution(
        solver=solver,
        make_ic_spec_fn=lambda i: S_ic,
        num_samples=1,  # 1 is enough because deterministic
        device=device,
    )

    # Convert var -> std
    R_field_std = torch.sqrt(R_field_var)
    G_field_std = torch.sqrt(G_field_var)
    S_field_std = torch.sqrt(S_field_var)

    R_wind_std = torch.sqrt(R_wind_var)
    G_wind_std = torch.sqrt(G_wind_var)
    S_wind_std = torch.sqrt(S_wind_var)

    # --- Your training-matching transform for steady-state normalization ---
    # mean*: Smean - Sstd*(Gmean - Rmean)/Gstd
    # std*:  Rstd*Sstd/Gstd
    field_mean_star = S_field_mean - S_field_std * (G_field_mean - R_field_mean) / (G_field_std + 1e-12)
    field_std_star  = R_field_std * S_field_std / (G_field_std + 1e-12)
    field_var_star  = torch.clamp(field_std_star**2, min=1e-8)

    wind_mean_star = S_wind_mean - S_wind_std * (G_wind_mean - R_wind_mean) / (G_wind_std + 1e-12)
    wind_std_star  = R_wind_std * S_wind_std / (G_wind_std + 1e-12)
    wind_var_star  = torch.clamp(wind_std_star**2, min=1e-8)

    print("\n=== STATS SUMMARY (per-channel mean/std) ===")
    for name, m, s in [
        ("R fields", R_field_mean, R_field_std),
        ("G fields", G_field_mean, G_field_std),
        ("S fields", S_field_mean, S_field_std),
        ("* fields", field_mean_star, field_std_star),
    ]:
        print(name, "mean:", m.view(-1).tolist(), "std:", s.view(-1).tolist())

    for name, m, s in [
        ("R winds", R_wind_mean, R_wind_std),
        ("G winds", G_wind_mean, G_wind_std),
        ("S winds", S_wind_mean, S_wind_std),
        ("* winds", wind_mean_star, wind_std_star),
    ]:
        print(name, "mean:", m.view(-1).tolist(), "std:", s.view(-1).tolist())

    # --- IMPORTANT: override dataset stats USED BY forecast normalization ---
    #dataset.inp_mean = field_mean_star.to(device)
    #dataset.inp_var  = field_var_star.to(device)

    #if hasattr(dataset, "wind_mean"):
    #    dataset.wind_mean = wind_mean_star.to(device)
    #    dataset.wind_var  = wind_var_star.to(device)
    
    
    
    #--------------  BEFORE ------------------------------------------------
    
    dataset.inp_mean = S_field_mean.to(device)
    dataset.inp_var  = torch.clamp(S_field_var.to(device), min=1e-8)

    if hasattr(dataset, "wind_mean"):
        dataset.wind_mean = S_wind_mean.to(device)
        dataset.wind_var  = torch.clamp(S_wind_var.to(device), min=1e-8)



    #-------------------          AFTER            --------------------------


    #dataset.inp_mean = S_field_mean.to(device)
    #S_field_std = torch.sqrt(S_field_var.to(device))
    #S_field_std = torch.clamp(S_field_std, min=1.0)   # <-- CHANGE THIS FLOOR if you want
    #dataset.inp_var = S_field_std**2
    
    #if hasattr(dataset, "wind_mean"):
    #    dataset.wind_mean = S_wind_mean.to(device)
    #    S_wind_std = torch.sqrt(S_wind_var.to(device))
    #    S_wind_std = torch.clamp(S_wind_std, min=1.0)  # <-- CHANGE THIS FLOOR if you want
    #    dataset.wind_var = S_wind_std**2


    dataset.sht = dataset.solver.sht
    metrics_dict = {
        "L1_error": model_module.metric_l1,
        "L2_error": model_module.metric_l2,
        "W11_error": model_module.metric_w11,
    }
    ic_spec = None
    #if args.williamson_case2:
    #    ic_spec = make_williamson_case2_ic_spec(dataset.solver, flip_vort=args.flip_vort)
    if args.williamson_case2:
        ic_spec = make_williamson_case2_ic_spec_from_winds(
            dataset.solver,
            gh0=29400.0,
            flip_vort=args.flip_vort,
        )
    if args.williamson_case6:
        ic_spec = make_williamson_case6_ic_spec_from_winds(
            dataset.solver,
            R=4,
            omega=7.848e-6,
            K=None,      # None -> K=omega
            h0=8000.0,
            flip_vort=args.flip_vort,
        )
    grid0 = dataset.solver.spec2grid(ic_spec)       # [phi, vort, div] on grid
    uv0   = dataset.solver.getuv(ic_spec[1:])       # winds recovered from vort/div

    print("phi mean/std:", grid0[0].mean().item(), grid0[0].std().item())
    print("vort mean/std:", grid0[1].mean().item(), grid0[1].std().item())
    print("div mean/std:", grid0[2].mean().item(), grid0[2].std().item())
    print("u mean/std:", uv0[0].mean().item(), uv0[0].std().item())
    print("v mean/std:", uv0[1].mean().item(), uv0[1].std().item())
    if use_winds:
        results = autoregressive_inference_with_winds(
            model=model_module,
            dataset=dataset,
            loss_fn=model_module.loss_fn,
            metrics_dict=metrics_dict,
            output_dir=args.output_dir,
            nsteps=nsteps,
            model_name=model_name,
            autoreg_steps=args.autoreg_steps,
            plot_channel=args.plot_channel,
            save_plots=(not args.no_plots),
            spectral_analysis=args.spectral_analysis,
            device=device,
            writer=writer,
            ic_spec=ic_spec
            #use_gbells=args.gbells,
            #gbells_cfg=gbells_cfg,
            #seed=args.seed,
        )
    else:
        results = autoregressive_inference(
            model=model_module,
            dataset=dataset,
            loss_fn=model_module.loss_fn,
            metrics_dict=metrics_dict,
            output_dir=args.output_dir,
            nsteps=nsteps,
            model_name=model_name,
            autoreg_steps=args.autoreg_steps,
            plot_channel=args.plot_channel,
            save_plots=(not args.no_plots),
            spectral_analysis=args.spectral_analysis,
            device=device,
            writer=writer,
            ic_spec=ic_spec
            #use_gbells=args.gbells,
            #gbells_cfg=gbells_cfg,
            #seed=args.seed,
        )

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    for key in ["loss", "L1_error", "L2_error", "W11_error"]:
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        if mean_key in results:
            print(f"{key:12s}: {results[mean_key]:.6f} ± {results[std_key]:.6f}")

    if writer is not None:
        for key in ["loss", "L1_error", "L2_error", "W11_error"]:
            mean_key = f"{key}_mean"
            std_key = f"{key}_std"
            if mean_key in results:
                writer.add_scalar(f"summary/{key}_mean", float(results[mean_key]), 0)
                writer.add_scalar(f"summary/{key}_std", float(results[std_key]), 0)

    df = pd.DataFrame([results])
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    df.to_csv(metrics_path, index=False)

    print("=" * 70)
    print(f"Metrics saved to: {metrics_path}")
    if not args.no_plots:
        print(f"Plots saved to: {args.output_dir}")
    if args.spectral_analysis:
        print(
            f"Spectral analysis saved to: {os.path.join(args.output_dir, 'spectral')}"
        )
    print("=" * 70)

    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()

