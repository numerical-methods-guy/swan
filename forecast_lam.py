#!/usr/bin/env python3
"""
forecast_lam.py

Autoregressive inference for the LAM PARADIS model trained on paired LR/HR
shallow water equation data.

Differences from forecast.py
-----------------------------
- Loads LAMLightningModule (not SWELightningModule) and build_lam_model.
- Reads IC data directly from the HDF5 dataset produced by generate_dataset.py
  rather than running a live solver — this guarantees the forecast starts from
  the same physically consistent (LR, HR) pairs used during training.
- The model forward pass takes (lr_halo, hr_patch_t0) and predicts hr_patch_t1.
- Rollout is patch-based: the HR domain is tiled into non-overlapping patches,
  each advanced one dt step independently, then stitched back into a full HR field.
- The reference "ground truth" at each step is taken from the stored HR trajectory
  in the HDF5 file.
- LR forcing is time-varying: at each autoregressive step the model is given the
  matching stored LR fields and winds for that lead time.
- Autoregressive HR fields are stitched into a full HR state after each step, and
  winds are reconstructed from predicted HR vorticity/divergence before the next step.
- Metrics are reported in physical (un-normalised) space to match forecast.py.
- Spectral energy analysis is performed on the full stitched HR field using
  a 2-D FFT (flat-earth approximation) rather than the SHT used in the global
  forecast.py, since the LAM domain is a limited-area Cartesian patch.

Usage
-----
python forecast_lam.py \
    --config  config_paradis_lam.yaml \
    --checkpoint  logs/.../checkpoints/best.ckpt \
    --h5_path  data/swe_paired.h5 \
    --output_dir  results_lam/ \
    --num_ics  5 \
    --autoreg_steps  10 \
    --split  val
"""

import argparse
import os
import time
from math import ceil

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.gridspec import GridSpec

from lam_lightning import LAMLightningModule
from lam_model import build_lam_model
from lam_patch_dataset import LAMPatchDataset, build_patch_manifest
from shallow_water_solver import ShallowWaterSolver

# ---------------------------------------------------------------------------
# Config / checkpoint helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_checkpoint(cfg: dict, ckpt_path: str, device: torch.device) -> LAMLightningModule:
    """Load LAMLightningModule weights from a PL checkpoint."""
    lit = LAMLightningModule(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    # strip "model." prefix if saved by PL's ModelCheckpoint
    state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()
             if k.startswith("model.")}
    missing, unexpected = lit.model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] Missing keys in checkpoint: {missing}")
    if unexpected:
        print(f"  [warn] Unexpected keys in checkpoint: {unexpected}")
    lit.eval()
    lit.to(device)
    return lit

def _make_hr_solver(hf, device: torch.device) -> ShallowWaterSolver:
    hr_nlat = int(hf.attrs["hr_nlat"])
    hr_nlon = int(hf.attrs["hr_nlon"])
    dt_solver = float(hf.attrs["dt_solver"])
    lmax = ceil(hr_nlat / 3)
    return (
        ShallowWaterSolver(
            hr_nlat,
            hr_nlon,
            dt_solver,
            lmax=lmax,
            mmax=lmax,
            grid="equiangular",
        )
        .to(device)
        .float()
    )

# ---------------------------------------------------------------------------
# Patch tiling / stitching
# ---------------------------------------------------------------------------

def _tile_patches(
    hr_field: torch.Tensor,
    lr_field: torch.Tensor,
    patch_nlat_lr: int,
    patch_nlon_lr: int,
    halo_radius: int,
    s: int,
) -> list:
    """Decompose one HR frame + its LR companion into a list of patch dicts.

    Parameters
    ----------
    hr_field : [C_hr, hr_nlat, hr_nlon]
    lr_field : [C_lr, lr_nlat, lr_nlon]  — LR fields + winds concatenated on C axis
    patch_nlat_lr, patch_nlon_lr : interior patch size in LR cells
    halo_radius : halo thickness in LR cells
    s : upscale factor

    Returns
    -------
    List of dicts with keys:
        lat0_lr, lon0_lr   — top-left of interior patch in LR coords
        hr_patch           — [C_hr, patch_nlat_hr, patch_nlon_hr]
        lr_halo            — [C_lr, win_nlat, win_nlon]
    """
    lr_nlat = lr_field.shape[-2]
    lr_nlon = lr_field.shape[-1]
    R = halo_radius
    win_nlat = patch_nlat_lr + 2 * R
    win_nlon  = patch_nlon_lr + 2 * R
    patch_nlat_hr = patch_nlat_lr * s
    patch_nlon_hr = patch_nlon_lr * s

    patches = []
    for lat0 in range(0, lr_nlat - patch_nlat_lr + 1, patch_nlat_lr):
        for lon0 in range(0, lr_nlon, patch_nlon_lr):
            # HR interior
            hr_lat0 = lat0 * s
            hr_lon0 = lon0 * s
            hr_p = _cyclic_crop(hr_field, hr_lat0, hr_lon0, patch_nlat_hr, patch_nlon_hr)

            # LR halo window
            win_lat0 = lat0 - R
            win_lon0 = lon0 - R
            # clamp lat (no polar wrap)
            if win_lat0 < 0 or win_lat0 + win_nlat > lr_nlat:
                continue  # skip patches too close to poles
            lr_h = _cyclic_crop(lr_field, win_lat0, win_lon0, win_nlat, win_nlon)

            patches.append({
                "lat0_lr": lat0,
                "lon0_lr": lon0,
                "hr_patch": hr_p,
                "lr_halo": lr_h,
            })
    return patches


def _cyclic_crop(
    tensor: torch.Tensor,
    lat0: int,
    lon0: int,
    nlat: int,
    nlon: int,
) -> torch.Tensor:
    """Crop [..., nlat, nlon] window with cyclic longitude."""
    nlon_total = tensor.shape[-1]
    lon0 = lon0 % nlon_total
    lat_slice = tensor[..., lat0 : lat0 + nlat, :]
    if lon0 + nlon <= nlon_total:
        return lat_slice[..., lon0 : lon0 + nlon].clone()
    part1 = lat_slice[..., lon0:]
    part2 = lat_slice[..., : lon0 + nlon - nlon_total]
    return torch.cat([part1, part2], dim=-1)


def _stitch_patches(
    patches: list,
    hr_nlat: int,
    hr_nlon: int,
    patch_nlat_lr: int,
    patch_nlon_lr: int,
    s: int,
    C: int,
    base=None,
) -> torch.Tensor:
    """Stitch predicted HR patch tensors back into a full HR field."""
    out = base.clone() if base is not None else torch.zeros(C, hr_nlat, hr_nlon)
    patch_nlat_hr = patch_nlat_lr * s
    patch_nlon_hr = patch_nlon_lr * s

    for p in patches:
        hr_lat0 = p["lat0_lr"] * s
        hr_lon0 = (p["lon0_lr"] * s) % hr_nlon
        pred = p["hr_patch_pred"]  # [C, patch_nlat_hr, patch_nlon_hr]

        if hr_lon0 + patch_nlon_hr <= hr_nlon:
            out[:, hr_lat0: hr_lat0 + patch_nlat_hr, hr_lon0: hr_lon0 + patch_nlon_hr] = pred
        else:
            split = hr_nlon - hr_lon0
            out[:, hr_lat0: hr_lat0 + patch_nlat_hr, hr_lon0:] = pred[:, :, :split]
            out[:, hr_lat0: hr_lat0 + patch_nlat_hr, : patch_nlon_hr - split] = pred[:, :, split:]

    return out
# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm_lr(tensor, f_mean, f_var, w_mean, w_var):
    """Normalise concatenated LR [fields(3) + winds(2)] tensor."""
    f = (tensor[:3] - f_mean) / f_var.sqrt()
    w = (tensor[3:] - w_mean) / w_var.sqrt()
    return torch.cat([f, w], dim=0)


def _norm_hr(tensor, mean, var):
    return (tensor - mean) / var.sqrt()


def _denorm_hr(tensor, mean, var):
    return tensor * var.sqrt() + mean

# ---------------------------------------------------------------------------
# Spectral analysis (2-D FFT, flat-earth approximation)
# ---------------------------------------------------------------------------

def compute_energy_spectra_fft(fields: torch.Tensor) -> dict:
    """2-D isotropic power spectrum via FFT.

    Parameters
    ----------
    fields : [C, nlat, nlon]  — un-normalised physical fields

    Returns
    -------
    dict with keys rotational, divergent, potential, total, wavenumbers
    """
    nlat, nlon = fields.shape[-2], fields.shape[-1]
    max_k = min(nlat, nlon) // 2

    def _spectrum(ch):
        f = fields[ch].float()
        F = torch.fft.rfft2(f)
        power = (F.real ** 2 + F.imag ** 2)
        ki = torch.arange(nlat, device=fields.device).reshape(-1, 1).float()
        kj = torch.arange(F.shape[-1], device=fields.device).reshape(1, -1).float()
        k = torch.sqrt(ki ** 2 + kj ** 2).long().clamp(0, max_k - 1)
        spec = torch.zeros(max_k, device=fields.device)
        spec.scatter_add_(0, k.flatten(), power.flatten())
        return spec.cpu().numpy()

    rot = _spectrum(1)   # vorticity ~ rotational KE
    div = _spectrum(2)   # divergence ~ divergent KE
    pot = _spectrum(0)   # geopotential ~ potential energy

    return {
        "rotational": rot,
        "divergent":  div,
        "potential":  pot,
        "total":      rot + div + pot,
        "wavenumbers": np.arange(max_k),
    }


def plot_energy_spectra(pred_spectra, truth_spectra, step, output_path):
    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    k_plot = pred_spectra["wavenumbers"][1:]
    k_ref  = np.array([5.0, k_plot[-1] * 0.5])

    for idx, (title, key) in enumerate([
        ("Rotational KE",  "rotational"),
        ("Divergent KE",   "divergent"),
        ("Potential E",    "potential"),
        ("Total Energy",   "total"),
    ]):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.loglog(k_plot, pred_spectra[key][1:],  "b-",  lw=2, label="LAM Prediction", alpha=0.85)
        ax.loglog(k_plot, truth_spectra[key][1:], "r--", lw=2, label="HR Truth",       alpha=0.85)
        ax.set_xlabel("Wavenumber", fontsize=11)
        ax.set_ylabel("Power Spectrum", fontsize=11)
        ax.set_title(f"{title} (step={step})", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=8)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------------------
# Field comparison plots
# ---------------------------------------------------------------------------

CHANNEL_NAMES = ["Geopotential h", "Vorticity", "Divergence"]


def plot_comparison(pred, truth, step, output_path,
                    lat_top=None, lat_bot=None, lon_left=None, lon_right=None):

    CHANNEL_NAMES = ["Geopotential h", "Vorticity ζ", "Divergence δ"]
    CMAPS = ["viridis", "RdBu_r", "RdBu_r"]
    col_titles = ["LAM Prediction", "HR Truth", "Error (Pred − Truth)"]  # was 4, now 3

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))  # was (3, 4)

    for ch in range(3):
        p_np = pred[ch].cpu().numpy()
        t_np = truth[ch].cpu().numpy()
        e_np = p_np - t_np
        vmin, vmax = t_np.min(), t_np.max()
        emax = max(abs(e_np.min()), abs(e_np.max())) + 1e-8

        imgs  = [p_np, t_np, e_np]               # removed l_np
        cmaps = [CMAPS[ch], CMAPS[ch], "RdBu_r"]
        vmins = [vmin, vmin, -emax]
        vmaxs = [vmax, vmax,  emax]

        for col, (img, cm, vn, vx) in enumerate(zip(imgs, cmaps, vmins, vmaxs)):
            ax = axes[ch, col]
            im = ax.imshow(img, cmap=cm, vmin=vn, vmax=vx, origin="upper")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis("off")
            if ch == 0:
                ax.set_title(col_titles[col], fontsize=12, fontweight="bold", pad=8)
            if col == 0:
                ax.set_ylabel(CHANNEL_NAMES[ch], fontsize=11, labelpad=6)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])

    if all(v is not None for v in [lat_top, lat_bot, lon_left, lon_right]):
        geo = f"Patch: {lat_bot:.1f}°–{lat_top:.1f}°N, {lon_left:.1f}°–{lon_right:.1f}°E"
    else:
        geo = ""
    fig.suptitle(f"LAM Forecast — Step {step}  {geo}",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_error_profiles(pred, truth, step, output_path, cfg):
    """
    3x2 figure:
    One row per SWE channel.
    Left column:  RMSE vs latitude
    Right column: RMSE vs longitude

    pred, truth: [3, hr_nlat, hr_nlon] in physical units
    """
    channel_names = ["Geopotential h", "Vorticity ζ", "Divergence δ"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    err = (pred - truth).pow(2)          # [3, hr_nlat, hr_nlon]
    rmse_lat = err.mean(dim=2).sqrt()    # [3, hr_nlat]
    rmse_lon = err.mean(dim=1).sqrt()    # [3, hr_nlon]

    hr_nlat = pred.shape[1]
    hr_nlon = pred.shape[2]
    lats = np.linspace(90, -90, hr_nlat)
    lons = np.linspace(0, 360, hr_nlon, endpoint=False)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    for ch in range(3):
        ax_lat = axes[ch, 0]
        ax_lon = axes[ch, 1]

        ax_lat.plot(
            lats,
            rmse_lat[ch].cpu().numpy(),
            color=colors[ch],
            lw=1.8,
        )
        ax_lat.set_xlabel("Latitude (°N)", fontsize=11)
        ax_lat.set_ylabel("RMSE (physical units)", fontsize=11)
        ax_lat.set_title(
            f"{channel_names[ch]} — RMSE vs Latitude (Step {step})",
            fontsize=12,
            fontweight="bold",
        )
        ax_lat.grid(True, alpha=0.3)
        ax_lat.invert_xaxis()

        ax_lon.plot(
            lons,
            rmse_lon[ch].cpu().numpy(),
            color=colors[ch],
            lw=1.8,
        )
        ax_lon.set_xlabel("Longitude (°E)", fontsize=11)
        ax_lon.set_ylabel("RMSE (physical units)", fontsize=11)
        ax_lon.set_title(
            f"{channel_names[ch]} — RMSE vs Longitude (Step {step})",
            fontsize=12,
            fontweight="bold",
        )
        ax_lon.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
# ---------------------------------------------------------------------------
# Single-IC rollout
# ---------------------------------------------------------------------------

def _run_single_ic(
    model,
    ic_idx: int,
    hf,
    norm: dict,
    cfg: dict,
    output_dir: str,
    autoreg_steps: int,
    plot_channel: int,
    save_plots: bool,
    spectral_analysis: bool,
    device: torch.device,
    plot_lat0_lr: int = None,
    plot_lon0_lr: int = None,
) -> tuple:
    """Run one autoregressive rollout for a single IC using time-matched trajectory truth."""
    lamc = cfg["lam"]
    s = int(lamc["refinement_factor_lat"])
    R = int(lamc["halo_radius"])
    pL = int(lamc["patch_nlat_lr"])
    pN = int(lamc["patch_nlon_lr"])

    hr_nlat = int(hf.attrs["hr_nlat"])
    hr_nlon = int(hf.attrs["hr_nlon"])
    lr_nlat = int(hf.attrs["lr_nlat"])
    lr_nlon = int(hf.attrs["lr_nlon"])

    assert "fields" in hf["lr"] and "winds" in hf["lr"], \
        "HDF5 file must contain /lr/fields and /lr/winds"
    assert "fields" in hf["hr"] and "winds" in hf["hr"], \
        "HDF5 file must contain /hr/fields and /hr/winds"

    rollout_steps_avail = int(hf.attrs["rollout_steps"])
    assert autoreg_steps <= rollout_steps_avail, \
        f"Requested autoreg_steps={autoreg_steps}, but dataset stores only {rollout_steps_avail}"

    lr_fields_traj = torch.tensor(np.array(hf["lr/fields"][ic_idx]), dtype=torch.float32)
    lr_winds_traj = torch.tensor(np.array(hf["lr/winds"][ic_idx]), dtype=torch.float32)
    hr_fields_traj = torch.tensor(np.array(hf["hr/fields"][ic_idx]), dtype=torch.float32)
    hr_winds_traj = torch.tensor(np.array(hf["hr/winds"][ic_idx]), dtype=torch.float32)

    lr_t0 = lr_fields_traj[0]
    lr_w0 = lr_winds_traj[0]
    hr_t0 = hr_fields_traj[0]
    hr_w0 = hr_winds_traj[0]

    hr_f_mean_dev = norm["hr_f_mean"].to(device)
    hr_f_var_dev = norm["hr_f_var"].to(device)
    hr_w_mean_dev = norm["hr_w_mean"].to(device)
    hr_w_var_dev = norm["hr_w_var"].to(device)

    lr_combined_raw = torch.cat([lr_t0, lr_w0], dim=0)
    lr_combined = _norm_lr(
        lr_combined_raw,
        norm["lr_f_mean"], norm["lr_f_var"],
        norm["lr_w_mean"], norm["lr_w_var"],
    )

    hr_w0_norm = (hr_w0 - norm["hr_w_mean"]) / norm["hr_w_var"].sqrt()
    hr_current_norm = torch.cat([
        _norm_hr(hr_t0, norm["hr_f_mean"], norm["hr_f_var"]),
        hr_w0_norm,
    ], dim=0)  # [5, hr_nlat, hr_nlon]

    solver = _make_hr_solver(hf, device)

    pad_width = len(str(autoreg_steps))
    prefix = f"ic{ic_idx:03d}_"
    step_metrics = {
        "RMSE": [],
        "MAE": [],
        "bias": [],
        "RMSE_h": [],
        "RMSE_vort": [],
        "RMSE_div": [],
        "RMSE_u": [],
        "RMSE_v": [],
        "RMSE_wind": [],
    }

    # --- timing pass --------------------------------------------------------
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    hr_fields_norm_t = hr_current_norm[:3].to(device)
    hr_winds_norm_t = hr_current_norm[3:].to(device)

    with torch.no_grad():
        for step in range(1, autoreg_steps + 1):
            lr_fields_raw_t = lr_fields_traj[step - 1]
            lr_winds_raw_t = lr_winds_traj[step - 1]
            lr_combined_t = _norm_lr(
                torch.cat([lr_fields_raw_t, lr_winds_raw_t], dim=0),
                norm["lr_f_mean"], norm["lr_f_var"],
                norm["lr_w_mean"], norm["lr_w_var"],
            ).to(device)

            hr_norm_rollout = torch.cat([hr_fields_norm_t, hr_winds_norm_t], dim=0)
            patches = _tile_patches(hr_norm_rollout, lr_combined_t, pL, pN, R, s)

            for p in patches:
                lh = p["lr_halo"].unsqueeze(0)
                hp = p["hr_patch"].unsqueeze(0)  # [1, 5, patch_nlat_hr, patch_nlon_hr]
                p["hr_patch_pred"] = model.model(lh, hp).squeeze(0).cpu()

            hr_pred_norm = _stitch_patches(
                patches,
                hr_nlat,
                hr_nlon,
                pL,
                pN,
                s,
                3,
                base=hr_fields_norm_t.detach().cpu(),
            )
            hr_fields_norm_t = hr_pred_norm.to(device)

            hr_pred_phys_dev = _denorm_hr(hr_fields_norm_t, hr_f_mean_dev, hr_f_var_dev)
            pred_vrtdiv_spec = solver.grid2spec(hr_pred_phys_dev[1:3])
            pred_winds_phys_dev = solver.getuv(pred_vrtdiv_spec)
            hr_winds_norm_t = (pred_winds_phys_dev - hr_w_mean_dev) / hr_w_var_dev.sqrt()

    if device.type == "cuda":
        torch.cuda.synchronize()
    ml_time = time.perf_counter() - t_start

    # --- metrics pass -------------------------------------------------------
    hr_fields_norm = hr_current_norm[:3].to(device)
    hr_winds_norm = hr_current_norm[3:].to(device)

    margin = R * s

    if plot_lat0_lr is not None and plot_lon0_lr is not None:
        crop_lat0 = plot_lat0_lr * s
        crop_lon0 = plot_lon0_lr * s
        crop_nlat = pL * s
        crop_nlon = pN * s
    else:
        crop_lat0 = margin
        crop_lon0 = 0
        crop_nlat = hr_nlat - 2 * margin
        crop_nlon = hr_nlon

    def _crop(field):
        lat_slice = field[..., crop_lat0: crop_lat0 + crop_nlat, :]
        lon0 = crop_lon0 % hr_nlon
        if lon0 + crop_nlon <= hr_nlon:
            return lat_slice[..., lon0: lon0 + crop_nlon]
        slice1 = lat_slice[..., lon0:]
        slice2 = lat_slice[..., : crop_nlon - (hr_nlon - lon0)]
        return torch.cat([slice1, slice2], dim=-1)

    with torch.no_grad():
        for step in range(1, autoreg_steps + 1):
            lr_fields_raw_t = lr_fields_traj[step - 1]
            lr_winds_raw_t = lr_winds_traj[step - 1]
            lr_combined_t = _norm_lr(
                torch.cat([lr_fields_raw_t, lr_winds_raw_t], dim=0),
                norm["lr_f_mean"], norm["lr_f_var"],
                norm["lr_w_mean"], norm["lr_w_var"],
            ).to(device)

            hr_norm_rollout = torch.cat([hr_fields_norm, hr_winds_norm], dim=0)
            patches = _tile_patches(hr_norm_rollout, lr_combined_t, pL, pN, R, s)

            for p in patches:
                lh = p["lr_halo"].unsqueeze(0)
                hp = p["hr_patch"].unsqueeze(0)  # [1, 5, patch_nlat_hr, patch_nlon_hr]
                p["hr_patch_pred"] = model.model(lh, hp).squeeze(0).cpu()

            hr_pred_norm = _stitch_patches(
                patches,
                hr_nlat,
                hr_nlon,
                pL,
                pN,
                s,
                3,
                base=hr_fields_norm.detach().cpu(),
            )
            hr_fields_norm = hr_pred_norm.to(device)

            hr_pred_phys = _denorm_hr(hr_pred_norm, norm["hr_f_mean"], norm["hr_f_var"])
            truth_phys = hr_fields_traj[step]

            hr_pred_phys_dev = _denorm_hr(hr_fields_norm, hr_f_mean_dev, hr_f_var_dev)
            pred_vrtdiv_spec = solver.grid2spec(hr_pred_phys_dev[1:3])
            pred_winds_phys_dev = solver.getuv(pred_vrtdiv_spec)
            pred_winds_phys = pred_winds_phys_dev.cpu()
            truth_winds = hr_winds_traj[step]

            pred_crop = _crop(hr_pred_phys)
            truth_crop = _crop(truth_phys)
            wind_pred_crop = _crop(pred_winds_phys)
            wind_truth_crop = _crop(truth_winds)

            err = pred_crop - truth_crop
            werr = wind_pred_crop - wind_truth_crop

            step_metrics["RMSE"].append(err.pow(2).mean().sqrt().item())
            step_metrics["MAE"].append(err.abs().mean().item())
            step_metrics["bias"].append(err.mean().item())

            for ch_i, key in enumerate(["RMSE_h", "RMSE_vort", "RMSE_div"]):
                step_metrics[key].append(err[ch_i].pow(2).mean().sqrt().item())

            for ch_i, key in enumerate(["RMSE_u", "RMSE_v"]):
                step_metrics[key].append(werr[ch_i].pow(2).mean().sqrt().item())

            step_metrics["RMSE_wind"].append(werr.pow(2).mean().sqrt().item())

            hr_winds_norm = (pred_winds_phys_dev - hr_w_mean_dev) / hr_w_var_dev.sqrt()

            if output_dir is not None:
                if save_plots:
                    dlat_hr = 180.0 / hr_nlat
                    dlon_hr = 360.0 / hr_nlon
                    lat_top = 90.0 - crop_lat0 * dlat_hr
                    lat_bot = lat_top - crop_nlat * dlat_hr
                    lon_left = (crop_lon0 * dlon_hr) % 360.0
                    lon_right = lon_left + crop_nlon * dlon_hr

                    plot_comparison(
                        pred_crop,
                        truth_crop,
                        step,
                        os.path.join(output_dir, f"{prefix}comparison_{step:0{pad_width}d}.png"),
                        lat_top=lat_top,
                        lat_bot=lat_bot,
                        lon_left=lon_left,
                        lon_right=lon_right,
                    )

                    plot_error_profiles(
                        pred_crop,
                        truth_crop,
                        step,
                        os.path.join(output_dir, f"{prefix}error_profiles_{step:0{pad_width}d}.png"),
                        cfg,
                    )

                if spectral_analysis:
                    pred_spec = compute_energy_spectra_fft(pred_crop)
                    truth_spec = compute_energy_spectra_fft(truth_crop)
                    plot_energy_spectra(
                        pred_spec,
                        truth_spec,
                        step,
                        os.path.join(output_dir, f"{prefix}spectra_{step:0{pad_width}d}.png"),
                    )

    return step_metrics, ml_time

# ---------------------------------------------------------------------------
# Multi-IC orchestration
# ---------------------------------------------------------------------------

def run_inference(
    model,
    h5_path: str,
    norm: dict,
    cfg: dict,
    output_dir: str,
    ic_indices: list,
    autoreg_steps: int,
    plot_channel: int,
    save_plots: bool,
    spectral_analysis: bool,
    device: torch.device,
    plot_lat0_lr: int=None,
    plot_lon0_lr: int=None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_metrics = []
    all_times   = []

    with h5py.File(h5_path, "r") as hf:
        for i, ic_idx in enumerate(ic_indices):
            print(f"\n--- IC {i+1}/{len(ic_indices)}  (HDF5 row {ic_idx}) ---")
            metrics, ml_time = _run_single_ic(
                model        = model,
                ic_idx       = ic_idx,
                hf           = hf,
                norm         = norm,
                cfg          = cfg,
                output_dir   = output_dir if i == 0 else None,
                autoreg_steps= autoreg_steps,
                plot_channel = plot_channel,
                save_plots   = save_plots and i == 0,
                spectral_analysis = spectral_analysis and i == 0,
                device       = device,
                plot_lat0_lr = plot_lat0_lr,
                plot_lon0_lr = plot_lon0_lr,
            )
            all_metrics.append(metrics)
            all_times.append(ml_time)
            print(f"  ML rollout time : {ml_time:.3f}s")
            rmse_final = metrics["RMSE"][-1] if metrics["RMSE"] else float("nan")
            print(f"  RMSE (final step): {rmse_final:.6f}")

    # aggregate
    summary = {}
    for key in all_metrics[0]:
        flat = [v for m in all_metrics for v in m[key]]
        summary[f"{key}_mean"] = float(np.mean(flat))
        summary[f"{key}_std"]  = float(np.std(flat))
    summary["ml_time_mean"] = float(np.mean(all_times))
    summary["ml_time_std"]  = float(np.std(all_times))
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LAM PARADIS autoregressive inference")
    parser.add_argument("--config",     required=True,             help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True,             help="Path to .ckpt file")
    parser.add_argument("--h5_path",    required=True,             help="Path to swe_paired.h5")
    parser.add_argument("--output_dir", default="results_lam",     help="Directory to save results")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_ics",    type=int, default=5,       help="Number of ICs to evaluate")
    parser.add_argument("--autoreg_steps", type=int, default=1,   help="Autoregressive rollout steps")
    parser.add_argument("--plot_channel",  type=int, default=0,   help="0=h, 1=vorticity, 2=divergence")
    parser.add_argument("--device",    default=None)
    parser.add_argument("--no_plots",  action="store_true")
    parser.add_argument("--no_spectra",action="store_true")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--ic_indices", type=str, default=None,
                        help="Comma-separated HDF5 row indices to evaluate, e.g. '700,701,705'. "
                        "Overrides --split and --num_ics entirely.")
    parser.add_argument("--plot_lat0_lr", type=int, default=None,
                        help="Top-left latitude of patch for visualization/evaluation, measured in LR cells")
    parser.add_argument("--plot_lon0_lr", type=int, default=None,
                        help="Top-left longitude of patch for visualization/evaluation, measured in LR cells")
    args = parser.parse_args()

    import pytorch_lightning as pl
    pl.seed_everything(args.seed)

    cfg    = load_config(args.config)
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 70)
    print("LAM FORECAST CONFIGURATION")
    print("=" * 70)
    print(f"  Checkpoint    : {args.checkpoint}")
    print(f"  HDF5 dataset  : {args.h5_path}")
    print(f"  Split         : {args.split}")
    print(f"  Device        : {device}")
    print(f"  Autoreg steps : {args.autoreg_steps}")
    print(f"  Num ICs       : {args.num_ics}")
    print(f"  Output dir    : {args.output_dir}")
    print("=" * 70)

    # Load normalisation stats from HDF5
    with h5py.File(args.h5_path, "r") as hf:
        def _t(k):
            return torch.tensor(np.array(hf.attrs[k]), dtype=torch.float32)

        norm = {
            "lr_f_mean": _t("lr_inp_mean").reshape(3, 1, 1),
            "lr_f_var": _t("lr_inp_var").reshape(3, 1, 1),
            "lr_w_mean": _t("lr_wind_mean").reshape(2, 1, 1),
            "lr_w_var": _t("lr_wind_var").reshape(2, 1, 1),
            "hr_f_mean": _t("hr_inp_mean").reshape(3, 1, 1),
            "hr_f_var": _t("hr_inp_var").reshape(3, 1, 1),
            "hr_w_mean": _t("hr_wind_mean").reshape(2, 1, 1),
            "hr_w_var": _t("hr_wind_var").reshape(2, 1, 1),
        }

        total_ics = int(hf.attrs["num_ics"])
        num_train = int(cfg["data"]["num_train_examples"])
        num_val = int(cfg["data"]["num_val_examples"])

        if args.ic_indices is not None:
            ic_indices = [int(x.strip()) for x in args.ic_indices.split(",")]
            assert all(0 <= i < total_ics for i in ic_indices), \
                f"One or more ic_indices out of range [0, {total_ics})"
        else:
            if args.split == "train":
                ic_pool = list(range(0, num_train))
            elif args.split == "val":
                ic_pool = list(range(num_train, num_train + num_val))
            else:
                ic_pool = list(range(num_train + num_val, total_ics))

            assert len(ic_pool) > 0, f"No ICs for split='{args.split}'"
            ic_indices = ic_pool[:args.num_ics]

    # Load model
    print(f"\nLoading checkpoint …")
    model = load_checkpoint(cfg, args.checkpoint, device)
    print("Checkpoint loaded.\n")

    # Run inference
    summary = run_inference(
        model          = model,
        h5_path        = args.h5_path,
        norm           = norm,
        cfg            = cfg,
        output_dir     = args.output_dir,
        ic_indices     = ic_indices,
        autoreg_steps  = args.autoreg_steps,
        plot_channel   = args.plot_channel,
        save_plots     = not args.no_plots,
        spectral_analysis = not args.no_spectra,
        device         = device,
        plot_lat0_lr = args.plot_lat0_lr,
        plot_lon0_lr = args.plot_lon0_lr,
    )

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    for key in ["RMSE", "MAE", "bias"]:
        m, s = summary[f"{key}_mean"], summary[f"{key}_std"]
        print(f"  {key:8s}: {m:.6f} ± {s:.6f}")
    print(f"  ML time : {summary['ml_time_mean']:.3f}s ± {summary['ml_time_std']:.3f}s")
    print("=" * 70)

    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    pd.DataFrame([summary]).to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to : {metrics_path}")
    if not args.no_plots:
        print(f"Plots saved to   : {args.output_dir}")


if __name__ == "__main__":
    main()
