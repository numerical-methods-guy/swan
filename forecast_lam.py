#!/usr/bin/env python3
"""
forecast_lam.py

Autoregressive inference for the LAM PARADIS model trained on paired LR/HR
shallow water equation data.

Rollout implemented here
-----------------------------------
For each autoregressive step t -> t+1:

- HR input state at time t:
    * step 0 uses database HR truth at t=0
    * later steps use the previous stitched post-blend model output
- LR input state at time t:
    * always taken from the stored LR trajectory for that same time index
- HR output state at time t+1:
    * predicted patchwise
    * optionally blended with LR in the patch edge zone
    * stitched into a full HR field
    * carried forward as the next HR input state

The reference "truth" used for evaluation at each step is the stored HR field
at the target time t+1.

Usage
-----
python forecast_lam.py \
    --config config_paradis_lam.yaml \
    --checkpoint logs/.../checkpoints/best.ckpt \
    --h5_path data/swe_paired.h5 \
    --output_dir results_lam/ \
    --num_ics 5 \
    --autoreg_steps 10 \
    --split val
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

from lam_helpers.lam_lightning import LAMLightningModule
from lam_helpers.lam_patch_dataset import build_patch_plan_tensors
from swe_solver.shallow_water_solver import ShallowWaterSolver
from lam_helpers.lam_blending import build_lr_blend_weight_hr, blend_hr_prediction_with_lr


# ---------------------------------------------------------------------------
# Config / checkpoint helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_checkpoint(
    cfg: dict, ckpt_path: str, device: torch.device
) -> LAMLightningModule:
    """Load LAMLightningModule weights from a PyTorch Lightning checkpoint."""
    lit = LAMLightningModule(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)

    state = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }

    missing, unexpected = lit.model.load_state_dict(state, strict=False)
    if missing:
        print(f" [warn] Missing keys in checkpoint: {missing}")
    if unexpected:
        print(f" [warn] Unexpected keys in checkpoint: {unexpected}")

    lit.eval()
    lit.to(device)
    return lit


def _make_hr_solver(hf, device: torch.device) -> ShallowWaterSolver:
    hr_nlat = int(hf.attrs["hr_nlat"])
    hr_nlon = int(hf.attrs["hr_nlon"])
    dt_solver = float(hf.attrs["dt_solver"])
    lmax = ceil(hr_nlat / 3)

    solver = ShallowWaterSolver(
        hr_nlat,
        hr_nlon,
        dt_solver,
        lmax=lmax,
        mmax=lmax,
        grid="equiangular",
    )
    solver = solver.to(device).float()
    return solver


# ---------------------------------------------------------------------------
# Patch extraction / stitching
# ---------------------------------------------------------------------------

def _cyclic_crop_stack(
    tensor: torch.Tensor,
    lat0_list,
    lon0_list,
    nlat: int,
    nlon: int,
) -> torch.Tensor:
    """
    Stack many cyclic crops from one [C, H, W] tensor into [P, C, nlat, nlon].

    Notes
    -----
    - longitude wraps cyclically
    - latitude must already be in-bounds
    - output stays on the same device as `tensor`
    """
    crops = []
    full_nlon = tensor.shape[-1]

    for lat0, lon0 in zip(lat0_list, lon0_list):
        lon0 = lon0 % full_nlon
        lat_slice = tensor[:, lat0 : lat0 + nlat, :]

        if lon0 + nlon <= full_nlon:
            crop = lat_slice[:, :, lon0 : lon0 + nlon]
        else:
            split = full_nlon - lon0
            crop = torch.cat(
                [lat_slice[:, :, lon0:], lat_slice[:, :, : nlon - split]],
                dim=-1,
            )

        crops.append(crop)

    return torch.stack(crops, dim=0)


def _stitch_pred_batch(
    pred_batch: torch.Tensor,
    hr_field_base: torch.Tensor,
    hr_lat0_list,
    hr_lon0_list,
    hr_nlon: int,
) -> torch.Tensor:
    """
    Stitch [P, C, Hpatch, Wpatch] back into one [C, H, W] HR state.

    For this model, C=5:
    [height, divergence, vorticity, u, v].
    Keeps everything on the same device.
    """
    out = hr_field_base.clone()
    patch_nlat_hr = pred_batch.shape[-2]
    patch_nlon_hr = pred_batch.shape[-1]

    for i, (hr_lat0, hr_lon0) in enumerate(zip(hr_lat0_list, hr_lon0_list)):
        hr_lon0 = hr_lon0 % hr_nlon
        pred = pred_batch[i]

        if hr_lon0 + patch_nlon_hr <= hr_nlon:
            out[:, hr_lat0 : hr_lat0 + patch_nlat_hr, hr_lon0 : hr_lon0 + patch_nlon_hr] = pred
        else:
            split = hr_nlon - hr_lon0
            out[:, hr_lat0 : hr_lat0 + patch_nlat_hr, hr_lon0:] = pred[:, :, :split]
            out[:, hr_lat0 : hr_lat0 + patch_nlat_hr, : patch_nlon_hr - split] = pred[:, :, split:]

    return out


def _predict_lam_step_batched(
    model,
    hr_fields_norm: torch.Tensor,
    hr_winds_norm: torch.Tensor,
    lr_fields_raw_curr: torch.Tensor,
    lr_winds_raw_curr: torch.Tensor,
    plan: dict,
    norm_dev: dict,
    hr_nlon: int,
    patch_batch_size: int,
    blend_cfg: dict | None = None,
) -> dict:
    """
    Advance one autoregressive LAM step.

    Inputs correspond to CURRENT time t:
      - hr_fields_norm / hr_winds_norm : carried HR state at time t
      - lr_fields_raw_curr / lr_winds_raw_curr : external LR forcing at time t

    Outputs correspond to NEXT time t+1:
      - hr_next_fields_norm : stitched HR field at t+1 after optional LR blending
      - hr_next_fields_phys : same field in physical units

    Extra patch-level outputs are returned for diagnostics.
    """
        # Normalized LR model input: [5, Hlr, Wlr].
    lr_combined_curr = torch.cat(
        [
            (lr_fields_raw_curr - norm_dev["lr_f_mean"])
            / norm_dev["lr_f_std"],
            (lr_winds_raw_curr - norm_dev["lr_w_mean"])
            / norm_dev["lr_w_std"],
        ],
        dim=0,
    )

    # Normalized HR autoregressive state: [5, Hhr, Whr].
    hr_state_norm_curr = torch.cat(
        [hr_fields_norm, hr_winds_norm],
        dim=0,
    )

    # Five-channel HR physical-unit normalization vectors.
    hr_mean = torch.cat(
        [norm_dev["hr_f_mean"], norm_dev["hr_w_mean"]],
        dim=0,
    )
    hr_std = torch.cat(
        [norm_dev["hr_f_std"], norm_dev["hr_w_std"]],
        dim=0,
    )

    lr_batch = _cyclic_crop_stack(
        lr_combined_curr,
        plan["win_lat0_list"],
        plan["win_lon0_list"],
        plan["win_nlat"],
        plan["win_nlon"],
    )

    hr_batch = _cyclic_crop_stack(
        hr_state_norm_curr,
        plan["hr_lat0_list"],
        plan["hr_lon0_list"],
        plan["patch_nlat_hr"],
        plan["patch_nlon_hr"],
    )

    # Current-time LR interior patches. These are distinct from lr_batch:
    # lr_batch is the masked LR halo supplied to the LR encoder, whereas
    # lr_patch_batch_raw is used to pre-blend the HR encoder input.
    lr_state_raw_curr = torch.cat(
        [lr_fields_raw_curr, lr_winds_raw_curr],
        dim=0,
    )

    lr_patch_batch_raw = _cyclic_crop_stack(
        lr_state_raw_curr,
        plan["lat0_lr_list"],
        plan["lon0_lr_list"],
        plan["patch_nlat_lr"],
        plan["patch_nlon_lr"],
    )

    # Mask the LR cells that geographically overlap the central HR patch.
    # Only the exterior LR halo ring is available to the model.
    R_lat = (plan["win_nlat"] - plan["patch_nlat_lr"]) // 2
    R_lon = (plan["win_nlon"] - plan["patch_nlon_lr"]) // 2

    if R_lat < 1 or R_lon < 1:
        raise ValueError(
            "Expected an LR halo around the patch, but inferred a zero-width halo."
        )

    lr_halo_mask = torch.ones(
        1,
        1,
        plan["win_nlat"],
        plan["win_nlon"],
        device=lr_batch.device,
        dtype=lr_batch.dtype,
    )

    lr_halo_mask[
        :,
        :,
        R_lat : R_lat + plan["patch_nlat_lr"],
        R_lon : R_lon + plan["patch_nlon_lr"],
    ] = 0.0

    lr_batch = lr_batch * lr_halo_mask

    # By default, preserve the existing unblended encoder input.
    hr_batch_for_encoder = hr_batch

    # Optional pre-encoder HR/LR blending. The LR patch is normalized with
    # HR statistics before mixing because hr_batch is HR-normalized.
    if blend_cfg is not None and blend_cfg.get("input_enabled", False):
        lr_patch_batch_hr_norm = (
            lr_patch_batch_raw - hr_mean
        ) / hr_std

        hr_batch_for_encoder = blend_hr_prediction_with_lr(
            hr_pred=hr_batch,
            lr_patch=lr_patch_batch_hr_norm,
            w_lr_hr=blend_cfg["w_lr_hr"],
            interpolation=blend_cfg.get("interpolation", "bilinear"),
        )

    pred_chunks = []
    num_patches = hr_batch_for_encoder.shape[0]

    for start in range(0, num_patches, patch_batch_size):
        end = min(start + patch_batch_size, num_patches)
        pred_chunks.append(
            model.model(
                lr_batch[start:end],
                hr_batch_for_encoder[start:end],
            )
        )

    # [P, 5, Hpatch_hr, Wpatch_hr], normalized.
    pred_batch_norm = torch.cat(pred_chunks, dim=0)
    if pred_batch_norm.shape[1] != 5:
        raise RuntimeError(
        "Expected five output channels "
        "[height, divergence, vorticity, u, v], "
        f"but model returned shape {tuple(pred_batch_norm.shape)}."
    )

    # Convert all five prediction channels to physical units.
    pred_batch_raw_phys = pred_batch_norm * hr_std + hr_mean

    if blend_cfg is not None and blend_cfg.get("output_enabled", False):
        pred_batch_out_phys = blend_hr_prediction_with_lr(
            hr_pred=pred_batch_raw_phys,
            lr_patch=lr_patch_batch_raw,
            w_lr_hr=blend_cfg["w_lr_hr"],
            interpolation=blend_cfg.get("interpolation", "bilinear"),
        )
    else:
        pred_batch_out_phys = pred_batch_raw_phys

    # Preserve the existing forecast outside the covered patch region.
    hr_state_base_phys = hr_state_norm_curr * hr_std + hr_mean

    # Stitch five-channel patch predictions into the global HR state.
    hr_next_state_phys = _stitch_pred_batch(
        pred_batch=pred_batch_out_phys,
        hr_field_base=hr_state_base_phys,
        hr_lat0_list=plan["hr_lat0_list"],
        hr_lon0_list=plan["hr_lon0_list"],
        hr_nlon=hr_nlon,
    )

    hr_next_state_norm = (hr_next_state_phys - hr_mean) / hr_std

    return {
        "hr_next_fields_norm": hr_next_state_norm[:3],
        "hr_next_winds_norm": hr_next_state_norm[3:],
        "hr_next_fields_phys": hr_next_state_phys[:3],
        "hr_next_winds_phys": hr_next_state_phys[3:],
        "pred_batch_raw_phys": pred_batch_raw_phys,
        "pred_batch_out_phys": pred_batch_out_phys,
        "lr_patch_batch_raw": lr_patch_batch_raw,
    }


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

def compute_energy_spectra_fft(fields: torch.Tensor) -> dict:
    """
    2-D isotropic power spectrum via FFT.

    Parameters
    ----------
    fields : [C, nlat, nlon] in physical units

    Returns
    -------
    dict with keys rotational, divergent, potential, total, wavenumbers
    """
    nlat, nlon = fields.shape[-2], fields.shape[-1]
    max_k = min(nlat, nlon) // 2

    def _spectrum(ch: int):
        f = fields[ch].float()
        F = torch.fft.rfft2(f)
        power = F.real**2 + F.imag**2

        ki = torch.arange(nlat, device=fields.device).reshape(-1, 1).float()
        kj = torch.arange(F.shape[-1], device=fields.device).reshape(1, -1).float()
        k = torch.sqrt(ki**2 + kj**2).long().clamp(0, max_k - 1)

        spec = torch.zeros(max_k, device=fields.device)
        spec.scatter_add_(0, k.flatten(), power.flatten())
        return spec.cpu().numpy()

    rot = _spectrum(1)
    div = _spectrum(2)
    pot = _spectrum(0)

    return {
        "rotational": rot,
        "divergent": div,
        "potential": pot,
        "total": rot + div + pot,
        "wavenumbers": np.arange(max_k),
    }


def plot_energy_spectra(pred_spectra, truth_spectra, step, output_path):
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    k_plot = pred_spectra["wavenumbers"][1:]
    if len(k_plot) == 0:
        plt.close()
        return

    panels = [
        ("Rotational KE", "rotational"),
        ("Divergent KE", "divergent"),
        ("Potential E", "potential"),
        ("Total Energy", "total"),
    ]

    for idx, (title, key) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.loglog(
            k_plot,
            pred_spectra[key][1:],
            "b-",
            lw=2,
            label="LAM Prediction",
            alpha=0.85,
        )
        ax.loglog(
            k_plot,
            truth_spectra[key][1:],
            "r--",
            lw=2,
            label="HR Truth",
            alpha=0.85,
        )
        ax.set_xlabel("Wavenumber", fontsize=11)
        ax.set_ylabel("Power Spectrum", fontsize=11)
        ax.set_title(f"{title} (step={step})", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=8)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_energy_spectra_ratios(
    pred_spectra,
    truth_spectra,
    blended_truth_spectra,
    step,
    output_path,
):
    """
    Plot spectral-energy ratios relative to HR truth.

    A ratio of 1 means agreement with HR truth:
      prediction / HR truth
      blended truth / HR truth
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    k_plot = truth_spectra["wavenumbers"][1:]
    if len(k_plot) == 0:
        plt.close(fig)
        return

    panels = [
        ("Rotational KE", "rotational"),
        ("Divergent KE", "divergent"),
        ("Potential E", "potential"),
        ("Total Energy", "total"),
    ]

    for idx, (title, key) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        hr_truth = truth_spectra[key][1:]
        pred = pred_spectra[key][1:]
        blended = blended_truth_spectra[key][1:]

        # Do not divide by numerically zero HR-truth spectral bins.
        valid = hr_truth > np.finfo(float).eps

        pred_ratio = np.full_like(hr_truth, np.nan, dtype=float)
        blended_ratio = np.full_like(hr_truth, np.nan, dtype=float)

        pred_ratio[valid] = pred[valid] / hr_truth[valid]
        blended_ratio[valid] = blended[valid] / hr_truth[valid]

        ax.axhline(
            1.0,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="HR Truth / HR Truth",
        )
        ax.plot(
            k_plot,
            pred_ratio,
            color="tab:blue",
            linewidth=2,
            label="LAM Prediction / HR Truth",
            alpha=0.9,
        )
        ax.plot(
            k_plot,
            blended_ratio,
            color="tab:orange",
            linewidth=2,
            label="Blended Truth / HR Truth",
            alpha=0.9,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Wavenumber", fontsize=11)
        ax.set_ylabel("Energy ratio to HR truth", fontsize=11)
        ax.set_title(
            f"{title} ratio (step={step})",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"Spectrum Ratios Relative to HR Truth — Step {step}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_energy_spectra_ratios_hr_only(
    pred_spectra,
    truth_spectra,
    step,
    output_path,
):
    """
    Plot prediction / HR-truth spectral-energy ratios for only the
    non-blended HR interior. A ratio of 1 means exact agreement.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    k_plot = truth_spectra["wavenumbers"][1:]
    if len(k_plot) == 0:
        plt.close(fig)
        return

    panels = [
        ("Rotational KE", "rotational"),
        ("Divergent KE", "divergent"),
        ("Potential E", "potential"),
        ("Total Energy", "total"),
    ]

    for idx, (title, key) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        hr_truth = truth_spectra[key][1:]
        pred = pred_spectra[key][1:]

        # Avoid division by zero or near-zero truth spectral bins.
        valid = hr_truth > np.finfo(float).eps
        pred_ratio = np.full_like(hr_truth, np.nan, dtype=float)
        pred_ratio[valid] = pred[valid] / hr_truth[valid]

        ax.axhline(
            1.0,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="HR Truth / HR Truth",
        )
        ax.plot(
            k_plot,
            pred_ratio,
            color="tab:blue",
            linewidth=2,
            label="LAM Prediction / HR Truth",
            alpha=0.9,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Wavenumber", fontsize=11)
        ax.set_ylabel("Energy ratio to HR truth", fontsize=11)
        ax.set_title(
            f"{title} ratio — HR-only interior (step={step})",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"Spectrum Ratios Relative to HR Truth — HR-only Interior — Step {step}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Field comparison plots
# ---------------------------------------------------------------------------

def plot_comparison(
    pred,
    truth,
    step,
    output_path,
    truth_blended=None,
    lat_top=None,
    lat_bot=None,
    lon_left=None,
    lon_right=None,
):
    channel_names = [
        "Geopotential h",
        "Divergence δ",
        "Vorticity ζ",
        "Zonal wind u",
        "Meridional wind v",
    ]

    cmaps = [
        "viridis",
        "RdBu_r",
        "RdBu_r",
        "RdBu_r",
        "RdBu_r",
    ]

    num_channels = pred.shape[0]

    if truth_blended is None:
        col_titles = ["LAM Prediction", "HR Truth", "Error (Pred − HR)"]
        fig, axes = plt.subplots(
            num_channels, 3, figsize=(18, 4.5 * num_channels)
        )
    else:
        col_titles = [
            "LAM Prediction",
            "HR Truth",
            "Blended Truth",
            "Error (Pred − HR)",
            "Error (Pred − Blended)",
        ]
        fig, axes = plt.subplots(
            num_channels, 5, figsize=(28, 4.5 * num_channels)
        )

    for ch in range(num_channels):
        p_np = pred[ch].cpu().numpy()
        t_np = truth[ch].cpu().numpy()

        if truth_blended is None:
            e_hr_np = p_np - t_np
            state_min = min(p_np.min(), t_np.min())
            state_max = max(p_np.max(), t_np.max())
            emax = np.abs(e_hr_np).max() + 1e-8

            imgs = [p_np, t_np, e_hr_np]
            plot_cmaps = [cmaps[ch], cmaps[ch], "RdBu_r"]
            vmins = [state_min, state_min, -emax]
            vmaxs = [state_max, state_max, emax]
        else:
            b_np = truth_blended[ch].cpu().numpy()
            e_hr_np = p_np - t_np
            e_bl_np = p_np - b_np

            state_min = min(p_np.min(), t_np.min(), b_np.min())
            state_max = max(p_np.max(), t_np.max(), b_np.max())
            emax = max(np.abs(e_hr_np).max(), np.abs(e_bl_np).max()) + 1e-8

            imgs = [p_np, t_np, b_np, e_hr_np, e_bl_np]
            plot_cmaps = [cmaps[ch], cmaps[ch], cmaps[ch], "RdBu_r", "RdBu_r"]
            vmins = [state_min, state_min, state_min, -emax, -emax]
            vmaxs = [state_max, state_max, state_max, emax, emax]

        for col, (img, cm, vn, vx) in enumerate(zip(imgs, plot_cmaps, vmins, vmaxs)):
            ax = axes[ch, col]
            im = ax.imshow(img, cmap=cm, vmin=vn, vmax=vx, origin="upper")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if ch == 0:
                ax.set_title(col_titles[col], fontsize=12, fontweight="bold", pad=8)
            if col == 0:
                ax.set_ylabel(channel_names[ch], fontsize=11, labelpad=6)
            ax.set_xticks([])
            ax.set_yticks([])

    if all(v is not None for v in [lat_top, lat_bot, lon_left, lon_right]):
        geo = f"Patch: {lat_bot:.1f}°–{lat_top:.1f}°N, {lon_left:.1f}°–{lon_right:.1f}°E"
    else:
        geo = ""

    fig.suptitle(
        f"LAM Forecast — Step {step} {geo}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_profiles(pred, truth, step, output_path):
    """
    Cx2 figure:
    One row per state channel.

    pred, truth: [C, hr_nlat, hr_nlon] in physical units.
    """
    channel_names = [
        "Geopotential h",
        "Divergence δ",
        "Vorticity ζ",
        "Zonal wind u",
        "Meridional wind v",
    ]

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
    ]

    num_channels = pred.shape[0]

    err = (pred - truth).pow(2)
    rmse_lat = err.mean(dim=2).sqrt()
    rmse_lon = err.mean(dim=1).sqrt()

    hr_nlat = pred.shape[1]
    hr_nlon = pred.shape[2]
    lats = np.linspace(90, -90, hr_nlat)
    lons = np.linspace(0, 360, hr_nlon, endpoint=False)

    fig, axes = plt.subplots(
        num_channels, 2, figsize=(14, 4.5 * num_channels)
    )

    for ch in range(num_channels):
        ax_lat = axes[ch, 0]
        ax_lon = axes[ch, 1]

        ax_lat.plot(lats, rmse_lat[ch].cpu().numpy(), color=colors[ch], lw=1.8)
        ax_lat.set_xlabel("Latitude (°N)", fontsize=11)
        ax_lat.set_ylabel("RMSE (physical units)", fontsize=11)
        ax_lat.set_title(
            f"{channel_names[ch]} — RMSE vs Latitude (Step {step})",
            fontsize=12,
            fontweight="bold",
        )
        ax_lat.grid(True, alpha=0.3)
        ax_lat.invert_xaxis()

        ax_lon.plot(lons, rmse_lon[ch].cpu().numpy(), color=colors[ch], lw=1.8)
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
# Blended truth field for diagnostics
# ---------------------------------------------------------------------------

def _build_blended_truth_field(
    hr_truth_state_phys: torch.Tensor,
    lr_state_raw_curr: torch.Tensor,
    plan: dict,
    hr_nlon: int,
    blend_cfg: dict | None,
) -> torch.Tensor:
    """
    Build a five-channel rollout-style blended reference field.

    State channel order:
    [height, divergence, vorticity, u, v].
    """
    if blend_cfg is None or not blend_cfg.get("output_enabled", False):
        return hr_truth_state_phys

    truth_patch_batch = _cyclic_crop_stack(
        hr_truth_state_phys,
        plan["hr_lat0_list"],
        plan["hr_lon0_list"],
        plan["patch_nlat_hr"],
        plan["patch_nlon_hr"],
    )

    lr_patch_batch_raw = _cyclic_crop_stack(
        lr_state_raw_curr,
        plan["lat0_lr_list"],
        plan["lon0_lr_list"],
        plan["patch_nlat_lr"],
        plan["patch_nlon_lr"],
    )

    truth_patch_blended = blend_hr_prediction_with_lr(
        hr_pred=truth_patch_batch,
        lr_patch=lr_patch_batch_raw,
        w_lr_hr=blend_cfg["w_lr_hr"],
        interpolation=blend_cfg.get("interpolation", "bilinear"),
    )

    return _stitch_pred_batch(
        pred_batch=truth_patch_blended,
        hr_field_base=hr_truth_state_phys,
        hr_lat0_list=plan["hr_lat0_list"],
        hr_lon0_list=plan["hr_lon0_list"],
        hr_nlon=hr_nlon,
    )

# ---------------------------------------------------------------------------
# Save per-pixel error CSV
# ---------------------------------------------------------------------------

def export_pixelwise_error_csv(
    pred_crop: torch.Tensor,
    truth_crop: torch.Tensor,
    output_path: str,
    *,
    ic_idx: int,
    step: int,
    crop_lat0: int,
    crop_lon0: int,
    hr_nlat: int,
    hr_nlon: int,
    truth_blended_crop: torch.Tensor | None = None,
):
    """
    Export per-pixel prediction/target/error values to a long-format CSV.

    pred_crop, truth_crop: [5, H, W] in physical units
    truth_blended_crop: optional [5, H, W] in physical units
    """
    pred_np = pred_crop.detach().cpu().numpy()
    truth_np = truth_crop.detach().cpu().numpy()
    blended_np = (
        truth_blended_crop.detach().cpu().numpy()
        if truth_blended_crop is not None
        else None
    )

    H, W = pred_np.shape[1], pred_np.shape[2]
    channel_names = [
        "Geopotential h",
        "Divergence",
        "Vorticity",
        "Zonal wind u",
        "Meridional wind v",
    ]
    dlat_hr = 180.0 / hr_nlat
    dlon_hr = 360.0 / hr_nlon

    rows = []
    
    for ch in range(pred_np.shape[0]):
        for i in range(H):
            global_lat_idx = crop_lat0 + i
            lat_deg = 90.0 - global_lat_idx * dlat_hr

            for j in range(W):
                global_lon_idx = (crop_lon0 + j) % hr_nlon
                lon_deg = global_lon_idx * dlon_hr

                pred_val = float(pred_np[ch, i, j])
                truth_val = float(truth_np[ch, i, j])

                err_hr = pred_val - truth_val
                row = {
                    "ic_idx": ic_idx,
                    "step": step,
                    "channel": ch,
                    "channel_name": channel_names[ch],
                    "lat_idx_local": i,
                    "lon_idx_local": j,
                    "lat_idx_global": global_lat_idx,
                    "lon_idx_global": global_lon_idx,
                    "lat_deg": lat_deg,
                    "lon_deg": lon_deg,
                    "pred": pred_val,
                    "truth_hr": truth_val,
                    "err_hr": err_hr,
                    "abs_err_hr": abs(err_hr),
                    "sq_err_hr": err_hr**2,
                }

                if blended_np is not None:
                    blended_val = float(blended_np[ch, i, j])
                    err_bl = pred_val - blended_val
                    row.update(
                        {
                            "truth_blended": blended_val,
                            "err_blended": err_bl,
                            "abs_err_blended": abs(err_bl),
                            "sq_err_blended": err_bl**2,
                        }
                    )
                else:
                    row.update(
                        {
                            "truth_blended": np.nan,
                            "err_blended": np.nan,
                            "abs_err_blended": np.nan,
                            "sq_err_blended": np.nan,
                        }
                    )

                rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Crop selection helpers
# ---------------------------------------------------------------------------

def _resolve_eval_crop(
    patch_plan: dict,
    lamc: dict,
    hr_nlat: int,
    hr_nlon: int,
    plot_lat0_lr: int | None,
    plot_lon0_lr: int | None,
) -> tuple[int, int, int, int]:
    """
    Resolve the evaluation/visualization crop.

    Priority:
    1. If plot_lat0_lr and plot_lon0_lr are provided, require that they match
       one of the valid patch coordinates in the current patch plan, then use
       the corresponding HR patch coordinates.
    2. Otherwise use configurable HR-space fallback keys:
         eval_crop_lat0_hr, eval_crop_lon0_hr, eval_crop_nlat_hr, eval_crop_nlon_hr
    """
    if (plot_lat0_lr is None) ^ (plot_lon0_lr is None):
        raise ValueError(
            "plot_lat0_lr and plot_lon0_lr must either both be provided or both be omitted."
        )

    if plot_lat0_lr is not None and plot_lon0_lr is not None:
        matches = (
            (patch_plan["lat0_lr"] == plot_lat0_lr)
            & (patch_plan["lon0_lr"] == plot_lon0_lr)
        ).nonzero(as_tuple=True)[0]

        if matches.numel() == 0:
            valid_pairs = sorted(
                {
                    (int(a), int(b))
                    for a, b in zip(
                        patch_plan["lat0_lr"].tolist(),
                        patch_plan["lon0_lr"].tolist(),
                    )
                }
            )
            valid_pairs_str = ", ".join(f"({a},{b})" for a, b in valid_pairs)
            raise ValueError(
                f"Requested plot patch (lat0_lr={plot_lat0_lr}, lon0_lr={plot_lon0_lr}) "
                f"is not valid for the current halo/patch geometry. "
                f"Valid (lat0_lr, lon0_lr) pairs are: {valid_pairs_str}"
            )

        patch_idx = int(matches[0].item())
        crop_lat0 = int(patch_plan["hr_lat0"][patch_idx].item())
        crop_lon0 = int(patch_plan["hr_lon0"][patch_idx].item())
        crop_nlat = int(patch_plan["patch_nlat_hr"])
        crop_nlon = int(patch_plan["patch_nlon_hr"])
        return crop_lat0, crop_lon0, crop_nlat, crop_nlon

    crop_lat0 = int(lamc.get("eval_crop_lat0_hr", 0))
    crop_lon0 = int(lamc.get("eval_crop_lon0_hr", 0))
    crop_nlat = int(lamc.get("eval_crop_nlat_hr", hr_nlat))
    crop_nlon = int(lamc.get("eval_crop_nlon_hr", hr_nlon))

    if crop_nlat < 1 or crop_nlat > hr_nlat:
        raise ValueError(
            f"Invalid eval_crop_nlat_hr={crop_nlat}; must be in [1, {hr_nlat}]"
        )
    if crop_nlon < 1 or crop_nlon > hr_nlon:
        raise ValueError(
            f"Invalid eval_crop_nlon_hr={crop_nlon}; must be in [1, {hr_nlon}]"
        )
    if crop_lat0 < 0 or crop_lat0 + crop_nlat > hr_nlat:
        raise ValueError(
            f"Invalid eval crop latitude range: "
            f"lat0={crop_lat0}, nlat={crop_nlat}, hr_nlat={hr_nlat}"
        )

    return crop_lat0, crop_lon0, crop_nlat, crop_nlon


# ---------------------------------------------------------------------------
# Single-IC rollout
# ---------------------------------------------------------------------------

def _run_single_ic(
    model,
    ic_idx: int,
    hf,
    norm: dict,
    cfg: dict,
    output_dir: str | None,
    autoreg_steps: int,
    plot_channel: int,
    save_plots: bool,
    spectral_analysis: bool,
    device: torch.device,
    plot_lat0_lr: int = None,
    plot_lon0_lr: int = None,
) -> tuple[dict, float]:
    """
    Run one autoregressive rollout for a single IC using time-matched trajectory truth.

    Rollout semantics:
      - current HR state at time t is carried forward from previous stitched output
      - current LR forcing at time t is always read from the stored LR trajectory
      - predicted stitched HR state represents time t+1
    """
    del plot_channel  # retained only for CLI compatibility

    lamc = cfg["lam"]
    s = int(lamc["refinement_factor_lat"])
    R = int(lamc["halo_radius"])
    pL = int(lamc["patch_nlat_lr"])
    pN = int(lamc["patch_nlon_lr"])
    patch_batch_size = int(lamc.get("inference_patch_batch_size", 32))

    hr_nlat = int(hf.attrs["hr_nlat"])
    hr_nlon = int(hf.attrs["hr_nlon"])
    lr_nlat = int(hf.attrs["lr_nlat"])
    lr_nlon = int(hf.attrs["lr_nlon"])

    assert "fields" in hf["lr"] and "winds" in hf["lr"], \
        "HDF5 file must contain /lr/fields and /lr/winds"
    assert "fields" in hf["hr"] and "winds" in hf["hr"], \
        "HDF5 file must contain /hr/fields and /hr/winds"

    rollout_steps_avail = int(hf.attrs["rollout_steps"])
    assert autoreg_steps <= rollout_steps_avail, (
        f"Requested autoreg_steps={autoreg_steps}, "
        f"but dataset stores only {rollout_steps_avail}"
    )

    lr_fields_traj = torch.tensor(np.array(hf["lr/fields"][ic_idx]), dtype=torch.float32)
    lr_winds_traj = torch.tensor(np.array(hf["lr/winds"][ic_idx]), dtype=torch.float32)
    hr_fields_traj = torch.tensor(np.array(hf["hr/fields"][ic_idx]), dtype=torch.float32)
    hr_winds_traj = torch.tensor(np.array(hf["hr/winds"][ic_idx]), dtype=torch.float32)

    hr_t0 = hr_fields_traj[0]
    hr_w0 = hr_winds_traj[0]

    norm_dev = {
        "lr_f_mean": norm["lr_f_mean"].to(device),
        "lr_f_std": norm["lr_f_var"].sqrt().to(device),
        "lr_w_mean": norm["lr_w_mean"].to(device),
        "lr_w_std": norm["lr_w_var"].sqrt().to(device),
        "hr_f_mean": norm["hr_f_mean"].to(device),
        "hr_f_std": norm["hr_f_var"].sqrt().to(device),
        "hr_w_mean": norm["hr_w_mean"].to(device),
        "hr_w_std": norm["hr_w_var"].sqrt().to(device),
    }

    hr_fields_norm_curr = (
        (hr_t0.to(device) - norm_dev["hr_f_mean"]) / norm_dev["hr_f_std"]
    )
    hr_winds_norm_curr = (
        (hr_w0.to(device) - norm_dev["hr_w_mean"]) / norm_dev["hr_w_std"]
    )

    patch_plan = build_patch_plan_tensors(
        lr_nlat=lr_nlat,
        lr_nlon=lr_nlon,
        patch_nlat_lr=pL,
        patch_nlon_lr=pN,
        halo_radius=R,
        upscale_factor=s,
        exclude_pole_rows=int(lamc.get("exclude_pole_rows", 4)),
    )
    patch_plan["win_lat0_list"] = patch_plan["win_lat0"].tolist()
    patch_plan["win_lon0_list"] = patch_plan["win_lon0"].tolist()
    patch_plan["hr_lat0_list"] = patch_plan["hr_lat0"].tolist()
    patch_plan["hr_lon0_list"] = patch_plan["hr_lon0"].tolist()
    patch_plan["lat0_lr_list"] = patch_plan["lat0_lr"].tolist()
    patch_plan["lon0_lr_list"] = patch_plan["lon0_lr"].tolist()
    patch_plan["patch_nlat_lr"] = pL
    patch_plan["patch_nlon_lr"] = pN

    blend_runtime = None
    blend_cfg = lamc.get("blending", {})

    blending_enabled = bool(blend_cfg.get("enabled", False))
    input_blending_enabled = (
        blending_enabled
        and bool(blend_cfg.get("apply_to_hr_input", False))
    )
    output_blending_enabled = (
        blending_enabled
        and bool(blend_cfg.get("apply_in_rollout", True))
    )

    # Build blend weights if either pre-encoder input blending or
    # post-prediction rollout blending is enabled.
    if input_blending_enabled or output_blending_enabled:
        blend_width_hr = int(blend_cfg.get("width_hr", 0))
        blend_runtime = {
            "input_enabled": input_blending_enabled,
            "output_enabled": output_blending_enabled,
            "interpolation": blend_cfg.get("interpolation", "bilinear"),
            "w_lr_hr": build_lr_blend_weight_hr(
                patch_nlat_hr=patch_plan["patch_nlat_hr"],
                patch_nlon_hr=patch_plan["patch_nlon_hr"],
                blend_width_hr=blend_width_hr,
                device=device,
                dtype=torch.float32,
            ),
        }

    pad_width = len(str(autoreg_steps))
    prefix = f"ic{ic_idx:03d}_"
    
    width_hr = int(blend_cfg.get("width_hr", 0)) if blend_runtime is not None else 0

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

        # Diagnostics calculated only in the unblended HR interior.
        "RMSE_hr_only": [],
        "MAE_hr_only": [],
        "bias_hr_only": [],
        "RMSE_h_hr_only": [],
        "RMSE_vort_hr_only": [],
        "RMSE_div_hr_only": [],
        "RMSE_u_hr_only": [],
        "RMSE_v_hr_only": [],
        "RMSE_wind_hr_only": [],
    }

    crop_lat0, crop_lon0, crop_nlat, crop_nlon = _resolve_eval_crop(
        patch_plan=patch_plan,
        lamc=lamc,
        hr_nlat=hr_nlat,
        hr_nlon=hr_nlon,
        plot_lat0_lr=plot_lat0_lr,
        plot_lon0_lr=plot_lon0_lr,
    )

    def _crop(field):
        lat_slice = field[..., crop_lat0 : crop_lat0 + crop_nlat, :]
        lon0 = crop_lon0 % hr_nlon
        if lon0 + crop_nlon <= hr_nlon:
            return lat_slice[..., lon0 : lon0 + crop_nlon]
        slice1 = lat_slice[..., lon0:]
        slice2 = lat_slice[..., : crop_nlon - (hr_nlon - lon0)]
        return torch.cat([slice1, slice2], dim=-1)

    if 2 * width_hr >= crop_nlat or 2 * width_hr >= crop_nlon:
        raise ValueError(
            "lam.blending.width_hr must leave at least one interior cell in "
            f"the evaluation crop; got width_hr={width_hr}, "
            f"crop shape=({crop_nlat}, {crop_nlon})."
        )

    def _crop_hr_only(field):
        """Exclude width_hr cells from every edge of the selected crop."""
        if width_hr == 0:
            return field
        return field[..., width_hr:-width_hr, width_hr:-width_hr]

    ml_time = 0.0

    with torch.inference_mode():
        for lead_idx in range(autoreg_steps):
            curr_t = lead_idx
            next_t = lead_idx + 1
            step = next_t

            lr_fields_raw_curr = lr_fields_traj[curr_t].to(device)
            lr_winds_raw_curr = lr_winds_traj[curr_t].to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            rollout = _predict_lam_step_batched(
                model=model,
                hr_fields_norm=hr_fields_norm_curr,
                hr_winds_norm=hr_winds_norm_curr,
                lr_fields_raw_curr=lr_fields_raw_curr,
                lr_winds_raw_curr=lr_winds_raw_curr,
                plan=patch_plan,
                norm_dev=norm_dev,
                hr_nlon=hr_nlon,
                patch_batch_size=patch_batch_size,
                blend_cfg=blend_runtime,
            )

            hr_next_fields_norm = rollout["hr_next_fields_norm"]
            hr_next_winds_norm = rollout["hr_next_winds_norm"]

            hr_next_fields_phys_dev = rollout["hr_next_fields_phys"]
            pred_winds_phys_dev = rollout["hr_next_winds_phys"]

            hr_fields_norm_curr = hr_next_fields_norm
            hr_winds_norm_curr = hr_next_winds_norm

            if device.type == "cuda":
                torch.cuda.synchronize()
            ml_time += time.perf_counter() - t0

            truth_phys_dev = hr_fields_traj[next_t].to(device)
            truth_winds_dev = hr_winds_traj[next_t].to(device)

            truth_blended_crop = None
            if blend_runtime is not None and blend_runtime.get("output_enabled", False):
                truth_state_phys_dev = torch.cat(
                    [truth_phys_dev, truth_winds_dev],
                    dim=0,
                )

                lr_state_raw_curr = torch.cat(
                    [lr_fields_raw_curr, lr_winds_raw_curr],
                    dim=0,
                )

                truth_blended_state_phys_dev = _build_blended_truth_field(
                    hr_truth_state_phys=truth_state_phys_dev,
                    lr_state_raw_curr=lr_state_raw_curr,
                    plan=patch_plan,
                    hr_nlon=hr_nlon,
                    blend_cfg=blend_runtime,
                )

                truth_blended_crop = _crop(truth_blended_state_phys_dev[:3])
                truth_blended_wind_crop = _crop(truth_blended_state_phys_dev[3:])

            pred_crop = _crop(hr_next_fields_phys_dev)
            truth_crop = _crop(truth_phys_dev)
            wind_pred_crop = _crop(pred_winds_phys_dev)
            wind_truth_crop = _crop(truth_winds_dev)

            pred_state_crop = torch.cat(
                [pred_crop, wind_pred_crop],
                dim=0,
            )

            truth_state_crop = torch.cat(
                [truth_crop, wind_truth_crop],
                dim=0,
            )

            blended_state_crop = None
            if truth_blended_crop is not None:
                blended_state_crop = torch.cat(
                    [truth_blended_crop, truth_blended_wind_crop],
                    dim=0,
                )

            if output_dir is not None:
                pixel_csv_path = os.path.join(
                    output_dir,
                    f"{prefix}pixel_errors_{step:0{pad_width}d}.csv",
                )
                export_pixelwise_error_csv(
                    pred_crop=pred_state_crop,
                    truth_crop=truth_state_crop,
                    truth_blended_crop=blended_state_crop,
                    output_path=pixel_csv_path,
                    ic_idx=ic_idx,
                    step=step,
                    crop_lat0=crop_lat0,
                    crop_lon0=crop_lon0,
                    hr_nlat=hr_nlat,
                    hr_nlon=hr_nlon,
                )

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

            # Compute an independent error set on only the non-blended interior.
            pred_hr_only = _crop_hr_only(pred_crop)
            truth_hr_only = _crop_hr_only(truth_crop)
            wind_pred_hr_only = _crop_hr_only(wind_pred_crop)
            wind_truth_hr_only = _crop_hr_only(wind_truth_crop)

            pred_state_hr_only = torch.cat(
                [pred_hr_only, wind_pred_hr_only],
                dim=0,
            )

            truth_state_hr_only = torch.cat(
                [truth_hr_only, wind_truth_hr_only],
                dim=0,
            )

            hr_only_err = pred_hr_only - truth_hr_only
            hr_only_werr = wind_pred_hr_only - wind_truth_hr_only

            step_metrics["RMSE_hr_only"].append(
                hr_only_err.pow(2).mean().sqrt().item()
            )
            step_metrics["MAE_hr_only"].append(
                hr_only_err.abs().mean().item()
            )
            step_metrics["bias_hr_only"].append(
                hr_only_err.mean().item()
            )

            for ch_i, key in enumerate(
                ["RMSE_h_hr_only", "RMSE_vort_hr_only", "RMSE_div_hr_only"]
            ):
                step_metrics[key].append(
                    hr_only_err[ch_i].pow(2).mean().sqrt().item()
                )

            for ch_i, key in enumerate(["RMSE_u_hr_only", "RMSE_v_hr_only"]):
                step_metrics[key].append(
                    hr_only_werr[ch_i].pow(2).mean().sqrt().item()
                )

            step_metrics["RMSE_wind_hr_only"].append(
                hr_only_werr.pow(2).mean().sqrt().item()
            )

            if output_dir is not None and save_plots:
                dlat_hr = 180.0 / hr_nlat
                dlon_hr = 360.0 / hr_nlon
                lat_top = 90.0 - crop_lat0 * dlat_hr
                lat_bot = lat_top - crop_nlat * dlat_hr
                lon_left = (crop_lon0 * dlon_hr) % 360.0
                lon_right = lon_left + crop_nlon * dlon_hr

                pred_state_crop = torch.cat(
                    [pred_crop, wind_pred_crop],
                    dim=0,
                )

                truth_state_crop = torch.cat(
                    [truth_crop, wind_truth_crop],
                    dim=0,
                )

                pred_crop_cpu = pred_state_crop.detach().cpu()
                truth_crop_cpu = truth_state_crop.detach().cpu()
                truth_blended_crop_cpu = (
                    blended_state_crop.detach().cpu()
                    if blended_state_crop is not None
                    else None
                )

                plot_comparison(
                    pred_crop_cpu,
                    truth_crop_cpu,
                    step,
                    os.path.join(output_dir, f"{prefix}comparison_{step:0{pad_width}d}.png"),
                    truth_blended=truth_blended_crop_cpu,
                    lat_top=lat_top,
                    lat_bot=lat_bot,
                    lon_left=lon_left,
                    lon_right=lon_right,
                )

                plot_error_profiles(
                    pred_crop_cpu,
                    truth_crop_cpu,
                    step,
                    os.path.join(output_dir, f"{prefix}error_profiles_{step:0{pad_width}d}.png"),
                )

                hr_only_lat_top = lat_top - width_hr * dlat_hr
                hr_only_lat_bot = lat_bot + width_hr * dlat_hr
                hr_only_lon_left = lon_left + width_hr * dlon_hr
                hr_only_lon_right = lon_right - width_hr * dlon_hr

                # These intentionally compare only model prediction to HR truth.
                # The blending zone and blended-truth comparison are excluded.
                plot_comparison(
                    pred_state_hr_only.detach().cpu(),
                    truth_state_hr_only.detach().cpu(),
                    step,
                    os.path.join(
                        output_dir,
                        f"{prefix}comparison_hr_only_{step:0{pad_width}d}.png",
                    ),
                    lat_top=hr_only_lat_top,
                    lat_bot=hr_only_lat_bot,
                    lon_left=hr_only_lon_left,
                    lon_right=hr_only_lon_right,
                )

                plot_error_profiles(
                    pred_state_hr_only.detach().cpu(),
                    truth_state_hr_only.detach().cpu(),
                    step,
                    os.path.join(
                        output_dir,
                        f"{prefix}error_profiles_hr_only_{step:0{pad_width}d}.png",
                    ),
                )

                if spectral_analysis:
                    pred_spec = compute_energy_spectra_fft(pred_crop)
                    truth_spec = compute_energy_spectra_fft(truth_crop)

                    # Existing absolute-energy comparison: prediction vs HR truth.
                    plot_energy_spectra(
                        pred_spec,
                        truth_spec,
                        step,
                        os.path.join(
                            output_dir,
                            f"{prefix}spectra_{step:0{pad_width}d}.png",
                        ),
                    )

                    # Additional ratio plot is available only when rollout blending is enabled.
                    if truth_blended_crop is not None:
                        blended_truth_spec = compute_energy_spectra_fft(truth_blended_crop)

                        plot_energy_spectra_ratios(
                            pred_spectra=pred_spec,
                            truth_spectra=truth_spec,
                            blended_truth_spectra=blended_truth_spec,
                            step=step,
                            output_path=os.path.join(
                                output_dir,
                                f"{prefix}spectra_ratio_{step:0{pad_width}d}.png",
                            ),
                        )
                    else:
                        print(
                            " [info] Skipping spectral-ratio plot because "
                            "rollout blending is disabled."
                        )
                    
                    pred_hr_only_spec = compute_energy_spectra_fft(pred_hr_only)
                    truth_hr_only_spec = compute_energy_spectra_fft(truth_hr_only)

                    plot_energy_spectra_ratios_hr_only(
                        pred_spectra=pred_hr_only_spec,
                        truth_spectra=truth_hr_only_spec,
                        step=step,
                        output_path=os.path.join(
                            output_dir,
                            f"{prefix}spectra_ratio_hr_only_"
                            f"{step:0{pad_width}d}.png",
                        ),
                    )

    return step_metrics, ml_time

def plot_mean_rmse_by_step(all_metrics, output_dir):
    """
    Save one mean-RMSE-by-forecast-step plot per variable.

    Each RMSE value is averaged over all evaluated ICs at its forecast step.
    """
    if not all_metrics:
        return

    steps = np.arange(1, len(all_metrics[0]["RMSE_h"]) + 1)

    plot_specs = [
        (
            "RMSE_h",
            "Geopotential Height",
            "tab:blue",
            "mean_rmse_geopotential_by_step.png",
        ),
        (
            "RMSE_vort",
            "Vorticity",
            "tab:orange",
            "mean_rmse_vorticity_by_step.png",
        ),
        (
            "RMSE_div",
            "Divergence",
            "tab:green",
            "mean_rmse_divergence_by_step.png",
        ),
                (
            "RMSE_u",
            "Zonal Wind u",
            "tab:red",
            "mean_rmse_zonal_wind_by_step.png",
        ),
        (
            "RMSE_v",
            "Meridional Wind v",
            "tab:purple",
            "mean_rmse_meridional_wind_by_step.png",
        ),

        (
            "RMSE_h_hr_only",
            "Geopotential Height (HR-only)",
            "tab:blue",
            "mean_rmse_geopotential_hr_only_by_step.png",
        ),
        (
            "RMSE_vort_hr_only",
            "Vorticity (HR-only)",
            "tab:orange",
            "mean_rmse_vorticity_hr_only_by_step.png",
        ),
        (
            "RMSE_div_hr_only",
            "Divergence (HR-only)",
            "tab:green",
            "mean_rmse_divergence_hr_only_by_step.png",
        ),
        (
            "RMSE_u_hr_only",
            "Zonal Wind u (HR-only)",
            "tab:red",
            "mean_rmse_zonal_wind_hr_only_by_step.png",
        ),
        (
            "RMSE_v_hr_only",
            "Meridional Wind v (HR-only)",
            "tab:purple",
            "mean_rmse_meridional_wind_hr_only_by_step.png",
        ),
    ]

    for metric_key, variable_name, color, filename in plot_specs:
        values = np.array(
            [metrics[metric_key] for metrics in all_metrics],
            dtype=float,
        )

        mean_rmse = values.mean(axis=0)
        std_rmse = values.std(axis=0)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            steps,
            mean_rmse,
            marker="o",
            linewidth=2,
            color=color,
            label=f"Mean {variable_name} RMSE",
        )

        ax.fill_between(
            steps,
            mean_rmse - std_rmse,
            mean_rmse + std_rmse,
            color=color,
            alpha=0.20,
            label="±1 standard deviation across ICs",
        )

        ax.set_xlabel("Forecast step", fontsize=11)
        ax.set_ylabel("RMSE", fontsize=11)
        ax.set_title(
            f"{variable_name}: Mean RMSE by Forecast Step "
            f"({len(all_metrics)} ICs)",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xticks(steps)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, filename),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

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
    plot_lat0_lr: int = None,
    plot_lon0_lr: int = None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_metrics = []
    all_times = []

    with h5py.File(h5_path, "r") as hf:
        for i, ic_idx in enumerate(ic_indices):
            print(f"\n--- IC {i + 1}/{len(ic_indices)} (HDF5 row {ic_idx}) ---")
            metrics, ml_time = _run_single_ic(
                model=model,
                ic_idx=ic_idx,
                hf=hf,
                norm=norm,
                cfg=cfg,
                output_dir=output_dir if i == 0 else None,
                autoreg_steps=autoreg_steps,
                plot_channel=plot_channel,
                save_plots=save_plots and i == 0,
                spectral_analysis=spectral_analysis and i == 0,
                device=device,
                plot_lat0_lr=plot_lat0_lr,
                plot_lon0_lr=plot_lon0_lr,
            )
            all_metrics.append(metrics)
            all_times.append(ml_time)

            print(f" ML rollout time : {ml_time:.3f}s")
            rmse_final = metrics["RMSE"][-1] if metrics["RMSE"] else float("nan")
            print(f" RMSE (final step): {rmse_final:.6f}")

    if save_plots:
        plot_mean_rmse_by_step(
            all_metrics=all_metrics,
            output_dir=output_dir,
        )

    summary = {}
    for key in all_metrics[0]:
        flat = [v for m in all_metrics for v in m[key]]
        summary[f"{key}_mean"] = float(np.mean(flat))
        summary[f"{key}_std"] = float(np.std(flat))

    summary["ml_time_mean"] = float(np.mean(all_times))
    summary["ml_time_std"] = float(np.std(all_times))
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LAM PARADIS autoregressive inference")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--h5_path", required=True, help="Path to swe_paired.h5")
    parser.add_argument(
        "--output_dir",
        default="results_lam",
        help="Directory to save results",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_ics", type=int, default=1, help="Number of ICs to evaluate")
    parser.add_argument("--autoreg_steps", type=int, default=1, help="Autoregressive rollout steps")
    parser.add_argument("--plot_channel", type=int, default=0, help="0=h, 1=vorticity, 2=divergence")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--no_spectra", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ic_indices",
        type=str,
        default=None,
        help=(
            "Comma-separated HDF5 row indices to evaluate, e.g. '700,701,705'. "
            "Overrides --split and --num_ics entirely."
        ),
    )
    parser.add_argument(
        "--plot_lat0_lr",
        type=int,
        default=None,
        help="Top-left latitude of patch for visualization/evaluation, measured in LR cells",
    )
    parser.add_argument(
        "--plot_lon0_lr",
        type=int,
        default=None,
        help="Top-left longitude of patch for visualization/evaluation, measured in LR cells",
    )
    parser.add_argument(
        "--norm_h5_path",
        default=None,
        help=(
            "Optional HDF5 file from which to load normalization statistics. "
            "Defaults to --h5_path."
        ),
    )
    args = parser.parse_args()

    import pytorch_lightning as pl
    pl.seed_everything(args.seed)

    cfg = load_config(args.config)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=" * 70)
    print("LAM FORECAST CONFIGURATION")
    print("=" * 70)
    print(f" Checkpoint   : {args.checkpoint}")
    print(f" HDF5 dataset : {args.h5_path}")
    print(f" Split        : {args.split}")
    print(f" Device       : {device}")
    print(f" Autoreg steps: {args.autoreg_steps}")
    print(f" Num ICs      : {args.num_ics}")
    print(f" Output dir   : {args.output_dir}")
    print("=" * 70)

    norm_h5_path = args.norm_h5_path or args.h5_path

    with h5py.File(norm_h5_path, "r") as norm_hf:
        def _t(key):
            return torch.tensor(
                np.array(norm_hf.attrs[key]),
                dtype=torch.float32,
            )

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

    with h5py.File(args.h5_path, "r") as hf:
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

    print("\nLoading checkpoint …")
    model = load_checkpoint(cfg, args.checkpoint, device)
    print("Checkpoint loaded.\n")

    summary = run_inference(
        model=model,
        h5_path=args.h5_path,
        norm=norm,
        cfg=cfg,
        output_dir=args.output_dir,
        ic_indices=ic_indices,
        autoreg_steps=args.autoreg_steps,
        plot_channel=args.plot_channel,
        save_plots=not args.no_plots,
        spectral_analysis=not args.no_spectra,
        device=device,
        plot_lat0_lr=args.plot_lat0_lr,
        plot_lon0_lr=args.plot_lon0_lr,
    )

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    for key in ["RMSE", "MAE", "bias"]:
        m = summary[f"{key}_mean"]
        s = summary[f"{key}_std"]
        print(f" {key:8s}: {m:.6f} ± {s:.6f}")
    print(f" ML time : {summary['ml_time_mean']:.3f}s ± {summary['ml_time_std']:.3f}s")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    pd.DataFrame([summary]).to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to : {metrics_path}")
    if not args.no_plots:
        print(f"Plots saved to   : {args.output_dir}")


if __name__ == "__main__":
    main()