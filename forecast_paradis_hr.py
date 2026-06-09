#!/usr/bin/env python3
"""
forecast_paradis_hr.py

Autoregressive inference for the global HR PARADIS model trained via
train_paradis_hr.py on the HDF5 dataset (swe_paired.h5).

Mirrors forecast_lam.py exactly in structure so metrics CSVs have identical
column names and compare_lam_paradis.py can ingest both without transformation.

Key differences from forecast_lam.py:
  - No patch tiling / stitching — one forward pass per step over the full
    HR global field [B, 3, hr_nlat, hr_nlon].
  - Model forward signature: model(fields, winds) — fields are advanced autoregressively,
  and winds are reconstructed from predicted HR vorticity/divergence at each step.
  - config["data"]["nlat"] / ["nlon"] are overridden to HR dims before
    loading the checkpoint so Paradis rebuilds at the correct mesh_size.

Usage:
    python forecast_paradis_hr.py \
        --config    config_paradis_lam.yaml \
        --checkpoint logs/.../hr_paradis-pretrain-best.ckpt \
        --h5_path   data/swe_paired.h5 \
        --output_dir results_paradis_hr/ \
        --num_ics   5 \
        --autoreg_steps 10 \
        --split     val
"""

import argparse
import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import ceil
import torch
import torch.nn.functional as F
import yaml
from matplotlib.gridspec import GridSpec

from model.paradis import Paradis
from train_paradis_hr import SWELightningModule   # reuse Lightning wrapper
from shallow_water_solver import ShallowWaterSolver


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _override_hr_dims(config: dict) -> dict:
    lc    = config["lam"]
    s_lat = int(lc["refinement_factor_lat"])
    s_lon = int(lc["refinement_factor_lon"])
    config["data"]["nlat"] = int(config["data"]["nlat"]) * s_lat
    config["data"]["nlon"] = int(config["data"]["nlon"]) * s_lon
    return config


def load_checkpoint(cfg: dict, ckpt_path: str, device: torch.device) -> SWELightningModule:
    lit  = SWELightningModule(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd   = {k.removeprefix("model."): v
            for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    missing, unexpected = lit.model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] Missing keys : {missing}")
    if unexpected:
        print(f"  [warn] Unexpected keys : {unexpected}")
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
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm(tensor, mean, var):
    return (tensor - mean) / var.sqrt()

def _denorm(tensor, mean, var):
    return tensor * var.sqrt() + mean


# ---------------------------------------------------------------------------
# Spectral analysis (identical to forecast_lam.py)
# ---------------------------------------------------------------------------

def compute_energy_spectra_fft(fields: torch.Tensor) -> dict:
    nlat, nlon = fields.shape[-2], fields.shape[-1]
    max_k = min(nlat, nlon) // 2

    def _spectrum(ch):
        f     = fields[ch].float()
        F_    = torch.fft.rfft2(f)
        power = F_.real ** 2 + F_.imag ** 2
        ki    = torch.arange(nlat,         device=fields.device).reshape(-1, 1).float()
        kj    = torch.arange(F_.shape[-1], device=fields.device).reshape(1, -1).float()
        k     = torch.sqrt(ki ** 2 + kj ** 2).long().clamp(0, max_k - 1)
        spec  = torch.zeros(max_k, device=fields.device)
        spec.scatter_add_(0, k.flatten(), power.flatten())
        return spec.cpu().numpy()

    rot = _spectrum(1)
    div = _spectrum(2)
    pot = _spectrum(0)
    return {"rotational": rot, "divergent": div, "potential": pot,
            "total": rot + div + pot,
            "wavenumbers": np.arange(max_k)}


def plot_energy_spectra(pred_spectra, truth_spectra, step, output_path):
    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    k_plot = pred_spectra["wavenumbers"][1:]
    k_ref  = np.array([5.0, k_plot[-1] * 0.5])

    for idx, (title, key) in enumerate([
        ("Rotational KE", "rotational"),
        ("Divergent KE",  "divergent"),
        ("Potential E",   "potential"),
        ("Total Energy",  "total"),
    ]):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.loglog(k_plot, pred_spectra[key][1:],  "b-",  lw=2, label="HR PARADIS", alpha=0.85)
        ax.loglog(k_plot, truth_spectra[key][1:], "r--", lw=2, label="HR Truth",   alpha=0.85)
        ax.loglog(k_ref, 1e4 * k_ref ** (-3),     "k:",  lw=1.5, alpha=0.5, label=r"$k^{-3}$")
        ax.loglog(k_ref, 1e3 * k_ref ** (-5/3),   "k-.", lw=1.5, alpha=0.5, label=r"$k^{-5/3}$")
        ax.set_xlabel("Wavenumber", fontsize=11)
        ax.set_ylabel("Power Spectrum", fontsize=11)
        ax.set_title(f"{title} (step={step})", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=8)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Comparison plots  (3x3: Prediction | Truth | Error)
# ---------------------------------------------------------------------------

CHANNEL_NAMES = ["Geopotential h", "Vorticity ζ", "Divergence δ"]

def plot_comparison(pred, truth, step, output_path, 
                    lat_top=None, lat_bot=None, lon_left=None, lon_right=None):
    CMAPS      = ["viridis", "RdBu_r", "RdBu_r"]
    col_titles = ["HR PARADIS Prediction", "HR Truth", "Error (Pred − Truth)"]

    if all(v is not None for v in [lat_top, lat_bot, lon_left, lon_right]):
        geo = (f"Patch: {lat_bot:.1f}°-{lat_top:.1f}°N, " f"{lon_left:.1f}°-{lon_right:.1f}°E")
    else:
        geo = ""
    
    # Geographic extent from grid dimensions directly
    # dlat      =  180.0 / hr_nlat
    # dlon      =  360.0 / hr_nlon
    # lat_top   =  90.0 - 0.5 * dlat
    # lat_bot   = -90.0 + 0.5 * dlat
    # lon_left  =   0.0 + 0.5 * dlon
    # lon_right = 360.0 - 0.5 * dlon
    # geo = (f"Domain: {lat_bot:.1f}°–{lat_top:.1f}°N, "
    #        f"{lon_left:.1f}°–{lon_right:.1f}°E")

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    for ch in range(3):
        p_np  = pred [ch].cpu().numpy()
        t_np  = truth[ch].cpu().numpy()
        e_np  = p_np - t_np
        vmin, vmax = t_np.min(), t_np.max()
        emax  = max(abs(e_np.min()), abs(e_np.max())) + 1e-8

        imgs  = [p_np, t_np, e_np]
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

    fig.suptitle(f"HR PARADIS Forecast — Step {step} {geo}",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Error profile plots
# ---------------------------------------------------------------------------

def plot_error_profiles(pred, truth, step, output_path):
    """
    3x2 figure:
    One row per SWE channel.
    Left column:  RMSE vs latitude
    Right column: RMSE vs longitude

    pred, truth: [3, hr_nlat, hr_nlon] in physical units
    """
    channel_names = ["Geopotential h", "Vorticity ζ", "Divergence δ"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    err = (pred - truth).pow(2)
    rmse_lat = err.mean(dim=2).sqrt()
    rmse_lon = err.mean(dim=1).sqrt()

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
    save_plots: bool,
    spectral_analysis: bool,
    device: torch.device,
    plot_lat0_lr: int = None,
    plot_lon0_lr: int = None,
) -> tuple:
    """Run one autoregressive rollout for a single IC using time-matched HR trajectory truth."""
    hr_nlat = int(hf.attrs["hr_nlat"])
    hr_nlon = int(hf.attrs["hr_nlon"])

    assert "fields" in hf["hr"] and "winds" in hf["hr"], \
        "HDF5 file must contain /hr/fields and /hr/winds"

    rollout_steps_avail = int(hf.attrs["rollout_steps"])
    assert autoreg_steps <= rollout_steps_avail, \
        f"Requested autoreg_steps={autoreg_steps}, but dataset stores only {rollout_steps_avail}"

    hr_fields_traj = torch.tensor(np.array(hf["hr/fields"][ic_idx]), dtype=torch.float32)
    hr_winds_traj = torch.tensor(np.array(hf["hr/winds"][ic_idx]), dtype=torch.float32)

    inp_f_raw = hr_fields_traj[0]
    inp_w_raw = hr_winds_traj[0]

    inp_f_norm = _norm(inp_f_raw, norm["hr_f_mean"], norm["hr_f_var"])
    inp_w_norm = _norm(inp_w_raw, norm["hr_w_mean"], norm["hr_w_var"])

    solver = _make_hr_solver(hf, device)

    hr_f_mean_dev = norm["hr_f_mean"].to(device)
    hr_f_var_dev = norm["hr_f_var"].to(device)
    hr_w_mean_dev = norm["hr_w_mean"].to(device)
    hr_w_var_dev = norm["hr_w_var"].to(device)

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

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    fields_norm = inp_f_norm.unsqueeze(0).to(device)
    winds_norm = inp_w_norm.unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(autoreg_steps):
            fields_norm = model.model(fields_norm, winds_norm)

            pred_phys_dev = _denorm(fields_norm.squeeze(0), hr_f_mean_dev, hr_f_var_dev)
            pred_vrtdiv_spec = solver.grid2spec(pred_phys_dev[1:3])
            pred_winds_phys_dev = solver.getuv(pred_vrtdiv_spec)
            winds_norm = _norm(
                pred_winds_phys_dev, hr_w_mean_dev, hr_w_var_dev
            ).unsqueeze(0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    ml_time = time.perf_counter() - t_start

    fields_norm = inp_f_norm.unsqueeze(0).to(device)
    winds_norm = inp_w_norm.unsqueeze(0).to(device)

    R = int(cfg["lam"]["halo_radius"])
    s_fac = int(cfg["lam"]["refinement_factor_lat"])
    margin = R * s_fac

    pL = int(cfg["lam"]["patch_nlat_lr"])
    pN = int(cfg["lam"]["patch_nlon_lr"])

    if plot_lat0_lr is not None and plot_lon0_lr is not None:
        crop_lat0 = plot_lat0_lr * s_fac
        crop_lon0 = plot_lon0_lr * s_fac
        crop_nlat = pL * s_fac
        crop_nlon = pN * s_fac
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
            fields_norm = model.model(fields_norm, winds_norm)

            pred_phys_dev = _denorm(fields_norm.squeeze(0), hr_f_mean_dev, hr_f_var_dev)
            pred_phys = pred_phys_dev.cpu()
            truth_phys = hr_fields_traj[step]

            pred_vrtdiv_spec = solver.grid2spec(pred_phys_dev[1:3])
            pred_winds_phys_dev = solver.getuv(pred_vrtdiv_spec)
            pred_winds_phys = pred_winds_phys_dev.cpu()
            truth_winds_phys = hr_winds_traj[step]

            pred_crop = _crop(pred_phys)
            truth_crop = _crop(truth_phys)
            wind_pred_crop = _crop(pred_winds_phys)
            wind_truth_crop = _crop(truth_winds_phys)

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

            winds_norm = _norm(
                pred_winds_phys_dev, hr_w_mean_dev, hr_w_var_dev
            ).unsqueeze(0)

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
    h5_path:  str,
    norm:     dict,
    cfg:      dict,
    output_dir: str,
    ic_indices: list,
    autoreg_steps: int,
    save_plots: bool,
    spectral_analysis: bool,
    device:   torch.device,
    plot_lat0_lr: int=None,
    plot_lon0_lr: int=None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_metrics = []
    all_times   = []

    with h5py.File(h5_path, "r") as hf:
        for i, ic_idx in enumerate(ic_indices):
            print(f"\n--- IC {i+1}/{len(ic_indices)} (HDF5 row {ic_idx}) ---")
            metrics, ml_time = _run_single_ic(
                model          = model,
                ic_idx         = ic_idx,
                hf             = hf,
                norm           = norm,
                cfg            = cfg,
                output_dir     = output_dir if i == 0 else None,
                autoreg_steps  = autoreg_steps,
                save_plots     = save_plots and i == 0,
                spectral_analysis = spectral_analysis and i == 0,
                device         = device,
                plot_lat0_lr = plot_lat0_lr,
                plot_lon0_lr = plot_lon0_lr,
            )
            all_metrics.append(metrics)
            all_times.append(ml_time)
            print(f"  ML rollout time  : {ml_time:.3f}s")
            rmse_final = metrics["RMSE"][-1] if metrics["RMSE"] else float("nan")
            print(f"  RMSE (final step): {rmse_final:.6f}")

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
    parser = argparse.ArgumentParser(description="HR PARADIS autoregressive inference")
    parser.add_argument("--config",       required=True)
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--h5_path",      required=True)
    parser.add_argument("--output_dir",   default="results_paradis_hr")
    parser.add_argument("--split",        default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--num_ics",      type=int, default=5)
    parser.add_argument("--autoreg_steps",type=int, default=1)
    parser.add_argument("--device",       default=None)
    parser.add_argument("--no_plots",     action="store_true")
    parser.add_argument("--no_spectra",   action="store_true")
    parser.add_argument("--seed",         type=int, default=42)
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
    cfg    = _override_hr_dims(cfg)
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 70)
    print("HR PARADIS FORECAST CONFIGURATION")
    print("=" * 70)
    print(f"  Checkpoint    : {args.checkpoint}")
    print(f"  HDF5 dataset  : {args.h5_path}")
    print(f"  HR grid       : {cfg['data']['nlat']} x {cfg['data']['nlon']}")
    print(f"  Split         : {args.split}")
    print(f"  Device        : {device}")
    print(f"  Autoreg steps : {args.autoreg_steps}")
    print(f"  Num ICs       : {args.num_ics}")
    print("=" * 70)

    # Load norm stats from HDF5
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

    print("\nLoading checkpoint …")
    model = load_checkpoint(cfg, args.checkpoint, device)
    print("Checkpoint loaded.\n")

    summary = run_inference(
        model          = model,
        h5_path        = args.h5_path,
        norm           = norm,
        cfg            = cfg,
        output_dir     = args.output_dir,
        ic_indices     = ic_indices,
        autoreg_steps  = args.autoreg_steps,
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

    metrics_path = os.path.join(args.output_dir, "metrics_paradis_hr.csv")
    pd.DataFrame([summary]).to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to : {metrics_path}")


if __name__ == "__main__":
    main()
