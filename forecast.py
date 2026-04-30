import os
import time
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import pytorch_lightning as pl

from torch_harmonics.examples.losses import (
    SquaredL2LossS2,
    L1LossS2,
    L2LossS2,
    W11LossS2,
)

from model.paradis import Paradis
from pde_dataset_with_winds import PdeDatasetWithWinds


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


class SWELightningModule(pl.LightningModule):
    """Lightning module for loading a PARADIS checkpoint."""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        self.nlat = config["data"]["nlat"]
        self.nlon = config["data"]["nlon"]
        self.grid = config["data"]["grid"]

        if "paradis" not in config["model"]:
            raise ValueError(
                "PARADIS model config not found. Add a 'model.paradis' section to your config."
            )
        self.model = Paradis(config)

        self.loss_fn = SquaredL2LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_l1 = L1LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_l2 = L2LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_w11 = W11LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)

    def forward(self, fields, winds):
        return self.model(fields, winds)


def compute_energy_spectra(fields, sht):
    """Compute energy spectra for shallow water equation fields.

    Args:
        fields: Tensor of shape (batch, 3, nlat, nlon) containing [h, vorticity, divergence].
        sht: RealSHT object for spherical harmonic transforms.

    Returns:
        Dictionary containing power spectra for rotational, divergent, and potential energy,
        plus the total and the wavenumber array.
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

    return {
        "rotational": rot_spectrum.mean(dim=0).cpu().numpy(),
        "divergent": div_spectrum.mean(dim=0).cpu().numpy(),
        "potential": pot_spectrum.mean(dim=0).cpu().numpy(),
        "total": total_spectrum.mean(dim=0).cpu().numpy(),
        "wavenumbers": np.arange(max_k),
    }


def plot_energy_spectra(
    pred_spectra, truth_spectra, step, output_path, model_name="Model"
):
    """Plot energy spectra comparing prediction and truth."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    k_plot = pred_spectra["wavenumbers"][1:]
    k_ref = np.array([5.0, 50.0])

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
        ax.loglog(
            k_ref,
            1e4 * k_ref ** (-3.0),
            "k:",
            linewidth=1.5,
            alpha=0.5,
            label=r"$k^{-3}$",
        )
        ax.loglog(
            k_ref,
            1e3 * k_ref ** (-5.0 / 3.0),
            "k-.",
            linewidth=1.5,
            alpha=0.5,
            label=r"$k^{-5/3}$",
        )
        ax.set_xlabel("Wavenumber $l$", fontsize=11)
        ax.set_ylabel("Power Spectrum", fontsize=11)
        ax.set_title(f"{title} (t={step})", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_xlim([1, k_plot[-1]])

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _move_dataset_to_device(dataset, device):
    """Move all dataset solver and normalization buffers to the target device."""
    dataset.solver = dataset.solver.to(device)
    dataset.sht = dataset.sht.to(device)
    dataset.inp_mean = dataset.inp_mean.to(device)
    dataset.inp_var = dataset.inp_var.to(device)
    dataset.wind_mean = dataset.wind_mean.to(device)
    dataset.wind_var = dataset.wind_var.to(device)


def _get_ic(solver, ic_type, mach=0.2):
    """Return a spectral initial condition from the solver.

    Args:
        solver: ShallowWaterSolver instance.
        ic_type: ``"random"`` or ``"galewsky"``.
        mach: Mach number passed to random_initial_condition (ignored for galewsky).
    """
    if ic_type == "random":
        return solver.random_initial_condition(mach=mach)
    elif ic_type == "galewsky":
        return solver.galewsky_initial_condition()
    else:
        raise ValueError(
            f"Unknown ic_type '{ic_type}'. Expected 'random' or 'galewsky'."
        )


def _run_single_ic_inference(
    model,
    dataset,
    loss_fn,
    metrics_dict,
    nsteps,
    autoreg_steps,
    device,
    ic_type="random",
    output_dir=None,
    ic_index=None,
    plot_channel=0,
    save_plots=True,
    spectral_analysis=True,
    model_name="Model",
):
    """Run one autoregressive rollout from a single initial condition.

    The ML model and the reference numerical solver are each timed over the
    full rollout horizon independently, with CUDA synchronization guards on GPU
    so the reported times are accurate. Per-step metrics and plot/tensor outputs
    are collected in a second pass to avoid I/O polluting the timing loop.

    Args:
        model: The PARADIS LightningModule in eval mode.
        dataset: PdeDatasetWithWinds instance with all buffers on the target device.
        loss_fn: Spherical L2 loss.
        metrics_dict: Dict mapping metric name to callable.
        nsteps: Number of solver substeps per model timestep.
        autoreg_steps: Number of autoregressive steps to roll out.
        device: Torch device.
        ic_type: Initial condition type — ``"random"`` (mach=0.2) or ``"galewsky"``.
        output_dir: If provided, tensors/plots/spectra are saved here.
        ic_index: IC index used as a filename prefix when output_dir is set.
        plot_channel: Field channel to visualise (0=h, 1=vorticity, 2=divergence).
        save_plots: Whether to save comparison plots.
        spectral_analysis: Whether to save spectral energy plots.
        model_name: Label used in plot titles.

    Returns:
        step_metrics: Dict mapping metric name to list of per-step values.
        ml_time: Wall-clock seconds for the ML rollout.
        solver_time: Wall-clock seconds for the reference solver rollout.
    """
    pad_width = len(str(autoreg_steps))
    prefix = f"ic{ic_index:03d}_" if ic_index is not None else ""

    inp_mean = dataset.inp_mean
    inp_var = dataset.inp_var
    wind_mean = dataset.wind_mean
    wind_var = dataset.wind_var

    # --- Timing pass: ML model ---
    ic = _get_ic(dataset.solver, ic_type)

    prd_fields = (dataset.solver.spec2grid(ic) - inp_mean) / torch.sqrt(inp_var)
    prd_fields = prd_fields.unsqueeze(0)
    prd_winds_raw = dataset.solver.getuv(ic[1:])
    prd_winds = (prd_winds_raw - wind_mean) / torch.sqrt(wind_var)
    prd_winds = prd_winds.unsqueeze(0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    ml_start = time.perf_counter()

    for _ in range(autoreg_steps):
        prd_fields = model(prd_fields, prd_winds)
        prd_unnorm = prd_fields * torch.sqrt(inp_var) + inp_mean
        prd_spec = dataset.sht(prd_unnorm.squeeze(0))
        prd_uv_grid = dataset.solver.getuv(prd_spec[1:])
        prd_winds = (prd_uv_grid - wind_mean) / torch.sqrt(wind_var)
        prd_winds = prd_winds.unsqueeze(0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    ml_time = time.perf_counter() - ml_start

    # --- Timing pass: reference numerical solver ---
    ref_uspec = ic.clone()

    if device.type == "cuda":
        torch.cuda.synchronize()
    solver_start = time.perf_counter()

    for _ in range(autoreg_steps):
        ref_uspec = dataset.solver.timestep(ref_uspec, nsteps)

    if device.type == "cuda":
        torch.cuda.synchronize()
    solver_time = time.perf_counter() - solver_start

    # --- Metrics and output pass ---
    ic = _get_ic(dataset.solver, ic_type)
    uspec = ic.clone()

    prd_fields = (dataset.solver.spec2grid(ic) - inp_mean) / torch.sqrt(inp_var)
    prd_fields = prd_fields.unsqueeze(0)
    prd_uv_grid_init = dataset.solver.getuv(ic[1:])
    prd_winds = (prd_uv_grid_init - wind_mean) / torch.sqrt(wind_var)
    prd_winds = prd_winds.unsqueeze(0)

    step_metrics = {key: [] for key in metrics_dict.keys()}
    step_metrics["loss"] = []

    if output_dir is not None:
        init_outputs = {"fields": prd_fields[0].cpu(), "winds": prd_uv_grid_init.cpu()}
        torch.save(
            init_outputs,
            os.path.join(output_dir, f"{prefix}prediction_{0:0{pad_width}d}.pt"),
        )
        torch.save(
            init_outputs,
            os.path.join(output_dir, f"{prefix}truth_{0:0{pad_width}d}.pt"),
        )

        if save_plots:
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(1, 1, 1)
            im = ax.imshow(
                prd_fields[0, plot_channel].cpu().numpy(),
                vmin=-4,
                vmax=4,
                cmap="twilight_shifted",
            )
            ax.set_title("Initial Condition (t=0)", fontsize=12, fontweight="bold")
            ax.axis("off")
            fig.subplots_adjust(bottom=0.15)
            fig.colorbar(
                im, cax=fig.add_axes([0.15, 0.05, 0.7, 0.03]), orientation="horizontal"
            )
            plt.savefig(
                os.path.join(output_dir, f"{prefix}comparison_{0:0{pad_width}d}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

        if spectral_analysis:
            init_spectra = compute_energy_spectra(prd_fields, dataset.sht)
            plot_energy_spectra(
                init_spectra,
                init_spectra,
                0,
                os.path.join(output_dir, f"{prefix}spectra_{0:0{pad_width}d}.png"),
                model_name,
            )

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

        for name, metric_fn in metrics_dict.items():
            step_metrics[name].append(metric_fn(prd_fields, ref_fields).item())
        step_metrics["loss"].append(loss_fn(prd_fields, ref_fields).item())

        if output_dir is not None:
            torch.save(
                {"fields": prd_fields[0].cpu(), "winds": prd_uv_grid.cpu()},
                os.path.join(output_dir, f"{prefix}prediction_{step:0{pad_width}d}.pt"),
            )
            torch.save(
                {"fields": ref_fields[0].cpu(), "winds": ref_uv_grid.cpu()},
                os.path.join(output_dir, f"{prefix}truth_{step:0{pad_width}d}.pt"),
            )

            if save_plots:
                pred_data = prd_fields[0, plot_channel].cpu().numpy()
                truth_data = ref_fields[0, plot_channel].cpu().numpy()
                error_data = pred_data - truth_data
                fig = plt.figure(figsize=(18, 5))
                ax1 = fig.add_subplot(1, 3, 1)
                im1 = ax1.imshow(pred_data, vmin=-4, vmax=4, cmap="twilight_shifted")
                ax1.set_title(f"Prediction (t={step})", fontsize=12, fontweight="bold")
                ax1.axis("off")
                ax2 = fig.add_subplot(1, 3, 2)
                im2 = ax2.imshow(truth_data, vmin=-4, vmax=4, cmap="twilight_shifted")
                ax2.set_title(
                    f"Ground Truth (t={step})", fontsize=12, fontweight="bold"
                )
                ax2.axis("off")
                ax3 = fig.add_subplot(1, 3, 3)
                error_max = max(abs(error_data.min()), abs(error_data.max()))
                im3 = ax3.imshow(
                    error_data, vmin=-error_max, vmax=error_max, cmap="RdBu_r"
                )
                ax3.set_title(f"Error (t={step})", fontsize=12, fontweight="bold")
                ax3.axis("off")
                fig.subplots_adjust(bottom=0.15, wspace=0.3)
                fig.colorbar(
                    im1,
                    cax=fig.add_axes([0.08, 0.08, 0.22, 0.03]),
                    orientation="horizontal",
                )
                fig.colorbar(
                    im2,
                    cax=fig.add_axes([0.39, 0.08, 0.22, 0.03]),
                    orientation="horizontal",
                )
                fig.colorbar(
                    im3,
                    cax=fig.add_axes([0.70, 0.08, 0.22, 0.03]),
                    orientation="horizontal",
                )
                plt.savefig(
                    os.path.join(
                        output_dir, f"{prefix}comparison_{step:0{pad_width}d}.png"
                    ),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close()

            if spectral_analysis:
                pred_spectra = compute_energy_spectra(prd_fields, dataset.sht)
                truth_spectra = compute_energy_spectra(ref_fields, dataset.sht)
                plot_energy_spectra(
                    pred_spectra,
                    truth_spectra,
                    step,
                    os.path.join(
                        output_dir, f"{prefix}spectra_{step:0{pad_width}d}.png"
                    ),
                    model_name,
                )

    return step_metrics, ml_time, solver_time


def _aggregate_multi_ic_metrics(all_step_metrics, all_ml_times, all_solver_times):
    """Aggregate per-IC step-metric lists into a flat summary dict.

    Per-step values are flattened across all ICs before computing mean/std so
    that the reported statistics reflect both temporal and IC-to-IC variability.
    Timing statistics include mean speedup of the ML model over the numerical solver.

    Args:
        all_step_metrics: List of per-IC dicts mapping metric name to list of per-step floats.
        all_ml_times: List of per-IC ML rollout wall-clock times in seconds.
        all_solver_times: List of per-IC numerical solver wall-clock times in seconds.

    Returns:
        Dict with {metric}_mean / {metric}_std entries plus timing statistics.
    """
    summary = {}

    for key in all_step_metrics[0].keys():
        all_values = [v for ic in all_step_metrics for v in ic[key]]
        summary[f"{key}_mean"] = (
            float(np.mean(all_values)) if all_values else float("nan")
        )
        summary[f"{key}_std"] = (
            float(np.std(all_values)) if all_values else float("nan")
        )

    summary["ml_time_mean"] = float(np.mean(all_ml_times))
    summary["ml_time_std"] = float(np.std(all_ml_times))
    summary["solver_time_mean"] = float(np.mean(all_solver_times))
    summary["solver_time_std"] = float(np.std(all_solver_times))
    summary["speedup_mean"] = (
        summary["solver_time_mean"] / summary["ml_time_mean"]
        if summary["ml_time_mean"] > 0
        else float("nan")
    )

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
    num_ics=1,
    ic_type="random",
    plot_channel=0,
    save_plots=True,
    spectral_analysis=True,
    device=torch.device("cpu"),
):
    """Perform autoregressive inference over one or more initial conditions.

    Plots, tensors, and spectral analyses are written only for the first IC to
    avoid excessive disk usage. Per-IC timing is printed to stdout and aggregated
    in the returned summary.

    For the Galewsky test case (``ic_type="galewsky"``) the solver has a single
    deterministic IC, so ``num_ics`` is forced to 1.

    Args:
        model: The PARADIS LightningModule.
        dataset: PdeDatasetWithWinds instance.
        loss_fn: Spherical L2 loss.
        metrics_dict: Dict mapping metric name to callable.
        output_dir: Directory to save results.
        nsteps: Number of solver substeps per model timestep.
        model_name: Label used in plot titles.
        autoreg_steps: Number of autoregressive steps per rollout.
        num_ics: Number of initial conditions to average over (forced to 1 for galewsky).
        ic_type: ``"random"`` or ``"galewsky"``.
        plot_channel: Field channel to visualise.
        save_plots: Whether to save comparison plots.
        spectral_analysis: Whether to save spectral energy plots.
        device: Torch device.

    Returns:
        Summary dict with aggregated mean/std metrics and timing statistics.
    """
    model.eval()
    model.to(device)
    _move_dataset_to_device(dataset, device)
    os.makedirs(output_dir, exist_ok=True)

    if ic_type == "galewsky":
        num_ics = 1

    all_step_metrics = []
    all_ml_times = []
    all_solver_times = []

    print(
        f"Starting Autoregressive Inference ({autoreg_steps} steps, {num_ics} IC(s), ic_type={ic_type})..."
    )

    with torch.no_grad():
        for ic_idx in range(num_ics):
            print(f"\n--- IC {ic_idx + 1}/{num_ics} ---")
            step_metrics, ml_time, solver_time = _run_single_ic_inference(
                model=model,
                dataset=dataset,
                loss_fn=loss_fn,
                metrics_dict=metrics_dict,
                nsteps=nsteps,
                autoreg_steps=autoreg_steps,
                device=device,
                ic_type=ic_type,
                output_dir=output_dir if ic_idx == 0 else None,
                ic_index=ic_idx,
                plot_channel=plot_channel,
                save_plots=save_plots and ic_idx == 0,
                spectral_analysis=spectral_analysis and ic_idx == 0,
                model_name=model_name,
            )
            all_step_metrics.append(step_metrics)
            all_ml_times.append(ml_time)
            all_solver_times.append(solver_time)

            speedup = solver_time / ml_time if ml_time > 0 else float("nan")
            print(f"  ML rollout:     {ml_time:.3f}s")
            print(f"  Solver rollout: {solver_time:.3f}s")
            print(f"  Speedup:        {speedup:.1f}x")

    return _aggregate_multi_ic_metrics(all_step_metrics, all_ml_times, all_solver_times)


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
        "--output_dir", type=str, default="./results", help="Directory to save results"
    )
    parser.add_argument(
        "--autoreg_steps", type=int, default=10, help="Number of autoregressive steps"
    )
    parser.add_argument(
        "--num_ics",
        type=int,
        default=1,
        help="Number of initial conditions to average over (ignored for galewsky)",
    )
    parser.add_argument(
        "--ic_type",
        type=str,
        default="random",
        choices=["random", "galewsky"],
        help="Initial condition type: random (mach=0.2) or galewsky barotropic jet",
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

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = load_config(args.config)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=" * 70)
    print("FORECAST CONFIGURATION")
    print("=" * 70)
    print(f"Checkpoint:          {args.checkpoint}")
    print(f"Device:              {device}")
    print(f"Autoregressive steps:{args.autoreg_steps}")
    print(f"IC type:             {args.ic_type}")
    print(f"Initial conditions:  {args.num_ics if args.ic_type == 'random' else 1}")
    print(f"Output directory:    {args.output_dir}")
    print(f"Save plots:          {not args.no_plots}")
    print(f"Spectral analysis:   {args.spectral_analysis}")
    print("=" * 70 + "\n")

    print(f"Loading checkpoint: {args.checkpoint}")
    model_module = SWELightningModule(config)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["state_dict"]
    for key in ["metric_w11.k_phi_mesh", "metric_w11.k_theta_mesh"]:
        if key in state_dict:
            print(f"Removing buffer: {key}")
            del state_dict[key]
    model_module.load_state_dict(state_dict, strict=False)
    model_module.eval()
    print("Checkpoint loaded successfully.\n")

    print("Setting up Shallow Water Solver...")
    dt = config["data"]["dt"]
    nsteps = dt // config["data"]["dt_solver"]

    dataset = PdeDatasetWithWinds(
        dt=dt,
        nsteps=nsteps,
        dims=(config["data"]["nlat"], config["data"]["nlon"]),
        normalize=True,
        device=device,
    )
    dataset.sht = dataset.solver.sht

    metrics_dict = {
        "L1_error": model_module.metric_l1,
        "L2_error": model_module.metric_l2,
        "W11_error": model_module.metric_w11,
    }

    results = autoregressive_inference(
        model=model_module,
        dataset=dataset,
        loss_fn=model_module.loss_fn,
        metrics_dict=metrics_dict,
        output_dir=args.output_dir,
        nsteps=nsteps,
        model_name="PARADIS",
        autoreg_steps=args.autoreg_steps,
        num_ics=args.num_ics,
        ic_type=args.ic_type,
        plot_channel=args.plot_channel,
        save_plots=(not args.no_plots),
        spectral_analysis=args.spectral_analysis,
        device=device,
    )

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    for key in ["loss", "L1_error", "L2_error", "W11_error"]:
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        if mean_key in results:
            print(f"{key:12s}: {results[mean_key]:.6f} ± {results[std_key]:.6f}")
    print("-" * 70)
    print(
        f"{'ML time (s)':12s}: {results['ml_time_mean']:.3f} ± {results['ml_time_std']:.3f}"
    )
    print(
        f"{'Solver (s)':12s}: {results['solver_time_mean']:.3f} ± {results['solver_time_std']:.3f}"
    )
    print(f"{'Speedup':12s}: {results['speedup_mean']:.1f}x")

    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    pd.DataFrame([results]).to_csv(metrics_path, index=False)

    print("=" * 70)
    print(f"Metrics saved to: {metrics_path}")
    if not args.no_plots:
        print(f"Plots saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
