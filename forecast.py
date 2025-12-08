import os
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
import pytorch_lightning as pl

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


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
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

    def forward(self, x):
        return self.model(x)


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


def plot_energy_spectra(
    pred_spectra, truth_spectra, step, output_path, model_name="Model"
):
    """Plot energy spectra comparing prediction and truth."""
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

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


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
):
    """Perform autoregressive inference and generate forecast plots."""
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
        ic = dataset.solver.random_initial_condition(mach=0.2)

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
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            pred_data = prd[0, plot_channel].cpu().numpy()
            im = ax.imshow(pred_data, vmin=-4, vmax=4, cmap="twilight_shifted")
            ax.set_title("Initial Condition (t=0)")
            ax.axis("off")
            fig.subplots_adjust(bottom=0.15)
            cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
            fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
            fname = f"comparison_{0:0{pad_width}d}.png"
            plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
            plt.close()

            wind_mag = (
                torch.sqrt(prd_uv_grid[0] ** 2 + prd_uv_grid[1] ** 2).cpu().numpy()
            )
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            im = ax.imshow(wind_mag, cmap="viridis")
            ax.set_title("Initial Wind Magnitude (t=0)")
            ax.axis("off")
            fig.subplots_adjust(bottom=0.15)
            cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
            fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
            fname = f"wind_{0:0{pad_width}d}.png"
            plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
            plt.close()

        if spectral_analysis:
            pred_spectra = compute_energy_spectra(prd, dataset.sht)
            truth_spectra = compute_energy_spectra(prd, dataset.sht)
            plot_path = os.path.join(spectral_dir, f"spectra_{0:0{pad_width}d}.png")
            plot_energy_spectra(pred_spectra, truth_spectra, 0, plot_path, model_name)

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

            if save_plots:
                fig = plt.figure(figsize=(12, 8))
                gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3)

                ax0 = fig.add_subplot(gs[0, 0])
                pred_data = prd[0, plot_channel].cpu().numpy()
                im0 = ax0.imshow(pred_data, vmin=-4, vmax=4, cmap="twilight_shifted")
                ax0.set_title(f"{model_name} Prediction (t={step})")
                ax0.axis("off")

                ax1 = fig.add_subplot(gs[0, 1])
                truth_data = ref[0, plot_channel].cpu().numpy()
                im1 = ax1.imshow(truth_data, vmin=-4, vmax=4, cmap="twilight_shifted")
                ax1.set_title(f"Truth (t={step})")
                ax1.axis("off")

                ax2 = fig.add_subplot(gs[1, :])
                error_data = pred_data - truth_data
                im2 = ax2.imshow(error_data, vmin=-1, vmax=1, cmap="RdBu_r")
                ax2.set_title(f"Error (Prediction - Truth) (t={step})")
                ax2.axis("off")

                cbar_ax1 = fig.add_axes([0.15, 0.52, 0.7, 0.02])
                fig.colorbar(im0, cax=cbar_ax1, orientation="horizontal")

                cbar_ax2 = fig.add_axes([0.15, 0.05, 0.7, 0.02])
                fig.colorbar(im2, cax=cbar_ax2, orientation="horizontal", label="Error")

                fname = f"comparison_{step:0{pad_width}d}.png"
                plt.savefig(
                    os.path.join(output_dir, fname), dpi=150, bbox_inches="tight"
                )
                plt.close()

                pred_wind_mag = (
                    torch.sqrt(prd_uv_grid[0] ** 2 + prd_uv_grid[1] ** 2).cpu().numpy()
                )
                truth_wind_mag = (
                    torch.sqrt(ref_uv_grid[0] ** 2 + ref_uv_grid[1] ** 2).cpu().numpy()
                )
                wind_error = pred_wind_mag - truth_wind_mag

                fig = plt.figure(figsize=(12, 8))
                gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3)

                ax0 = fig.add_subplot(gs[0, 0])
                im0 = ax0.imshow(pred_wind_mag, cmap="viridis")
                ax0.set_title(f"{model_name} Wind Magnitude (t={step})")
                ax0.axis("off")

                ax1 = fig.add_subplot(gs[0, 1])
                im1 = ax1.imshow(truth_wind_mag, cmap="viridis")
                ax1.set_title(f"Truth Wind Magnitude (t={step})")
                ax1.axis("off")

                ax2 = fig.add_subplot(gs[1, :])
                vmax_err = max(abs(wind_error.min()), abs(wind_error.max()))
                im2 = ax2.imshow(
                    wind_error, vmin=-vmax_err, vmax=vmax_err, cmap="RdBu_r"
                )
                ax2.set_title(f"Wind Magnitude Error (t={step})")
                ax2.axis("off")

                cbar_ax1 = fig.add_axes([0.15, 0.52, 0.7, 0.02])
                fig.colorbar(im0, cax=cbar_ax1, orientation="horizontal")

                cbar_ax2 = fig.add_axes([0.15, 0.05, 0.7, 0.02])
                fig.colorbar(im2, cax=cbar_ax2, orientation="horizontal", label="Error")

                fname = f"wind_{step:0{pad_width}d}.png"
                plt.savefig(
                    os.path.join(output_dir, fname), dpi=150, bbox_inches="tight"
                )
                plt.close()

            if spectral_analysis:
                pred_spectra = compute_energy_spectra(prd, dataset.sht)
                truth_spectra = compute_energy_spectra(ref, dataset.sht)
                plot_path = os.path.join(
                    spectral_dir, f"spectra_{step:0{pad_width}d}.png"
                )
                plot_energy_spectra(
                    pred_spectra, truth_spectra, step, plot_path, model_name
                )

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

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = load_config(args.config)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

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

    dataset = PdeDataset(
        dt=dt,
        nsteps=nsteps,
        dims=(config["data"]["nlat"], config["data"]["nlon"]),
        grid=config["data"]["grid"],
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
        model_name=model_name,
        autoreg_steps=args.autoreg_steps,
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


if __name__ == "__main__":
    main()
