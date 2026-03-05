# avi_forecast_4.py
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


def compute_energy_spectra(fields, sht):
    """
    Compute isotropic energy spectra using spherical harmonic degrees l.

    Args:
        fields: Tensor (batch, 3, nlat, nlon)
                channels = [height, vorticity, divergence]
        sht: RealSHT object

    Returns:
        dict with spectra keyed by degree l
    """

    h = fields[:, 0:1]
    vort = fields[:, 1:2]
    div = fields[:, 2:3]

    # Spherical harmonic transform
    h_spec = sht(h)
    vort_spec = sht(vort)
    div_spec = sht(div)

    # Power spectra
    h_power = torch.abs(h_spec) ** 2
    vort_power = torch.abs(vort_spec) ** 2
    div_power = torch.abs(div_spec) ** 2

    # Remove channel dim if present
    if h_power.dim() == 4:
        h_power = h_power.squeeze(1)
        vort_power = vort_power.squeeze(1)
        div_power = div_power.squeeze(1)

    batch_size, Lmax, Mmax = h_power.shape
    max_l = min(Lmax, Mmax)

    rot_spec = torch.zeros(batch_size, max_l, device=fields.device)
    div_spec_out = torch.zeros(batch_size, max_l, device=fields.device)
    pot_spec = torch.zeros(batch_size, max_l, device=fields.device)

    for l in range(max_l):
        # Sum over all m for fixed l
        rot_spec[:, l] = vort_power[:, l, :l + 1].sum(dim=-1)
        div_spec_out[:, l] = div_power[:, l, :l + 1].sum(dim=-1)
        pot_spec[:, l] = h_power[:, l, :l + 1].sum(dim=-1)

    total_spec = rot_spec + div_spec_out + pot_spec

    return {
        "rotational": rot_spec.mean(dim=0).cpu().numpy(),
        "divergent": div_spec_out.mean(dim=0).cpu().numpy(),
        "potential": pot_spec.mean(dim=0).cpu().numpy(),
        "total": total_spec.mean(dim=0).cpu().numpy(),
        "wavenumbers": np.arange(max_l),
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
        ic = dataset.solver.random_initial_condition(mach=0.2)

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
            im = ax.imshow(pred_data, vmin=-4, vmax=4, cmap="twilight_shifted")
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
            fig = plt.figure(figsize=(6, 5))
            pred_data = prd[0, plot_channel].cpu().numpy()
            ax = fig.add_subplot(1, 1, 1)
            im = ax.imshow(pred_data, vmin=-4, vmax=4, cmap="twilight_shifted")
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

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = load_config(args.config)

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

    dataset.sht = dataset.solver.sht

    metrics_dict = {
        "L1_error": model_module.metric_l1,
        "L2_error": model_module.metric_l2,
        "W11_error": model_module.metric_w11,
    }

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
