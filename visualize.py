#!/usr/bin/env python3
"""
visualize.py
============

Public visualization interface for comparing optimizers in the SWAN project.

Users interact with this file only.  The two backend modules are:

* ``history_utils.py``: reads/prepares TensorBoard or CSV scalar histories.
* ``rollout_utils.py``: prepares forecast/rollout data, optionally by calling
  the original ``forecast.py`` helper functions inside the SWAN repository.

Commands
--------
1. ``plot_history``
   Compare training/validation histories from TensorBoard logs.

2. ``forecast``
   Run or synthesize rollout comparisons and generate forecast-level plots.

The plotting code lives here because this is the user-facing file.  The helpers
return clean data structures, and this script turns them into figures.

Examples
--------
Training/validation history::

    python visualize.py plot_history \
      --runs ./results/adam/version_0 ./results/mud/version_0 ./results/muon/version_0 \
      --labels Adam MUD Muon \
      --stage validation \
      --plot both \
      --error_metric l2 \
      --efficiency_metric both \
      --outdir ./figures_history

Forecast comparison from trained runs::

    python visualize.py forecast \
      --runs ./results/adam/version_0 ./results/mud/version_0 ./results/muon/version_0 \
      --labels Adam MUD Muon \
      --config config_paradis.yaml \
      --autoreg_steps 100 \
      --output_freq 10 \
      --channel vorticity \
      --rollout_dir ./rollout_results \
      --outdir ./figures_forecast

Quick synthetic forecast demo::

    python visualize.py forecast \
      --synthetic_demo \
      --labels Adam MUD Muon \
      --autoreg_steps 20 \
      --output_freq 5
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import history_utils as hist
import rollout_utils as roll


# ---------------------------------------------------------------------------
# General plotting helpers
# ---------------------------------------------------------------------------

def ensure_outdir(path: str | Path) -> Path:
    """Create and return an output directory."""
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a figure with consistent options and close it."""
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def resource_axis_name(resource: str) -> str:
    """Human-readable resource axis name."""
    if resource == "step":
        return "global training step"
    if resource == "time":
        return "relative wall-clock time (s)"
    return resource


def history_line_style(index: int) -> Dict[str, object]:
    """Return a clean solid-line style for crowded optimizer history plots."""
    colors = plt.get_cmap("tab20").colors
    return {
        "color": colors[index % len(colors)],
        "linestyle": "-",
        "linewidth": 2.2 + 0.15 * (index % 3),
        "alpha": 0.72,
        "zorder": 10 + index,
    }


# ---------------------------------------------------------------------------
# plot_history command
# ---------------------------------------------------------------------------

def run_plot_history(args: argparse.Namespace) -> None:
    """Entry point for ``python visualize.py plot_history``."""
    outdir = ensure_outdir(args.outdir)
    runs = hist.load_history_runs(args.runs, args.labels)

    for stage in hist.concrete_stages(args.stage):
        metric = hist.metric_for_stage(stage, args.error_metric)

        if args.plot in ("learning_curve", "both"):
            for resource in resources_from_arg(args.efficiency_metric):
                plot_history_learning_curve(
                    runs=runs,
                    stage=stage,
                    error_metric=metric,
                    resource=resource,
                    outdir=outdir,
                )

        if args.plot in ("hitting_curve", "both"):
            for resource in resources_from_arg(args.efficiency_metric):
                plot_history_hitting_curve(
                    runs=runs,
                    stage=stage,
                    error_metric=metric,
                    resource=resource,
                    outdir=outdir,
                )


def resources_from_arg(efficiency_metric: str) -> List[str]:
    """Expand ``both`` into ``step`` and ``time``."""
    if efficiency_metric == "both":
        return ["step", "time"]
    return [efficiency_metric]


def plot_history_learning_curve(
    runs: Sequence[hist.RunScalars],
    stage: str,
    error_metric: str,
    resource: str,
    outdir: Path,
) -> None:
    """Plot metric value against training step or relative wall-clock time."""
    series_by_label = hist.prepare_learning_curve_data(runs, stage, error_metric)
    y_label = hist.display_metric_name(stage, error_metric)
    metric_name = hist.filename_metric_name(stage, error_metric)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, (label, series) in enumerate(series_by_label.items()):
        x = series.step if resource == "step" else series.relative_time_sec
        ax.plot(x, series.value, label=label, **history_line_style(i))

    ax.set_xlabel(resource_axis_name(resource))
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} vs {resource_axis_name(resource)}")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, framealpha=0.92)
    fig.tight_layout()

    output = outdir / f"learning_curve_{stage}_{resource}_{metric_name}.png"
    save_figure(fig, output)


def plot_history_hitting_curve(
    runs: Sequence[hist.RunScalars],
    stage: str,
    error_metric: str,
    resource: str,
    outdir: Path,
) -> None:
    """Plot first-hitting resource as a function of target error threshold."""
    curves = hist.prepare_hitting_curve_data(runs, stage, error_metric, resource)
    y_label = "first hitting step" if resource == "step" else "first hitting time (s)"
    metric_label = hist.display_metric_name(stage, error_metric)
    metric_name = hist.filename_metric_name(stage, error_metric)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, (label, (thresholds, hits)) in enumerate(curves.items()):
        style = history_line_style(i)
        ax.plot(thresholds, hits, label=label, **style)

    ax.set_xlabel(f"target {metric_label} threshold")
    ax.set_ylabel(y_label)
    ax.set_title(f"First hitting {resource} vs target {metric_label}")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, framealpha=0.92)
    # Thresholds are generated from loose to strict.  Inverting the x-axis makes
    # the plot read naturally: moving right asks for a stricter/lower error.
    ax.invert_xaxis()
    fig.tight_layout()

    output = outdir / f"hitting_curve_{stage}_{resource}_{metric_name}.png"
    save_figure(fig, output)


# ---------------------------------------------------------------------------
# forecast command
# ---------------------------------------------------------------------------

def run_forecast(args: argparse.Namespace) -> None:
    """Entry point for ``python visualize.py forecast``.

    In normal SWAN use, this command receives training run folders.  It finds
    checkpoints, runs rollouts through rollout_utils.py, then plots comparison
    figures.  For testing, ``--synthetic_demo`` creates fake rollout folders
    with the same structure.
    """
    outdir = ensure_outdir(args.outdir)
    rollout_dir = ensure_outdir(args.rollout_dir)

    if args.synthetic_demo:
        labels = args.labels or ["Adam", "MUD", "Muon"]
        rollout_runs = roll.create_synthetic_rollouts(
            labels=labels,
            rollout_dir=rollout_dir,
            autoreg_steps=args.autoreg_steps,
            output_freq=args.output_freq,
            num_ics=args.num_ics,
            seed=args.seed,
        )
    else:
        if not args.runs:
            raise ValueError("forecast requires --runs unless --synthetic_demo is used.")
        if not args.labels:
            raise ValueError("forecast requires --labels unless --synthetic_demo is used.")
        checkpoints = roll.checkpoints_from_runs(args.runs, args.checkpoint_choice)
        rollout_runs = roll.run_real_rollouts(
            checkpoints=checkpoints,
            labels=args.labels,
            config_path=args.config,
            rollout_dir=rollout_dir,
            autoreg_steps=args.autoreg_steps,
            output_freq=args.output_freq,
            num_ics=args.num_ics,
            ic_type=args.ic_type,
            seed=args.seed,
            channel=args.channel,
            device=args.device,
        )

    # Always reload from disk after generation.  This tests the same path users
    # rely on later and avoids hidden state in memory.
    rollout_runs = roll.load_rollout_runs([run.rollout_dir for run in rollout_runs], [run.label for run in rollout_runs])
    snapshots = roll.load_snapshots_for_step(rollout_runs, args.summary_step)

    plot_forecast_error_curve(rollout_runs, args.error_metric, outdir)
    plot_forecast_accuracy_bar(rollout_runs, args.error_metric, outdir)
    plot_forecast_speedup_bar(rollout_runs, outdir)
    plot_prediction_grid(snapshots, args.channel, args.grid_cols, args.output_freq, outdir)
    plot_error_grid(snapshots, args.channel, args.error_mode, args.grid_cols, args.output_freq, outdir)
    plot_combined_spectra(snapshots, args.output_freq, outdir)


# ---------------------------------------------------------------------------
# Forecast scalar plots
# ---------------------------------------------------------------------------

def plot_forecast_error_curve(rollout_runs: Sequence[roll.RolloutRun], error_metric: str, outdir: Path) -> None:
    """Plot forecast error history versus autoregressive rollout step."""
    column = roll.metric_column(error_metric)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for run in rollout_runs:
        if column not in run.per_step.columns:
            raise KeyError(f"{column} not found in {run.rollout_dir / 'per_step_metrics.csv'}")
        grouped = run.per_step.groupby("step")[column]
        mean = grouped.mean()
        std = grouped.std().fillna(0.0)
        steps = mean.index.to_numpy(dtype=float)
        values = mean.to_numpy(dtype=float)
        ax.plot(steps, values, marker="o", markersize=3, linewidth=1.8, label=run.label)
        if len(rollout_runs) <= 6 and std.max() > 0:
            ax.fill_between(steps, values - std.to_numpy(), values + std.to_numpy(), alpha=0.12)

    ax.set_xlabel("autoregressive rollout step")
    ax.set_ylabel(column)
    ax.set_title(f"Forecast {column} over rollout")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, outdir / f"forecast_error_curve_{error_metric}.png")


def _bar_colors(n: int):
    """Return a visually distinct but restrained color list for bar charts."""
    cmap = plt.get_cmap("tab20" if n > 10 else "tab10")
    return [cmap(i % cmap.N) for i in range(n)]


def _style_bar_axes(ax: plt.Axes) -> None:
    """Apply consistent styling to forecast summary bar charts."""
    ax.grid(True, axis="y", alpha=0.28, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelrotation=25)


def _annotate_bars(ax: plt.Axes, bars) -> None:
    """Place compact numeric values above bars when values are finite."""
    finite_heights = [bar.get_height() for bar in bars if np.isfinite(bar.get_height())]
    if not finite_heights:
        return
    ymax = max(finite_heights)
    pad = 0.015 * ymax if ymax != 0 else 0.02
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + pad,
            f"{height:.3g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(top=max(ax.get_ylim()[1], ymax + 5 * pad))


def plot_forecast_accuracy_bar(rollout_runs: Sequence[roll.RolloutRun], error_metric: str, outdir: Path) -> None:
    """Plot aggregate forecast error from metrics.csv.

    This is a bar chart, not a histogram: there is one aggregate scalar per
    optimizer.  The value is typically averaged over rollout steps and initial
    conditions by rollout_utils.py / forecast.py.
    """
    column = roll.metric_mean_column(error_metric)
    labels = [run.label for run in rollout_runs]
    values = np.array([run.metrics.get(column, np.nan) for run in rollout_runs], dtype=float)

    fig, ax = plt.subplots(figsize=(max(8.5, 1.25 * len(labels)), 5.4))
    bars = ax.bar(labels, values, color=_bar_colors(len(labels)), edgecolor="black", linewidth=0.7, alpha=0.88)
    ax.set_ylabel(f"{column} (lower is better)")
    ax.set_title(f"Aggregate forecast accuracy: {column}", fontweight="bold")
    _style_bar_axes(ax)
    _annotate_bars(ax, bars)
    fig.tight_layout()
    save_figure(fig, outdir / f"forecast_accuracy_bar_{error_metric}.png")


def plot_forecast_speedup_bar(rollout_runs: Sequence[roll.RolloutRun], outdir: Path) -> None:
    """Plot ML-vs-solver speedup from metrics.csv."""
    labels = [run.label for run in rollout_runs]
    values = np.array([run.metrics.get("speedup_mean", np.nan) for run in rollout_runs], dtype=float)

    fig, ax = plt.subplots(figsize=(max(8.5, 1.25 * len(labels)), 5.4))
    bars = ax.bar(labels, values, color=_bar_colors(len(labels)), edgecolor="black", linewidth=0.7, alpha=0.88)
    ax.set_ylabel("speedup_mean = solver_time_mean / ml_time_mean")
    ax.set_title("Forecast rollout speedup (higher is better)", fontweight="bold")
    _style_bar_axes(ax)
    _annotate_bars(ax, bars)
    fig.tight_layout()
    save_figure(fig, outdir / "forecast_speedup_bar.png")


# ---------------------------------------------------------------------------
# Forecast spatial grid plots
# ---------------------------------------------------------------------------

def plot_prediction_grid(
    snapshots: Sequence[roll.FieldSnapshot],
    channel: str,
    grid_cols: int,
    output_freq: int,
    outdir: Path,
) -> None:
    """Plot final ground truth and optimizer predictions with one shared scale."""
    ch = roll.CHANNEL_TO_INDEX[channel]
    truth = snapshots[0].truth_fields[ch]
    panels = [("Ground Truth", truth)] + [(snap.label, snap.prediction_fields[ch]) for snap in snapshots]
    title = f"Final prediction comparison ({channel}, step {snapshots[0].step}; saved every {output_freq} step(s))"
    output = outdir / f"forecast_prediction_grid_{channel}_final.png"
    plot_image_panels(
        panels,
        title,
        grid_cols,
        output,
        cmap="twilight_shifted",
        symmetric=True,
        colorbar_label=f"{channel} value (shared color scale)",
    )


def plot_error_grid(
    snapshots: Sequence[roll.FieldSnapshot],
    channel: str,
    error_mode: str,
    grid_cols: int,
    output_freq: int,
    outdir: Path,
) -> None:
    """Plot pointwise error maps with one shared scale."""
    ch = roll.CHANNEL_TO_INDEX[channel]
    panels = []
    for snap in snapshots:
        diff = snap.prediction_fields[ch] - snap.truth_fields[ch]
        if error_mode == "signed":
            data = diff
        elif error_mode == "abs":
            data = np.abs(diff)
        elif error_mode == "squared":
            data = diff**2
        else:
            raise ValueError("error_mode must be signed, abs, or squared")
        panels.append((snap.label, data))

    symmetric = error_mode == "signed"
    cmap = "RdBu_r" if symmetric else "viridis"
    title = f"Final pointwise error ({channel}, {error_mode}, step {snapshots[0].step}; saved every {output_freq} step(s))"
    output = outdir / f"forecast_error_grid_{channel}_final_{error_mode}.png"
    if error_mode == "signed":
        cbar_label = f"prediction − truth for {channel} (shared color scale)"
    elif error_mode == "abs":
        cbar_label = f"|prediction − truth| for {channel} (shared color scale)"
    else:
        cbar_label = f"(prediction − truth)^2 for {channel} (shared color scale)"
    plot_image_panels(panels, title, grid_cols, output, cmap=cmap, symmetric=symmetric, colorbar_label=cbar_label)


def plot_image_panels(
    panels: Sequence[Tuple[str, np.ndarray]],
    title: str,
    grid_cols: int,
    output: Path,
    cmap: str,
    symmetric: bool,
    colorbar_label: str,
) -> None:
    """Plot many 2D fields with automatic row wrapping and one colorbar."""
    if grid_cols < 1:
        raise ValueError("grid_cols must be >= 1")
    n = len(panels)
    ncols = min(grid_cols, n)
    nrows = int(math.ceil(n / ncols))

    all_values = np.concatenate([np.asarray(data).ravel() for _, data in panels])
    if symmetric:
        vmax = float(np.nanmax(np.abs(all_values)))
        vmin = -vmax
    else:
        vmin = float(np.nanmin(all_values))
        vmax = float(np.nanmax(all_values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = -1.0, 1.0

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.6 * nrows), squeeze=False)
    last_im = None
    for ax, (panel_title, data) in zip(axes.ravel(), panels):
        last_im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.axis("off")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.82, location="bottom", pad=0.05)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        cbar.formatter = formatter
        cbar.update_ticks()
        cbar.set_label(colorbar_label, fontsize=10)
        # When values are very small or large, ScalarFormatter shows a compact
        # multiplier such as ×10^{-3} next to the colorbar ticks.
        cbar.ax.xaxis.get_offset_text().set_fontsize(9)
    save_figure(fig, output)


# ---------------------------------------------------------------------------
# Forecast spectral plot
# ---------------------------------------------------------------------------

def _scaled_power_law(k: np.ndarray, spectrum: np.ndarray, exponent: float) -> np.ndarray:
    """Return a reference k^exponent line scaled to the spectrum magnitude.

    The original forecast.py overlays k^{-3} and k^{-5/3} reference slopes.
    Those reference lines use fixed constants because the original spectra come
    from spherical harmonic coefficients.  This comparison plot may use saved
    grid fields and a normalized FFT fallback, so a fixed constant can be wildly
    off-scale.  We therefore scale the reference line to pass through a finite
    ground-truth spectrum value near the middle of the available wavenumbers.
    """
    k = np.asarray(k, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    valid = (k > 0) & np.isfinite(spectrum) & (spectrum > 0)
    if not np.any(valid):
        return np.full_like(k, np.nan, dtype=float)
    valid_indices = np.flatnonzero(valid)
    ref_index = valid_indices[len(valid_indices) // 2]
    scale = spectrum[ref_index] / (k[ref_index] ** exponent)
    return scale * (k ** exponent)


def plot_combined_spectra(snapshots: Sequence[roll.FieldSnapshot], output_freq: int, outdir: Path) -> None:
    """Plot one 2x2 spectral figure containing truth and all optimizers.

    The layout intentionally mirrors the original SWAN forecast.py spectral
    figure: rotational kinetic energy, divergent kinetic energy, potential
    energy, and total energy.  The comparison version differs in that every
    optimizer is plotted on the same axes together with the same ground truth.
    """
    truth_spectra = roll.compute_simple_energy_spectra(snapshots[0].truth_fields)
    pred_spectra = [(snap.label, roll.compute_simple_energy_spectra(snap.prediction_fields)) for snap in snapshots]

    titles = [
        "Rotational kinetic energy",
        "Divergent kinetic energy",
        "Potential energy",
        "Total energy",
    ]
    keys = ["rotational", "divergent", "potential", "total"]
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    colors = _bar_colors(max(1, len(pred_spectra)))

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6))
    legend_handles = []
    legend_labels = []

    for ax, title, key in zip(axes.ravel(), titles, keys):
        k = truth_spectra["wavenumbers"][1:]
        truth_values = truth_spectra[key][1:]
        truth_line, = ax.loglog(
            k,
            truth_values,
            color="black",
            linestyle="--",
            linewidth=2.6,
            label="Ground Truth",
            zorder=5,
        )
        if not legend_handles:
            legend_handles.append(truth_line)
            legend_labels.append("Ground Truth")

        for i, (label, spectra) in enumerate(pred_spectra):
            k_pred = spectra["wavenumbers"][1:]
            line, = ax.loglog(
                k_pred,
                spectra[key][1:],
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=1.9,
                marker="o" if len(k_pred) < 40 else None,
                markersize=3,
                alpha=0.92,
                label=label,
            )
            if ax is axes.ravel()[0]:
                legend_handles.append(line)
                legend_labels.append(label)

        # Add the two reference slopes used by the original forecast.py.  They
        # are scaled to the current subplot so they remain visible under the
        # normalized FFT fallback used by this comparison script.
        if len(k) > 2:
            k_ref = np.array([float(k[0]), float(k[-1])])
            ref_k_all = np.asarray(k, dtype=float)
            ref_3_all = _scaled_power_law(ref_k_all, truth_values, -3.0)
            ref_53_all = _scaled_power_law(ref_k_all, truth_values, -5.0 / 3.0)
            ref_3 = np.interp(k_ref, ref_k_all, ref_3_all)
            ref_53 = np.interp(k_ref, ref_k_all, ref_53_all)
            ref_line_3, = ax.loglog(k_ref, ref_3, color="0.35", linestyle=":", linewidth=1.5, alpha=0.75, label=r"scaled $k^{-3}$")
            ref_line_53, = ax.loglog(k_ref, ref_53, color="0.35", linestyle="-.", linewidth=1.5, alpha=0.75, label=r"scaled $k^{-5/3}$")
            if ax is axes.ravel()[0]:
                legend_handles.extend([ref_line_3, ref_line_53])
                legend_labels.extend([r"scaled $k^{-3}$", r"scaled $k^{-5/3}$"])

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Wavenumber $l$")
        ax.set_ylabel("Power spectrum")
        ax.grid(True, which="both", alpha=0.3)
        if len(k) > 0:
            ax.set_xlim(left=max(1, k[0]), right=k[-1])

    fig.suptitle(
        f"Final rollout spectra comparison (step {snapshots[0].step}; saved every {output_freq} step(s))",
        fontsize=14,
        fontweight="bold",
    )
    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=min(4, len(legend_labels)), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    save_figure(fig, outdir / "forecast_spectra_final.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare SWAN optimizer runs using TensorBoard histories and forecast rollouts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # plot_history command ----------------------------------------------------
    ph = subparsers.add_parser(
        "plot_history",
        help="Plot training/validation histories from TensorBoard or CSV scalar logs.",
    )
    ph.add_argument("--runs", nargs="+", required=True, help="Run directories, e.g. ./results/adam/version_0")
    ph.add_argument("--labels", nargs="+", required=True, help="Legend labels, one per run directory")
    ph.add_argument("--stage", choices=hist.STAGES, default="validation", help="History stage to plot. Default: validation")
    ph.add_argument("--plot", choices=hist.HISTORY_PLOTS, default="learning_curve", help="History plot type. Default: learning_curve")
    ph.add_argument("--error_metric", choices=hist.ERROR_METRICS, default="loss", help="Error/loss metric. Default: loss")
    ph.add_argument("--efficiency_metric", choices=hist.EFFICIENCY_METRICS, default="both", help="X-axis resource. Default: both")
    ph.add_argument("--outdir", default="./figures", help="Directory for history figures. Default: ./figures")
    ph.set_defaults(func=run_plot_history)

    # forecast command --------------------------------------------------------
    fc = subparsers.add_parser(
        "forecast",
        help="Run/compare forecast rollouts from trained run folders.",
    )
    fc.add_argument("--runs", nargs="+", help="Training run directories containing checkpoints/. Required unless --synthetic_demo is used.")
    fc.add_argument("--labels", nargs="+", help="Optimizer labels. Required unless --synthetic_demo is used; synthetic defaults to Adam MUD Muon.")
    fc.add_argument("--config", default="config_paradis.yaml", help="SWAN config file. Default: config_paradis.yaml")
    fc.add_argument("--checkpoint_choice", choices=("best", "last"), default="best", help="Checkpoint to use from each run. Default: best")
    fc.add_argument("--autoreg_steps", type=int, default=100, help="Number of autoregressive rollout steps. Default: 100")
    fc.add_argument("--output_freq", type=int, default=10, help="Save rollout tensors/plots every N steps. Default: 10")
    fc.add_argument("--num_ics", type=int, default=1, help="Number of forecast initial conditions. Default: 1")
    fc.add_argument("--ic_type", choices=("random", "galewsky"), default="random", help="Forecast initial condition type. Default: random")
    fc.add_argument("--seed", type=int, default=42, help="Forecast-time random seed. Default: 42")
    fc.add_argument("--channel", choices=tuple(roll.CHANNEL_TO_INDEX.keys()), default="vorticity", help="Field channel for spatial plots. Default: vorticity")
    fc.add_argument("--error_metric", choices=tuple(roll.ERROR_METRIC_TO_COLUMN.keys()), default="l2", help="Scalar forecast metric. Default: l2")
    fc.add_argument("--error_mode", choices=("signed", "abs", "squared"), default="signed", help="Pointwise error map mode. Default: signed")
    fc.add_argument("--summary_step", default="final", help="Step for final grid/spectra plots: final/latest or an integer. Default: final")
    fc.add_argument("--grid_cols", type=int, default=3, help="Maximum columns in spatial grids. Default: 3")
    fc.add_argument("--rollout_dir", default="./rollout_results", help="Directory for per-optimizer rollout outputs. Default: ./rollout_results")
    fc.add_argument("--outdir", default="./figures_forecast", help="Directory for final forecast figures. Default: ./figures_forecast")
    fc.add_argument("--device", default=None, help="Optional real-rollout device, e.g. cuda or cpu. Default: auto")
    fc.add_argument("--synthetic_demo", action="store_true", help="Generate artificial rollout data instead of loading real checkpoints. For tests only.")
    fc.set_defaults(func=run_forecast)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
