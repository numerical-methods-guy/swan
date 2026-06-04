"""
plots.py
========

All matplotlib figure-building functions for the SWAN visualize package.

This module contains only rendering logic: it receives clean data structures
from ``history`` and ``rollout`` and turns them into figures.  It does not
parse command-line arguments and it does not load data from disk itself.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from visualize import history as hist
from visualize import rollout as roll


# ---------------------------------------------------------------------------
# General helpers
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
# plot_history plots
# ---------------------------------------------------------------------------

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
# forecast scalar plots
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
    """Plot aggregate forecast error from metrics.csv."""
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
# forecast spatial grid plots
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
# forecast spectral plot
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


def plot_combined_spectra(
    snapshots: Sequence[roll.FieldSnapshot],
    output_freq: int,
    outdir: Path,
    spectra_method: str = "fft",
    sht=None,
) -> None:
    """Plot one 2x2 spectral figure containing truth and all optimizers.

    The layout intentionally mirrors the original SWAN forecast.py spectral
    figure: rotational kinetic energy, divergent kinetic energy, potential
    energy, and total energy.  The comparison version differs in that every
    optimizer is plotted on the same axes together with the same ground truth.
    """
    if spectra_method == "spherical":
        for snap in snapshots[1:]:
            if not np.allclose(snap.truth_fields, snapshots[0].truth_fields, rtol=1e-5, atol=1e-6):
                raise ValueError(
                    "Combined spectra require every rollout to use the same truth fields. "
                    f"{snap.label!r} differs from {snapshots[0].label!r}; rerun with the same "
                    "forecast seed/config or plot each rollout's individual spectra."
                )
        if sht is None:
            raise ValueError("spherical combined spectra require an SHT object.")
        truth_spectra = roll.compute_spherical_energy_spectra(snapshots[0].truth_fields, sht)
        pred_spectra = [
            (snap.label, roll.compute_spherical_energy_spectra(snap.prediction_fields, sht))
            for snap in snapshots
        ]
    elif spectra_method == "fft":
        truth_spectra = roll.compute_simple_energy_spectra(snapshots[0].truth_fields)
        pred_spectra = [
            (snap.label, roll.compute_simple_energy_spectra(snap.prediction_fields))
            for snap in snapshots
        ]
    else:
        raise ValueError("spectra_method must be 'spherical' or 'fft'")

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
