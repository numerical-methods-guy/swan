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
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
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


def display_metric_text(metric: str) -> str:
    """Convert metric column names into plot-friendly display text."""
    text = metric.replace("_mean", " mean").replace("_", " ")
    replacements = {
        "l1": "L1",
        "l2": "L2",
        "w11": "W11",
        "ml": "ML",
    }
    return " ".join(replacements.get(part.lower(), part) for part in text.split())


def title_case(text: str) -> str:
    """Apply lightweight title capitalization for generated plot titles."""
    small_words = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "vs."}
    words = text.split()
    titled: List[str] = []
    for i, word in enumerate(words):
        lower = word.lower()
        if i > 0 and lower in small_words:
            titled.append(lower)
        elif word.upper() in {"L1", "L2", "W11", "ML", "SGD"}:
            titled.append(word.upper())
        else:
            titled.append(word[:1].upper() + word[1:])
    return " ".join(titled)


def resource_axis_name(resource: str) -> str:
    """Human-readable resource axis name."""
    if resource == "step":
        return "global training step"
    if resource == "time":
        return "relative wall-clock time (s)"
    return resource


def history_line_style(index: int) -> Dict[str, object]:
    """Return a distinguishable line style for crowded optimizer history plots."""
    colors = plt.get_cmap("tab10").colors
    linestyles = ("-", "--", "-.", ":")
    markers = ("o", "s", "^", "D", "v", "P", "X", "*")
    color = colors[index % len(colors)]
    return {
        "color": color,
        "linestyle": linestyles[index % len(linestyles)],
        "linewidth": 2.0,
        "alpha": 0.9,
        "marker": markers[index % len(markers)],
        "markersize": 3.2,
        "markerfacecolor": "white",
        "markeredgecolor": color,
        "markeredgewidth": 0.9,
        "zorder": 10 + index,
    }


def history_marker_positions(values: Sequence[float]) -> List[int]:
    """Choose sparse marker positions while always including visible endpoints."""
    n_points = len(values)
    if n_points <= 0:
        return []
    if n_points <= 12:
        return [i for i, value in enumerate(values) if np.isfinite(value)]

    finite_indices = [i for i, value in enumerate(values) if np.isfinite(value)]
    if not finite_indices:
        return []

    stride = max(1, n_points // 10)
    positions = list(range(0, n_points, stride))
    positions = [i for i in positions if i in finite_indices]
    for endpoint in (finite_indices[0], finite_indices[-1]):
        if endpoint not in positions:
            positions.append(endpoint)
    return sorted(positions)


# ---------------------------------------------------------------------------
# plot_history plots
# ---------------------------------------------------------------------------

def plot_history_learning_curve(
    runs: Sequence[hist.RunScalars],
    stage: str,
    error_metric: str,
    resource: str,
    outdir: Path,
    yscale: str = "linear",
) -> None:
    """Plot metric value against training step or relative wall-clock time."""
    series_by_label = hist.prepare_learning_curve_data(runs, stage, error_metric)
    y_label = hist.display_metric_name(stage, error_metric)
    metric_name = hist.filename_metric_name(stage, error_metric)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, (label, series) in enumerate(series_by_label.items()):
        x = series.step if resource == "step" else series.relative_time_sec
        style = history_line_style(i)
        if stage == "training":
            style.pop("marker", None)
            style.pop("markersize", None)
            style.pop("markerfacecolor", None)
            style.pop("markeredgecolor", None)
            style.pop("markeredgewidth", None)
        ax.plot(x, series.value, label=label, **style)

    ax.set_xlabel(resource_axis_name(resource))
    ax.set_ylabel(y_label)
    ax.set_title(title_case(f"{y_label} vs. {resource_axis_name(resource)}"))
    ax.set_yscale(yscale)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, framealpha=0.92)
    fig.tight_layout(rect=(0, 0, 0.82, 1))

    output = outdir / f"learning_curve_{stage}_{resource}_{metric_name}.png"
    save_figure(fig, output)


def plot_history_hitting_curve(
    runs: Sequence[hist.RunScalars],
    stage: str,
    error_metric: str,
    resource: str,
    outdir: Path,
    threshold_xscale: str = "linear",
) -> None:
    """Plot first-hitting resource as a function of target error threshold."""
    curves = hist.prepare_hitting_curve_data(runs, stage, error_metric, resource)
    y_label = "first hitting step" if resource == "step" else "first hitting time (s)"
    metric_label = hist.display_metric_name(stage, error_metric)
    metric_name = hist.filename_metric_name(stage, error_metric)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, (label, (thresholds, hits)) in enumerate(curves.items()):
        style = history_line_style(i)
        style["markevery"] = history_marker_positions(hits)
        ax.plot(thresholds, hits, label=label, **style)

    ax.set_xlabel(f"target {metric_label} threshold")
    ax.set_ylabel(y_label)
    ax.set_title(title_case(f"first hitting {resource} vs. target {metric_label}"))
    ax.set_xscale(threshold_xscale)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, framealpha=0.92)
    # Thresholds are generated from loose to strict.  Inverting the x-axis makes
    # the plot read naturally: moving right asks for a stricter/lower error.
    ax.invert_xaxis()
    fig.tight_layout()

    output = outdir / f"hitting_curve_{stage}_{resource}_{metric_name}.png"
    save_figure(fig, output)


# ---------------------------------------------------------------------------
# forecast scalar plots
# ---------------------------------------------------------------------------

def plot_forecast_error_curve(
    rollout_runs: Sequence[roll.RolloutRun],
    error_metric: str,
    outdir: Path,
    yscale: str = "linear",
) -> None:
    """Plot forecast error history versus autoregressive rollout step."""
    column = roll.metric_column(error_metric)
    display_column = display_metric_text(column)
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
    ax.set_ylabel(display_column)
    ax.set_title(title_case(f"forecast {display_column} over rollout"))
    ax.set_yscale(yscale)
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
    display_column = display_metric_text(column)
    labels = [run.label for run in rollout_runs]
    values = np.array([run.metrics.get(column, np.nan) for run in rollout_runs], dtype=float)

    fig, ax = plt.subplots(figsize=(max(8.5, 1.25 * len(labels)), 5.4))
    bars = ax.bar(labels, values, color=_bar_colors(len(labels)), edgecolor="black", linewidth=0.7, alpha=0.88)
    ax.set_ylabel(f"{display_column} (lower is better)")
    ax.set_title(title_case(f"aggregate forecast accuracy: {display_column}"), fontweight="bold")
    _style_bar_axes(ax)
    _annotate_bars(ax, bars)
    fig.tight_layout()
    save_figure(fig, outdir / f"forecast_accuracy_bar_{error_metric}.png")


def plot_forecast_runtime_ratio_bar(rollout_runs: Sequence[roll.RolloutRun], outdir: Path) -> None:
    """Plot ML-vs-non-ML solver runtime ratio from metrics.csv."""
    labels = [run.label for run in rollout_runs]
    ml_times = np.array([run.metrics.get("ml_time_mean", np.nan) for run in rollout_runs], dtype=float)
    solver_times = np.array([run.metrics.get("solver_time_mean", np.nan) for run in rollout_runs], dtype=float)
    # Runtime ratio is easier to read when the non-ML solver is often faster:
    # 1.0 means equal runtime, values above 1.0 mean the ML rollout is slower,
    # and values below 1.0 mean the ML rollout is faster.
    values = np.divide(
        ml_times,
        solver_times,
        out=np.full_like(ml_times, np.nan, dtype=float),
        where=np.isfinite(solver_times) & (solver_times > 0),
    )

    fig, ax = plt.subplots(figsize=(max(8.5, 1.25 * len(labels)), 5.4))
    bars = ax.bar(labels, values, color=_bar_colors(len(labels)), edgecolor="black", linewidth=0.7, alpha=0.88)
    ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.2, alpha=0.8, label="equal runtime")
    ax.set_ylabel("Runtime ratio")
    ax.set_title("Forecast Rollout Runtime vs. Non-ML Solver (Lower Is Better)", fontweight="bold")
    _style_bar_axes(ax)
    finite_heights = [bar.get_height() for bar in bars if np.isfinite(bar.get_height())]
    if finite_heights:
        ymax = max(max(finite_heights), 1.0)
        pad = 0.015 * ymax if ymax != 0 else 0.02
        for bar in bars:
            height = bar.get_height()
            if np.isfinite(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + pad,
                    f"{height:.3g}x",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax.set_ylim(top=max(ax.get_ylim()[1], ymax + 5 * pad))
    ax.legend(frameon=False)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    save_figure(fig, outdir / "forecast_runtime_ratio_bar.png")


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
    channel_label = display_metric_text(channel)
    title = f"Final Prediction Comparison ({channel_label}, Step {snapshots[0].step}; Saved Every {output_freq} Step(s))"
    output = outdir / f"forecast_prediction_grid_{channel}_final.png"
    plot_image_panels(
        panels,
        title,
        grid_cols,
        output,
        cmap="twilight_shifted",
        symmetric=True,
        colorbar_label=f"{channel_label} value (shared color scale)",
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
    channel_label = display_metric_text(channel)
    error_mode_label = display_metric_text(error_mode)
    title = f"Final Pointwise Error ({channel_label}, {error_mode_label}, Step {snapshots[0].step}; Saved Every {output_freq} Step(s))"
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
        # Spherical combined spectra are only meaningful when every optimizer
        # was rolled out against the same truth trajectory.  Comparing against
        # different random ICs would mix optimizer error with different target
        # spectra, so fail loudly instead of drawing a misleading overlay.
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
        # Keep the FFT branch available only when explicitly requested or when
        # synthetic demo data cannot provide forecast.py's spherical transform.
        # Real comparison runs should normally use the spherical branch above.
        truth_spectra = roll.compute_simple_energy_spectra(snapshots[0].truth_fields)
        pred_spectra = [
            (snap.label, roll.compute_simple_energy_spectra(snap.prediction_fields))
            for snap in snapshots
        ]
    else:
        raise ValueError("spectra_method must be 'spherical' or 'fft'")

    titles = [
        "Rotational Kinetic Energy",
        "Divergent Kinetic Energy",
        "Potential Energy",
        "Total Energy",
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
            ref_line_3, = ax.loglog(k_ref, ref_3, color="0.35", linestyle=":", linewidth=1.5, alpha=0.75, label=r"Scaled $k^{-3}$")
            ref_line_53, = ax.loglog(k_ref, ref_53, color="0.35", linestyle="-.", linewidth=1.5, alpha=0.75, label=r"Scaled $k^{-5/3}$")
            if ax is axes.ravel()[0]:
                legend_handles.extend([ref_line_3, ref_line_53])
                legend_labels.extend([r"Scaled $k^{-3}$", r"Scaled $k^{-5/3}$"])

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Wavenumber $l$")
        ax.set_ylabel("Power spectrum")
        ax.grid(True, which="both", alpha=0.3)
        if len(k) > 0:
            ax.set_xlim(left=max(1, k[0]), right=k[-1])

    fig.suptitle(
        f"Final Rollout Spectra Comparison (Step {snapshots[0].step}; Saved Every {output_freq} Step(s))",
        fontsize=14,
        fontweight="bold",
    )
    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=min(4, len(legend_labels)), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    save_figure(fig, outdir / "forecast_spectra_final.png")


# ---------------------------------------------------------------------------
# Rollout animation
# ---------------------------------------------------------------------------

def make_rollout_animation(
    frames: Sequence[Sequence[roll.FieldSnapshot]],
    channel: str,
    fps: int = 8,
    output: Optional[Union[str, Path]] = None,
    show_error: bool = False,
) -> None:
    """Create a multi-optimizer comparison animation from pre-computed rollouts.

    Each frame is a timestep.  The top row shows ground truth alongside every
    optimizer's prediction on a shared colorscale.  When ``show_error=True`` a
    second row shows the signed pointwise error (prediction − truth) for each
    optimizer on a shared error colorscale.

    Parameters
    ----------
    frames:
        Output of ``rollout.load_animation_frames`` — a list of steps, each
        step being a list of ``FieldSnapshot`` objects (one per optimizer).
    channel:
        Field channel name to animate (``'h'``, ``'vorticity'``, or
        ``'divergence'``).
    fps:
        Frames per second.
    output:
        Destination file path (.gif or .mp4).  Passing ``None`` shows the
        animation interactively.
    show_error:
        When ``True``, adds a second row of signed error panels.
    """
    if not frames:
        raise ValueError("frames is empty — nothing to animate.")

    ch = roll.CHANNEL_TO_INDEX[channel]
    n_opts = len(frames[0])
    labels = [snap.label for snap in frames[0]]

    # Pre-compute global color limits across all steps and optimizers so the
    # scale stays fixed throughout the animation.
    all_field_values = np.concatenate([
        values
        for step_snaps in frames
        for snap in step_snaps
        for values in (snap.truth_fields[ch].ravel(), snap.prediction_fields[ch].ravel())
    ])
    vmax_field = float(np.nanpercentile(np.abs(all_field_values), 98))
    vmin_field, vmax_field = -vmax_field, vmax_field

    err_abs_max = 0.0
    if show_error:
        all_errors = np.concatenate([
            (snap.prediction_fields[ch] - snap.truth_fields[ch]).ravel()
            for step_snaps in frames
            for snap in step_snaps
        ])
        err_abs_max = float(max(
            abs(np.nanpercentile(all_errors, 2)),
            abs(np.nanpercentile(all_errors, 98)),
        ))

    # Layout: top row = Truth + N optimizer predictions,
    #         bottom row (optional) = N error maps.
    n_rows = 2 if show_error else 1
    n_cols = 1 + n_opts  # truth + one column per optimizer

    panel_w, panel_h = 4.2, 3.6
    fig_w = panel_w * n_cols
    fig_h = panel_h * n_rows + 0.9  # extra for title

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0d1117")

    axes = fig.subplots(n_rows, n_cols, squeeze=False)
    for row_axes in axes:
        for ax in row_axes:
            ax.axis("off")
            ax.set_facecolor("#0d1117")

    # Top row: truth + predictions.
    im_truth = axes[0][0].imshow(
        np.zeros_like(frames[0][0].truth_fields[ch]),
        cmap="twilight_shifted", vmin=vmin_field, vmax=vmax_field,
        origin="upper", aspect="auto",
    )
    axes[0][0].set_title("Ground Truth", color="white", fontsize=11, fontweight="bold", pad=6)
    _dark_colorbar(fig, im_truth, axes[0][0])

    im_preds = []
    for col, label in enumerate(labels):
        im = axes[0][col + 1].imshow(
            np.zeros_like(frames[0][col].prediction_fields[ch]),
            cmap="twilight_shifted", vmin=vmin_field, vmax=vmax_field,
            origin="upper", aspect="auto",
        )
        axes[0][col + 1].set_title(label, color="white", fontsize=11, fontweight="bold", pad=6)
        _dark_colorbar(fig, im, axes[0][col + 1])
        im_preds.append(im)

    # Bottom row: error maps (optional).
    im_errors = []
    if show_error:
        for col, label in enumerate(labels):
            im = axes[1][col + 1].imshow(
                np.zeros_like(frames[0][col].prediction_fields[ch]),
                cmap="RdBu_r", vmin=-err_abs_max, vmax=err_abs_max,
                origin="upper", aspect="auto",
            )
            axes[1][col + 1].set_title(
                f"{label} error", color="white", fontsize=10, fontweight="bold", pad=6
            )
            _dark_colorbar(fig, im, axes[1][col + 1])
            im_errors.append(im)
        # Hide the unused bottom-left cell (under the truth panel).
        axes[1][0].set_visible(False)

    # Use a figure-level title rather than per-axes text so the shared title is
    # retained by Pillow/FFMpeg writers across every rendered animation frame.
    fig.suptitle(f"Rollout Comparison: {display_metric_text(channel)}", color="white", fontsize=15, fontweight="bold")
    step_label = fig.text(
        0.5, 0.945,
        "",
        ha="center", va="top",
        color="white", fontsize=12, fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0.0, 1, 0.92])

    def update(frame_index: int):
        step_snaps = frames[frame_index]
        step = step_snaps[0].step
        truth = step_snaps[0].truth_fields[ch]
        im_truth.set_data(truth)
        for im, snap in zip(im_preds, step_snaps):
            im.set_data(snap.prediction_fields[ch])
        for im, snap in zip(im_errors, step_snaps):
            im.set_data(snap.prediction_fields[ch] - snap.truth_fields[ch])
        step_label.set_text(f"Rollout Step {step}")
        return [im_truth, *im_preds, *im_errors, step_label]

    ani = manimation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        # Full redraws are a little slower but more reliable for figure-level
        # titles, colorbars, and text annotations in saved GIF/MP4 outputs.
        blit=False,
    )

    if output is None:
        plt.show()
    else:
        output = Path(output)
        ext = output.suffix.lower()
        if ext == ".gif":
            writer = manimation.PillowWriter(fps=fps)
        else:
            writer = manimation.FFMpegWriter(fps=fps, bitrate=1800)
        print(f"Saving animation to {output} ...")
        ani.save(str(output), writer=writer, dpi=120)
        print(f"Saved: {output}")

    plt.close(fig)


def make_spectral_image_animation(
    frames: Sequence[Tuple[int, Sequence[Tuple[str, Path]]]],
    fps: int = 8,
    output: Optional[Union[str, Path]] = None,
) -> None:
    """Animate saved per-optimizer spectral-analysis PNGs over rollout steps."""
    if not frames:
        raise ValueError("spectral frames are empty - nothing to animate.")

    first_step, first_panels = frames[0]
    n_panels = len(first_panels)
    n_cols = min(3, n_panels)
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.2 * n_rows + 0.7), squeeze=False)
    fig.patch.set_facecolor("#111827")

    images = []
    for ax, (label, path) in zip(axes.ravel(), first_panels):
        img = plt.imread(path)
        im = ax.imshow(img)
        ax.set_title(label, color="white", fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")
        ax.set_facecolor("#111827")
        images.append(im)
    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")
        ax.set_facecolor("#111827")

    # This mode displays the exact spectra PNGs emitted by forecast.py for each
    # optimizer.  It intentionally treats those PNGs as image panels rather than
    # recomputing spectra, preserving the original per-optimizer diagnostics.
    fig.suptitle("Spectral Analysis Comparison", color="white", fontsize=15, fontweight="bold")
    step_label = fig.text(
        0.5, 0.955,
        f"Rollout Step {first_step}",
        ha="center", va="top",
        color="white", fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.0, 1, 0.93])

    def update(frame_index: int):
        step, panels = frames[frame_index]
        for im, (_, path) in zip(images, panels):
            im.set_data(plt.imread(path))
        step_label.set_text(f"Rollout Step {step}")
        return [*images, step_label]

    ani = manimation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    _save_or_show_animation(fig, ani, output, fps=fps, dpi=110)


def make_combined_spectral_animation(
    frames: Sequence[Sequence[roll.FieldSnapshot]],
    fps: int = 8,
    output: Optional[Union[str, Path]] = None,
    config_path: Union[str, Path] = "config_paradis.yaml",
) -> None:
    """Animate all optimizer spherical-harmonic spectra in one comparison figure."""
    if not frames:
        raise ValueError("frames is empty - nothing to animate.")

    # Build the same SHT object used by forecast.py and reuse it for every frame.
    # The combined animation should not silently fall back to FFT: if the SWAN
    # spherical-harmonic diagnostic cannot be constructed, the command should
    # fail clearly so the plotted spectra are not mistaken for forecast.py output.
    sht = roll.build_spherical_sht(config_path, frames[0][0].truth_fields.shape)
    spectral_frames = []
    y_values = []
    for step_snaps in frames:
        # Precompute spectra and global positive y-limits before constructing
        # the animation.  Fixed log-scale limits keep optimizer differences from
        # being hidden by per-frame autoscaling.
        truth_spectra = roll.compute_spherical_energy_spectra(step_snaps[0].truth_fields, sht)
        pred_spectra = [
            (snap.label, roll.compute_spherical_energy_spectra(snap.prediction_fields, sht))
            for snap in step_snaps
        ]
        spectral_frames.append((step_snaps[0].step, truth_spectra, pred_spectra))
        for spectra in [truth_spectra, *[spec for _, spec in pred_spectra]]:
            for key in ("rotational", "divergent", "potential", "total"):
                values = np.asarray(spectra[key])
                y_values.extend(values[np.isfinite(values) & (values > 0)].tolist())

    spec_keys = [
        ("rotational", "Rotational"),
        ("divergent", "Divergent"),
        ("potential", "Potential"),
        ("total", "Total"),
    ]
    labels = [snap.label for snap in frames[0]]
    colors = _bar_colors(max(1, len(labels)))
    y_min = max(min(y_values) * 0.6, 1e-14) if y_values else 1e-14
    y_max = max(y_values) * 1.6 if y_values else 1.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Combined Rollout Spectra Comparison", fontsize=15, fontweight="bold")
    step_label = fig.text(0.5, 0.935, "", ha="center", va="top", fontsize=11, fontweight="bold")

    line_sets = []
    for ax, (key, title) in zip(axes.ravel(), spec_keys):
        truth_line, = ax.loglog([], [], color="black", linewidth=2.2, linestyle="--", label="Ground Truth")
        pred_lines = []
        for i, label in enumerate(labels):
            line, = ax.loglog([], [], color=colors[i % len(colors)], linewidth=1.8, label=label)
            pred_lines.append((label, line))
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Wavenumber $l$")
        ax.set_ylabel("Power spectrum")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, which="both", alpha=0.3)
        line_sets.append((key, truth_line, pred_lines))

    handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=min(4, len(legend_labels)), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.91))

    def update(frame_index: int):
        step, truth_spectra, pred_spectra = spectral_frames[frame_index]
        pred_by_label = dict(pred_spectra)
        artists = [step_label]
        step_label.set_text(f"Rollout Step {step}")
        for ax, (key, truth_line, pred_lines) in zip(axes.ravel(), line_sets):
            k = np.asarray(truth_spectra["wavenumbers"])[1:]
            truth_values = np.asarray(truth_spectra[key])[1:]
            truth_line.set_data(k, truth_values)
            artists.append(truth_line)
            for label, line in pred_lines:
                spectra = pred_by_label[label]
                line.set_data(np.asarray(spectra["wavenumbers"])[1:], np.asarray(spectra[key])[1:])
                artists.append(line)
            if len(k) > 0:
                ax.set_xlim(left=max(1, k[0]), right=k[-1])
        return artists

    ani = manimation.FuncAnimation(
        fig,
        update,
        frames=len(spectral_frames),
        interval=1000 / fps,
        blit=False,
    )
    _save_or_show_animation(fig, ani, output, fps=fps, dpi=120)


def _save_or_show_animation(fig: plt.Figure, ani, output: Optional[Union[str, Path]], fps: int, dpi: int) -> None:
    """Save an animation to GIF/MP4 or show it interactively."""
    if output is None:
        plt.show()
    else:
        output = Path(output)
        ext = output.suffix.lower()
        if ext == ".gif":
            writer = manimation.PillowWriter(fps=fps)
        else:
            writer = manimation.FFMpegWriter(fps=fps, bitrate=1800)
        print(f"Saving animation to {output} ...")
        ani.save(str(output), writer=writer, dpi=dpi)
        print(f"Saved: {output}")
    plt.close(fig)


def _dark_colorbar(fig: plt.Figure, im, ax: plt.Axes) -> None:
    """Attach a compact horizontal colorbar with white ticks to a dark panel."""
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, orientation="horizontal")
    cbar.ax.tick_params(colors="white", labelsize=7)
    cbar.outline.set_edgecolor("white")
