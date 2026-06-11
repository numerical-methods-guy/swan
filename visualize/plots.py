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
from visualize import mpl_style  # noqa: F401  # apply M2PI report typography
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterMathtext, LogLocator, MaxNLocator, NullFormatter, ScalarFormatter

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


def axis_label_text(text: str) -> str:
    """Format generated axis labels with title-style capitalization."""
    return title_case(text) if text else text


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


OPTIMIZER_STYLES: Dict[str, Dict[str, object]] = {
    "adam": {"color": "#4E79A7", "linestyle": "-", "marker": "o", "linewidth": 1.9, "alpha": 0.88},
    "adamw": {"color": "#59A14F", "linestyle": "--", "marker": "s", "linewidth": 1.9, "alpha": 0.88},
    "mud": {"color": "#F28E2B", "linestyle": "-", "marker": "^", "linewidth": 2.25, "alpha": 0.96},
    "muon": {"color": "#E15759", "linestyle": "-", "marker": "D", "linewidth": 2.25, "alpha": 0.96},
    "sgd": {"color": "#9C755F", "linestyle": ":", "marker": "v", "linewidth": 1.75, "alpha": 0.78},
    "gaussnewton": {"color": "#B07AA1", "linestyle": "-.", "marker": "x", "linewidth": 1.75, "alpha": 0.78},
}


def _style_key(label: str) -> str:
    """Normalize optimizer display labels to stable style keys."""
    return "".join(ch for ch in label.lower() if ch.isalnum())


def optimizer_line_style(label: str, index: int = 0, include_marker: bool = True) -> Dict[str, object]:
    """Return a consistent line style for an optimizer label across plots."""
    fallback_colors = plt.get_cmap("tab10").colors
    base = dict(OPTIMIZER_STYLES.get(_style_key(label), {}))
    if not base:
        base = {
            "color": fallback_colors[index % len(fallback_colors)],
            "linestyle": ("-", "--", "-.", ":")[index % 4],
            "marker": ("o", "s", "^", "D", "v", "P", "X", "*")[index % 8],
            "linewidth": 1.9,
            "alpha": 0.9,
        }
    if include_marker:
        color = base["color"]
        base.update({
            "markersize": 3.0,
            "markerfacecolor": "white",
            "markeredgecolor": color,
            "markeredgewidth": 0.85,
        })
    else:
        base.pop("marker", None)
    return base


def history_line_style(label: str, index: int) -> Dict[str, object]:
    """Return a distinguishable line style for crowded optimizer history plots."""
    style = optimizer_line_style(label, index=index, include_marker=True)
    style["zorder"] = 10 + index
    return style


def apply_log_axis_style(ax: plt.Axes, axis: str) -> None:
    """Add readable minor ticks and faint gridlines to a log-scaled axis."""
    major_locator = LogLocator(base=10.0)
    major_formatter = LogFormatterMathtext(base=10.0)
    minor_locator = LogLocator(base=10.0, subs=tuple(float(i) for i in range(2, 10)))
    formatter = FuncFormatter(_log_minor_tick_label)
    if axis == "x":
        _expand_log_axis_limits(ax, "x")
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_formatter)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.xaxis.set_minor_formatter(formatter)
        ax.tick_params(axis="x", which="minor", labelsize=7, pad=2)
        ax.grid(True, axis="x", which="minor", alpha=0.12, linewidth=0.55)
    elif axis == "y":
        _expand_log_axis_limits(ax, "y")
        ax.yaxis.set_major_locator(major_locator)
        ax.yaxis.set_major_formatter(major_formatter)
        ax.yaxis.set_minor_locator(minor_locator)
        ax.yaxis.set_minor_formatter(formatter)
        ax.tick_params(axis="y", which="minor", labelsize=7, pad=2)
        ax.grid(True, axis="y", which="minor", alpha=0.12, linewidth=0.55)


def apply_symlog_axis_style(ax: plt.Axes, linthresh: float = 10.0) -> None:
    """Add readable signed log ticks, including the linear region near zero."""
    ax.yaxis.set_major_locator(FixedLocator(_signed_percent_major_ticks(ax, linthresh)))
    ax.yaxis.set_major_formatter(FuncFormatter(_signed_percent_tick_label))
    ax.yaxis.set_minor_locator(FixedLocator(_signed_percent_minor_ticks(ax, linthresh)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, axis="y", which="minor", alpha=0.12, linewidth=0.55)


def apply_percent_difference_y_axis(ax: plt.Axes, y_limit: float, linthresh: float = 10.0) -> None:
    """Style signed percent-difference axes with readable tick density."""
    ax.set_ylim(-y_limit, y_limit)
    ax.set_yscale("symlog", linthresh=linthresh)
    apply_symlog_axis_style(ax, linthresh=linthresh)
    ax.tick_params(axis="y", which="minor", length=3, width=0.6)


def _signed_percent_major_ticks(ax: plt.Axes, linthresh: float) -> List[float]:
    """Return symmetric major ticks for signed percentage symlog axes."""
    low, high = ax.get_ylim()
    limit = max(abs(float(low)), abs(float(high)), linthresh)
    positive_ticks = [5.0, linthresh]
    decade = linthresh * 10.0
    while decade <= limit * 1.0001:
        positive_ticks.append(decade)
        decade *= 10.0
    ticks = [-tick for tick in reversed(positive_ticks) if tick <= limit * 1.0001]
    ticks.extend([0.0, *[tick for tick in positive_ticks if tick <= limit * 1.0001]])
    return ticks


def _signed_percent_minor_ticks(ax: plt.Axes, linthresh: float) -> List[float]:
    """Return dense minor ticks for both sides of a signed percentage symlog axis."""
    low, high = ax.get_ylim()
    limit = max(abs(float(low)), abs(float(high)), linthresh)
    minor_abs = [value for value in np.arange(1.0, linthresh, 1.0) if not math.isclose(value, 5.0)]
    decade = linthresh
    while decade <= limit * 1.0001:
        for multiplier in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0):
            value = decade * multiplier
            if value <= limit * 1.0001:
                minor_abs.append(value)
        decade *= 10.0
    return [-tick for tick in reversed(minor_abs)] + minor_abs


def _signed_percent_tick_label(value: float, _pos: int) -> str:
    """Format signed percent ticks without scientific notation near zero."""
    if math.isclose(value, 0.0, abs_tol=1e-12):
        return "0"
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    if abs_value < 1000:
        return f"{sign}{abs_value:g}"
    exponent = int(round(math.log10(abs_value)))
    if math.isclose(abs_value, 10 ** exponent, rel_tol=1e-8):
        return rf"${sign}10^{{{exponent}}}$"
    return f"{sign}{abs_value:g}"


def _log_minor_tick_label(value: float, _pos: int) -> str:
    """Label only selected minor ticks inside each log decade."""
    if value <= 0 or not np.isfinite(value):
        return ""
    exponent = math.floor(math.log10(value))
    mantissa = value / (10 ** exponent)
    for reference in (2.0, 3.0, 5.0, 8.0):
        if math.isclose(mantissa, reference, rel_tol=1e-4):
            return rf"${int(reference)}{{\times}}10^{{{exponent}}}$"
    return ""


def _expand_log_axis_limits(ax: plt.Axes, axis: str) -> None:
    """Expand narrow log axes to nearby readable reference ticks."""
    get_limits = ax.get_xlim if axis == "x" else ax.get_ylim
    set_limits = ax.set_xlim if axis == "x" else ax.set_ylim
    low, high = get_limits()
    if low <= 0 or high <= 0 or not np.isfinite(low) or not np.isfinite(high):
        return
    if high < low:
        low, high = high, low
        inverted = True
    else:
        inverted = False

    lower = _log_reference_floor(low)
    upper = _log_reference_ceil(high)
    if inverted:
        set_limits(upper, lower)
    else:
        set_limits(lower, upper)


def _log_reference_floor(value: float) -> float:
    exponent = math.floor(math.log10(value))
    scaled = value / (10 ** exponent)
    for reference in (10.0, 8.0, 5.0, 2.0, 1.0):
        if scaled >= reference:
            return reference * (10 ** exponent)
    return 10 ** (exponent - 1)


def _log_reference_ceil(value: float) -> float:
    exponent = math.floor(math.log10(value))
    scaled = value / (10 ** exponent)
    for reference in (1.0, 2.0, 5.0, 8.0, 10.0):
        if scaled <= reference:
            return reference * (10 ** exponent)
    return 10 ** (exponent + 1)


def style_plot_legend(ax: plt.Axes, outside: bool = False):
    """Apply a consistent legend style for optimizer comparison plots."""
    if outside:
        return ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            framealpha=0.94,
            borderaxespad=0.0,
        )
    return ax.legend(loc="best", frameon=True, framealpha=0.9)


def padded_axis_limits(values: np.ndarray, scale: str, pad_fraction: float = 0.025) -> Tuple[float, float]:
    """Return lightly padded min/max limits for linear or log axes."""
    values = values[np.isfinite(values)]
    if scale == "log":
        values = values[values > 0]
    if values.size == 0:
        return (0.0, 1.0)
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if low == high:
        pad = abs(low) * pad_fraction if low != 0 else pad_fraction
        return low - pad, high + pad
    if scale == "log":
        log_low = math.log10(low)
        log_high = math.log10(high)
        pad = max((log_high - log_low) * pad_fraction, 1e-3)
        return 10 ** (log_low - pad), 10 ** (log_high + pad)
    pad = (high - low) * pad_fraction
    return low - pad, high + pad


def log_reference_axis_limits(values: np.ndarray, pad_fraction: float = 0.025) -> Tuple[float, float]:
    """Return log limits snapped to nearby reference ticks for readable axes."""
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return (1e-3, 1e-2)
    low = _log_reference_floor(float(np.nanmin(values)))
    high = _log_reference_ceil(float(np.nanmax(values)))
    log_low = math.log10(low)
    log_high = math.log10(high)
    pad = max((log_high - log_low) * pad_fraction, 1e-3)
    return 10 ** (log_low - pad), 10 ** (log_high + pad)


def robust_log_axis_limits(values: Sequence[float], pad_fraction: float = 0.04) -> Tuple[float, float]:
    """Return stable log limits that ignore tiny tails and rare spikes."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size == 0:
        return (1e-14, 1.0)
    if finite.size < 20:
        return log_reference_axis_limits(finite, pad_fraction=pad_fraction)

    log_values = np.log10(finite)
    low = 10 ** float(np.nanpercentile(log_values, 1.0))
    high = 10 ** float(np.nanpercentile(log_values, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
    return log_reference_axis_limits(np.asarray([low, high]), pad_fraction=pad_fraction)


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
    effective_yscale = "linear" if stage == "training" else yscale

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, (label, series) in enumerate(series_by_label.items()):
        x = series.step if resource == "step" else series.relative_time_sec
        style = history_line_style(label, i)
        style["markevery"] = history_marker_positions(series.value)
        ax.plot(x, series.value, label=label, **style)

    ax.set_xlabel(axis_label_text(resource_axis_name(resource)))
    ax.set_ylabel(axis_label_text(y_label))
    ax.set_title(title_case(f"{y_label} vs. {resource_axis_name(resource)}"))
    ax.set_yscale(effective_yscale)
    if effective_yscale == "log":
        apply_log_axis_style(ax, "y")
    ax.grid(True, axis="y", which="major", alpha=0.3)
    style_plot_legend(ax)
    fig.tight_layout(rect=(0, 0, 1, 1))

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
    effective_threshold_xscale = "linear" if stage == "training" else threshold_xscale

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    threshold_arrays = []
    for i, (label, (thresholds, hits)) in enumerate(curves.items()):
        threshold_arrays.append(np.asarray(thresholds, dtype=float))
        style = history_line_style(label, i)
        style["markevery"] = history_marker_positions(hits)
        ax.plot(thresholds, hits, label=label, **style)

    ax.set_xlabel(axis_label_text(f"target {metric_label} threshold"))
    ax.set_ylabel(axis_label_text(y_label))
    ax.set_title(title_case(f"first hitting {resource} vs. target {metric_label}"))
    ax.set_xscale(effective_threshold_xscale)
    if effective_threshold_xscale == "log":
        apply_log_axis_style(ax, "x")
    ax.grid(True, axis="y", which="major", alpha=0.3)
    style_plot_legend(ax)
    # Thresholds are generated from loose to strict.  Inverting the x-axis makes
    # the plot read naturally: moving right asks for a stricter/lower error.
    if threshold_arrays:
        all_thresholds = np.concatenate(threshold_arrays)
        all_thresholds = all_thresholds[np.isfinite(all_thresholds)]
        if effective_threshold_xscale == "log":
            all_thresholds = all_thresholds[all_thresholds > 0]
        if all_thresholds.size:
            if effective_threshold_xscale == "log":
                low, high = log_reference_axis_limits(all_thresholds)
            else:
                low, high = padded_axis_limits(all_thresholds, effective_threshold_xscale)
            ax.set_xlim(high, low)
    else:
        ax.invert_xaxis()
    fig.tight_layout(rect=(0, 0, 1, 1))

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

    for i, run in enumerate(rollout_runs):
        if column not in run.per_step.columns:
            raise KeyError(f"{column} not found in {run.rollout_dir / 'per_step_metrics.csv'}")
        grouped = run.per_step.groupby("step")[column]
        mean = grouped.mean()
        std = grouped.std().fillna(0.0)
        steps = mean.index.to_numpy(dtype=float)
        values = mean.to_numpy(dtype=float)
        style = optimizer_line_style(run.label, index=i, include_marker=True)
        ax.plot(steps, values, label=run.label, **style)
        if len(rollout_runs) <= 6 and std.max() > 0:
            ax.fill_between(steps, values - std.to_numpy(), values + std.to_numpy(), alpha=0.12)

    ax.set_xlabel("Autoregressive Rollout Step")
    ax.set_ylabel(axis_label_text(display_column))
    title = f"Forecast {display_metric_text(error_metric)} Over Rollout: All Channels"
    subtitle = "Spatial and channel mean per step; rollout initial condition only; no initial-condition averaging"
    output_name = f"forecast_error_curve_{error_metric}.png"
    ax.set_title(f"{title}\n{subtitle}", fontweight="bold")
    ax.set_yscale(yscale)
    if yscale == "log":
        apply_log_axis_style(ax, "y")
    ax.grid(True, which="major", alpha=0.3)
    style_plot_legend(ax)
    fig.tight_layout(rect=(0, 0, 1, 1))
    save_figure(fig, outdir / output_name)


def _bar_colors(n: int):
    """Return a visually distinct but restrained color list for bar charts."""
    cmap = plt.get_cmap("tab20" if n > 10 else "tab10")
    return [cmap(i % cmap.N) for i in range(n)]


def optimizer_bar_colors(labels: Sequence[str]) -> List[object]:
    """Return bar colors that match the optimizer line-color convention."""
    return [optimizer_line_style(label, index=i, include_marker=False)["color"] for i, label in enumerate(labels)]


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


def plot_forecast_accuracy_bar(
    rollout_runs: Sequence[roll.RolloutRun],
    error_metric: str,
    outdir: Path,
) -> None:
    """Plot aggregate forecast error from metrics.csv."""
    column = roll.metric_mean_column(error_metric)
    display_column = display_metric_text(column)
    labels = [run.label for run in rollout_runs]
    values = np.array([run.metrics.get(column, np.nan) for run in rollout_runs], dtype=float)

    fig, ax = plt.subplots(figsize=(max(8.5, 1.25 * len(labels)), 5.4))
    bars = ax.bar(labels, values, color=optimizer_bar_colors(labels), edgecolor="black", linewidth=0.7, alpha=0.88)
    ax.set_ylabel(title_case(f"{display_column} (lower is better)"))
    title = f"Aggregate Forecast {display_metric_text(error_metric)}: All Channels"
    subtitle = "Temporal mean over rollout steps of spatial and channel mean error; rollout initial condition only"
    output_name = f"forecast_accuracy_bar_{error_metric}.png"
    ax.set_title(f"{title}\n{subtitle}", fontweight="bold")
    _style_bar_axes(ax)
    _annotate_bars(ax, bars)
    fig.tight_layout()
    save_figure(fig, outdir / output_name)


def plot_skill_horizon_vs_gamma(
    rollout_runs: Sequence[roll.RolloutRun],
    gammas: Optional[Sequence[float]],
    outdir: Path,
) -> None:
    """Plot forecast skill horizon as a function of relative-error threshold."""
    curves = roll.compute_skill_horizon_curves(rollout_runs, gammas)
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.set_facecolor("#fbfbfd")

    has_uncrossed = False
    for i, curve in enumerate(curves):
        style = optimizer_line_style(curve.label, index=i, include_marker=True)
        style["linewidth"] = max(float(style.get("linewidth", 1.9)), 2.1)
        style["markersize"] = 4.0
        style["markevery"] = history_marker_positions(curve.gammas)
        ax.plot(curve.gammas, curve.horizons, label=curve.label, **style)
        uncrossed = ~curve.crossed
        if np.any(uncrossed):
            has_uncrossed = True
            marker = style.get("marker", "o")
            color = style.get("color", "black")
            ax.scatter(
                curve.gammas[uncrossed],
                curve.horizons[uncrossed],
                marker=marker,
                facecolors="none",
                edgecolors=color,
                linewidths=1.4,
                s=42,
                zorder=5,
            )

    all_gammas = np.concatenate([curve.gammas for curve in curves])
    all_horizons = np.concatenate([curve.horizons for curve in curves])
    if all_gammas.size:
        x_pad = max(0.02, 0.025 * float(np.nanmax(all_gammas) - np.nanmin(all_gammas)))
        ax.set_xlim(float(np.nanmin(all_gammas)) - x_pad, float(np.nanmax(all_gammas)) + x_pad)
    if all_horizons.size:
        y_high = float(np.nanmax(all_horizons))
        ax.set_ylim(bottom=0.0, top=max(1.0, y_high * 1.06))

    ax.set_xlabel("Allowed relative error threshold γ")
    ax.set_ylabel("Reliable rollout length before error exceeds threshold [steps]")
    ax.set_title("Forecast Reliability", fontweight="bold", pad=12)
    if has_uncrossed:
        ax.text(
            0.99,
            0.02,
            "open markers: threshold was not crossed within the saved rollout horizon",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="0.25",
        )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", which="major", alpha=0.28, linewidth=0.8)
    ax.grid(True, axis="x", which="major", alpha=0.14, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    style_plot_legend(ax, outside=len(curves) > 5)
    fig.tight_layout(rect=(0, 0, 0.84 if len(curves) > 5 else 1, 1))
    save_figure(fig, outdir / "skill_horizon_vs_gamma.png")


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
    bars = ax.bar(labels, values, color=optimizer_bar_colors(labels), edgecolor="black", linewidth=0.7, alpha=0.88)
    ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.2, alpha=0.8, label="equal runtime")
    ax.set_ylabel("Runtime Ratio")
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


def _comparison_spectra(
    snapshots: Sequence[roll.FieldSnapshot],
    spectra_method: str,
    sht=None,
) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, Dict[str, np.ndarray]]]]:
    """Compute one truth spectrum and one prediction spectrum per optimizer."""
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
    return truth_spectra, pred_spectra


def _signed_percent_difference(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return signed pointwise percent difference from a truth spectrum."""
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    finite_truth = np.abs(truth[np.isfinite(truth)])
    # A tiny spectrum value makes raw percent error numerically explosive even
    # when the absolute spectral mismatch is unimportant.  Use a small floor
    # relative to the panel's truth-spectrum magnitude to keep the plot focused
    # on physically meaningful discrepancies.
    eps = max(float(np.nanmax(finite_truth)) * 1e-6, 1e-30) if finite_truth.size else 1e-30
    denom = np.maximum(np.abs(truth), eps)
    return 100.0 * (prediction - truth) / denom


def _spectral_percent_axis_limit(values: Sequence[np.ndarray]) -> float:
    """Choose a robust symmetric limit for signed spectral percent plots."""
    arrays = [
        np.abs(np.asarray(value, dtype=float).ravel())
        for value in values
        if np.asarray(value).size
    ]
    if not arrays:
        return 100.0
    finite = np.concatenate(arrays)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 100.0
    return max(25.0, float(np.nanpercentile(finite, 98)) * 1.15)


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
    truth_spectra, pred_spectra = _comparison_spectra(snapshots, spectra_method, sht)

    titles = ["Rotational", "Divergent", "Potential", "Total"]
    keys = ["rotational", "divergent", "potential", "total"]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6))
    legend_handles = []
    legend_labels = []

    for idx, (ax, title, key) in enumerate(zip(axes.ravel(), titles, keys)):
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
            style = optimizer_line_style(label, index=i, include_marker=len(k_pred) < 40)
            line, = ax.loglog(
                k_pred,
                spectra[key][1:],
                label=label,
                **style,
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
        if idx // 2 == 1:
            ax.set_xlabel("Wavenumber $l$")
        if idx % 2 == 0:
            ax.set_ylabel("Power Spectrum")
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


def plot_combined_spectra_percent_difference(
    snapshots: Sequence[roll.FieldSnapshot],
    output_freq: int,
    outdir: Path,
    spectra_method: str = "fft",
    sht=None,
) -> None:
    """Plot signed spectral percent difference from ground truth in a 2x2 grid."""
    truth_spectra, pred_spectra = _comparison_spectra(snapshots, spectra_method, sht)

    titles = ["Rotational", "Divergent", "Potential", "Total"]
    keys = ["rotational", "divergent", "potential", "total"]
    percent_by_key = {}
    all_percent_values = []
    for key in keys:
        truth_values = np.asarray(truth_spectra[key])[1:]
        entries = []
        for label, spectra in pred_spectra:
            diff = _signed_percent_difference(np.asarray(spectra[key])[1:], truth_values)
            entries.append((label, np.asarray(spectra["wavenumbers"])[1:], diff))
            all_percent_values.append(diff)
        percent_by_key[key] = entries
    y_limit = _spectral_percent_axis_limit(all_percent_values)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6))
    legend_handles = []
    legend_labels = []

    for idx, (ax, title, key) in enumerate(zip(axes.ravel(), titles, keys)):
        k = np.asarray(truth_spectra["wavenumbers"])[1:]
        for i, (label, k_pred, diff) in enumerate(percent_by_key[key]):
            style = optimizer_line_style(label, index=i, include_marker=len(k_pred) < 40)
            line, = ax.semilogx(
                k_pred,
                diff,
                label=label,
                **style,
            )
            if ax is axes.ravel()[0]:
                legend_handles.append(line)
                legend_labels.append(label)

        ax.axhline(0.0, color="0.25", linewidth=1.1, alpha=0.75)
        ax.set_title(title, fontweight="bold")
        if idx // 2 == 1:
            ax.set_xlabel("Wavenumber $l$")
        if idx % 2 == 0:
            ax.set_ylabel("Difference from Ground Truth (%)")
        apply_percent_difference_y_axis(ax, y_limit, linthresh=10.0)
        ax.grid(True, which="both", alpha=0.3)
        if len(k) > 0:
            ax.set_xlim(left=max(1, k[0]), right=k[-1])

    fig.suptitle(
        f"Final Rollout Spectral Percentage Difference (Step {snapshots[0].step}; Saved Every {output_freq} Step(s))",
        fontsize=14,
        fontweight="bold",
    )
    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=min(4, len(legend_labels)), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    save_figure(fig, outdir / "forecast_spectra_percent_difference_final.png")


# ---------------------------------------------------------------------------
# Rollout animation
# ---------------------------------------------------------------------------

ANIMATION_MAX_COLS = 3
ANIMATION_PANEL_W = 4.0
ANIMATION_PANEL_H = 3.4
_ANIMATION_DARK_BG = "#0d1117"


def _compact_grid_shape(n_panels: int, max_cols: int = ANIMATION_MAX_COLS) -> Tuple[int, int]:
    """Return a roughly square grid with at most ``max_cols`` columns."""
    if n_panels < 1:
        raise ValueError("n_panels must be >= 1")
    n_cols = min(max_cols, n_panels)
    n_rows = int(math.ceil(n_panels / n_cols))
    return n_rows, n_cols


def _init_animation_panel_axes(
    axes_grid: np.ndarray,
    n_rows: int,
    n_cols: int,
    n_panels: int,
) -> List[plt.Axes]:
    """Activate the first ``n_panels`` cells in a grid and hide the rest."""
    active: List[plt.Axes] = []
    for idx, ax in enumerate(axes_grid.ravel()):
        if idx < n_panels:
            ax.axis("off")
            ax.set_facecolor(_ANIMATION_DARK_BG)
            active.append(ax)
        else:
            ax.set_visible(False)
    return active


def make_rollout_animation(
    frames: Sequence[Sequence[roll.FieldSnapshot]],
    channel: str,
    fps: int = 8,
    output: Optional[Union[str, Path]] = None,
    show_error: bool = False,
) -> None:
    """Create a multi-optimizer comparison animation from pre-computed rollouts.

    Each frame is a timestep.  Ground truth and optimizer predictions are shown
    in a compact grid (at most three columns) instead of one very wide row.
    When ``show_error=True``, a second grid below shows signed pointwise error
    maps (prediction − truth) for each optimizer on a shared error colorscale.

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

    # Compact grid layout: at most three columns so five optimizers do not
    # produce an unwieldy 1x6 strip.  Static forecast grids already use the
    # same column cap via --grid_cols.
    n_pred_panels = 1 + n_opts
    pred_rows, pred_cols = _compact_grid_shape(n_pred_panels)
    if show_error:
        err_rows, err_cols = _compact_grid_shape(n_opts)
        fig_rows = pred_rows + err_rows
        fig_cols = max(pred_cols, err_cols)
    else:
        err_rows = err_cols = 0
        fig_rows, fig_cols = pred_rows, pred_cols

    fig_w = ANIMATION_PANEL_W * fig_cols
    fig_h = ANIMATION_PANEL_H * fig_rows + 0.9

    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.patch.set_facecolor(_ANIMATION_DARK_BG)
    for ax in axes.ravel():
        ax.axis("off")
        ax.set_facecolor(_ANIMATION_DARK_BG)

    pred_axes = _init_animation_panel_axes(axes[:pred_rows, :pred_cols], pred_rows, pred_cols, n_pred_panels)
    for ax in axes[:pred_rows, pred_cols:fig_cols].ravel():
        ax.set_visible(False)

    panel_titles = ["Ground Truth", *labels]
    panel_fields = [
        frames[0][0].truth_fields[ch],
        *[snap.prediction_fields[ch] for snap in frames[0]],
    ]
    im_truth = None
    im_preds = []
    for ax, title, field in zip(pred_axes, panel_titles, panel_fields):
        im = ax.imshow(
            np.zeros_like(field),
            cmap="twilight_shifted",
            vmin=vmin_field,
            vmax=vmax_field,
            origin="upper",
            aspect="auto",
        )
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6)
        _dark_colorbar(fig, im, ax)
        if title == "Ground Truth":
            im_truth = im
        else:
            im_preds.append(im)

    im_errors = []
    if show_error:
        err_axes = _init_animation_panel_axes(
            axes[pred_rows:pred_rows + err_rows, :err_cols],
            err_rows,
            err_cols,
            n_opts,
        )
        for ax in axes[pred_rows:pred_rows + err_rows, err_cols:fig_cols].ravel():
            ax.set_visible(False)
        for ax, label in zip(err_axes, labels):
            im = ax.imshow(
                np.zeros_like(frames[0][0].prediction_fields[ch]),
                cmap="RdBu_r",
                vmin=-err_abs_max,
                vmax=err_abs_max,
                origin="upper",
                aspect="auto",
            )
            ax.set_title(f"{label} error", color="white", fontsize=10, fontweight="bold", pad=6)
            _dark_colorbar(fig, im, ax)
            im_errors.append(im)

    # Use a figure-level title rather than per-axes text so the shared title is
    # retained by Pillow/FFMpeg writers across every rendered animation frame.
    fig.suptitle(
        f"Rollout Comparison: {display_metric_text(channel)}",
        color="white",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    step_label = fig.text(
        0.5, 0.925,
        "",
        ha="center", va="top",
        color="white", fontsize=12, fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0.0, 1, 0.875])

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


def make_rollout_error_animation(
    frames: Sequence[Sequence[roll.FieldSnapshot]],
    channel: str,
    fps: int = 8,
    output: Optional[Union[str, Path]] = None,
) -> None:
    """Create an error-only animation for a selected optimizer group.

    The layout wraps optimizer panels into a compact grid and keeps one shared
    signed-error color scale across all frames, making it suitable for focused
    slide exports where the value fields are shown in a separate GIF.
    """
    if not frames:
        raise ValueError("frames is empty - nothing to animate.")

    ch = roll.CHANNEL_TO_INDEX[channel]
    labels = [snap.label for snap in frames[0]]
    n_opts = len(labels)
    n_cols = min(3, max(1, int(math.ceil(math.sqrt(n_opts)))))
    n_rows = int(math.ceil(n_opts / n_cols))

    all_errors = np.concatenate([
        (snap.prediction_fields[ch] - snap.truth_fields[ch]).ravel()
        for step_snaps in frames
        for snap in step_snaps
    ])
    err_abs_max = float(max(
        abs(np.nanpercentile(all_errors, 2)),
        abs(np.nanpercentile(all_errors, 98)),
    ))
    if not np.isfinite(err_abs_max) or err_abs_max <= 0:
        err_abs_max = 1.0

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.7 * n_rows + 0.9),
        squeeze=False,
    )
    fig.patch.set_facecolor("#0d1117")

    im_errors = []
    for ax, label, snap in zip(axes.ravel(), labels, frames[0]):
        im = ax.imshow(
            np.zeros_like(snap.prediction_fields[ch]),
            cmap="RdBu_r",
            vmin=-err_abs_max,
            vmax=err_abs_max,
            origin="upper",
            aspect="auto",
        )
        ax.set_title(label, color="white", fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")
        ax.set_facecolor("#0d1117")
        _dark_colorbar(fig, im, ax)
        im_errors.append(im)

    for ax in axes.ravel()[n_opts:]:
        ax.axis("off")
        ax.set_facecolor("#0d1117")

    fig.suptitle(
        f"Rollout Error: {display_metric_text(channel)}",
        color="white",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    step_label = fig.text(
        0.5,
        0.925,
        "",
        ha="center",
        va="top",
        color="white",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.0, 1, 0.875])

    def update(frame_index: int):
        step_snaps = frames[frame_index]
        step_label.set_text(f"Rollout Step {step_snaps[0].step}")
        for im, snap in zip(im_errors, step_snaps):
            im.set_data(snap.prediction_fields[ch] - snap.truth_fields[ch])
        return [*im_errors, step_label]

    ani = manimation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=False,
    )
    _save_or_show_animation(fig, ani, output, fps=fps, dpi=120)


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
    n_rows, n_cols = _compact_grid_shape(n_panels)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(ANIMATION_PANEL_W * n_cols, ANIMATION_PANEL_H * n_rows + 0.7),
        squeeze=False,
    )
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
    y_values_by_key = {key: [] for key in ("rotational", "divergent", "potential", "total")}
    for step_snaps in frames:
        # Precompute spectra and fixed per-channel y-limits before constructing
        # the animation.  Per-channel limits keep each panel readable without
        # per-frame autoscaling jitter.
        truth_spectra = roll.compute_spherical_energy_spectra(step_snaps[0].truth_fields, sht)
        pred_spectra = [
            (snap.label, roll.compute_spherical_energy_spectra(snap.prediction_fields, sht))
            for snap in step_snaps
        ]
        spectral_frames.append((step_snaps[0].step, truth_spectra, pred_spectra))
        for spectra in [truth_spectra, *[spec for _, spec in pred_spectra]]:
            for key in ("rotational", "divergent", "potential", "total"):
                values = np.asarray(spectra[key])
                y_values_by_key[key].extend(values[np.isfinite(values) & (values > 0)].tolist())

    spec_keys = [
        ("rotational", "Rotational"),
        ("divergent", "Divergent"),
        ("potential", "Potential"),
        ("total", "Total"),
    ]
    labels = [snap.label for snap in frames[0]]
    y_limits_by_key = {
        key: robust_log_axis_limits(y_values_by_key[key])
        for key, _ in spec_keys
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Combined Rollout Spectra Comparison", fontsize=15, fontweight="bold")
    step_label = fig.text(0.5, 0.935, "", ha="center", va="top", fontsize=11, fontweight="bold")

    line_sets = []
    for idx, (ax, (key, title)) in enumerate(zip(axes.ravel(), spec_keys)):
        truth_line, = ax.loglog([], [], color="black", linewidth=2.2, linestyle="--", label="Ground Truth")
        pred_lines = []
        for i, label in enumerate(labels):
            style = optimizer_line_style(label, index=i, include_marker=False)
            line, = ax.loglog([], [], label=label, **style)
            pred_lines.append((label, line))
        ax.set_title(title, fontweight="bold")
        if idx // 2 == 1:
            ax.set_xlabel("Wavenumber $l$")
        if idx % 2 == 0:
            ax.set_ylabel("Power Spectrum")
        ax.set_ylim(*y_limits_by_key[key])
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


def make_combined_spectral_percent_difference_animation(
    frames: Sequence[Sequence[roll.FieldSnapshot]],
    fps: int = 8,
    output: Optional[Union[str, Path]] = None,
    config_path: Union[str, Path] = "config_paradis.yaml",
) -> None:
    """Animate signed spectral percent difference from ground truth."""
    if not frames:
        raise ValueError("frames is empty - nothing to animate.")

    # This companion intentionally uses the same SHT-backed spectra as the
    # combined spectral animation, but it removes the truth and reference-slope
    # curves so optimizer error is the only quantity being compared.
    sht = roll.build_spherical_sht(config_path, frames[0][0].truth_fields.shape)
    spectral_frames = []
    percent_values = []
    for step_snaps in frames:
        truth_spectra = roll.compute_spherical_energy_spectra(step_snaps[0].truth_fields, sht)
        pred_spectra = [
            (snap.label, roll.compute_spherical_energy_spectra(snap.prediction_fields, sht))
            for snap in step_snaps
        ]
        spectral_frames.append((step_snaps[0].step, truth_spectra, pred_spectra))
        for _, spectra in pred_spectra:
            for key in ("rotational", "divergent", "potential", "total"):
                percent_values.append(
                    _signed_percent_difference(np.asarray(spectra[key])[1:], np.asarray(truth_spectra[key])[1:])
                )

    spec_keys = [
        ("rotational", "Rotational"),
        ("divergent", "Divergent"),
        ("potential", "Potential"),
        ("total", "Total"),
    ]
    labels = [snap.label for snap in frames[0]]
    y_limit = _spectral_percent_axis_limit(percent_values)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Combined Rollout Spectral Percentage Difference", fontsize=15, fontweight="bold")
    step_label = fig.text(0.5, 0.935, "", ha="center", va="top", fontsize=11, fontweight="bold")

    line_sets = []
    for idx, (ax, (key, title)) in enumerate(zip(axes.ravel(), spec_keys)):
        pred_lines = []
        for i, label in enumerate(labels):
            style = optimizer_line_style(label, index=i, include_marker=False)
            line, = ax.semilogx([], [], label=label, **style)
            pred_lines.append((label, line))
        ax.axhline(0.0, color="0.25", linewidth=1.1, alpha=0.75)
        ax.set_title(title, fontweight="bold")
        if idx // 2 == 1:
            ax.set_xlabel("Wavenumber $l$")
        if idx % 2 == 0:
            ax.set_ylabel("Difference from Ground Truth (%)")
        apply_percent_difference_y_axis(ax, y_limit, linthresh=10.0)
        ax.grid(True, which="both", alpha=0.3)
        line_sets.append((key, pred_lines))

    handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=min(4, len(legend_labels)), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.91))

    def update(frame_index: int):
        step, truth_spectra, pred_spectra = spectral_frames[frame_index]
        pred_by_label = dict(pred_spectra)
        artists = [step_label]
        step_label.set_text(f"Rollout Step {step}")
        for ax, (key, pred_lines) in zip(axes.ravel(), line_sets):
            k = np.asarray(truth_spectra["wavenumbers"])[1:]
            truth_values = np.asarray(truth_spectra[key])[1:]
            for label, line in pred_lines:
                spectra = pred_by_label[label]
                diff = _signed_percent_difference(np.asarray(spectra[key])[1:], truth_values)
                line.set_data(np.asarray(spectra["wavenumbers"])[1:], diff)
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
