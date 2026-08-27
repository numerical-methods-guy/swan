#!/usr/bin/env python3
"""
compare_lam_paradis.py

Side-by-side comparison of LAM vs HR PARADIS forecast performance.

Reads the metrics CSVs produced by forecast_lam.py and
forecast_paradis_hr.py respectively, then produces:
  1. RMSE vs autoregressive step (overall + per-channel)
  2. MAE  vs autoregressive step
  3. Bias vs autoregressive step
  4. Speedup bar chart (LAM vs HR PARADIS wall-clock time)

Both CSVs must have been produced with the same --autoreg_steps value
and the same set of ICs for the comparison to be meaningful.

Usage:
    python compare_lam_paradis.py \
        --lam_metrics     results_lam/metrics.csv \
        --paradis_metrics results_paradis_hr/metrics_paradis_hr.csv \
        --autoreg_steps   10 \
        --output_dir      results_comparison/
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: str) -> dict:
    """Load a single-row metrics CSV into a flat dict."""
    df = pd.read_csv(path)
    assert len(df) == 1, f"Expected single-row CSV, got {len(df)} rows in {path}"
    return df.iloc[0].to_dict()


def _per_step(row: dict, key: str, autoreg_steps: int) -> np.ndarray:
    """
    Both forecast scripts aggregate over all ICs and all steps into a single
    mean scalar.  For a step-by-step curve we need per-step output.

    If the CSVs contain per-step columns (step_1_RMSE, step_2_RMSE …) those
    are used.  Otherwise the single mean value is broadcast across all steps
    as a flat line (indicating no per-step breakdown was saved).
    """
    step_cols = [f"step_{s}_{key}" for s in range(1, autoreg_steps + 1)]
    if all(c in row for c in step_cols):
        return np.array([row[c] for c in step_cols])
    # Fallback: broadcast the overall mean
    mean_key = f"{key}_mean"
    val = row.get(mean_key, float("nan"))
    return np.full(autoreg_steps, val)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def _line_plot(steps, lam_vals, par_vals, ylabel, title, output_path,
               lam_label="LAM", par_label="HR PARADIS"):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, lam_vals, "b-o",  lw=2, ms=5, label=lam_label)
    ax.plot(steps, par_vals, "r-s",  lw=2, ms=5, label=par_label)
    ax.set_xlabel("Autoregressive Step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def _per_channel_rmse(steps, lam_rows, par_rows, autoreg_steps,
                      output_path):
    CHANNEL_NAMES  = ["Geopotential h", "Vorticity ζ", "Divergence δ"]
    CHANNEL_KEYS   = ["RMSE_h", "RMSE_vort", "RMSE_div"]
    LAM_COLORS     = ["#1f77b4", "#aec7e8", "#6baed6"]
    PAR_COLORS     = ["#d62728", "#fc8d59", "#fdae6b"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ch, (name, key) in enumerate(zip(CHANNEL_NAMES, CHANNEL_KEYS)):
        ax = axes[ch]
        lam_v = _per_step(lam_rows, key, autoreg_steps)
        par_v = _per_step(par_rows, key, autoreg_steps)
        ax.plot(steps, lam_v, "o-", color=LAM_COLORS[ch], lw=2, ms=5, label="LAM")
        ax.plot(steps, par_v, "s-", color=PAR_COLORS[ch], lw=2, ms=5, label="HR PARADIS")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Autoregressive Step", fontsize=11)
        ax.set_ylabel("RMSE (physical units)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.suptitle("Per-Channel RMSE: LAM vs HR PARADIS",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def _speedup_bar(lam_row, par_row, output_path):
    lam_t = lam_row.get("ml_time_mean", float("nan"))
    par_t = par_row.get("ml_time_mean", float("nan"))

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["LAM", "HR PARADIS"], [lam_t, par_t],
                  color=["#1f77b4", "#d62728"], width=0.4)
    ax.bar_label(bars, fmt="%.3fs", fontsize=11, padding=4)
    ax.set_ylabel("Wall-clock time (s)", fontsize=12)
    ax.set_title("Rollout Time: LAM vs HR PARADIS", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare LAM vs HR PARADIS metrics")
    parser.add_argument("--lam_metrics",     required=True,
                        help="Path to results_lam/metrics.csv")
    parser.add_argument("--paradis_metrics", required=True,
                        help="Path to results_paradis_hr/metrics_paradis_hr.csv")
    parser.add_argument("--autoreg_steps",   type=int, default=1,
                        help="Number of autoregressive steps used in both forecasts")
    parser.add_argument("--output_dir",      default="results_comparison")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    lam_row = _load(args.lam_metrics)
    par_row = _load(args.paradis_metrics)
    steps   = np.arange(1, args.autoreg_steps + 1)

    print(f"\nComparing over {args.autoreg_steps} autoregressive step(s):")
    print(f"  LAM     : {args.lam_metrics}")
    print(f"  PARADIS : {args.paradis_metrics}")
    print()

    def out(name): return os.path.join(args.output_dir, name)

    # 1. Overall RMSE vs step
    _line_plot(
        steps,
        _per_step(lam_row, "RMSE", args.autoreg_steps),
        _per_step(par_row, "RMSE", args.autoreg_steps),
        ylabel="RMSE (physical units)",
        title="Overall RMSE vs Autoregressive Step",
        output_path=out("rmse_vs_step.png"),
    )

    # 2. MAE vs step
    _line_plot(
        steps,
        _per_step(lam_row, "MAE", args.autoreg_steps),
        _per_step(par_row, "MAE", args.autoreg_steps),
        ylabel="MAE (physical units)",
        title="MAE vs Autoregressive Step",
        output_path=out("mae_vs_step.png"),
    )

    # 3. Bias vs step
    _line_plot(
        steps,
        _per_step(lam_row, "bias", args.autoreg_steps),
        _per_step(par_row, "bias", args.autoreg_steps),
        ylabel="Bias (physical units)",
        title="Bias vs Autoregressive Step",
        output_path=out("bias_vs_step.png"),
    )

    # 4. Per-channel RMSE
    _per_channel_rmse(
        steps, lam_row, par_row, args.autoreg_steps,
        output_path=out("rmse_per_channel.png"),
    )

    # 5. Speedup bar
    _speedup_bar(lam_row, par_row, out("rollout_time.png"))

    # 6. Print table
    print("=" * 50)
    print(f"{'Metric':<18} {'LAM':>12} {'HR PARADIS':>12}")
    print("=" * 50)
    for key in ["RMSE", "MAE", "bias"]:
        lv = lam_row.get(f"{key}_mean", float("nan"))
        pv = par_row.get(f"{key}_mean", float("nan"))
        print(f"  {key:<16} {lv:>12.6f} {pv:>12.6f}")
    print(f"  {'ML time (s)':<16} "
          f"{lam_row.get('ml_time_mean', float('nan')):>12.3f} "
          f"{par_row.get('ml_time_mean', float('nan')):>12.3f}")
    print("=" * 50)
    print(f"\nAll comparison plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
