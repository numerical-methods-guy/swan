# ============================================================
# Pixel-error visualizer for regional/LAM autoregressive runs
# ============================================================

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Patch

# ----------------------------------------------------------------
# 1. COLAB / GOOGLE DRIVE PATHS
# ----------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser(
    description="Create fixed-scale LAM pixel-error maps from CSV files."
)

parser.add_argument(
    "input_dir",
    type=Path,
    help="Folder containing pixel-error CSV files.",
)

parser.add_argument(
    "output_dir",
    type=Path,
    help="Folder where PNG figures and fixed_scales.json are written.",
)

parser.add_argument(
    "--reference-scales",
    type=Path,
    default=None,
    help=(
        "Optional existing fixed_scales.json to reuse. "
        "Use this for strictly comparable model-to-model plots."
    ),
)

parser.add_argument(
    "--glob",
    default="*pixel_errors*.csv",
    help="Filename glob used to select input CSVs (default: %(default)s).",
)

parser.add_argument(
    "--percentile",
    type=float,
    default=99.5,
    help="Robust global percentile for fixed limits (default: %(default)s).",
)

parser.add_argument(
    "--no-absolute",
    action="store_true",
    help="Do not create supplementary absolute-error maps.",
)

parser.add_argument(
    "--blend-tolerance",
    type=float,
    default=1e-12,
    help="Tolerance for detecting the inferred blending zone (default: %(default)s).",
)

parser.add_argument(
    "--no-hr-only",
    action="store_true",
    help="Do not create additional HR-only interior error maps.",
)

parser.add_argument(
    "--geo-ticks",
    type=int,
    default=5,
    help="Number of lat/lon ticks per plot axis"
)

args = parser.parse_args()

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
REFERENCE_SCALE_FILE = args.reference_scales
ROBUST_PERCENTILE = args.percentile
MAKE_ABSOLUTE_ERROR_MAPS = not args.no_absolute
MAKE_HR_ONLY_MAPS = not args.no_hr_only
BLEND_TOLERANCE = args.blend_tolerance
N_GEO_TICKS = args.geo_ticks
FILE_GLOB = args.glob


# ----------------------------------------------------------------
# 2. INPUT VALIDATION AND HELPERS
# ----------------------------------------------------------------
REQUIRED_COLUMNS = {
    "step", "channel_name",
    "lat_idx_local", "lon_idx_local",
    "lat_deg", "lon_deg",
    "truth_hr", "err_hr", "abs_err_hr",
    "truth_blended", "err_blended", "abs_err_blended",
}

def safe_name(text):
    """Convert a channel/file name into a portable filename component."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")

def read_error_csv(csv_path):
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required columns: {sorted(missing)}"
        )

    numeric_columns = [
        "step", "lat_idx_local", "lon_idx_local",
        "lat_deg", "lon_deg",
        "truth_hr", "err_hr", "abs_err_hr",
        "truth_blended", "err_blended", "abs_err_blended",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(
        subset=["step", "channel_name", "lat_idx_local", "lon_idx_local"]
    )

def make_grid(frame, value_column):
    """
    Return a 2-D local-grid array indexed as [lat_idx_local, lon_idx_local].
    Missing cells remain NaN.
    """
    lat_ids = np.sort(frame["lat_idx_local"].unique())
    lon_ids = np.sort(frame["lon_idx_local"].unique())

    grid = (
        frame.pivot_table(
            index="lat_idx_local",
            columns="lon_idx_local",
            values=value_column,
            aggfunc="first",
        )
        .reindex(index=lat_ids, columns=lon_ids)
        .to_numpy(dtype=float)
    )
    return grid, lat_ids, lon_ids

def blending_mask(frame):
    """
    Infer blending-transition cells from a difference between the HR and
    blended truth fields. The contour marks the boundary of this set.
    """
    difference = np.abs(
        frame["truth_blended"].to_numpy() - frame["truth_hr"].to_numpy()
    )
    frame = frame.copy()
    frame["is_blended"] = difference > BLEND_TOLERANCE
    mask, _, _ = make_grid(frame, "is_blended")
    return np.nan_to_num(mask, nan=0.0).astype(bool)

def hr_only_frame(frame):
    """
    Keep only cells whose blended target equals the HR target, i.e. cells
    outside the inferred blending zone. The returned rows retain their
    geographic and local-index metadata.
    """
    difference = np.abs(
        frame["truth_blended"].to_numpy(dtype=float)
        - frame["truth_hr"].to_numpy(dtype=float)
    )
    keep = np.isfinite(difference) & (difference <= BLEND_TOLERANCE)

    if not np.any(keep):
        raise ValueError(
            "No HR-only cells remain after excluding the inferred blending zone. "
            "Check truth_blended, truth_hr, and --blend-tolerance."
        )

    return frame.loc[keep].copy()

def geographic_ticks(frame, lat_ids, lon_ids, n_ticks=N_GEO_TICKS):
    """
    Produce local-grid tick positions labelled with latitude/longitude.
    This preserves a rectangular numerical-grid display.
    """
    lat_lookup = (
        frame.groupby("lat_idx_local")["lat_deg"].median().reindex(lat_ids)
    )
    lon_lookup = (
        frame.groupby("lon_idx_local")["lon_deg"].median().reindex(lon_ids)
    )

    x_positions = np.unique(
        np.linspace(0, len(lon_ids) - 1, min(n_ticks, len(lon_ids))).round().astype(int)
    )
    y_positions = np.unique(
        np.linspace(0, len(lat_ids) - 1, min(n_ticks, len(lat_ids))).round().astype(int)
    )

    x_labels = [f"{lon_lookup.iloc[i]:.1f}°" for i in x_positions]
    y_labels = [f"{lat_lookup.iloc[i]:.1f}°" for i in y_positions]

    return x_positions, x_labels, y_positions, y_labels

def decorate_axis(ax, x_pos, x_labels, y_pos, y_labels, show_ylabel=True):
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels if show_ylabel else [], fontsize=8)

    ax.set_xlabel("Longitude", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Latitude", fontsize=9)

# def draw_blend_outline(ax, blend_zone):
#     """
#     Draw a black contour around cells for which blended truth differs from HR
#     truth. Nothing is drawn if there is no inferred blending zone.
#     """
#     if blend_zone.any() and (~blend_zone).any():
#         ax.contour(
#             blend_zone.astype(float),
#             levels=[0.5],
#             colors="black",
#             linewidths=0.8,
#             origin="upper",
#         )

def save_signed_error_figure(
    frame, csv_stem, step, channel, signed_limit, output_path
):
    hr_grid, lat_ids, lon_ids = make_grid(frame, "err_hr")
    blended_grid, _, _ = make_grid(frame, "err_blended")
    blend_zone = blending_mask(frame)

    x_pos, x_labels, y_pos, y_labels = geographic_ticks(
        frame, lat_ids, lon_ids
    )

    norm = TwoSlopeNorm(vmin=-signed_limit, vcenter=0.0, vmax=signed_limit)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    panels = [
        (hr_grid, "Error: Prediction − HR truth"),
        (blended_grid, "Error: Prediction − Blended truth"),
    ]

    for i, (ax, (grid, title)) in enumerate(zip(axes, panels)):
        im = ax.imshow(
            grid,
            cmap="RdBu_r",
            norm=norm,
            interpolation="nearest",
            aspect="auto",
            origin="upper",
        )
        # draw_blend_outline(ax, blend_zone)
        decorate_axis(
            ax, x_pos, x_labels, y_pos, y_labels, show_ylabel=(i == 0)
        )
        ax.set_title(title, fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("Error", fontsize=10)

    # if blend_zone.any() and (~blend_zone).any():
    #     axes[1].legend(
    #         handles=[Patch(facecolor="none", edgecolor="black",
    #                         label="Inferred blending-zone boundary")],
    #         loc="upper right",
    #         fontsize=8,
    #         framealpha=0.9,
    #     )

    fig.suptitle(
        f"{csv_stem} | Step {step} | {channel}\n"
        f"Fixed symmetric scale: ±{signed_limit:.3e}",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def save_hr_only_signed_error_figure(
    frame, csv_stem, step, channel, signed_limit, output_path
):
    """
    Plot prediction-minus-HR-truth signed error only for pixels outside the
    inferred blending zone.
    """
    hr_only = hr_only_frame(frame)
    hr_grid, lat_ids, lon_ids = make_grid(hr_only, "err_hr")

    x_pos, x_labels, y_pos, y_labels = geographic_ticks(
        hr_only, lat_ids, lon_ids
    )

    norm = TwoSlopeNorm(vmin=-signed_limit, vcenter=0.0, vmax=signed_limit)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    im = ax.imshow(
        hr_grid,
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
        aspect="auto",
        origin="upper",
    )

    decorate_axis(ax, x_pos, x_labels, y_pos, y_labels, show_ylabel=True)
    ax.set_title(
        "Error: Prediction − HR truth\n(HR-only interior)",
        fontsize=11,
        fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label("Error", fontsize=10)

    fig.suptitle(
        f"{csv_stem} | Step {step} | {channel}\n"
        f"HR-only interior; fixed symmetric scale: ±{signed_limit:.3e}",
        fontsize=12,
        fontweight="bold",
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def save_absolute_error_figure(
    frame, csv_stem, step, channel, abs_limit, output_path
):
    hr_grid, lat_ids, lon_ids = make_grid(frame, "abs_err_hr")
    blended_grid, _, _ = make_grid(frame, "abs_err_blended")
    blend_zone = blending_mask(frame)

    x_pos, x_labels, y_pos, y_labels = geographic_ticks(
        frame, lat_ids, lon_ids
    )

    norm = Normalize(vmin=0.0, vmax=abs_limit)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    panels = [
        (hr_grid, "Absolute error: |Prediction − HR truth|"),
        (blended_grid, "Absolute error: |Prediction − Blended truth|"),
    ]

    for i, (ax, (grid, title)) in enumerate(zip(axes, panels)):
        im = ax.imshow(
            grid,
            cmap="magma",
            norm=norm,
            interpolation="nearest",
            aspect="auto",
            origin="upper",
        )
        # draw_blend_outline(ax, blend_zone)
        decorate_axis(
            ax, x_pos, x_labels, y_pos, y_labels, show_ylabel=(i == 0)
        )
        ax.set_title(title, fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("Absolute error", fontsize=10)

    fig.suptitle(
        f"{csv_stem} | Step {step} | {channel}\n"
        f"Fixed robust scale: 0 to {abs_limit:.3e}",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def save_hr_only_absolute_error_figure(
    frame, csv_stem, step, channel, abs_limit, output_path
):
    """
    Plot absolute prediction-minus-HR-truth error only outside the inferred
    blending zone.
    """
    hr_only = hr_only_frame(frame)
    hr_grid, lat_ids, lon_ids = make_grid(hr_only, "abs_err_hr")

    x_pos, x_labels, y_pos, y_labels = geographic_ticks(
        hr_only, lat_ids, lon_ids
    )

    norm = Normalize(vmin=0.0, vmax=abs_limit)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    im = ax.imshow(
        hr_grid,
        cmap="magma",
        norm=norm,
        interpolation="nearest",
        aspect="auto",
        origin="upper",
    )

    decorate_axis(ax, x_pos, x_labels, y_pos, y_labels, show_ylabel=True)
    ax.set_title(
        "Absolute error: |Prediction − HR truth|\n(HR-only interior)",
        fontsize=11,
        fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label("Absolute error", fontsize=10)

    fig.suptitle(
        f"{csv_stem} | Step {step} | {channel}\n"
        f"HR-only interior; fixed robust scale: 0 to {abs_limit:.3e}",
        fontsize=12,
        fontweight="bold",
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def calculate_metrics(df, csv_name):
    rows = []

    for (step, channel), group in df.groupby(["step", "channel_name"]):
        blended_error = group["err_blended"].to_numpy(dtype=float)
        hr_error = group["err_hr"].to_numpy(dtype=float)

        blended_error = blended_error[np.isfinite(blended_error)]
        hr_error = hr_error[np.isfinite(hr_error)]

        mse_blended = (
            np.mean(blended_error ** 2) if blended_error.size else np.nan
        )
        mse_hr = np.mean(hr_error ** 2) if hr_error.size else np.nan

        rows.append(
            {
                "csv_file": csv_name,
                "step": int(step),
                "channel_name": str(channel),
                "n_blended_pixels": int(blended_error.size),
                "mse_blended": mse_blended,
                "rmse_blended": np.sqrt(mse_blended),
                "mse_hr": mse_hr,
                "rmse_hr": np.sqrt(mse_hr),
            }
        )

    return pd.DataFrame(rows)

# ----------------------------------------------------------------
# 3. GLOBAL SCALE CALCULATION
# ----------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
signed_dir = OUTPUT_DIR / "signed_error"
absolute_dir = OUTPUT_DIR / "absolute_error"
hr_only_signed_dir = OUTPUT_DIR / "signed_error_hr_only"
hr_only_absolute_dir = OUTPUT_DIR / "absolute_error_hr_only"

signed_dir.mkdir(exist_ok=True)
absolute_dir.mkdir(exist_ok=True)
hr_only_signed_dir.mkdir(exist_ok=True)
hr_only_absolute_dir.mkdir(exist_ok=True)

csv_files = sorted(INPUT_DIR.glob(FILE_GLOB))
if not csv_files:
    raise FileNotFoundError(
        f"No files matching '{FILE_GLOB}' found in {INPUT_DIR}"
    )

if REFERENCE_SCALE_FILE is not None:
    with open(REFERENCE_SCALE_FILE, "r") as f:
        fixed_scales = json.load(f)
    print(f"Using reference scales from: {REFERENCE_SCALE_FILE}")
else:
    all_abs_errors = {}

    for csv_path in csv_files:
        df = read_error_csv(csv_path)
        for channel, group in df.groupby("channel_name"):
            # Use absolute blended signed error rather than abs_err_blended
            # so the scale always corresponds exactly to err_blended.
            values = np.abs(group["err_blended"].to_numpy(dtype=float))
            values = values[np.isfinite(values)]
            values = values[values > 0.0]

            if len(values):
                all_abs_errors.setdefault(str(channel), []).append(values)

    fixed_scales = {}
    for channel, chunks in all_abs_errors.items():
        values = np.concatenate(chunks)
        limit = float(np.percentile(values, ROBUST_PERCENTILE))

        # Prevent a pathological all-zero channel from breaking TwoSlopeNorm.
        fixed_scales[channel] = max(limit, np.finfo(float).eps)

    scale_path = OUTPUT_DIR / "fixed_scales.json"
    with open(scale_path, "w") as f:
        json.dump(
            {
                "percentile": ROBUST_PERCENTILE,
                "error_field_for_scale": "abs(err_blended)",
                "signed_limits_by_channel": fixed_scales,
            },
            f,
            indent=2,
        )

    print(f"Calculated and saved fixed scales: {scale_path}")

print("\nFixed limits:")
for channel, limit in fixed_scales.items():
    print(f"  {channel}: ±{float(limit):.6e}")


# ----------------------------------------------------------------
# 4. CREATE ONE PNG PER FILE / STEP / CHANNEL
# ----------------------------------------------------------------
all_metrics = []

for csv_path in csv_files:
    df = read_error_csv(csv_path)
    file_metrics = calculate_metrics(df, csv_path.name)
    all_metrics.append(file_metrics)
    csv_stem = safe_name(csv_path.stem)

    print(f"\n{'=' * 90}")
    print(f"Blended-target pixel metrics: {csv_path.name}")
    print(file_metrics.to_string(
        index=False,
        formatters={
            "mse_blended": "{:.6e}".format,
            "rmse_blended": "{:.6e}".format,
            "mse_hr": "{:.6e}".format,
            "rmse_hr": "{:.6e}".format,
        },
    ))
    

    for (step, channel), group in df.groupby(["step", "channel_name"]):
        channel = str(channel)
        if channel not in fixed_scales:
            print(f"Skipping {csv_path.name}, step {step}, {channel}: no scale.")
            continue

        limit = float(fixed_scales[channel])
        label = f"{csv_stem}_step-{int(step):03d}_{safe_name(channel)}"

        save_signed_error_figure(
            frame=group,
            csv_stem=csv_stem,
            step=int(step),
            channel=channel,
            signed_limit=limit,
            output_path=signed_dir / f"{label}_fixedscale.png",
        )

        if MAKE_ABSOLUTE_ERROR_MAPS:
            save_absolute_error_figure(
                frame=group,
                csv_stem=csv_stem,
                step=int(step),
                channel=channel,
                abs_limit=limit,
                output_path=absolute_dir / f"{label}_absolute.png",
            )
        
        if MAKE_HR_ONLY_MAPS:
            save_hr_only_signed_error_figure(
                frame=group,
                csv_stem=csv_stem,
                step=int(step),
                channel=channel,
                signed_limit=limit,
                output_path=(
                    hr_only_signed_dir
                    / f"{label}_hr_only_fixedscale.png"
                ),
            )

            if MAKE_ABSOLUTE_ERROR_MAPS:
                save_hr_only_absolute_error_figure(
                    frame=group,
                    csv_stem=csv_stem,
                    step=int(step),
                    channel=channel,
                    abs_limit=limit,
                    output_path=(
                        hr_only_absolute_dir
                        / f"{label}_hr_only_absolute.png"
                    ),
                )

print(f"\nFinished. PNG output directory: {OUTPUT_DIR}")
if all_metrics:
    metrics_df = pd.concat(all_metrics, ignore_index=True)

    metrics_path = OUTPUT_DIR / "blended_error_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\nSaved per-file/per-step/per-channel metrics: {metrics_path}")

    print(f"\n{'=' * 90}")
    print("All blended-target metrics")
    print(metrics_df.to_string(
        index=False,
        formatters={
            "mse_blended": "{:.6e}".format,
            "rmse_blended": "{:.6e}".format,
        },
    ))