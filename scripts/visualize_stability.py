"""Visualize saved precomputed trajectories for one sample.

For each step, plots all 3 fields (h, vorticity, divergence) for:
  - main solver (row 1)
  - reference solver from stability_check/ (row 2)
  - difference (row 3)

Saves one PNG per step to --output_dir.

Usage (from repo root):
    python scripts/visualize_stability.py \
        --dataset_folder /home/avg000/swan/datasets/train/williamson_case6_60_20260618 \
        --sample 0 \
        --output_dir /home/avg000/swan/weights/20260623_run1/stability_viz
"""

import argparse
import os
from math import ceil

import matplotlib.pyplot as plt
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataset.shallow_water_solver import ShallowWaterSolver


def build_solver(nlat, nlon, dt_solver, device):
    lmax = ceil(nlat / 3)
    return ShallowWaterSolver(
        nlat, nlon, dt_solver, lmax=lmax, mmax=lmax, grid="equiangular"
    ).to(device).float()


def plot_step(dataset_folder, sample, step, solver, output_dir):
    main_path = os.path.join(dataset_folder, f"{sample}_{step}.pt")
    ref_path  = os.path.join(dataset_folder, "stability_check", f"{sample}_{step}_ref.pt")

    if not os.path.exists(main_path):
        print(f"  skipping step {step}: main file not found")
        return

    device = next(solver.parameters()).device
    spec  = torch.load(main_path, map_location=device)
    fields = solver.spec2grid(spec).cpu()  # (3, nlat, nlon)

    has_ref = os.path.exists(ref_path)
    if has_ref:
        spec_ref   = torch.load(ref_path, map_location=device)
        fields_ref = solver.spec2grid(spec_ref).cpu()

    field_names = ["Geopotential (h)", "Vorticity", "Divergence"]
    n_rows = 3 if has_ref else 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for col, name in enumerate(field_names):
        f    = fields[col].numpy()
        vmin, vmax = f.min(), f.max()

        im = axes[0, col].imshow(f, origin="lower", aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"{name}\nsample {sample}, step {step} (main)")
        plt.colorbar(im, ax=axes[0, col])

        if has_ref:
            im_ref = axes[1, col].imshow(
                fields_ref[col].numpy(), origin="lower", aspect="auto",
                cmap="RdBu_r", vmin=vmin, vmax=vmax
            )
            axes[1, col].set_title(f"{name} (ref)")
            plt.colorbar(im_ref, ax=axes[1, col])

            diff = (fields[col] - fields_ref[col]).numpy()
            im_diff = axes[2, col].imshow(
                diff, origin="lower", aspect="auto", cmap="RdBu_r"
            )
            axes[2, col].set_title(f"{name} (main - ref)")
            plt.colorbar(im_diff, ax=axes[2, col])

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"step_{step:03d}.png")
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--nlat", type=int, default=128)
    parser.add_argument("--nlon", type=int, default=256)
    parser.add_argument("--dt_solver", type=float, default=60.0)
    parser.add_argument("--n_steps", type=int, default=40,
                        help="Number of steps to visualize (step 0 to n_steps)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    solver = build_solver(args.nlat, args.nlon, args.dt_solver, args.device)

    print(f"Visualizing sample {args.sample}, steps 0 to {args.n_steps}")
    for step in range(args.n_steps + 1):
        plot_step(args.dataset_folder, args.sample, step, solver, args.output_dir)


if __name__ == "__main__":
    main()
