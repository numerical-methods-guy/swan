import os
import json
import argparse
import datetime
import subprocess
from math import ceil

import torch

from dataset.shallow_water_solver import ShallowWaterSolver


SAVED_DATASETS_DIR = os.path.join(os.path.dirname(__file__), "Saved_Datasets")


def build_solver(nlat, nlon, dt, dt_solver, device):
    nsteps = dt // dt_solver
    lmax = ceil(nlat / 3)
    mmax = lmax
    solver = (
        ShallowWaterSolver(nlat, nlon, dt_solver, lmax=lmax, mmax=mmax, grid="equiangular")
        .to(device)
        .float()
    )
    return solver, nsteps


def make_output_folder(ictype, dt_solver):
    date_str = datetime.date.today().strftime("%Y%m%d")
    folder_name = f"{ictype}_{dt_solver}_{date_str}"
    folder_path = os.path.join(SAVED_DATASETS_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(os.path.join(folder_path, "stability_check"), exist_ok=True)
    return folder_path


def generate_ic(solver, ictype, gbells_ref_mean=None, gbells_ref_std=None, gbells_kwargs=None):
    if ictype == "random":
        return solver.random_initial_condition(mach=0.2)
    elif ictype == "galewsky":
        return solver.galewsky_initial_condition()
    elif ictype == "gbells":
        if gbells_ref_mean is None or gbells_ref_std is None:
            raise ValueError("gbells_ref_mean and gbells_ref_std must be provided for ictype='gbells'")
        kwargs = gbells_kwargs or {}
        return solver.gaussian_bells_initial_condition(gbells_ref_mean, gbells_ref_std, **kwargs)
    elif ictype == "williamson_case2":
        return solver.williamson_case2_initial_condition()
    else:
        raise ValueError(f"Unsupported ictype: {ictype}")


def _compute_gbells_ref_stats(solver, n_samples=50):
    """Compute per-channel mean and std from random ICs for Gaussian bell scaling."""
    device = solver.lats.device
    means = torch.zeros(3, device=device)
    stds  = torch.zeros(3, device=device)
    with torch.no_grad():
        for _ in range(n_samples):
            spec = solver.random_initial_condition(mach=0.2)
            grid = solver.spec2grid(spec)  # (3, nlat, nlon)
            means += grid.mean(dim=(-1, -2))
            stds  += grid.std(dim=(-1, -2))
    return means / n_samples, stds / n_samples


def welford_update(count, mean, M2, new_value):
    """One step of Welford's online mean/variance algorithm."""
    count += 1
    delta = new_value - mean
    mean = mean + delta / count
    delta2 = new_value - mean
    M2 = M2 + delta * delta2
    return count, mean, M2


def save_trajectories(solver, ictype, n_samples, n_steps_per_trajectory, nsteps,
                      output_folder, device, dt, dt_solver_ref,
                      n_stability_samples, n_stability_steps, stability_threshold,
                      gbells_kwargs=None):
    """Generate trajectories, save .pt files, compute normalization stats online,
    and run the stability check inline for the first n_stability_samples trajectories."""

    stability_dir = os.path.join(output_folder, "stability_check")

    # build reference solver once
    nsteps_ref = dt // dt_solver_ref
    lmax = ceil(solver.nlat / 3)
    ref_solver = (
        ShallowWaterSolver(solver.nlat, solver.nlon, dt_solver_ref,
                           lmax=lmax, mmax=lmax, grid="equiangular")
        .to(device)
        .float()
    )

    # pre-compute Gaussian bell reference stats if needed
    gbells_ref_mean = None
    gbells_ref_std  = None
    if ictype == "gbells":
        print("Computing Gaussian bell reference statistics...")
        gbells_ref_mean, gbells_ref_std = _compute_gbells_ref_stats(solver)

    # Welford accumulators for fields and winds
    field_count = 0
    field_mean  = None
    field_M2    = None
    wind_count  = 0
    wind_mean   = None
    wind_M2     = None

    stability_errors = []

    with torch.no_grad():
        for i in range(n_samples):
            spec = generate_ic(solver, ictype,
                               gbells_ref_mean=gbells_ref_mean,
                               gbells_ref_std=gbells_ref_std,
                               gbells_kwargs=gbells_kwargs)

            # for stability samples, clone the IC so both solvers start identically
            do_stability = i < n_stability_samples
            if do_stability:
                spec_ref = spec.clone()

            for step in range(n_steps_per_trajectory + 1):
                # save main spectral state
                path = os.path.join(output_folder, f"{i}_{step}.pt")
                torch.save(spec.cpu(), path)

                # update normalization stats online
                fields = solver.spec2grid(spec)       # (3, nlat, nlon)
                winds  = solver.getuv(spec[1:])       # (2, nlat, nlon)
                f_val  = fields.mean(dim=(-1, -2))    # (3,)
                w_val  = winds.mean(dim=(-1, -2))     # (2,)

                if field_mean is None:
                    field_mean = torch.zeros_like(f_val)
                    field_M2   = torch.zeros_like(f_val)
                    wind_mean  = torch.zeros_like(w_val)
                    wind_M2    = torch.zeros_like(w_val)

                field_count, field_mean, field_M2 = welford_update(field_count, field_mean, field_M2, f_val)
                wind_count,  wind_mean,  wind_M2  = welford_update(wind_count,  wind_mean,  wind_M2,  w_val)

                # stability check: advance reference solver and compare at each step
                if do_stability and step > 0 and step <= n_stability_steps:
                    ref_path = os.path.join(stability_dir, f"{i}_{step}_ref.pt")
                    torch.save(spec_ref.cpu(), ref_path)

                    fields_main = solver.spec2grid(spec)
                    fields_ref  = ref_solver.spec2grid(spec_ref)
                    diff    = (fields_main - fields_ref).norm()
                    denom   = fields_ref.norm().clamp(min=1e-8)
                    rel_err = (diff / denom).item()
                    stability_errors.append({"sample": i, "step": step, "rel_l2_error": rel_err})
                    print(f"  [stability] sample {i}, step {step}: rel L2 error = {rel_err:.6f}")

                # advance solvers
                if step < n_steps_per_trajectory:
                    spec = solver.timestep(spec, nsteps)
                    if do_stability and step < n_stability_steps:
                        spec_ref = ref_solver.timestep(spec_ref, nsteps_ref)

            if (i + 1) % 10 == 0:
                print(f"  saved trajectory {i + 1}/{n_samples}")

    # finalise Welford variance
    field_var  = (field_M2 / field_count).reshape(-1, 1, 1)
    wind_var   = (wind_M2  / wind_count ).reshape(-1, 1, 1)
    field_mean = field_mean.reshape(-1, 1, 1)
    wind_mean  = wind_mean.reshape(-1, 1, 1)

    stats = {
        "inp_mean":  field_mean,
        "inp_var":   field_var,
        "wind_mean": wind_mean,
        "wind_var":  wind_var,
    }

    # stability summary
    if stability_errors:
        max_err  = max(e["rel_l2_error"] for e in stability_errors)
        mean_err = sum(e["rel_l2_error"] for e in stability_errors) / len(stability_errors)
        passed   = max_err < stability_threshold
        status   = "PASSED" if passed else "FAILED"
        print(f"\nStability check {status}  (max rel L2 error = {max_err:.6f}, threshold = {stability_threshold})")
    else:
        max_err = mean_err = 0.0
        passed  = True

    stability_summary = {
        "n_stability_samples": n_stability_samples,
        "n_stability_steps":   n_stability_steps,
        "dt_solver_ref":       dt_solver_ref,
        "stability_threshold": stability_threshold,
        "max_rel_l2_error":    max_err,
        "mean_rel_l2_error":   mean_err,
        "passed":              passed,
        "per_sample_errors":   stability_errors,
    }

    return stats, stability_summary


def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unavailable"


def save_metadata(output_folder, args, nsteps, stats, stability_summary):
    """Save metadata.json and metadata.txt to the output folder."""

    def tensor_to_list(t):
        return t.squeeze().tolist()

    meta = {
        "ictype":                  args.ictype,
        "dt":                      args.dt,
        "dt_solver":               args.dt_solver,
        "nsteps":                  nsteps,
        "nlat":                    args.nlat,
        "nlon":                    args.nlon,
        "n_samples":               args.n_samples,
        "n_steps_per_trajectory":  args.n_steps_per_trajectory,
        "date":                    datetime.date.today().isoformat(),
        "timestamp":               datetime.datetime.now().isoformat(),
        "git_hash":                get_git_hash(),
        "inp_mean":                tensor_to_list(stats["inp_mean"]),
        "inp_var":                 tensor_to_list(stats["inp_var"]),
        "wind_mean":               tensor_to_list(stats["wind_mean"]),
        "wind_var":                tensor_to_list(stats["wind_var"]),
        "stability_check":         stability_summary,
    }

    # JSON
    json_path = os.path.join(output_folder, "metadata.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    # human-readable TXT
    sc = stability_summary
    txt_lines = [
        "=" * 60,
        "SWAN Precomputed Dataset Metadata",
        "=" * 60,
        f"  IC type                  : {meta['ictype']}",
        f"  Model timestep (dt)      : {meta['dt']} s",
        f"  Solver sub-step          : {meta['dt_solver']} s",
        f"  Solver sub-steps per dt  : {meta['nsteps']}",
        f"  Grid                     : {meta['nlat']} x {meta['nlon']} (lat x lon)",
        f"  Number of samples        : {meta['n_samples']}",
        f"  Steps per trajectory     : {meta['n_steps_per_trajectory']}  (step 0 = IC)",
        f"  Date                     : {meta['date']}",
        f"  Timestamp                : {meta['timestamp']}",
        f"  Git hash                 : {meta['git_hash']}",
        "",
        "Normalization stats (per channel):",
        f"  inp_mean  : {meta['inp_mean']}",
        f"  inp_var   : {meta['inp_var']}",
        f"  wind_mean : {meta['wind_mean']}",
        f"  wind_var  : {meta['wind_var']}",
        "",
        "Stability check:",
        f"  dt_solver_ref            : {sc['dt_solver_ref']} s",
        f"  Samples checked          : {sc['n_stability_samples']}",
        f"  Steps checked            : {sc['n_stability_steps']}",
        f"  Threshold                : {sc['stability_threshold']}",
        f"  Max rel L2 error         : {sc['max_rel_l2_error']:.6f}",
        f"  Mean rel L2 error        : {sc['mean_rel_l2_error']:.6f}",
        f"  Result                   : {'PASSED' if sc['passed'] else 'FAILED'}",
        "=" * 60,
    ]
    txt_path = os.path.join(output_folder, "metadata.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines) + "\n")

    print(f"\nMetadata saved to {json_path} and {txt_path}")


def visualize(output_folder, index, step, solver, compare_ref=False):
    """Load and plot a saved spectral state as physical fields.

    Args:
        output_folder: path to the dataset folder
        index: trajectory index
        step: which step to load
        solver: ShallowWaterSolver instance (used for spec2grid)
        compare_ref: if True, also load the stability reference and plot side-by-side
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for visualization. Install it with: pip install matplotlib")
        return

    path = os.path.join(output_folder, f"{index}_{step}.pt")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    spec = torch.load(path, map_location="cpu")
    fields = solver.spec2grid(spec).cpu()   # (3, nlat, nlon)

    field_names = ["Geopotential (h)", "Vorticity", "Divergence"]

    if compare_ref:
        ref_path = os.path.join(output_folder, "stability_check", f"{index}_{step}_ref.pt")
        if not os.path.exists(ref_path):
            print(f"Reference file not found: {ref_path}  (running without comparison)")
            compare_ref = False
        else:
            spec_ref  = torch.load(ref_path, map_location="cpu")
            fields_ref = solver.spec2grid(spec_ref).cpu()

    n_rows = 3 if compare_ref else 1
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]   # make 2D for uniform indexing

    for col, name in enumerate(field_names):
        im = axes[0, col].imshow(fields[col].numpy(), origin="lower", aspect="auto", cmap="RdBu_r")
        axes[0, col].set_title(f"{name}\nsample {index}, step {step}")
        plt.colorbar(im, ax=axes[0, col])

        if compare_ref:
            im_ref = axes[1, col].imshow(fields_ref[col].numpy(), origin="lower", aspect="auto", cmap="RdBu_r")
            axes[1, col].set_title(f"{name} (ref, dt_solver_ref)")
            plt.colorbar(im_ref, ax=axes[1, col])

            diff = (fields[col] - fields_ref[col]).numpy()
            im_diff = axes[2, col].imshow(diff, origin="lower", aspect="auto", cmap="RdBu_r")
            axes[2, col].set_title(f"{name} (difference)")
            plt.colorbar(im_diff, ax=axes[2, col])

    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and save precomputed SWE trajectory datasets.")
    parser.add_argument("--ictype", type=str, default="random", choices=["random", "galewsky", "gbells", "williamson_case2"],
                        help="Initial condition type")
    # Gaussian bells options (used when --ictype gbells)
    parser.add_argument("--gbells_k_min", type=int, default=1, help="Minimum number of bells per channel")
    parser.add_argument("--gbells_k_max", type=int, default=8, help="Maximum number of bells per channel")
    parser.add_argument("--gbells_sigma_min_deg", type=float, default=5.0, help="Minimum bell width in degrees")
    parser.add_argument("--gbells_sigma_max_deg", type=float, default=20.0, help="Maximum bell width in degrees")
    parser.add_argument("--gbells_mean_scale", type=float, default=1.0, help="Scale applied to ref_mean")
    parser.add_argument("--gbells_std_scale", type=float, default=1.0, help="Scale applied to ref_std")
    parser.add_argument("--gbells_unsigned", action="store_true", default=False,
                        help="If set, bell amplitudes are drawn from U(0,1) instead of U(-1,1)")
    parser.add_argument("--dt", type=int, default=900,
                        help="Model timestep in seconds")
    parser.add_argument("--dt_solver", type=int, default=150,
                        help="Solver sub-step in seconds")
    parser.add_argument("--dt_solver_ref", type=int, default=5,
                        help="Solver sub-step in seconds for stability reference runs (should be smaller than dt_solver)")
    parser.add_argument("--nlat", type=int, default=128)
    parser.add_argument("--nlon", type=int, default=256)
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of trajectories to generate")
    parser.add_argument("--n_steps_per_trajectory", type=int, default=5,
                        help="Number of steps to save per trajectory (step 0 is the IC)")
    parser.add_argument("--n_stability_samples", type=int, default=10,
                        help="Number of samples to use for the stability check")
    parser.add_argument("--n_stability_steps", type=int, default=2,
                        help="Number of rollout steps to compare in the stability check")
    parser.add_argument("--stability_threshold", type=float, default=0.01,
                        help="Relative L2 error threshold for stability pass/fail")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--visualize_index", type=int, default=None,
                        help="If set, visualize this sample index after saving")
    parser.add_argument("--visualize_step", type=int, default=0,
                        help="Which step to visualize (used with --visualize_index)")
    parser.add_argument("--compare_ref", action="store_true", default=False,
                        help="If set, show stability reference alongside the main sample when visualizing")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)

    print(f"\nBuilding solver (nlat={args.nlat}, nlon={args.nlon}, dt={args.dt}, dt_solver={args.dt_solver})...")
    solver, nsteps = build_solver(args.nlat, args.nlon, args.dt, args.dt_solver, device)

    output_folder = make_output_folder(args.ictype, args.dt_solver)
    print(f"Output folder: {output_folder}")

    gbells_kwargs = None
    if args.ictype == "gbells":
        gbells_kwargs = dict(
            k_min=args.gbells_k_min,
            k_max=args.gbells_k_max,
            sigma_min_deg=args.gbells_sigma_min_deg,
            sigma_max_deg=args.gbells_sigma_max_deg,
            mean_scale=args.gbells_mean_scale,
            std_scale=args.gbells_std_scale,
            signed=not args.gbells_unsigned,
        )

    print(f"\nGenerating {args.n_samples} trajectories ({args.n_steps_per_trajectory} steps each)...")
    stats, stability_summary = save_trajectories(
        solver=solver,
        ictype=args.ictype,
        n_samples=args.n_samples,
        n_steps_per_trajectory=args.n_steps_per_trajectory,
        nsteps=nsteps,
        output_folder=output_folder,
        device=device,
        dt=args.dt,
        dt_solver_ref=args.dt_solver_ref,
        n_stability_samples=args.n_stability_samples,
        n_stability_steps=args.n_stability_steps,
        stability_threshold=args.stability_threshold,
        gbells_kwargs=gbells_kwargs,
    )

    save_metadata(output_folder, args, nsteps, stats, stability_summary)

    if args.visualize_index is not None:
        print(f"\nVisualizing sample {args.visualize_index}, step {args.visualize_step}...")
        visualize(
            output_folder=output_folder,
            index=args.visualize_index,
            step=args.visualize_step,
            solver=solver,
            compare_ref=args.compare_ref,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
