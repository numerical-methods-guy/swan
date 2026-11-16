import os
import json
import time
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from kdv.tkdv_solver import TKdVSolver
from kdv.multistep_tkdv_dataset import MultiStepTKdVDataset
from Training.tkdv_train import TKdVLightningModule


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def update_config_from_args(config, unknown_args):
    """Update config with command-line arguments in dot notation."""
    for i in range(0, len(unknown_args), 2):
        if i + 1 >= len(unknown_args):
            break
        key = unknown_args[i].lstrip("-")
        val = unknown_args[i + 1]
        keys = key.split(".")
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        try:
            if "." in val:
                val = float(val)
            else:
                val = int(val)
        except ValueError:
            pass
        current[keys[-1]] = val
    return config


def _plot_spacetime(pred_traj, truth_traj, x_arr, t_arr, out_path):
    """Space-time heatmaps: Truth | Prediction | Error side by side.

    pred_traj, truth_traj: (T, J) numpy arrays (physical space).
    x_arr: (J,) spatial grid. t_arr: (T,) time grid.
    """
    err = pred_traj - truth_traj
    vmax = max(np.abs(truth_traj).max(), np.abs(pred_traj).max())
    emax = np.abs(err).max() if np.abs(err).max() > 0 else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    extent = [t_arr[0], t_arr[-1], x_arr[0], x_arr[-1]]

    for ax, data, title, vm, cmap in [
        (axes[0], truth_traj.T, "Truth",      vmax, "RdYlBu_r"),
        (axes[1], pred_traj.T,  "Prediction", vmax, "RdYlBu_r"),
        (axes[2], err.T,        "Error",       emax, "RdBu_r"),
    ]:
        pcm = ax.imshow(
            data, aspect="auto", origin="lower", extent=extent,
            cmap=cmap, vmin=-vm, vmax=vm, interpolation="nearest",
        )
        ax.set_xlabel("time")
        ax.set_ylabel("x")
        ax.set_title(title, fontsize=12, fontweight="bold")
        fig.colorbar(pcm, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_energy_hamiltonian(pred_energy, truth_energy, pred_H, truth_H, t_arr, out_path):
    """Energy and Hamiltonian over trajectory: prediction (blue) vs truth (red dashed)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(t_arr, truth_energy, "r--", lw=1.5, label="Truth")
    axes[0].plot(t_arr, pred_energy,  "b-",  lw=1.5, label="Prediction")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("Energy")
    axes[0].set_title("Energy over trajectory", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_arr, truth_H, "r--", lw=1.5, label="Truth")
    axes[1].plot(t_arr, pred_H,  "b-",  lw=1.5, label="Prediction")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("Hamiltonian")
    axes[1].set_title("Hamiltonian over trajectory", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_fourier(pred_traj, truth_traj, t_arr, out_path, max_modes=16):
    """|û_k(t)| for each wavenumber k: prediction (blue) vs truth (red dashed).

    pred_traj, truth_traj: (T, J) numpy arrays (physical space).
    """
    pred_fft  = np.abs(np.fft.rfft(pred_traj,  axis=1))   # (T, J//2+1)
    truth_fft = np.abs(np.fft.rfft(truth_traj, axis=1))

    n_modes = min(max_modes, pred_fft.shape[1])
    ncols = 4
    nrows = (n_modes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.5), squeeze=False)

    for k in range(n_modes):
        r, c = divmod(k, ncols)
        ax = axes[r][c]
        ax.plot(t_arr, truth_fft[:, k], "r--", lw=1.2, label="Truth")
        ax.plot(t_arr, pred_fft[:, k],  "b-",  lw=1.2, label="Pred")
        ax.set_title(f"k={k}")
        ax.set_xlabel("t")
        ax.set_ylabel("|û_k|")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7)

    for k in range(n_modes, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")

    plt.suptitle("|Fourier coefficients| over time", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_physics(traj_phys, solver):
    """Compute energy and Hamiltonian for each frame of a physical trajectory.

    Args:
        traj_phys: (T, J) numpy array.
        solver: TKdVSolver instance.

    Returns:
        energy_arr: (T,) numpy array.
        H_arr: (T,) numpy array.
    """
    energy_arr, H_arr = [], []
    for frame in traj_phys:
        u = torch.tensor(frame, dtype=torch.float32, device=solver.x.device)
        u_hat = solver.phys2spec(u)
        energy_arr.append(solver.energy(u_hat).item())
        H_arr.append(solver.hamiltonian(u_hat).item())
    return np.array(energy_arr), np.array(H_arr)


def _run_single_ic_inference(
    model,
    dataset,
    loss_fn,
    metrics_dict,
    nsteps,
    autoreg_steps,
    device,
    output_dir=None,
    ic_index=None,
    save_plots=True,
    model_name="Model",
):
    """Run one autoregressive rollout from a single initial condition drawn from dataset.

    The dataset (MultiStepTKdVDataset) handles IC generation and normalization.
    Stats (inp_mean/var) must already be set on the dataset before calling this function.

    Returns:
        step_metrics: Dict mapping metric name to list of per-step values.
        ml_time: Wall-clock seconds for the ML rollout.
        solver_time: Wall-clock seconds for the reference solver rollout.
    """
    inp_mean = dataset.inp_mean
    inp_var  = dataset.inp_var
    inp_std  = torch.sqrt(inp_var)
    solver   = dataset.solver

    # dataset returns normalized (inp, tar): inp shape (J,), tar shape (n_rollout_steps, J)
    inp_fields, tar_fields = dataset[ic_index]
    inp_fields = inp_fields.to(device)
    tar_fields  = tar_fields.to(device)
    n_metric_steps = min(autoreg_steps, tar_fields.shape[0])

    # unnormalized IC for the reference solver
    inp_phys = inp_fields * inp_std + inp_mean   # (J,) physical

    # --- Timing pass: ML model ---
    prd = inp_fields.unsqueeze(0).unsqueeze(0)   # (1, 1, J) normalized

    if device.type == "cuda":
        torch.cuda.synchronize()
    ml_start = time.perf_counter()

    for _ in range(autoreg_steps):
        prd = model.model(prd)                   # (1, 1, J) normalized

    if device.type == "cuda":
        torch.cuda.synchronize()
    ml_time = time.perf_counter() - ml_start

    # --- Timing pass: reference numerical solver ---
    u_hat_ref = solver.phys2spec(inp_phys)

    if device.type == "cuda":
        torch.cuda.synchronize()
    solver_start = time.perf_counter()

    for _ in range(autoreg_steps):
        u_hat_ref = solver.timestep(u_hat_ref, nsteps=nsteps)

    if device.type == "cuda":
        torch.cuda.synchronize()
    solver_time = time.perf_counter() - solver_start

    # --- Metrics and output pass ---
    prd = inp_fields.unsqueeze(0).unsqueeze(0)   # (1, 1, J) normalized

    step_metrics = {key: [] for key in metrics_dict.keys()}
    step_metrics["loss"] = []

    # collect full trajectories for plotting (physical / unnormalized)
    pred_frames  = [inp_phys.cpu().numpy()]
    truth_frames = [inp_phys.cpu().numpy()]

    for step in range(1, autoreg_steps + 1):
        prd = model.model(prd)                              # (1, 1, J) normalized
        prd_phys = prd.squeeze(0).squeeze(0) * inp_std + inp_mean   # (J,) physical
        pred_frames.append(prd_phys.cpu().numpy())

        if step <= n_metric_steps:
            ref_norm = tar_fields[step - 1]                # (J,) normalized
            ref_phys = ref_norm * inp_std + inp_mean       # (J,) physical
            truth_frames.append(ref_phys.cpu().numpy())

            for name, metric_fn in metrics_dict.items():
                step_metrics[name].append(
                    metric_fn(prd.squeeze(0), ref_norm.unsqueeze(0)).item()
                )
            step_metrics["loss"].append(
                loss_fn(prd.squeeze(0), ref_norm.unsqueeze(0)).item()
            )
        else:
            # pad truth trajectory with last available frame if autoreg_steps > n_metric_steps
            truth_frames.append(truth_frames[-1])

    if output_dir is not None and save_plots:
        pred_traj  = np.stack(pred_frames,  axis=0)   # (T, J)
        truth_traj = np.stack(truth_frames, axis=0)   # (T, J)
        x_arr = solver.x.cpu().numpy()
        dt    = dataset.dt
        t_arr = np.arange(pred_traj.shape[0]) * dt
        prefix = f"ic{ic_index:03d}" if ic_index is not None else "ic"

        _plot_spacetime(
            pred_traj, truth_traj, x_arr, t_arr,
            os.path.join(output_dir, f"{prefix}_spacetime.png"),
        )

        pred_energy,  pred_H  = compute_physics(pred_traj,  solver)
        truth_energy, truth_H = compute_physics(truth_traj, solver)
        _plot_energy_hamiltonian(
            pred_energy, truth_energy, pred_H, truth_H, t_arr,
            os.path.join(output_dir, f"{prefix}_energy_hamiltonian.png"),
        )

        _plot_fourier(
            pred_traj, truth_traj, t_arr,
            os.path.join(output_dir, f"{prefix}_fourier.png"),
        )

    return step_metrics, ml_time, solver_time


def _aggregate_multi_ic_metrics(all_step_metrics, all_ml_times, all_solver_times):
    """Aggregate per-IC step-metric lists into a flat summary dict.

    Per-step values are flattened across all ICs before computing mean/std so
    that the reported statistics reflect both temporal and IC-to-IC variability.
    Timing statistics include mean speedup of the ML model over the numerical solver.

    Args:
        all_step_metrics: List of per-IC dicts mapping metric name to list of per-step floats.
        all_ml_times: List of per-IC ML rollout wall-clock times in seconds.
        all_solver_times: List of per-IC numerical solver wall-clock times in seconds.

    Returns:
        Dict with {metric}_mean / {metric}_std entries plus timing statistics.
    """
    summary = {}

    for key in all_step_metrics[0].keys():
        all_values = [v for ic in all_step_metrics for v in ic[key]]
        summary[f"{key}_mean"] = (
            float(np.mean(all_values)) if all_values else float("nan")
        )
        summary[f"{key}_std"] = (
            float(np.std(all_values)) if all_values else float("nan")
        )

    summary["ml_time_mean"] = float(np.mean(all_ml_times))
    summary["ml_time_std"] = float(np.std(all_ml_times))
    summary["solver_time_mean"] = float(np.mean(all_solver_times))
    summary["solver_time_std"] = float(np.std(all_solver_times))
    summary["speedup_mean"] = (
        summary["solver_time_mean"] / summary["ml_time_mean"]
        if summary["ml_time_mean"] > 0
        else float("nan")
    )

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
    num_ics=1,
    save_plots=True,
    device=torch.device("cpu"),
    ic_start_index=0,
):
    """Perform autoregressive inference over one or more initial conditions.

    Plots are written for every IC. Per-IC timing is printed to stdout and aggregated
    in the returned summary.

    Args:
        model: TKdVLightningModule.
        dataset: MultiStepTKdVDataset instance with training stats set.
        loss_fn: Loss function callable.
        metrics_dict: Dict mapping metric name to callable.
        output_dir: Directory to save results.
        nsteps: Number of solver substeps per model timestep.
        model_name: Label used in plot titles.
        autoreg_steps: Number of autoregressive steps per rollout.
        num_ics: Number of initial conditions to evaluate.
        save_plots: Whether to save comparison plots.
        device: Torch device.

    Returns:
        Summary dict with aggregated mean/std metrics and timing statistics.
    """
    model.eval()
    model.to(device)
    os.makedirs(output_dir, exist_ok=True)

    all_step_metrics = []
    all_ml_times = []
    all_solver_times = []

    print(
        f"Starting Autoregressive Inference ({autoreg_steps} steps, {num_ics} IC(s))..."
    )

    with torch.no_grad():
        for i, ic_idx in enumerate(range(ic_start_index, ic_start_index + num_ics)):
            print(f"\n--- IC {i + 1}/{num_ics} (dataset index {ic_idx}) ---")
            step_metrics, ml_time, solver_time = _run_single_ic_inference(
                model=model,
                dataset=dataset,
                loss_fn=loss_fn,
                metrics_dict=metrics_dict,
                nsteps=nsteps,
                autoreg_steps=autoreg_steps,
                device=device,
                output_dir=output_dir,
                ic_index=ic_idx,
                save_plots=save_plots,
                model_name=model_name,
            )
            all_step_metrics.append(step_metrics)
            all_ml_times.append(ml_time)
            all_solver_times.append(solver_time)

            speedup = solver_time / ml_time if ml_time > 0 else float("nan")
            print(f"  ML rollout:     {ml_time:.3f}s")
            print(f"  Solver rollout: {solver_time:.3f}s")
            print(f"  Speedup:        {speedup:.1f}x")

    return _aggregate_multi_ic_metrics(all_step_metrics, all_ml_times, all_solver_times)


def main():
    parser = argparse.ArgumentParser(
        description="Autoregressive inference for tKdV Paradis model"
    )
    parser.add_argument(
        "--config", type=str, default="config_tkdv.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./tkdv_forecast_results",
        help="Base directory under which a timestamped forecast_N subfolder is created",
    )
    parser.add_argument(
        "--autoreg_steps", type=int, default=10, help="Number of autoregressive steps"
    )
    parser.add_argument(
        "--num_ics",
        type=int,
        default=1,
        help="Number of initial conditions to evaluate",
    )
    parser.add_argument(
        "--ic_start_index",
        type=int,
        default=0,
        help="Dataset index of the first IC to use (default 0)",
    )
    parser.add_argument(
        "--ic_type",
        type=str,
        default="precomputed",
        choices=["precomputed"],
        help="Initial condition type (only precomputed supported for tKdV)",
    )
    parser.add_argument(
        "--precomputed_folder",
        type=str,
        default=None,
        help="Path to precomputed dataset folder (required when --ic_type precomputed)",
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        required=True,
        help="Path to stats.pt saved by a training run (contains inp_mean/var, input_step_idx)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda/cpu)"
    )
    parser.add_argument("--no_plots", action="store_true", help="Disable saving plots")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n_rollout_steps",
        type=int,
        default=40,
        help="Autoregressive rollout steps for the dataset's reference trajectory",
    )
    parser.add_argument(
        "--input_step_idx",
        type=int,
        default=None,
        help="Solver warm-up steps before NN input. Defaults to the value saved in stats.pt.",
    )

    args, unknown_args = parser.parse_known_args()

    # auto-generate a timestamped, non-colliding output subfolder
    import datetime as _dt
    _date = _dt.date.today().strftime("%Y%m%d")
    _idx = 1
    while os.path.exists(os.path.join(args.output_dir, f"{_date}_forecast{_idx}")):
        _idx += 1
    output_dir = os.path.join(args.output_dir, f"{_date}_forecast{_idx}")
    os.makedirs(output_dir, exist_ok=True)

    pl.seed_everything(args.seed)

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = load_config(args.config)
    config = update_config_from_args(config, unknown_args)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=" * 70)
    print("FORECAST CONFIGURATION")
    print("=" * 70)
    print(f"Checkpoint:          {args.checkpoint}")
    print(f"Device:              {device}")
    print(f"Autoregressive steps:{args.autoreg_steps}")
    print(f"IC type:             {args.ic_type}")
    print(f"Initial conditions:  {args.num_ics}")
    print(f"Output directory:    {output_dir}")
    print(f"Save plots:          {not args.no_plots}")
    print("=" * 70 + "\n")

    print("Loading normalization stats...")
    stats = torch.load(args.stats_path, map_location=device)
    inp_mean = stats["inp_mean"].to(device)
    inp_var  = stats["inp_var"].to(device)
    input_step_idx = args.input_step_idx if args.input_step_idx is not None else int(stats.get("input_step_idx", 0))
    print(f"  stats_path:     {args.stats_path}")
    print(f"  input_step_idx: {input_step_idx}\n")

    print(f"Loading checkpoint: {args.checkpoint}")
    model_module = TKdVLightningModule(config, inp_mean, inp_var)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_module.load_state_dict(checkpoint["state_dict"], strict=False)
    model_module.eval()
    print("Checkpoint loaded successfully.\n")

    if args.ic_type == "precomputed" and not args.precomputed_folder:
        raise ValueError("--precomputed_folder is required when --ic_type precomputed")

    print("Setting up dataset and solver...")
    dt = config["data"]["dt"]
    nsteps = int(config["data"]["dt"] / config["data"]["dt_solver"])

    # read physics params from folder metadata
    with open(os.path.join(args.precomputed_folder, "metadata.json")) as f:
        meta = json.load(f)
    J     = int(meta["J"])
    D     = float(meta["D"])
    E0    = float(meta["E0"])
    L0    = float(meta["L0"])
    theta = float(meta["theta"])

    dataset = MultiStepTKdVDataset(
        dt=dt,
        nsteps=nsteps,
        n_rollout_steps=args.n_rollout_steps,
        input_step_idx=input_step_idx,
        J=J,
        D=D,
        E0=E0,
        L0=L0,
        theta=theta,
        initial_condition="precomputed",
        num_examples=args.num_ics,
        device=device,
        normalize=True,
        precomputed_folder=args.precomputed_folder,
    )

    # override dataset stats with training stats so forecast uses identical normalization
    dataset.inp_mean = inp_mean
    dataset.inp_var  = inp_var

    # attach solver for reference trajectory and physics diagnostics
    dataset.solver = TKdVSolver(J=J, dt=dt, D=D, E0=E0, L0=L0).to(device)
    dataset.dt = dt

    metrics_dict = {
        "MSE": lambda p, t: F.mse_loss(p, t),
        "MAE": lambda p, t: F.l1_loss(p, t),
    }

    results = autoregressive_inference(
        model=model_module,
        dataset=dataset,
        loss_fn=model_module.loss_fn,
        metrics_dict=metrics_dict,
        output_dir=output_dir,
        nsteps=nsteps,
        model_name="ParadisTKdV",
        autoreg_steps=args.autoreg_steps,
        num_ics=args.num_ics,
        save_plots=(not args.no_plots),
        device=device,
        ic_start_index=args.ic_start_index,
    )

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    for key in ["loss", "MSE", "MAE"]:
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        if mean_key in results:
            print(f"{key:12s}: {results[mean_key]:.6f} ± {results[std_key]:.6f}")
    print("-" * 70)
    print(
        f"{'ML time (s)':12s}: {results['ml_time_mean']:.3f} ± {results['ml_time_std']:.3f}"
    )
    print(
        f"{'Solver (s)':12s}: {results['solver_time_mean']:.3f} ± {results['solver_time_std']:.3f}"
    )
    print(f"{'Speedup':12s}: {results['speedup_mean']:.1f}x")

    metrics_path = os.path.join(output_dir, "metrics.csv")
    pd.DataFrame([results]).to_csv(metrics_path, index=False)

    print("=" * 70)
    print(f"Metrics saved to: {metrics_path}")
    if not args.no_plots:
        print(f"Plots saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
