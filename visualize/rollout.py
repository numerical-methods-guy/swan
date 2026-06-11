"""
rollout.py
==========

Backend utilities for ``visualize forecast``.

This module is intentionally not a public command-line tool.  Users should run
``python -m visualize forecast ...``.  The CLI calls the functions here to do
the heavy forecast/rollout preparation.

Design goals
------------
1. Do not modify the original SWAN ``forecast.py``.
2. Reuse the original forecast helpers when running inside the SWAN repository.
3. Save additional comparison-friendly files, especially ``per_step_metrics.csv``.
4. Provide a synthetic rollout generator so the visualization pipeline can be
   tested without the real model, PyTorch Lightning, torch_harmonics, or the
   shallow-water solver.

Real SWAN workflow
------------------
When ``synthetic`` is false, this module imports the original ``forecast.py`` and
uses its classes/functions, including:

* ``load_config``
* ``SWELightningModule``
* ``PdeDatasetWithWinds``
* ``_run_single_ic_inference``
* ``_aggregate_multi_ic_metrics``
* ``compute_energy_spectra``

The original helper already returns per-step metrics from a rollout.  The
original script only writes aggregate ``metrics.csv``; this module additionally
writes ``per_step_metrics.csv`` without changing ``forecast.py``.

Synthetic workflow
------------------
For tests, ``create_synthetic_rollouts`` writes fake forecast folders with the
same general structure as real outputs:

    rollout_results/Adam/
      ic000_prediction_000.pt
      ic000_truth_000.pt
      ...
      metrics.csv
      per_step_metrics.csv

The synthetic fields are smooth 2D arrays with three channels
``[h, vorticity, divergence]``.  This makes it possible to test all plotting
functions deterministically.
"""

from __future__ import annotations

import csv
import importlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
from visualize import mpl_style  # noqa: F401  # apply M2PI report typography
import matplotlib.pyplot as plt

try:  # torch is available in the SWAN environment and in this execution image.
    import torch
except Exception:  # pragma: no cover - kept for robustness outside this env.
    torch = None  # type: ignore


CHANNEL_TO_INDEX = {"h": 0, "vorticity": 1, "divergence": 2}
ERROR_METRIC_TO_COLUMN = {
    "loss": "loss",
    "l1": "L1_error",
    "l2": "L2_error",
    "w11": "W11_error",
}


@dataclass
class RolloutRun:
    """Prepared rollout information for one optimizer label."""

    label: str
    rollout_dir: Path
    metrics: Dict[str, float]
    per_step: pd.DataFrame


@dataclass
class FieldSnapshot:
    """Prediction/truth tensors loaded at one rollout step."""

    label: str
    step: int
    prediction_fields: np.ndarray  # shape: (3, nlat, nlon)
    truth_fields: np.ndarray       # shape: (3, nlat, nlon)


@dataclass
class SkillHorizonCurve:
    """Skill horizon curve for one optimizer label."""

    label: str
    gammas: np.ndarray
    horizons: np.ndarray
    crossed: np.ndarray


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_checkpoint(run_dir: str | Path, checkpoint_choice: str = "best") -> Path:
    """Find a checkpoint inside a training run directory.

    Parameters
    ----------
    run_dir:
        Directory such as ``results/adam/version_0``.
    checkpoint_choice:
        ``best`` or ``last``.  ``best`` means the non-``last.ckpt`` checkpoint
        produced by PyTorch Lightning's ``ModelCheckpoint`` callback.  If
        several candidate best checkpoints exist, the function tries to parse
        ``val_loss=...`` from the filename and selects the lowest value.
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")

    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")

    last = ckpt_dir / "last.ckpt"
    if checkpoint_choice == "last":
        if last.exists():
            return last
        raise FileNotFoundError(f"Requested last checkpoint but last.ckpt was not found in {ckpt_dir}")

    if checkpoint_choice != "best":
        raise ValueError("checkpoint_choice must be 'best' or 'last'")

    candidates = [p for p in ckpts if p.name != "last.ckpt"]
    if not candidates:
        # If training saved only last.ckpt, use it rather than failing.
        if last.exists():
            return last
        return ckpts[0]

    def parsed_val_loss(path: Path) -> float:
        match = re.search(r"val_loss[=\-]([0-9.eE+\-]+)", path.name)
        if not match:
            return float("inf")
        try:
            return float(match.group(1))
        except ValueError:
            return float("inf")

    scored = [(parsed_val_loss(path), path.stat().st_mtime, path) for path in candidates]
    if any(np.isfinite(score[0]) for score in scored):
        return min(scored, key=lambda item: item[0])[2]

    # No parseable val_loss.  Use the most recently modified candidate as a
    # reasonable fallback.
    return max(scored, key=lambda item: item[1])[2]


def checkpoints_from_runs(run_dirs: Sequence[str | Path], checkpoint_choice: str) -> List[Path]:
    """Resolve one checkpoint path per training run directory."""
    return [find_checkpoint(run_dir, checkpoint_choice) for run_dir in run_dirs]


# ---------------------------------------------------------------------------
# Real rollout execution using original forecast.py
# ---------------------------------------------------------------------------

def run_real_rollouts(
    checkpoints: Sequence[str | Path],
    labels: Sequence[str],
    config_path: str | Path,
    rollout_dir: str | Path,
    autoreg_steps: int,
    output_freq: int,
    num_ics: int,
    ic_type: str,
    seed: int,
    channel: str,
    device: Optional[str] = None,
    nlat: Optional[int] = None,
    nlon: Optional[int] = None,
) -> List[RolloutRun]:
    """Run real SWAN rollouts using original ``forecast.py`` helpers.

    This function is expected to be run from inside the SWAN repository.  It
    imports ``forecast.py`` at runtime.  If dependencies are missing, the error
    message will point users toward the synthetic mode for tests.

    Forecast fairness
    -----------------
    The seed is reset before each optimizer rollout.  Therefore, if
    ``ic_type=random`` and all other forecast flags match, each optimizer is
    evaluated on the same sequence of random initial conditions.  This forecast
    seed is separate from the training seed used by ``train_*.py``.
    """
    if len(checkpoints) != len(labels):
        raise ValueError("Expected the same number of checkpoints and labels.")
    if channel not in CHANNEL_TO_INDEX:
        raise ValueError(f"Unknown channel {channel!r}. Expected one of {list(CHANNEL_TO_INDEX)}")
    if output_freq < 1:
        raise ValueError("output_freq must be >= 1")

    try:
        forecast = importlib.import_module("forecast")
    except Exception as exc:
        raise RuntimeError(
            "Could not import the original forecast.py. Run this command from the SWAN "
            "repository root, or use --synthetic_demo to test the plotting pipeline."
        ) from exc

    if torch is None:
        raise RuntimeError("PyTorch is required for real rollout execution.")

    try:
        import pytorch_lightning as pl
    except Exception as exc:
        raise RuntimeError("pytorch_lightning is required for real rollout execution.") from exc

    fallback_config = forecast.load_config(str(config_path))
    device_obj = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rollout_dir = Path(rollout_dir)
    rollout_dir.mkdir(parents=True, exist_ok=True)

    results: List[RolloutRun] = []
    for checkpoint_path, label in zip(checkpoints, labels):
        # Reset the forecast-time RNG before each optimizer.  forecast.py draws
        # random ICs inside _run_single_ic_inference; reseeding here makes every
        # optimizer consume the same IC sequence while keeping this evaluation
        # seed independent from the seed used during training.
        pl.seed_everything(seed, workers=True)

        output_dir = rollout_dir / sanitize_label(label)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print(f"Running rollout for {label}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Output:     {output_dir}")
        print("=" * 70)

        checkpoint = torch.load(str(checkpoint_path), map_location=device_obj)
        run_config = checkpoint.get("hyper_parameters", {}).get("config")
        if not isinstance(run_config, dict):
            run_config = fallback_config
            print("Checkpoint does not contain a saved config; using --config instead.")
        else:
            dims = run_config.get("data", {})
            print(
                "Using checkpoint config: "
                f"nlat={dims.get('nlat')}, nlon={dims.get('nlon')}, "
                f"dt={dims.get('dt')}, dt_solver={dims.get('dt_solver')}"
            )

        # Resolution overrides are evaluation-time settings.  They update the
        # data grid used by the forecast dataset and spherical transforms while
        # leaving checkpoint weights unchanged.
        if nlat is not None:
            run_config["data"]["nlat"] = int(nlat)
        if nlon is not None:
            run_config["data"]["nlon"] = int(nlon)
        if nlat is not None or nlon is not None:
            print(
                "Using forecast resolution override: "
                f"nlat={run_config['data']['nlat']}, nlon={run_config['data']['nlon']}"
            )

        model_module = forecast.SWELightningModule(run_config)
        state_dict = checkpoint["state_dict"]
        # Match original forecast.py behavior: remove old W11 buffers if present.
        for key in ["metric_w11.k_phi_mesh", "metric_w11.k_theta_mesh"]:
            if key in state_dict:
                del state_dict[key]
        try:
            model_module.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not load checkpoint {checkpoint_path}. The checkpoint architecture "
                "does not match its saved config or the fallback --config. Re-train with the "
                "current config, or pass/use the config that was used to train this checkpoint."
            ) from exc
        model_module.eval()

        dt = run_config["data"]["dt"]
        nsteps = dt // run_config["data"]["dt_solver"]
        dataset = forecast.PdeDatasetWithWinds(
            dt=dt,
            nsteps=nsteps,
            dims=(run_config["data"]["nlat"], run_config["data"]["nlon"]),
            normalize=True,
            device=device_obj,
        )
        dataset.sht = dataset.solver.sht
        metrics_dict = {
            "L1_error": model_module.metric_l1,
            "L2_error": model_module.metric_l2,
            "W11_error": model_module.metric_w11,
        }

        # Forecast comparison figures should match the saved rollout tensors:
        # the first rollout IC only, with no averaging across additional ICs.
        actual_num_ics = 1

        all_step_metrics: List[Dict[str, List[float]]] = []
        all_ml_times: List[float] = []
        all_solver_times: List[float] = []

        model_module.eval()
        model_module.to(device_obj)
        forecast._move_dataset_to_device(dataset, device_obj)

        with torch.no_grad():
            for ic_index in range(actual_num_ics):
                step_metrics, ml_time, solver_time = forecast._run_single_ic_inference(
                    model=model_module,
                    dataset=dataset,
                    loss_fn=model_module.loss_fn,
                    metrics_dict=metrics_dict,
                    nsteps=nsteps,
                    autoreg_steps=autoreg_steps,
                    device=device_obj,
                    ic_type=ic_type,
                    output_dir=str(output_dir) if ic_index == 0 else None,
                    ic_index=ic_index,
                    plot_channel=CHANNEL_TO_INDEX[channel],
                    save_plots=(ic_index == 0),
                    spectral_analysis=(ic_index == 0),
                    model_name=label,
                    output_freq=output_freq,
                )
                all_step_metrics.append(step_metrics)
                all_ml_times.append(float(ml_time))
                all_solver_times.append(float(solver_time))

        summary = forecast._aggregate_multi_ic_metrics(all_step_metrics, all_ml_times, all_solver_times)
        pd.DataFrame([summary]).to_csv(output_dir / "metrics.csv", index=False)
        write_per_step_metrics_csv(output_dir / "per_step_metrics.csv", all_step_metrics)
        results.append(
            RolloutRun(
                label=label,
                rollout_dir=output_dir,
                metrics={k: float(v) for k, v in summary.items()},
                per_step=pd.read_csv(output_dir / "per_step_metrics.csv"),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Synthetic rollout generation for tests and examples
# ---------------------------------------------------------------------------

def create_synthetic_rollouts(
    labels: Sequence[str],
    rollout_dir: str | Path,
    autoreg_steps: int = 20,
    output_freq: int = 5,
    num_ics: int = 1,
    seed: int = 42,
    nlat: int = 32,
    nlon: int = 64,
) -> List[RolloutRun]:
    """Create deterministic artificial rollout folders for testing.

    The fake data are not physically meaningful.  They simply mimic the file
    structure and metric trends of real rollouts so every plotting path can be
    tested quickly.
    """
    if torch is None:
        raise RuntimeError("Synthetic rollout generation uses torch.save and requires PyTorch.")
    if output_freq < 1:
        raise ValueError("output_freq must be >= 1")

    rng = np.random.default_rng(seed)
    rollout_dir = Path(rollout_dir)
    rollout_dir.mkdir(parents=True, exist_ok=True)
    pad_width = len(str(autoreg_steps))
    saved_steps = saved_rollout_steps(autoreg_steps, output_freq)

    results: List[RolloutRun] = []
    for label_index, label in enumerate(labels):
        safe_label = sanitize_label(label)
        out = rollout_dir / safe_label
        out.mkdir(parents=True, exist_ok=True)

        # Give each optimizer a different error growth rate so the plots are
        # visibly distinct in tests.
        quality = 0.65 + 0.18 * label_index
        all_step_metrics: List[Dict[str, List[float]]] = []
        ml_times: List[float] = []
        solver_times: List[float] = []

        for ic in range(1):
            phase = 0.3 * ic
            step_metrics = {"L1_error": [], "L2_error": [], "W11_error": [], "loss": []}

            for step in range(1, autoreg_steps + 1):
                truth = synthetic_truth_field(step, nlat, nlon, phase=phase)
                pred = synthetic_prediction_field(truth, step, quality, rng)
                diff = pred - truth
                l1 = float(np.mean(np.abs(diff)))
                l2 = float(np.sqrt(np.mean(diff**2)))
                w11 = float(l2 + 0.25 * np.sqrt(np.mean(np.gradient(diff[1])[0] ** 2)))
                loss = float(np.mean(diff**2))
                step_metrics["L1_error"].append(l1)
                step_metrics["L2_error"].append(l2)
                step_metrics["W11_error"].append(w11)
                step_metrics["loss"].append(loss)

            all_step_metrics.append(step_metrics)
            # Fake timing is per full rollout horizon and per IC, matching the
            # meaning of the real forecast.py aggregate timing metrics.
            ml_times.append(float((0.02 + 0.006 * label_index) * autoreg_steps))
            solver_times.append(float(0.35 * autoreg_steps))

            if ic == 0:
                for step in saved_steps:
                    truth = synthetic_truth_field(step, nlat, nlon, phase=phase)
                    pred = synthetic_prediction_field(truth, step, quality, rng)
                    save_field_pt(out / f"ic{ic:03d}_prediction_{step:0{pad_width}d}.pt", pred)
                    save_field_pt(out / f"ic{ic:03d}_truth_{step:0{pad_width}d}.pt", truth)
                    # Also save a small individual comparison PNG so the fake
                    # rollout folders resemble real forecast.py output.
                    save_individual_comparison_png(
                        out / f"ic{ic:03d}_comparison_{step:0{pad_width}d}.png", pred, truth, channel_index=1, step=step
                    )
                    save_individual_spectra_png(
                        out / f"ic{ic:03d}_spectra_{step:0{pad_width}d}.png", pred, truth, step=step, label=label
                    )

        summary = aggregate_step_metrics(all_step_metrics, ml_times, solver_times)
        pd.DataFrame([summary]).to_csv(out / "metrics.csv", index=False)
        write_per_step_metrics_csv(out / "per_step_metrics.csv", all_step_metrics)
        results.append(
            RolloutRun(
                label=label,
                rollout_dir=out,
                metrics=summary,
                per_step=pd.read_csv(out / "per_step_metrics.csv"),
            )
        )

    return results


def synthetic_truth_field(step: int, nlat: int, nlon: int, phase: float = 0.0) -> np.ndarray:
    """Create a smooth three-channel fake shallow-water field."""
    y = np.linspace(-np.pi, np.pi, nlat)
    x = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    t = 0.08 * step + phase
    h = np.sin(xx + 0.6 * t) * np.cos(yy) + 0.2 * np.cos(2 * xx - t)
    vort = np.sin(2 * xx - t) * np.sin(yy + 0.3 * t) + 0.35 * np.cos(3 * yy)
    div = 0.5 * np.cos(xx - 0.4 * t) * np.cos(2 * yy + t)
    return np.stack([h, vort, div], axis=0).astype(np.float32)


def synthetic_prediction_field(truth: np.ndarray, step: int, quality: float, rng: np.random.Generator) -> np.ndarray:
    """Perturb truth with smooth bias plus small random noise."""
    _, nlat, nlon = truth.shape
    y = np.linspace(-np.pi, np.pi, nlat)
    x = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    amp = quality * (0.015 + 0.0025 * step)
    smooth_error = np.stack(
        [
            amp * np.sin(xx + 0.2 * step) * np.sin(yy),
            amp * np.cos(2 * xx - 0.1 * step) * np.cos(yy),
            amp * np.sin(xx - yy + 0.1 * step),
        ],
        axis=0,
    )
    noise = rng.normal(scale=amp * 0.15, size=truth.shape)
    return (truth + smooth_error + noise).astype(np.float32)


def save_field_pt(path: Path, fields: np.ndarray) -> None:
    """Save a field dictionary in the same spirit as forecast.py output."""
    if torch is None:
        raise RuntimeError("PyTorch is required to write .pt files.")
    winds = np.stack([fields[1], fields[2]], axis=0).astype(np.float32)
    torch.save({"fields": torch.tensor(fields), "winds": torch.tensor(winds)}, path)


def save_individual_comparison_png(path: Path, pred: np.ndarray, truth: np.ndarray, channel_index: int, step: int) -> None:
    """Save a simple prediction/truth/error PNG for synthetic data."""
    error = pred[channel_index] - truth[channel_index]
    vmax = float(max(abs(pred[channel_index]).max(), abs(truth[channel_index]).max(), 1e-8))
    errmax = float(max(abs(error).max(), 1e-8))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, data, title, vmin, vmax_i, cmap in [
        (axes[0], pred[channel_index], f"Prediction t={step}", -vmax, vmax, "twilight_shifted"),
        (axes[1], truth[channel_index], f"Truth t={step}", -vmax, vmax, "twilight_shifted"),
        (axes[2], error, f"Error t={step}", -errmax, errmax, "RdBu_r"),
    ]:
        im = ax.imshow(data, vmin=vmin, vmax=vmax_i, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_individual_spectra_png(path: Path, pred: np.ndarray, truth: np.ndarray, step: int, label: str) -> None:
    """Save a simple forecast.py-style spectra PNG for synthetic data."""
    # Synthetic rollouts do not have a shallow-water solver or SHT attached, so
    # this helper deliberately uses the FFT fallback only to create test images
    # for the split spectral animation path.
    pred_spectra = compute_simple_energy_spectra(pred)
    truth_spectra = compute_simple_energy_spectra(truth)
    titles = [
        "Rotational kinetic energy",
        "Divergent kinetic energy",
        "Potential energy",
        "Total energy",
    ]
    keys = ["rotational", "divergent", "potential", "total"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, title, key in zip(axes.ravel(), titles, keys):
        k = truth_spectra["wavenumbers"][1:]
        ax.loglog(k, pred_spectra[key][1:], "b-", linewidth=1.8, label=f"{label} prediction")
        ax.loglog(k, truth_spectra[key][1:], "r--", linewidth=1.8, label="Ground truth")
        ax.set_title(f"{title} (t={step})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Wavenumber $l$")
        ax.set_ylabel("Power spectrum")
        ax.grid(True, which="both", alpha=0.3)
        if len(k) > 0:
            ax.set_xlim(left=max(1, k[0]), right=k[-1])
        ax.legend(fontsize=8)
    fig.suptitle(f"Synthetic spectra: {label} step {step}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loading and preparing rollout data for plotting
# ---------------------------------------------------------------------------

def load_rollout_runs(rollout_dirs: Sequence[str | Path], labels: Sequence[str]) -> List[RolloutRun]:
    """Load metrics and per-step metrics for existing rollout directories."""
    if len(rollout_dirs) != len(labels):
        raise ValueError("Expected the same number of rollout_dirs and labels.")
    runs: List[RolloutRun] = []
    for path, label in zip(rollout_dirs, labels):
        rollout_path = Path(path)
        metrics_path = rollout_path / "metrics.csv"
        per_step_path = rollout_path / "per_step_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics.csv in {rollout_path}")
        if not per_step_path.exists():
            raise FileNotFoundError(
                f"Missing per_step_metrics.csv in {rollout_path}. This file is written by visualize.rollout."
            )
        metrics_df = pd.read_csv(metrics_path)
        metrics = {col: float(metrics_df.iloc[0][col]) for col in metrics_df.columns}
        runs.append(
            RolloutRun(
                label=label,
                rollout_dir=rollout_path,
                metrics=metrics,
                per_step=pd.read_csv(per_step_path),
            )
        )
    return runs


def saved_rollout_steps(autoreg_steps: int, output_freq: int) -> List[int]:
    """Return steps saved by forecast.py style output rules."""
    steps = [0]
    steps.extend(step for step in range(1, autoreg_steps + 1) if step % output_freq == 0)
    if autoreg_steps not in steps and autoreg_steps % output_freq == 0:
        steps.append(autoreg_steps)
    return sorted(set(steps))


def parse_saved_steps(rollout_dir: str | Path) -> List[int]:
    """Parse saved prediction steps from one rollout directory."""
    rollout_dir = Path(rollout_dir)
    return sorted({step for step, _ in _preferred_step_files(rollout_dir, "prediction", ".pt")})


def common_summary_step(rollout_dirs: Sequence[str | Path], requested: str) -> int:
    """Choose the rollout step used for final grid/spectral comparisons."""
    all_steps = [set(parse_saved_steps(path)) for path in rollout_dirs]
    if not all_steps or any(not steps for steps in all_steps):
        raise FileNotFoundError("Could not find saved prediction_*.pt files in one or more rollout directories.")

    common = set.intersection(*all_steps)
    if not common:
        raise ValueError("No common saved rollout step exists across all optimizers.")

    if requested in ("final", "latest"):
        return max(common)

    try:
        step = int(requested)
    except ValueError as exc:
        raise ValueError("summary_step must be 'final', 'latest', or an integer step") from exc
    if step not in common:
        raise ValueError(f"Requested step {step} is not available in every rollout directory. Common steps: {sorted(common)}")
    return step


def load_field_snapshot(rollout_dir: str | Path, label: str, step: int) -> FieldSnapshot:
    """Load prediction/truth field tensors for one label and step."""
    rollout_dir = Path(rollout_dir)
    pred_matches = [p for s, p in _preferred_step_files(rollout_dir, "prediction", ".pt") if s == step]
    truth_matches = [p for s, p in _preferred_step_files(rollout_dir, "truth", ".pt") if s == step]
    if not pred_matches or not truth_matches:
        raise FileNotFoundError(f"Could not find prediction/truth .pt files for step {step} in {rollout_dir}")

    pred = load_field_pt(sorted(pred_matches)[0])
    truth = load_field_pt(sorted(truth_matches)[0])
    return FieldSnapshot(label=label, step=step, prediction_fields=pred, truth_fields=truth)


def compute_skill_horizon_curves(
    rollout_runs: Sequence[RolloutRun],
    gammas: Optional[Sequence[float]] = None,
) -> List[SkillHorizonCurve]:
    """Compute gamma -> first threshold-crossing rollout step from saved tensors.

    The relative error uses all saved field channels and grid points.  When a
    rollout directory contains multiple initial conditions, horizons are
    summarized by the median across those initial conditions.
    """
    gamma_values = _validate_skill_gammas(gammas)
    curves: List[SkillHorizonCurve] = []
    for run in rollout_runs:
        per_ic_errors = _relative_rollout_errors_by_ic(run.rollout_dir)
        if not per_ic_errors:
            raise FileNotFoundError(f"Could not find prediction/truth .pt files in {run.rollout_dir}")

        horizons_by_ic = []
        crossed_by_ic = []
        for step_errors in per_ic_errors.values():
            steps = np.asarray(sorted(step_errors), dtype=int)
            errors = np.asarray([step_errors[int(step)] for step in steps], dtype=float)
            horizons, crossed = _skill_horizon_for_errors(steps, errors, gamma_values)
            horizons_by_ic.append(horizons)
            crossed_by_ic.append(crossed)

        horizons_arr = np.vstack(horizons_by_ic)
        crossed_arr = np.vstack(crossed_by_ic)
        curves.append(
            SkillHorizonCurve(
                label=run.label,
                gammas=gamma_values,
                horizons=np.nanmedian(horizons_arr, axis=0),
                crossed=np.mean(crossed_arr, axis=0) >= 0.5,
            )
        )
    return curves


def _validate_skill_gammas(gammas: Optional[Sequence[float]]) -> np.ndarray:
    """Return sorted positive gamma thresholds."""
    if gammas is None:
        values = np.linspace(0.25, 2.0, 36)
    else:
        values = np.asarray(gammas, dtype=float)
    values = np.unique(values)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Skill-horizon gamma values must be positive finite numbers.")
    return values


def _relative_rollout_errors_by_ic(rollout_dir: str | Path) -> Dict[int, Dict[int, float]]:
    """Load saved prediction/truth tensors and compute relative errors by IC."""
    rollout_dir = Path(rollout_dir)
    pred_files = _preferred_ic_step_files(rollout_dir, "prediction", ".pt")
    truth_files = _preferred_ic_step_files(rollout_dir, "truth", ".pt")
    per_ic_errors: Dict[int, Dict[int, float]] = {}

    for ic in sorted(set(pred_files) & set(truth_files)):
        common_steps = sorted(set(pred_files[ic]) & set(truth_files[ic]))
        if not common_steps:
            continue
        per_ic_errors[ic] = {}
        for step in common_steps:
            pred = load_field_pt(pred_files[ic][step])
            truth = load_field_pt(truth_files[ic][step])
            per_ic_errors[ic][step] = _relative_field_error(pred, truth)
    return per_ic_errors


def _skill_horizon_for_errors(
    steps: np.ndarray,
    errors: np.ndarray,
    gammas: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return first saved step where error exceeds each threshold."""
    valid = ~np.isnan(errors)
    finite_steps = steps[valid]
    finite_errors = errors[valid]
    if finite_steps.size == 0:
        raise ValueError("Cannot compute skill horizon from all-NaN rollout errors.")

    max_step = float(np.max(finite_steps))
    horizons = np.full(gammas.shape, max_step, dtype=float)
    crossed = np.zeros(gammas.shape, dtype=bool)
    for i, gamma in enumerate(gammas):
        hit_indices = np.flatnonzero(finite_errors > gamma)
        if hit_indices.size:
            horizons[i] = float(finite_steps[int(hit_indices[0])])
            crossed[i] = True
    return horizons, crossed


def _relative_field_error(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Relative L2 field error over all channels and spatial grid points."""
    diff = prediction - truth
    numerator = float(np.sqrt(np.nansum(diff * diff)))
    denominator = float(np.sqrt(np.nansum(truth * truth)))
    if denominator <= 0.0:
        return 0.0 if numerator <= 0.0 else float("inf")
    return numerator / denominator


def _preferred_ic_step_files(rollout_dir: Path, kind: str, suffix: str) -> Dict[int, Dict[int, Path]]:
    """Return preferred saved tensor files grouped by initial condition."""
    candidates: List[Tuple[int, int, int, Path]] = []
    pattern = re.compile(rf"^(?:ic(?P<ic>\d+)_)?{re.escape(kind)}_(?P<step>\d+){re.escape(suffix)}$")
    for path in rollout_dir.glob(f"*{kind}_*{suffix}"):
        match = pattern.match(path.name)
        if not match:
            continue
        ic = int(match.group("ic") or 0)
        step_digits = match.group("step")
        candidates.append((ic, int(step_digits), len(step_digits), path))
    if not candidates:
        return {}

    preferred_width = max(width for _, _, width, _ in candidates)
    grouped: Dict[int, Dict[int, Path]] = {}
    for ic, step, width, path in sorted(candidates, key=lambda item: (item[0], item[1], item[3].name)):
        if width == preferred_width:
            grouped.setdefault(ic, {})[step] = path
    return grouped


def _path_step(path: Path) -> Optional[int]:
    """Extract the numeric rollout step from saved tensor/PNG filenames."""
    match = re.search(r"_(\d+)\.(?:pt|png)$", path.name)
    return int(match.group(1)) if match else None


def _preferred_step_files(rollout_dir: Path, kind: str, suffix: str) -> List[Tuple[int, Path]]:
    """Return one consistent filename-padding family for saved rollout files.

    A rollout folder can contain stale files from a shorter test run, for
    example ``ic000_prediction_0.pt`` next to real-run
    ``ic000_prediction_000.pt``.  Mixing those families can make animations jump
    between unrelated rollouts.  Prefer the widest step-number padding, which is
    the convention used by longer real forecasts, and keep only that family.
    """
    candidates: List[Tuple[int, int, Path]] = []
    patterns = [f"ic000_{kind}_*{suffix}", f"{kind}_*{suffix}"]
    for pattern in patterns:
        for path in rollout_dir.glob(pattern):
            match = re.search(rf"{re.escape(kind)}_(\d+){re.escape(suffix)}$", path.name)
            if match:
                digits = match.group(1)
                candidates.append((int(digits), len(digits), path))
        if candidates:
            break

    if not candidates:
        return []

    # Width is a practical proxy for one coherent run.  A real 100-step rollout
    # writes 000/010/.../100, while short smoke tests may leave 0/2/4 behind in
    # the same directory.  Choosing the widest family avoids frame jumps caused
    # by mixing old and new outputs.
    preferred_width = max(width for _, width, _ in candidates)
    preferred = [(step, path) for step, width, path in candidates if width == preferred_width]
    return sorted(preferred, key=lambda item: (item[0], item[1].name))


def load_field_pt(path: str | Path) -> np.ndarray:
    """Load a forecast.py-style .pt field file and return a numpy array."""
    if torch is None:
        raise RuntimeError("PyTorch is required to read .pt rollout files.")
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, Mapping) and "fields" in obj:
        fields = obj["fields"]
    else:
        fields = obj
    if hasattr(fields, "detach"):
        fields = fields.detach().cpu().numpy()
    fields = np.asarray(fields)
    if fields.ndim == 4 and fields.shape[0] == 1:
        fields = fields[0]
    if fields.ndim != 3:
        raise ValueError(f"Expected fields shape (3,nlat,nlon), got {fields.shape} from {path}")
    return fields.astype(float)


def load_snapshots_for_step(rollout_runs: Sequence[RolloutRun], summary_step: str = "final") -> List[FieldSnapshot]:
    """Load final/common-step prediction/truth snapshots for several runs."""
    step = common_summary_step([run.rollout_dir for run in rollout_runs], summary_step)
    return [load_field_snapshot(run.rollout_dir, run.label, step) for run in rollout_runs]


def load_animation_frames(
    rollout_dirs: Sequence[str | Path],
    labels: Sequence[str],
) -> List[List[FieldSnapshot]]:
    """Load all saved snapshots across every common step for animation.

    Returns a list of frames ordered by step.  Each frame is a list of
    ``FieldSnapshot`` objects — one per optimizer — all at the same rollout
    step.  Only steps present in every rollout directory are included so the
    animation is always consistent across optimizers.
    """
    if len(rollout_dirs) != len(labels):
        raise ValueError("Expected the same number of rollout_dirs and labels.")

    # Animate only the intersection of saved steps.  This keeps every optimizer
    # synchronized even when one rollout has fewer saved tensors than another.
    per_run_steps = [set(parse_saved_steps(d)) for d in rollout_dirs]
    if any(not steps for steps in per_run_steps):
        raise FileNotFoundError(
            "Could not find saved prediction_*.pt files in one or more rollout directories. "
            "Run 'python -m visualize forecast' first."
        )

    common_steps = sorted(set.intersection(*per_run_steps))
    if not common_steps:
        raise ValueError("No common saved rollout steps found across all rollout directories.")

    return [
        [load_field_snapshot(d, l, step) for d, l in zip(rollout_dirs, labels)]
        for step in common_steps
    ]


def parse_saved_spectra_steps(rollout_dir: str | Path) -> List[int]:
    """Parse saved spectra PNG steps from one rollout directory."""
    rollout_dir = Path(rollout_dir)
    return sorted({step for step, _ in _preferred_step_files(rollout_dir, "spectra", ".png")})


def load_spectral_animation_frames(
    rollout_dirs: Sequence[str | Path],
    labels: Sequence[str],
) -> List[Tuple[int, List[Tuple[str, Path]]]]:
    """Load saved per-optimizer spectra PNG paths across every common step."""
    if len(rollout_dirs) != len(labels):
        raise ValueError("Expected the same number of rollout_dirs and labels.")

    # The split spectral GIF is built from saved PNGs, not recomputed spectra.
    # Using the common step intersection preserves a one-to-one correspondence
    # with the field animation timeline.
    per_run_steps = [set(parse_saved_spectra_steps(d)) for d in rollout_dirs]
    if any(not steps for steps in per_run_steps):
        raise FileNotFoundError(
            "Could not find saved spectra_*.png files in one or more rollout directories. "
            "Run 'python -m visualize forecast' first with spectral analysis enabled."
        )

    common_steps = sorted(set.intersection(*per_run_steps))
    if not common_steps:
        raise ValueError("No common saved spectral-analysis steps found across all rollout directories.")

    frames: List[Tuple[int, List[Tuple[str, Path]]]] = []
    for step in common_steps:
        panels: List[Tuple[str, Path]] = []
        for rollout_dir, label in zip(rollout_dirs, labels):
            rollout_path = Path(rollout_dir)
            matches = [p for s, p in _preferred_step_files(rollout_path, "spectra", ".png") if s == step]
            if not matches:
                raise FileNotFoundError(f"Could not find spectra PNG for step {step} in {rollout_path}")
            panels.append((label, sorted(matches)[0]))
        frames.append((step, panels))
    return frames


def metric_column(error_metric: str) -> str:
    """Map user-facing error metric to per-step CSV column name."""
    if error_metric not in ERROR_METRIC_TO_COLUMN:
        raise ValueError(f"Unknown error_metric {error_metric!r}. Expected one of {list(ERROR_METRIC_TO_COLUMN)}")
    return ERROR_METRIC_TO_COLUMN[error_metric]


def metric_mean_column(error_metric: str) -> str:
    """Map user-facing metric to aggregate metrics.csv column name."""
    return f"{metric_column(error_metric)}_mean"


def aggregate_step_metrics(
    all_step_metrics: Sequence[Mapping[str, Sequence[float]]],
    all_ml_times: Sequence[float],
    all_solver_times: Sequence[float],
) -> Dict[str, float]:
    """Aggregate per-step metrics in the same spirit as forecast.py."""
    summary: Dict[str, float] = {}
    for key in all_step_metrics[0].keys():
        values = [float(v) for ic in all_step_metrics for v in ic[key]]
        summary[f"{key}_mean"] = float(np.mean(values)) if values else float("nan")
        summary[f"{key}_std"] = float(np.std(values)) if values else float("nan")
    summary["ml_time_mean"] = float(np.mean(all_ml_times))
    summary["ml_time_std"] = float(np.std(all_ml_times))
    summary["solver_time_mean"] = float(np.mean(all_solver_times))
    summary["solver_time_std"] = float(np.std(all_solver_times))
    summary["speedup_mean"] = summary["solver_time_mean"] / summary["ml_time_mean"] if summary["ml_time_mean"] > 0 else float("nan")
    return summary


def write_per_step_metrics_csv(path: str | Path, all_step_metrics: Sequence[Mapping[str, Sequence[float]]]) -> None:
    """Write the per-step metrics returned by forecast.py's rollout helper."""
    path = Path(path)
    rows: List[Dict[str, float | int]] = []
    for ic_index, metrics in enumerate(all_step_metrics):
        if not metrics:
            continue
        n = max(len(values) for values in metrics.values())
        for i in range(n):
            row: Dict[str, float | int] = {"ic": ic_index, "step": i + 1}
            for key, values in metrics.items():
                row[key] = float(values[i]) if i < len(values) else float("nan")
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def sanitize_label(label: str) -> str:
    """Convert a plot label into a safe folder name."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return safe or "run"


# ---------------------------------------------------------------------------
# Spectral preparation
# ---------------------------------------------------------------------------

def compute_simple_energy_spectra(fields: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute simple grid-FFT spectra for comparison plots.

    Real SWAN spectra use spherical harmonic transforms in ``forecast.py``.
    This fallback uses a 2D FFT radial average and is used for synthetic tests
    and for combined plotting when only saved fields are available.  The plot is
    still useful for comparing optimizers on the same grid, but users should
    remember that the exact definition differs from the spherical-harmonic
    diagnostic in the original forecast.py.
    """
    h, vort, div = fields[0], fields[1], fields[2]
    rot = radial_power_spectrum(vort)
    divspec = radial_power_spectrum(div)
    pot = radial_power_spectrum(h)
    n = min(len(rot), len(divspec), len(pot))
    rot = rot[:n]
    divspec = divspec[:n]
    pot = pot[:n]
    total = rot + divspec + pot
    return {
        "rotational": rot,
        "divergent": divspec,
        "potential": pot,
        "total": total,
        "wavenumbers": np.arange(n),
    }


def build_spherical_sht(config_path: str | Path, field_shape: Sequence[int], device: Optional[str] = None):
    """Build the same spherical harmonic transform used by forecast.py."""
    if torch is None:
        raise RuntimeError("PyTorch is required for spherical-harmonic spectra.")

    try:
        forecast = importlib.import_module("forecast")
    except Exception as exc:
        raise RuntimeError("Could not import forecast.py for spherical-harmonic spectra.") from exc

    # Construct a minimal forecast dataset solely to obtain its solver.sht.  This
    # mirrors forecast.py instead of duplicating low-level torch_harmonics setup,
    # which keeps the diagnostic definition aligned with the original script.
    config = forecast.load_config(str(config_path))
    nlat, nlon = int(field_shape[-2]), int(field_shape[-1])
    data_config = config["data"]
    device_obj = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = data_config["dt"]
    nsteps = dt // data_config["dt_solver"]
    dataset = forecast.PdeDatasetWithWinds(
        dt=dt,
        nsteps=nsteps,
        dims=(nlat, nlon),
        normalize=True,
        device=device_obj,
    )
    return dataset.solver.sht.to(device_obj)


def compute_spherical_energy_spectra(fields: np.ndarray, sht) -> Dict[str, np.ndarray]:
    """Compute spectra using forecast.py's spherical-harmonic diagnostic."""
    if torch is None:
        raise RuntimeError("PyTorch is required for spherical-harmonic spectra.")

    try:
        forecast = importlib.import_module("forecast")
    except Exception as exc:
        raise RuntimeError("Could not import forecast.py for spherical-harmonic spectra.") from exc

    try:
        device = next(sht.parameters()).device
    except (AttributeError, StopIteration):
        try:
            device = next(sht.buffers()).device
        except (AttributeError, StopIteration):
            device = torch.device("cpu")
    # Saved rollout tensors are stored as numpy arrays with shape
    # (channels, nlat, nlon).  forecast.compute_energy_spectra expects a batch
    # dimension, so add one without changing the field values.
    tensor = torch.as_tensor(fields, dtype=torch.float32, device=device)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return forecast.compute_energy_spectra(tensor, sht)


def radial_power_spectrum(field: np.ndarray) -> np.ndarray:
    """Radially average a normalized 2D FFT power spectrum.

    The original forecast.py computes spectra from spherical-harmonic
    coefficients.  For the comparison utility we often only have saved grid
    tensors, so the fallback must use a grid FFT.  Using NumPy's default FFT
    normalization makes powers scale like the square of the grid size, which can
    produce visually huge values unrelated to the original diagnostic scale.
    ``norm="forward"`` divides by the number of grid points in the forward
    transform, giving a scale closer to mean-square field amplitude and making
    cross-optimizer comparisons more interpretable.
    """
    field = np.asarray(field, dtype=float)
    field = field - np.nanmean(field)
    fft = np.fft.rfft2(field, norm="forward")
    power = np.abs(fft) ** 2
    nlat, nmodes = power.shape
    ky = np.fft.fftfreq(nlat) * nlat
    kx = np.fft.rfftfreq((nmodes - 1) * 2) * ((nmodes - 1) * 2)
    yy, xx = np.meshgrid(ky, kx, indexing="ij")
    kr = np.sqrt(xx**2 + yy**2).astype(int)
    max_k = max(2, min(nlat // 2, nmodes))
    spectrum = np.zeros(max_k, dtype=float)
    counts = np.zeros(max_k, dtype=float)
    for k in range(max_k):
        mask = kr == k
        if np.any(mask):
            spectrum[k] = power[mask].mean()
            counts[k] = mask.sum()
    # Avoid exact zeros on log axes.
    return np.maximum(spectrum, 1e-30)
