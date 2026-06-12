"""
_cli.py
=======

Argument parsing and command entry points for the SWAN visualize package.

Users interact through::

    python -m visualize <command> [options]

Commands
--------
plot_history
    Compare training/validation histories from TensorBoard or CSV logs.

forecast
    Run or synthesize rollout comparisons and generate forecast-level plots.

Examples
--------
Training/validation history::

    python -m visualize plot_history \\
      --runs ./results/adam/version_0 ./results/mud/version_0 ./results/muon/version_0 \\
      --labels Adam MUD Muon \\
      --stage validation \\
      --plot both \\
      --error_metric l2 \\
      --efficiency_metric both \\
      --outdir ./figures_history

Forecast comparison from trained runs::

    python -m visualize forecast \\
      --runs ./results/adam/version_0 ./results/mud/version_0 ./results/muon/version_0 \\
      --labels Adam MUD Muon \\
      --config config_paradis.yaml \\
      --autoreg_steps 100 \\
      --output_freq 10 \\
      --channel vorticity \\
      --rollout_dir ./rollout_results \\
      --outdir ./figures_forecast

Rollout animation from pre-computed forecast data::

    python -m visualize animate \\
      --rollout_dir ./rollout_results \\
      --labels Adam MUD Muon \\
      --channel vorticity \\
      --output comparison.gif

Quick synthetic forecast demo::

    python -m visualize forecast \\
      --synthetic_demo \\
      --labels Adam MUD Muon \\
      --autoreg_steps 20 \\
      --output_freq 5
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Sequence

from visualize import history as hist
from visualize import rollout as roll
from visualize.plots import (
    ensure_outdir,
    plot_history_learning_curve,
    plot_history_hitting_curve,
    plot_forecast_error_curve,
    plot_forecast_accuracy_bar,
    plot_skill_horizon_vs_gamma,
    plot_forecast_runtime_ratio_bar,
    plot_spectral_horizon,
    plot_prediction_grid,
    plot_error_grid,
    plot_combined_spectra,
    plot_combined_spectra_percent_difference,
    make_rollout_animation,
    make_rollout_error_animation,
    make_combined_spectral_animation,
    make_combined_spectral_percent_difference_animation,
    make_spectral_image_animation,
)


def run_plot_history(args: argparse.Namespace) -> None:
    """Entry point for the ``plot_history`` command."""
    outdir = ensure_outdir(args.outdir)
    runs = hist.load_history_runs(args.runs, args.labels)

    for stage in hist.concrete_stages(args.stage):
        metric = hist.metric_for_stage(stage, args.error_metric)

        if args.plot in ("learning_curve", "both"):
            for resource in _resources_from_arg(args.efficiency_metric):
                plot_history_learning_curve(
                    runs=runs,
                    stage=stage,
                    error_metric=metric,
                    resource=resource,
                    outdir=outdir,
                    yscale=args.history_scale,
                )

        if args.plot in ("hitting_curve", "both"):
            for resource in _resources_from_arg(args.efficiency_metric):
                plot_history_hitting_curve(
                    runs=runs,
                    stage=stage,
                    error_metric=metric,
                    resource=resource,
                    outdir=outdir,
                    threshold_xscale=args.history_scale,
                )


def _resources_from_arg(efficiency_metric: str) -> List[str]:
    """Expand ``both`` into ``step`` and ``time``."""
    if efficiency_metric == "both":
        return ["step", "time"]
    return [efficiency_metric]


def run_forecast(args: argparse.Namespace) -> None:
    """Entry point for the ``forecast`` command."""
    outdir = ensure_outdir(args.outdir)
    rollout_dir = ensure_outdir(args.rollout_dir)

    if args.reuse_rollouts:
        if args.synthetic_demo:
            raise ValueError("--reuse_rollouts loads existing data and cannot be combined with --synthetic_demo.")
        if not args.labels:
            raise ValueError("forecast --reuse_rollouts requires --labels.")

        # Reuse mode intentionally skips checkpoint loading and model execution.
        # It is for making additional grouped figures from rollout folders that
        # were already generated with the same forecast settings.
        rollout_runs = roll.load_rollout_runs(
            [rollout_dir / roll.sanitize_label(label) for label in args.labels],
            args.labels,
        )
    elif args.synthetic_demo:
        labels = args.labels or ["Adam", "MUD", "Muon"]
        rollout_runs = roll.create_synthetic_rollouts(
            labels=labels,
            rollout_dir=rollout_dir,
            autoreg_steps=args.autoreg_steps,
            output_freq=args.output_freq,
            num_ics=args.num_ics,
            seed=args.seed,
            nlat=args.nlat or 32,
            nlon=args.nlon or 64,
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
            nlat=args.nlat,
            nlon=args.nlon,
        )

    if not args.reuse_rollouts:
        # Always reload from disk after generation so every plotting path
        # exercises the same code users rely on when re-loading pre-computed
        # rollouts.
        rollout_runs = roll.load_rollout_runs(
            [run.rollout_dir for run in rollout_runs],
            [run.label for run in rollout_runs],
        )
    snapshots = roll.load_snapshots_for_step(rollout_runs, args.summary_step)

    plot_forecast_error_curve(rollout_runs, args.error_metric, outdir, yscale=args.forecast_error_scale)
    plot_forecast_accuracy_bar(rollout_runs, args.error_metric, outdir)
    plot_forecast_runtime_ratio_bar(rollout_runs, outdir)
    if args.skill_horizon:
        plot_skill_horizon_vs_gamma(rollout_runs, args.skill_horizon_gammas, outdir)
    plot_prediction_grid(snapshots, args.channel, args.grid_cols, args.output_freq, outdir)
    plot_error_grid(snapshots, args.channel, args.error_mode, args.grid_cols, args.output_freq, outdir)
    # The static combined spectra should match forecast.py's spherical-harmonic
    # diagnostic for real rollouts.  Synthetic demo data does not have the SWAN
    # solver/SHT context needed for that diagnostic, so the demo path keeps the
    # lightweight FFT fallback only for synthetic testing.
    sht = None
    spectra_method = args.spherical_method
    if spectra_method == "spherical" and not args.synthetic_demo:
        sht = roll.build_spherical_sht(args.config, snapshots[0].truth_fields.shape, args.device)
    elif spectra_method == "spherical":
        spectra_method = "fft"
    if args.spectral_horizon:
        plot_spectral_horizon(
            rollout_runs,
            outdir,
            spectra_method=spectra_method,
            mode=args.spectral_horizon_mode,
            eta_factor=args.spectral_eta_factor,
            sht=sht,
        )
    plot_combined_spectra(snapshots, args.output_freq, outdir, spectra_method=spectra_method, sht=sht)
    plot_combined_spectra_percent_difference(
        snapshots,
        args.output_freq,
        outdir,
        spectra_method=spectra_method,
        sht=sht,
    )


def run_animate(args: argparse.Namespace) -> None:
    """Entry point for the ``animate`` command."""
    rollout_dir = args.rollout_dir
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.synthetic_demo:
        labels = args.labels or ["Adam", "MUD", "Muon"]
        rollout_runs = roll.create_synthetic_rollouts(
            labels=labels,
            rollout_dir=rollout_dir,
            autoreg_steps=args.autoreg_steps,
            output_freq=args.output_freq,
            seed=args.seed,
        )
        rollout_dirs = [run.rollout_dir for run in rollout_runs]
    else:
        if not args.labels:
            raise ValueError("animate requires --labels unless --synthetic_demo is used.")
        labels = args.labels
        rollout_names = args.rollout_names or labels
        if len(rollout_names) != len(labels):
            raise ValueError("--rollout_names must have the same length as --labels.")
        rollout_dirs = [
            Path(rollout_dir) / roll.sanitize_label(name) for name in rollout_names
        ]

    frames = roll.load_animation_frames(rollout_dirs, labels)
    make_rollout_animation(
        frames=frames,
        channel=args.channel,
        fps=args.fps,
        output=args.output,
        show_error=args.show_error,
    )
    if args.with_error_output:
        Path(args.with_error_output).parent.mkdir(parents=True, exist_ok=True)
        make_rollout_animation(
            frames=frames,
            channel=args.channel,
            fps=args.fps,
            output=args.with_error_output,
            show_error=True,
        )
    if args.error_output:
        Path(args.error_output).parent.mkdir(parents=True, exist_ok=True)
        make_rollout_error_animation(
            frames=frames,
            channel=args.channel,
            fps=args.fps,
            output=args.error_output,
        )
    spectral_output = args.spectral_output or _default_spectral_output(args.output)
    Path(spectral_output).parent.mkdir(parents=True, exist_ok=True)
    # Two spectral animation modes are intentionally separated:
    # - default: recompute spherical-harmonic spectra from saved tensors and put
    #   every optimizer on the same animated axes for direct comparison;
    # - split: stitch the per-optimizer spectra PNGs already written by
    #   forecast.py, which is useful when each optimizer should be inspected in
    #   its own original forecast-style figure.
    if args.split_spectral:
        spectral_frames = roll.load_spectral_animation_frames(rollout_dirs, labels)
        make_spectral_image_animation(
            frames=spectral_frames,
            fps=args.fps,
            output=spectral_output,
        )
    else:
        make_combined_spectral_animation(
            frames=frames,
            fps=args.fps,
            output=spectral_output,
            config_path="config_paradis.yaml",
        )
        make_combined_spectral_percent_difference_animation(
            frames=frames,
            fps=args.fps,
            output=_default_spectral_percent_output(spectral_output),
            config_path="config_paradis.yaml",
        )


def _default_spectral_output(output: str | Path) -> Path:
    """Return a sibling path for the spectral animation."""
    output = Path(output)
    suffix = output.suffix or ".gif"
    stem = output.stem
    if stem.endswith("_fields"):
        stem = f"{stem[:-7]}_spectra"
    else:
        stem = f"{stem}_spectra"
    return output.with_name(f"{stem}{suffix}")


def _default_spectral_percent_output(output: str | Path) -> Path:
    """Return a sibling path for the signed-percent spectral animation."""
    output = Path(output)
    suffix = output.suffix or ".gif"
    match = re.match(r"^(?P<base>.+?)(?P<version>_v\d+)$", output.stem)
    if match:
        stem = f"{match.group('base')}_percent_difference{match.group('version')}"
    else:
        stem = f"{output.stem}_percent_difference"
    return output.with_name(f"{stem}{suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare SWAN optimizer runs using TensorBoard histories and forecast rollouts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # plot_history -----------------------------------------------------------
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
    ph.add_argument(
        "--history_scale",
        "--yscale",
        choices=("linear", "log"),
        default="linear",
        help="Scale for history plots: learning-curve y-axis and hitting-curve threshold x-axis. Default: linear",
    )
    ph.add_argument("--outdir", default="./figures", help="Directory for history figures. Default: ./figures")
    ph.set_defaults(func=run_plot_history)

    # forecast ---------------------------------------------------------------
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
    fc.add_argument("--forecast_error_scale", choices=("linear", "log"), default="linear", help="Y-axis scale for forecast_error_curve_<metric>.png. Default: linear")
    fc.add_argument("--skill_horizon", action="store_true", help="Also plot skill_horizon_vs_gamma.png from saved rollout tensors.")
    fc.add_argument("--skill_horizon_gammas", nargs="+", type=float, default=None, help="Gamma thresholds for --skill_horizon. Default: denser sweep below 1.0 from 0.01 to 2.0")
    fc.add_argument("--spectral_horizon", action="store_true", help="Also plot a spectral horizon diagnostic from saved rollout tensors.")
    fc.add_argument("--spectral_horizon_mode", choices=("abs", "positive"), default="abs", help="Threshold mode for --spectral_horizon. Default: abs")
    fc.add_argument("--spectral_eta_factor", type=float, default=2.0, help="Multiplicative spectral threshold factor for --spectral_horizon. Default: 2.0")
    fc.add_argument("--error_mode", choices=("signed", "abs", "squared"), default="signed", help="Pointwise error map mode. Default: signed")
    fc.add_argument("--summary_step", default="final", help="Step for final grid/spectra plots: final/latest or an integer. Default: final")
    fc.add_argument("--spherical_method", choices=("spherical", "fft"), default="spherical", help="Method for forecast_spectra_final.png. Default: spherical")
    fc.add_argument("--grid_cols", type=int, default=3, help="Maximum columns in spatial grids. Default: 3")
    fc.add_argument("--nlat", type=int, default=None, help="Optional forecast grid latitude count override. Default: config/checkpoint value")
    fc.add_argument("--nlon", type=int, default=None, help="Optional forecast grid longitude count override. Default: config/checkpoint value")
    fc.add_argument("--rollout_dir", default="./rollout_results", help="Directory for per-optimizer rollout outputs. Default: ./rollout_results")
    fc.add_argument("--outdir", default="./figures_forecast", help="Directory for final forecast figures. Default: ./figures_forecast")
    fc.add_argument("--device", default=None, help="Optional real-rollout device, e.g. cuda or cpu. Default: auto")
    fc.add_argument("--synthetic_demo", action="store_true", help="Generate artificial rollout data instead of loading real checkpoints. For tests only.")
    fc.add_argument("--reuse_rollouts", action="store_true", help="Skip model rollout execution and plot from existing per-optimizer folders in --rollout_dir.")
    fc.set_defaults(func=run_forecast)

    # animate ----------------------------------------------------------------
    an = subparsers.add_parser(
        "animate",
        help="Animate pre-computed forecast rollouts as a GIF or MP4.",
    )
    an.add_argument("--rollout_dir", default="./rollout_results", help="Directory containing per-optimizer rollout folders. Default: ./rollout_results")
    an.add_argument("--labels", nargs="+", help="Optimizer labels matching the subfolder names. Required unless --synthetic_demo is used.")
    an.add_argument("--rollout_names", nargs="+", help="Optional rollout subfolder names matching --labels. Useful when display labels differ from folder names.")
    an.add_argument("--channel", choices=tuple(roll.CHANNEL_TO_INDEX.keys()), default="vorticity", help="Field channel to animate. Default: vorticity")
    an.add_argument("--output", default="./figures_forecast/rollout_fields.gif", help="Output file path for field animation (.gif or .mp4). Default: ./figures_forecast/rollout_fields.gif")
    an.add_argument("--with_error_output", default=None, help="Optional companion field animation with signed error panels.")
    an.add_argument("--error_output", default=None, help="Optional error-only animation output path.")
    an.add_argument("--spectral_output", default=None, help="Output file path for spectral-analysis animation. Default: derive from --output")
    an.add_argument("--split_spectral", "--split-spectral", action="store_true", help="Show each optimizer's saved spectral-analysis image separately. Default: combine all optimizers in one spherical-harmonic graph.")
    an.add_argument("--fps", type=int, default=8, help="Frames per second. Default: 8")
    an.add_argument("--show_error", action="store_true", help="Add a second row of signed error maps (prediction - truth).")
    an.add_argument("--synthetic_demo", action="store_true", help="Generate synthetic rollout data before animating. For tests only.")
    an.add_argument("--autoreg_steps", type=int, default=20, help="Autoregressive steps for synthetic demo. Default: 20")
    an.add_argument("--output_freq", type=int, default=4, help="Save-every-N-steps for synthetic demo. Default: 4")
    an.add_argument("--seed", type=int, default=42, help="Random seed for synthetic demo. Default: 42")
    an.set_defaults(func=run_animate)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
