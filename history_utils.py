#!/usr/bin/env python3
"""
history_utils.py
================

Backend utilities for ``visualize.py plot_history``.

This module deliberately contains only data loading and preparation logic for
training/validation histories.  It does **not** create figures and it does not
parse command-line arguments.  Keeping the TensorBoard/CSV fetching code here
makes ``visualize.py`` easier to read: the public script can focus on the user
interface and plotting, while this module focuses on converting raw logs into
clean arrays.

The implementation is intentionally close to the earlier standalone
``visualize.py`` behavior:

* read one directory per optimizer run;
* prefer a simple CSV fallback when present, which is convenient for tests;
* otherwise read TensorBoard event files;
* map user-friendly metric names such as ``l2`` to real log tags such as
  ``val_l2``;
* provide learning-curve data and first-hitting-curve data.

Supported run directory layouts
-------------------------------
Real PyTorch Lightning / TensorBoard run::

    results/adam/version_0/events.out.tfevents...

CSV fallback run, useful for tests or exported TensorBoard scalars::

    results/adam/version_0/history.csv

Supported CSV formats
---------------------
Long format::

    tag,step,wall_time,value
    val_l2,0,1000.0,0.9
    val_l2,1,1010.0,0.7

Wide format::

    step,wall_time,train_loss_epoch,val_loss,val_l2
    0,1000.0,1.0,0.9,0.85
    1,1010.0,0.8,0.7,0.65
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ScalarSeries:
    """One scalar metric series from one run.

    Parameters
    ----------
    tag:
        Raw TensorBoard/CSV tag, for example ``val_l2``.
    step:
        Global step values recorded with the scalar.
    relative_time_sec:
        Wall-clock time in seconds relative to the first scalar event in the
        run.  This is what TensorBoard calls the "relative" x-axis.
    value:
        Scalar values.
    """

    tag: str
    step: np.ndarray
    relative_time_sec: np.ndarray
    value: np.ndarray


@dataclass
class RunScalars:
    """All scalar histories belonging to one optimizer run."""

    run_dir: Path
    label: str
    scalars: Dict[str, ScalarSeries]

    def available_tags(self) -> List[str]:
        """Return scalar tags sorted for readable error messages."""
        return sorted(self.scalars.keys())


# ---------------------------------------------------------------------------
# Public names and metric mappings
# ---------------------------------------------------------------------------

ERROR_METRICS = ("loss", "l1", "l2", "sq_l2", "w11")
STAGES = ("training", "validation", "both")
EFFICIENCY_METRICS = ("step", "time", "both")
HISTORY_PLOTS = ("learning_curve", "hitting_curve", "both")

# The command-line metric names do not include val_ or train_.  The stage
# selects the correct group of aliases.
VALIDATION_TAG_ALIASES: Mapping[str, Sequence[str]] = {
    "loss": ("val_loss", "validation_loss", "validation/loss"),
    "l1": ("val_l1", "validation_l1", "validation/l1"),
    "l2": ("val_l2", "validation_l2", "validation/l2"),
    "sq_l2": ("val_sq_l2", "validation_sq_l2", "validation/sq_l2"),
    "w11": ("val_w11", "validation_w11", "validation/w11"),
}

TRAINING_TAG_ALIASES: Mapping[str, Sequence[str]] = {
    # PyTorch Lightning often writes train_loss_epoch and train_loss_step when
    # self.log("train_loss", on_step=True, on_epoch=True) is used.  For an
    # optimizer comparison plot, the epoch-level value is usually cleaner than
    # the batch-level value, so it is the first alias.
    "loss": (
        "train_loss_epoch",
        "train_loss",
        "train_loss_step",
        "training_loss",
        "training/loss",
    ),
}

CSV_FALLBACK_NAMES = ("scalars.csv", "history.csv", "metrics_history.csv")


def stage_metric_to_aliases(stage: str, error_metric: str) -> Sequence[str]:
    """Return raw scalar tags for a user-facing stage/metric pair.

    ``stage`` must be a concrete stage: ``training`` or ``validation``.  The
    caller should expand ``both`` into those two stages before calling this.
    """
    if stage == "validation":
        return VALIDATION_TAG_ALIASES[error_metric]

    if stage == "training":
        if error_metric != "loss":
            raise ValueError(
                "The original SWAN training scripts only log training loss. "
                f"stage=training with error_metric={error_metric!r} is not "
                "supported unless your team adds train_l1/train_l2/etc."
            )
        return TRAINING_TAG_ALIASES["loss"]

    raise ValueError(f"Unsupported concrete stage {stage!r}.")


def display_metric_name(stage: str, error_metric: str) -> str:
    """Human-readable metric name for labels/titles."""
    if stage == "training":
        return "training loss"
    if stage == "validation":
        if error_metric == "loss":
            return "validation loss"
        if error_metric == "sq_l2":
            return "validation squared L2 error"
        if error_metric == "w11":
            return "validation W11 error"
        return f"validation {error_metric.upper()} error"
    return f"{stage} {error_metric}"


def filename_metric_name(stage: str, error_metric: str) -> str:
    """Compact metric component for output filenames."""
    return "loss" if stage == "training" else error_metric


def concrete_stages(stage: str) -> List[str]:
    """Expand ``both`` into concrete stages."""
    if stage == "both":
        return ["training", "validation"]
    return [stage]


def metric_for_stage(stage: str, requested_error_metric: str) -> str:
    """Choose the actual metric for a stage.

    When the user requests ``--stage both --error_metric l2``, validation uses
    ``l2`` but training still uses ``loss`` because the original training logs
    do not contain training L2.
    """
    if stage == "training":
        return "loss"
    return requested_error_metric


# ---------------------------------------------------------------------------
# Loading scalar histories
# ---------------------------------------------------------------------------

def read_run_scalars(run_dir: Path | str, label: str) -> RunScalars:
    """Read all scalar histories from one run directory.

    The loader first checks for a CSV fallback file.  This keeps the test path
    lightweight and also lets a team use exported scalars if TensorBoard is not
    installed.  If no CSV exists, TensorBoard event files are read through
    TensorBoard's ``EventAccumulator``.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    csv_path = find_csv_fallback(run_dir)
    if csv_path is not None:
        return RunScalars(run_dir=run_dir, label=label, scalars=read_scalars_from_csv(csv_path))

    return RunScalars(run_dir=run_dir, label=label, scalars=read_scalars_from_tensorboard(run_dir))


def load_history_runs(run_dirs: Sequence[str | Path], labels: Sequence[str]) -> List[RunScalars]:
    """Load several optimizer histories and validate list lengths."""
    if len(run_dirs) != len(labels):
        raise ValueError(
            f"Expected the same number of --runs and --labels, got "
            f"{len(run_dirs)} runs and {len(labels)} labels."
        )
    return [read_run_scalars(Path(run_dir), label) for run_dir, label in zip(run_dirs, labels)]


def find_csv_fallback(run_dir: Path) -> Optional[Path]:
    """Return the first supported CSV scalar file inside ``run_dir``."""
    for name in CSV_FALLBACK_NAMES:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def read_scalars_from_tensorboard(run_dir: Path) -> Dict[str, ScalarSeries]:
    """Read TensorBoard scalar histories from ``run_dir``.

    Notes
    -----
    TensorBoard stores both global step and absolute wall-clock time.  For fair
    optimizer comparison we convert absolute wall time into relative seconds
    since the first scalar event in the run.
    """
    if not list(run_dir.glob("events.out.tfevents*")):
        raise FileNotFoundError(
            f"No CSV fallback and no TensorBoard event file found in {run_dir}. "
            f"Expected one of {CSV_FALLBACK_NAMES} or events.out.tfevents*"
        )

    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError as exc:
        raise ImportError(
            "TensorBoard is required to read events.out.tfevents files. Install it with:\n\n"
            "    pip install tensorboard\n\n"
            "For tests, place a scalars.csv/history.csv file in each run folder."
        ) from exc

    accumulator = event_accumulator.EventAccumulator(str(run_dir))
    accumulator.Reload()

    tags = accumulator.Tags().get("scalars", [])
    if not tags:
        raise ValueError(f"No scalar tags found in TensorBoard logs under {run_dir}")

    raw_by_tag = {}
    all_wall_times: List[float] = []
    for tag in tags:
        events = accumulator.Scalars(tag)
        if events:
            raw_by_tag[tag] = events
            all_wall_times.extend(float(event.wall_time) for event in events)

    if not all_wall_times:
        raise ValueError(f"No scalar events found under {run_dir}")

    run_start = min(all_wall_times)
    scalars: Dict[str, ScalarSeries] = {}
    for tag, events in raw_by_tag.items():
        series = ScalarSeries(
            tag=tag,
            step=np.array([float(e.step) for e in events], dtype=float),
            relative_time_sec=np.array([float(e.wall_time) - run_start for e in events], dtype=float),
            value=np.array([float(e.value) for e in events], dtype=float),
        )
        scalars[tag] = clean_and_sort_series(series)
    return scalars


def read_scalars_from_csv(csv_path: Path) -> Dict[str, ScalarSeries]:
    """Read scalar histories from a CSV file in long or wide format."""
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")

    field_set = set(fieldnames)
    if {"tag", "step", "value"}.issubset(field_set):
        return _read_long_csv_rows(rows, csv_path)
    if "step" in field_set:
        return _read_wide_csv_rows(rows, fieldnames, csv_path)

    raise ValueError(
        f"Unsupported CSV format in {csv_path}. Expected long format with "
        "tag,step,value or wide format with step plus metric columns."
    )


def _time_column(rows: Sequence[Mapping[str, str]], fieldnames: Sequence[str]) -> Tuple[str, bool, float]:
    """Choose the time column and return ``(column, is_relative, start)``."""
    if "relative_time_sec" in fieldnames:
        return "relative_time_sec", True, 0.0
    if "elapsed_sec" in fieldnames:
        return "elapsed_sec", True, 0.0
    if "wall_time" in fieldnames:
        col = "wall_time"
        vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
        return col, False, min(vals) if vals else 0.0
    if "time" in fieldnames:
        col = "time"
        vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
        return col, False, min(vals) if vals else 0.0
    return "step", True, 0.0


def _read_long_csv_rows(rows: Sequence[Mapping[str, str]], csv_path: Path) -> Dict[str, ScalarSeries]:
    """Parse long-format scalar rows."""
    fieldnames = list(rows[0].keys())
    time_col, time_is_relative, start = _time_column(rows, fieldnames)
    by_tag: Dict[str, Dict[str, List[float]]] = {}

    for row in rows:
        tag = (row.get("tag") or "").strip()
        if not tag:
            continue
        try:
            step = float(row["step"])
            value = float(row["value"])
            t = float(row[time_col])
        except (KeyError, TypeError, ValueError):
            continue
        if not time_is_relative:
            t -= start
        entry = by_tag.setdefault(tag, {"step": [], "time": [], "value": []})
        entry["step"].append(step)
        entry["time"].append(t)
        entry["value"].append(value)

    scalars = {
        tag: clean_and_sort_series(
            ScalarSeries(
                tag=tag,
                step=np.array(data["step"], dtype=float),
                relative_time_sec=np.array(data["time"], dtype=float),
                value=np.array(data["value"], dtype=float),
            )
        )
        for tag, data in by_tag.items()
    }
    if not scalars:
        raise ValueError(f"No valid scalar rows found in {csv_path}")
    return scalars


def _read_wide_csv_rows(
    rows: Sequence[Mapping[str, str]], fieldnames: Sequence[str], csv_path: Path
) -> Dict[str, ScalarSeries]:
    """Parse wide-format scalar rows."""
    time_col, time_is_relative, start = _time_column(rows, fieldnames)
    excluded = {
        "step",
        "wall_time",
        "time",
        "relative_time_sec",
        "elapsed_sec",
        "epoch",
        "optimizer",
        "run_id",
        "seed",
    }
    metric_cols = [col for col in fieldnames if col not in excluded]
    if not metric_cols:
        raise ValueError(f"No metric columns found in wide CSV {csv_path}")

    scalars: Dict[str, ScalarSeries] = {}
    for metric in metric_cols:
        steps: List[float] = []
        times: List[float] = []
        values: List[float] = []
        for row in rows:
            raw_value = row.get(metric)
            if raw_value in (None, ""):
                continue
            try:
                step = float(row["step"])
                t = float(row[time_col])
                value = float(raw_value)
            except (KeyError, TypeError, ValueError):
                continue
            if not time_is_relative:
                t -= start
            steps.append(step)
            times.append(t)
            values.append(value)
        if values:
            scalars[metric] = clean_and_sort_series(
                ScalarSeries(
                    tag=metric,
                    step=np.array(steps, dtype=float),
                    relative_time_sec=np.array(times, dtype=float),
                    value=np.array(values, dtype=float),
                )
            )

    if not scalars:
        raise ValueError(f"No valid scalar values found in {csv_path}")
    return scalars


def clean_and_sort_series(series: ScalarSeries) -> ScalarSeries:
    """Drop non-finite values and sort by step.

    If repeated steps exist, the last value for a step is kept.  This mirrors the
    usual interpretation of TensorBoard scalar streams, where a later event at
    the same step supersedes an earlier one.
    """
    mask = np.isfinite(series.step) & np.isfinite(series.relative_time_sec) & np.isfinite(series.value)
    step = series.step[mask]
    time = series.relative_time_sec[mask]
    value = series.value[mask]

    if step.size == 0:
        return ScalarSeries(series.tag, step, time, value)

    order = np.argsort(step, kind="stable")
    step = step[order]
    time = time[order]
    value = value[order]

    # Keep last event for each repeated step.
    unique_steps, last_indices = np.unique(step, return_index=False, return_inverse=False, return_counts=False), None
    last_positions = []
    for s in unique_steps:
        last_positions.append(np.where(step == s)[0][-1])
    last_positions = np.array(last_positions, dtype=int)

    return ScalarSeries(
        tag=series.tag,
        step=step[last_positions],
        relative_time_sec=time[last_positions],
        value=value[last_positions],
    )


# ---------------------------------------------------------------------------
# Preparation helpers used by plotting code
# ---------------------------------------------------------------------------

def select_series(run: RunScalars, stage: str, error_metric: str) -> ScalarSeries:
    """Select the best matching scalar series from one run.

    The first available alias is returned.  If none exists, a detailed error
    tells the user which tags are available in that run.
    """
    aliases = stage_metric_to_aliases(stage, error_metric)
    for tag in aliases:
        if tag in run.scalars:
            return run.scalars[tag]

    raise KeyError(
        f"Could not find scalar for label={run.label!r}, stage={stage!r}, "
        f"error_metric={error_metric!r}. Tried aliases: {list(aliases)}. "
        f"Available tags: {run.available_tags()}"
    )


def prepare_learning_curve_data(
    runs: Sequence[RunScalars], stage: str, error_metric: str
) -> Dict[str, ScalarSeries]:
    """Return selected scalar series for learning-curve plots."""
    return {run.label: rebase_relative_time(select_series(run, stage, error_metric)) for run in runs}


def rebase_relative_time(series: ScalarSeries) -> ScalarSeries:
    """Return a copy whose first plotted timestamp is zero.

    Raw TensorBoard and CSV histories can contain different first-event times
    for different scalar tags.  Rebasing after tag selection makes training and
    validation curves share the same relative-time origin in plot_history.
    """
    if series.relative_time_sec.size == 0:
        return series
    return ScalarSeries(
        tag=series.tag,
        step=series.step.copy(),
        relative_time_sec=series.relative_time_sec - float(series.relative_time_sec[0]),
        value=series.value.copy(),
    )


def prepare_hitting_curve_data(
    runs: Sequence[RunScalars],
    stage: str,
    error_metric: str,
    resource: str,
    num_thresholds: int = 100,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Compute first-hitting curves for a chosen resource.

    A first-hitting curve maps a target error threshold epsilon to the first
    resource value at which the best error seen so far is at or below epsilon.

    Parameters
    ----------
    resource:
        ``step`` or ``time``.  This decides whether the y-axis is global step or
        relative wall-clock time.
    num_thresholds:
        Number of threshold samples used to draw the curve.

    Returns
    -------
    Mapping label -> (thresholds, hitting_resource).  Missing points are stored
    as NaN if a run never reaches a strict threshold.
    """
    if resource not in ("step", "time"):
        raise ValueError("resource must be 'step' or 'time'")

    selected = {run.label: rebase_relative_time(select_series(run, stage, error_metric)) for run in runs}
    if not selected:
        return {}

    # Choose a common threshold range that is meaningful for all runs.  The upper
    # end is the worst initial value; the lower end is the best value reached by
    # at least one optimizer.  If the range degenerates, expand it slightly.
    initial_values = [float(series.value[0]) for series in selected.values() if series.value.size]
    best_values = [float(np.nanmin(series.value)) for series in selected.values() if series.value.size]
    if not initial_values or not best_values:
        raise ValueError("No scalar values are available for hitting-curve computation.")

    threshold_high = max(initial_values)
    threshold_low = min(best_values)
    if not np.isfinite(threshold_low) or not np.isfinite(threshold_high):
        raise ValueError("Non-finite threshold range for hitting curve.")
    if threshold_low == threshold_high:
        eps = max(abs(threshold_low), 1.0) * 1e-3
        threshold_low -= eps
        threshold_high += eps

    thresholds = np.linspace(threshold_high, threshold_low, num_thresholds)

    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for label, series in selected.items():
        best_so_far = np.minimum.accumulate(series.value)
        resource_values = series.step if resource == "step" else series.relative_time_sec
        hits = np.full_like(thresholds, np.nan, dtype=float)
        for i, threshold in enumerate(thresholds):
            idx = np.where(best_so_far <= threshold)[0]
            if idx.size:
                hits[i] = float(resource_values[idx[0]])
        result[label] = (thresholds.copy(), hits)

    return result
