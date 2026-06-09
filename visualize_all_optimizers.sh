#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# User-facing visualization settings
# ---------------------------------------------------------------------------
# SGD is always part of the base optimizer comparison.  This can be overridden
# by the cluster wrapper with INCLUDE_GAUSS_NEWTON=false so one wrapper flag can
# control both training and visualization.
include_gauss_newton="${INCLUDE_GAUSS_NEWTON:-true}"

channel="${VIS_CHANNEL:-vorticity}"
history_scale="${HISTORY_SCALE:-log}"
forecast_error_scale="${FORECAST_ERROR_SCALE:-log}"
autoreg_steps="${AUTOREG_STEPS:-100}"
output_freq="${OUTPUT_FREQ:-5}"

# Rollout tensors are much larger than the final figures.  Keep this true for
# cluster runs when the saved .pt rollout files are only an intermediate product
# used to make plots and animations.  Set it to false if you want to inspect or
# reuse the rollout tensors later.
delete_rollouts_after_plotting=true

# ---------------------------------------------------------------------------
# Optimizer groups
# ---------------------------------------------------------------------------
# Base comparison: first-order/baseline optimizers only, always including SGD.
# These are the folders produced by train_all_optimizers.sh when
# include_gauss_newton=false.
base_optimizers=(adam adamw mud muon sgd)
base_labels=(Adam AdamW MUD Muon SGD)

# Optional all-optimizer comparison: the base group plus Gauss-Newton in one
# graph.  Use this when the Gauss-Newton training budget is intentionally part
# of the comparison story.
all_optimizers=(adam adamw gauss_newton mud muon sgd)
all_labels=(Adam AdamW Gauss-Newton MUD Muon SGD)

# Optional diagnostic view: Gauss-Newton alone.  This is useful when it was
# trained for fewer epochs and should not be visually judged as a fair peer of
# the first-order optimizers.
gauss_newton_optimizers=(gauss_newton)
gauss_newton_labels=(Gauss-Newton)

validate_group() {
  local group_name="$1"
  local -n optimizers_ref="$2"
  local -n labels_ref="$3"

  if [[ "${#optimizers_ref[@]}" -ne "${#labels_ref[@]}" ]]; then
    echo "Error: ${group_name} optimizers and labels must have the same length." >&2
    exit 1
  fi
}

build_runs() {
  local -n optimizers_ref="$1"
  local -n runs_ref="$2"

  runs_ref=()
  for opt in "${optimizers_ref[@]}"; do
    runs_ref+=("./logs/${opt}/version_0")
  done
}

run_history_group() {
  local group_name="$1"
  local history_dir="$2"
  local -n runs_ref="$3"
  local -n labels_ref="$4"

  mkdir -p "${history_dir}"
  echo "=== Plotting ${group_name} training history ==="
  echo "Output: ${history_dir}"
  python -m visualize plot_history \
    --runs "${runs_ref[@]}" \
    --labels "${labels_ref[@]}" \
    --stage both \
    --plot both \
    --error_metric l2 \
    --efficiency_metric both \
    --history_scale "${history_scale}" \
    --outdir "${history_dir}"
}

run_forecast_group() {
  local group_name="$1"
  local forecast_dir="$2"
  local rollout_dir="$3"
  local reuse_rollouts="$4"
  local -n runs_ref="$5"
  local -n labels_ref="$6"

  mkdir -p "${forecast_dir}" "${rollout_dir}"
  if [[ "${reuse_rollouts}" == "true" ]]; then
    echo "=== Plotting ${group_name} forecast comparison from existing rollouts ==="
  else
    echo "=== Running ${group_name} forecast rollouts and plotting comparison ==="
  fi
  echo "Figures: ${forecast_dir}"
  echo "Rollouts: ${rollout_dir}"

  local -a forecast_cmd
  forecast_cmd=(
    python -m visualize forecast
    --labels "${labels_ref[@]}" \
    --config config_paradis.yaml \
    --autoreg_steps "${autoreg_steps}" \
    --spherical_method spherical \
    --summary_step final \
    --output_freq "${output_freq}" \
    --channel "${channel}" \
    --rollout_dir "${rollout_dir}" \
    --forecast_error_scale "${forecast_error_scale}" \
    --outdir "${forecast_dir}"
  )
  if [[ "${reuse_rollouts}" == "true" ]]; then
    forecast_cmd+=(--reuse_rollouts)
  else
    forecast_cmd+=(--runs "${runs_ref[@]}")
  fi
  "${forecast_cmd[@]}"

  echo "=== Animating ${group_name} forecast comparison ==="
  python -m visualize animate \
    --rollout_dir "${rollout_dir}" \
    --labels "${labels_ref[@]}" \
    --channel "${channel}" \
    --show_error \
    --output "${forecast_dir}/rollout_fields.gif" \
    --spectral_output "${forecast_dir}/rollout_spectra.gif"
}

# ---------------------------------------------------------------------------
# Output folders
# ---------------------------------------------------------------------------
# The script writes three top-level folders: history figures, forecast figures,
# and raw rollout outputs.  Each selected comparison scope gets a matching
# subfolder under every root, which keeps files from different optimizer sets
# easy to browse without mixing them together.
history_root="./figures_history"
forecast_root="./figures_forecast"
rollout_root="./rollout_results"

base_history_dir="${history_root}/base_optimizers"
base_forecast_dir="${forecast_root}/base_optimizers"
base_rollout_dir="${rollout_root}/base_optimizers"

all_history_dir="${history_root}/with_gauss_newton"
all_forecast_dir="${forecast_root}/with_gauss_newton"

gauss_newton_history_dir="${history_root}/gauss_newton_only"
gauss_newton_forecast_dir="${forecast_root}/gauss_newton_only"

# When Gauss-Newton is included, every optimizer is rolled out once into this
# shared folder.  The separate forecast figure folders then reuse the matching
# per-optimizer subfolders, so grouped comparisons stay accurate without
# repeating expensive model rollouts.
shared_rollout_dir="${rollout_root}/shared_rollouts"

folders_to_clean=("${history_root}" "${forecast_root}" "${rollout_root}")

read -r -p "Clear visualization output folders before plotting? [y/N] " clear_outputs
case "${clear_outputs}" in
  [yY]|[yY][eE][sS])
    echo "Clearing visualization outputs:"
    printf '  %s\n' "${folders_to_clean[@]}"
    rm -rf "${folders_to_clean[@]}"
    ;;
  *)
    echo "Keeping existing visualization outputs"
    ;;
esac

# ---------------------------------------------------------------------------
# Run selected visualization groups
# ---------------------------------------------------------------------------
# Build every selected group before plotting.  Keeping these arrays explicit
# makes each output folder correspond to exactly one comparison scope.
validate_group "base" base_optimizers base_labels
build_runs base_optimizers base_runs

if [[ "${include_gauss_newton}" == "true" ]]; then
  validate_group "all-with-Gauss-Newton" all_optimizers all_labels
  build_runs all_optimizers all_runs

  validate_group "Gauss-Newton-only" gauss_newton_optimizers gauss_newton_labels
  build_runs gauss_newton_optimizers gauss_newton_runs
fi

# History plots are cheaper than rollouts and do not depend on forecast output.
# Run every selected history comparison first so Gauss-Newton folders/messages
# are produced even if a later checkpoint lookup or rollout job fails.
echo "=== History plot phase ==="
run_history_group "base optimizer" "${base_history_dir}" base_runs base_labels

if [[ "${include_gauss_newton}" == "true" ]]; then
  run_history_group "all optimizers including Gauss-Newton" "${all_history_dir}" all_runs all_labels
  run_history_group "Gauss-Newton-only" "${gauss_newton_history_dir}" gauss_newton_runs gauss_newton_labels
fi

echo "=== Forecast and animation phase ==="

if [[ "${include_gauss_newton}" == "true" ]]; then
  run_forecast_group "all optimizers including Gauss-Newton" "${all_forecast_dir}" "${shared_rollout_dir}" false all_runs all_labels
  run_forecast_group "base optimizer" "${base_forecast_dir}" "${shared_rollout_dir}" true base_runs base_labels
  run_forecast_group "Gauss-Newton-only" "${gauss_newton_forecast_dir}" "${shared_rollout_dir}" true gauss_newton_runs gauss_newton_labels
else
  run_forecast_group "base optimizer" "${base_forecast_dir}" "${base_rollout_dir}" false base_runs base_labels
fi

if [[ "${delete_rollouts_after_plotting}" == "true" ]]; then
  echo "=== Removing saved rollout tensors after plotting ==="
  echo "Deleting: ${rollout_root}"
  rm -rf "${rollout_root}"
else
  echo "Keeping saved rollout tensors in: ${rollout_root}"
fi
