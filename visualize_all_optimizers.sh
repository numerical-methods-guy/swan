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
nlat="${VIS_NLAT:-128}"
nlon="${VIS_NLON:-256}"
animation_pacing="${ANIMATION_PACING:-standard}"

# Optional focused animation export.  The normal history/forecast comparison
# folders remain unchanged; this creates an extra folder containing only the
# selected optimizer subset for slide-friendly inspection.  Edit the optimizer
# and label arrays below to focus on a different subset.
enable_focused_animations="${ENABLE_FOCUSED_ANIMATIONS:-true}"
focused_animation_group_name="${FOCUSED_ANIMATION_GROUP_NAME:-focused_mud_muon}"
read -r -a focused_animation_optimizers <<< "${FOCUSED_ANIMATION_OPTIMIZERS:-mud muon}"
read -r -a focused_animation_labels <<< "${FOCUSED_ANIMATION_LABELS:-MUD Muon}"

# Optional suffix for exported animation files.  The cluster wrapper sets this
# to values such as "_v3" so copied GIFs still identify their source version.
animation_file_suffix="${VIS_EXPORT_SUFFIX:-}"

# Rollout tensors are much larger than the final figures.  Keep this true for
# cluster runs when the saved .pt rollout files are only an intermediate product
# used to make plots and animations.  Set it to false if you want to inspect or
# reuse the rollout tensors later.
delete_rollouts_after_plotting=true

# ---------------------------------------------------------------------------
# Optimizer groups
# ---------------------------------------------------------------------------
# Full comparison when Gauss-Newton is enabled.
all_optimizers=(adam adamw gauss_newton mud muon sgd)
all_labels=(Adam AdamW Gauss-Newton MUD Muon SGD)

# Full comparison when Gauss-Newton is disabled.
all_without_gauss_optimizers=(adam adamw mud muon sgd)
all_without_gauss_labels=(Adam AdamW MUD Muon SGD)

# Diagnostic comparison for the two unstable/baseline optimizers.
sgd_gauss_optimizers=(gauss_newton sgd)
sgd_gauss_labels=(Gauss-Newton SGD)

# Main high-performing comparison, excluding both SGD and Gauss-Newton.
without_sgd_gauss_optimizers=(adam adamw mud muon)
without_sgd_gauss_labels=(Adam AdamW MUD Muon)

# Same high-performing comparison when Gauss-Newton is not included.
without_sgd_optimizers=(adam adamw mud muon)
without_sgd_labels=(Adam AdamW MUD Muon)

validate_group() {
  local group_name="$1"
  local -n optimizers_ref="$2"
  local -n labels_ref="$3"

  if [[ "${#optimizers_ref[@]}" -ne "${#labels_ref[@]}" ]]; then
    echo "Error: ${group_name} optimizers and labels must have the same length." >&2
    exit 1
  fi
}

animation_fps_for_pacing() {
  local frame_count=$(( (autoreg_steps + output_freq - 1) / output_freq + 1 ))
  local fps

  case "${animation_pacing}" in
    standard)
      fps=8
      ;;
    smooth)
      # Smooth playback is most effective when output_freq is small enough to
      # save dense rollout frames.  This mode plays those frames a bit faster.
      fps=12
      ;;
    slow)
      # Aim for a presentation-friendly 12-second playback duration.  Clamp the
      # result so short rollouts do not become painfully slow.
      fps=$(( (frame_count + 11) / 12 ))
      if (( fps < 2 )); then
        fps=2
      elif (( fps > 6 )); then
        fps=6
      fi
      ;;
    *)
      echo "Error: animation_pacing must be standard, smooth, or slow; got '${animation_pacing}'." >&2
      exit 1
      ;;
  esac
  printf '%s\n' "${fps}"
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
    --nlat "${nlat}" \
    --nlon "${nlon}" \
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
  local animation_fps
  animation_fps="$(animation_fps_for_pacing)"
  echo "Animation pacing: ${animation_pacing} (${animation_fps} fps)"
  python -m visualize animate \
    --rollout_dir "${rollout_dir}" \
    --labels "${labels_ref[@]}" \
    --channel "${channel}" \
    --fps "${animation_fps}" \
    --show_error \
    --output "${forecast_dir}/rollout_fields.gif" \
    --spectral_output "${forecast_dir}/rollout_spectra.gif"
}

run_focused_animation_group() {
  local group_name="$1"
  local animation_dir="$2"
  local rollout_dir="$3"
  local -n optimizers_ref="$4"
  local -n labels_ref="$5"

  mkdir -p "${animation_dir}"
  echo "=== Rendering focused animation group: ${group_name} ==="
  echo "Animations: ${animation_dir}"

  local file_prefix="${group_name}_rollout"
  local animation_fps
  animation_fps="$(animation_fps_for_pacing)"
  echo "Animation pacing: ${animation_pacing} (${animation_fps} fps)"
  python -m visualize animate \
    --rollout_dir "${rollout_dir}" \
    --labels "${labels_ref[@]}" \
    --rollout_names "${labels_ref[@]}" \
    --channel "${channel}" \
    --fps "${animation_fps}" \
    --output "${animation_dir}/${file_prefix}_fields${animation_file_suffix}.gif" \
    --with_error_output "${animation_dir}/${file_prefix}_fields_with_error${animation_file_suffix}.gif" \
    --error_output "${animation_dir}/${file_prefix}_error${animation_file_suffix}.gif" \
    --spectral_output "${animation_dir}/${file_prefix}_spectra${animation_file_suffix}.gif"
}

# ---------------------------------------------------------------------------
# Output folders
# ---------------------------------------------------------------------------
# The script writes three top-level folders: history figures, forecast figures,
# and raw rollout outputs.  Each selected comparison scope gets a matching
# subfolder under every root, which keeps files from different optimizer sets
# easy to browse without mixing them together.
history_root="${FIGURES_HISTORY_ROOT:-./figures_history}"
forecast_root="${FIGURES_FORECAST_ROOT:-./figures_forecast}"
rollout_root="${ROLLOUT_ROOT:-./rollout_results}"
focused_animation_root="${FOCUSED_ANIMATION_ROOT:-./focused_animations}"

all_history_dir="${history_root}/all_optimizers"
all_forecast_dir="${forecast_root}/all_optimizers"

sgd_gauss_history_dir="${history_root}/sgd_and_gauss_newton"
sgd_gauss_forecast_dir="${forecast_root}/sgd_and_gauss_newton"

without_sgd_gauss_history_dir="${history_root}/without_sgd_gauss_newton"
without_sgd_gauss_forecast_dir="${forecast_root}/without_sgd_gauss_newton"

without_sgd_history_dir="${history_root}/without_sgd"
without_sgd_forecast_dir="${forecast_root}/without_sgd"

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
if [[ "${include_gauss_newton}" == "true" ]]; then
  validate_group "all optimizers" all_optimizers all_labels
  build_runs all_optimizers all_runs

  validate_group "SGD-and-Gauss-Newton" sgd_gauss_optimizers sgd_gauss_labels
  build_runs sgd_gauss_optimizers sgd_gauss_runs

  validate_group "without-SGD-and-Gauss-Newton" without_sgd_gauss_optimizers without_sgd_gauss_labels
  build_runs without_sgd_gauss_optimizers without_sgd_gauss_runs
else
  validate_group "all optimizers without Gauss-Newton" all_without_gauss_optimizers all_without_gauss_labels
  build_runs all_without_gauss_optimizers all_without_gauss_runs

  validate_group "without-SGD" without_sgd_optimizers without_sgd_labels
  build_runs without_sgd_optimizers without_sgd_runs
fi

if [[ "${enable_focused_animations}" == "true" ]]; then
  validate_group "focused animations" focused_animation_optimizers focused_animation_labels
fi

# History plots are cheaper than rollouts and do not depend on forecast output.
# Run every selected history comparison first so Gauss-Newton folders/messages
# are produced even if a later checkpoint lookup or rollout job fails.
echo "=== History plot phase ==="

if [[ "${include_gauss_newton}" == "true" ]]; then
  run_history_group "all optimizers including SGD and Gauss-Newton" "${all_history_dir}" all_runs all_labels
  run_history_group "SGD and Gauss-Newton" "${sgd_gauss_history_dir}" sgd_gauss_runs sgd_gauss_labels
  run_history_group "optimizers excluding SGD and Gauss-Newton" "${without_sgd_gauss_history_dir}" without_sgd_gauss_runs without_sgd_gauss_labels
else
  run_history_group "all optimizers" "${all_history_dir}" all_without_gauss_runs all_without_gauss_labels
  run_history_group "optimizers excluding SGD" "${without_sgd_history_dir}" without_sgd_runs without_sgd_labels
fi

echo "=== Forecast and animation phase ==="

if [[ "${include_gauss_newton}" == "true" ]]; then
  run_forecast_group "all optimizers including SGD and Gauss-Newton" "${all_forecast_dir}" "${shared_rollout_dir}" false all_runs all_labels
  run_forecast_group "SGD and Gauss-Newton" "${sgd_gauss_forecast_dir}" "${shared_rollout_dir}" true sgd_gauss_runs sgd_gauss_labels
  run_forecast_group "optimizers excluding SGD and Gauss-Newton" "${without_sgd_gauss_forecast_dir}" "${shared_rollout_dir}" true without_sgd_gauss_runs without_sgd_gauss_labels
else
  run_forecast_group "all optimizers" "${all_forecast_dir}" "${shared_rollout_dir}" false all_without_gauss_runs all_without_gauss_labels
  run_forecast_group "optimizers excluding SGD" "${without_sgd_forecast_dir}" "${shared_rollout_dir}" true without_sgd_runs without_sgd_labels
fi

if [[ "${enable_focused_animations}" == "true" ]]; then
  run_focused_animation_group \
    "${focused_animation_group_name}" \
    "${focused_animation_root}/${focused_animation_group_name}" \
    "${shared_rollout_dir}" \
    focused_animation_optimizers \
    focused_animation_labels
fi

if [[ "${delete_rollouts_after_plotting}" == "true" ]]; then
  echo "=== Removing saved rollout tensors after plotting ==="
  echo "Deleting: ${rollout_root}"
  rm -rf "${rollout_root}"
else
  echo "Keeping saved rollout tensors in: ${rollout_root}"
fi
