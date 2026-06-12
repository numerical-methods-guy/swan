#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# User-facing visualization settings
# ---------------------------------------------------------------------------
channel="${VIS_CHANNEL:-vorticity}"
config_path="${VIS_CONFIG:-./swan_checkpoints/config_paradis.yaml}"
runs_root="${RUNS_ROOT:-./swan_checkpoints/logs}"
history_scale="${HISTORY_SCALE:-log}"
forecast_error_scale="${FORECAST_ERROR_SCALE:-log}"
plot_skill_horizon="${PLOT_SKILL_HORIZON:-false}"
skill_horizon_gammas="${SKILL_HORIZON_GAMMAS:-}"
plot_spectral_horizon="${PLOT_SPECTRAL_HORIZON:-false}"
spectral_horizon_modes_text="${SPECTRAL_HORIZON_MODES:-abs positive}"
spectral_eta_factors_text="${SPECTRAL_ETA_FACTORS:-1.05 1.1 1.25 1.5 2}"
autoreg_steps="${AUTOREG_STEPS:-100}"
output_freq="${OUTPUT_FREQ:-5}"
nlat="${VIS_NLAT:-128}"
nlon="${VIS_NLON:-256}"
animation_pacing="${ANIMATION_PACING:-standard}"
run_history_plots="${RUN_HISTORY_PLOTS:-false}"
visualization_root="${VISUALIZATION_ROOT:-./forecast_group_visualizations}"
visualization_version="${VISUALIZATION_VERSION:-auto}"
overwrite_visualization_outputs="${OVERWRITE_VISUALIZATION_OUTPUTS:-false}"
read -r -a vis_groups <<< "${VIS_GROUPS:-all_optimizers without_spectral_blow_up without_spatial_blow_up stable_core}"
reuse_legacy_rollouts="${REUSE_LEGACY_ROLLOUTS:-false}"

# Rollout tensors are much larger than the final figures.  Keep this true for
# cluster runs when the saved .pt rollout files are only an intermediate product
# used to make plots and animations.  Set it to false if you want to inspect or
# reuse the rollout tensors later.
delete_rollouts_after_plotting="${DELETE_ROLLOUTS_AFTER_PLOTTING:-true}"

# ---------------------------------------------------------------------------
# Optimizer groups
# ---------------------------------------------------------------------------
# Explicit group definitions provided by the user-facing wrapper.
read -r -a all_optimizers <<< "${ALL_OPTIMIZERS:-adam adamw mud mud_new muon muon_new sgd}"
read -r -a all_labels <<< "${ALL_LABELS:-Adam AdamW MUD MUD-new Muon Muon-new SGD}"
read -r -a without_spectral_blow_up_optimizers <<< "${WITHOUT_SPECTRAL_BLOW_UP_OPTIMIZERS:-adam adamw mud_new muon_new sgd}"
read -r -a without_spectral_blow_up_labels <<< "${WITHOUT_SPECTRAL_BLOW_UP_LABELS:-Adam AdamW MUD-new Muon-new SGD}"
read -r -a without_spatial_blow_up_optimizers <<< "${WITHOUT_SPATIAL_BLOW_UP_OPTIMIZERS:-adam adamw mud muon_new sgd}"
read -r -a without_spatial_blow_up_labels <<< "${WITHOUT_SPATIAL_BLOW_UP_LABELS:-Adam AdamW MUD Muon-new SGD}"
read -r -a stable_core_optimizers <<< "${STABLE_CORE_OPTIMIZERS:-adam adamw muon_new sgd}"
read -r -a stable_core_labels <<< "${STABLE_CORE_LABELS:-Adam AdamW Muon-new SGD}"

# Optional user-defined comparison group.
read -r -a custom_optimizers <<< "${CUSTOM_OPTIMIZERS:-}"
read -r -a custom_labels <<< "${CUSTOM_LABELS:-}"

validate_group() {
  local group_name="$1"
  local -n optimizers_ref="$2"
  local -n labels_ref="$3"

  if [[ "${#optimizers_ref[@]}" -ne "${#labels_ref[@]}" ]]; then
    echo "Error: ${group_name} optimizers and labels must have the same length." >&2
    exit 1
  fi
}

validate_bool() {
  case "$2" in
    true|false) ;;
    *) echo "Error: $1 must be true or false, got '$2'." >&2; exit 1 ;;
  esac
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
    runs_ref+=("${runs_root}/${opt}/version_0")
  done
}

check_inputs() {
  local run
  local missing=0
  local -a runs_to_check=()
  local group

  if [[ ! -f "${config_path}" ]]; then
    echo "Error: config does not exist: ${config_path}" >&2
    missing=1
  fi

  for group in "${vis_groups[@]}"; do
    case "${group}" in
      all_optimizers)
        runs_to_check+=("${all_runs[@]}")
        ;;
      without_spectral_blow_up)
        runs_to_check+=("${without_spectral_blow_up_runs[@]}")
        ;;
      without_spatial_blow_up)
        runs_to_check+=("${without_spatial_blow_up_runs[@]}")
        ;;
      stable_core)
        runs_to_check+=("${stable_core_runs[@]}")
        ;;
      custom)
        runs_to_check+=("${custom_runs[@]}")
        ;;
      *)
        echo "Error: unknown visualization group '${group}'." >&2
        exit 1
        ;;
    esac
  done

  readarray -t runs_to_check < <(printf '%s\n' "${runs_to_check[@]}" | awk 'NF && !seen[$0]++')

  for run in "${runs_to_check[@]}"; do
    if [[ ! -d "${run}" ]]; then
      echo "Error: run directory does not exist: ${run}" >&2
      missing=1
    fi
  done

  if [[ "${missing}" -ne 0 ]]; then
    echo "Expected swan_checkpoints layout:" >&2
    echo "  swan_checkpoints/config_paradis.yaml" >&2
    echo "  swan_checkpoints/logs/<optimizer>/version_0" >&2
    exit 1
  fi
}

prepare_visualization_version() {
  local root="$1"
  local version="$2"
  local candidate
  local latest=""
  local idx=0

  mkdir -p "${root}"

  case "${version}" in
    auto)
      if [[ "${overwrite_visualization_outputs}" == "true" ]]; then
        while true; do
          candidate="${root}/version_${idx}"
          if [[ -e "${candidate}" ]]; then
            latest="${candidate}"
            idx=$((idx + 1))
          else
            break
          fi
        done
        if [[ -n "${latest}" ]]; then
          printf '%s\n' "${latest}"
        else
          candidate="${root}/version_0"
          mkdir -p "${candidate}"
          printf '%s\n' "${candidate}"
        fi
        return 0
      fi
      while true; do
        candidate="${root}/version_${idx}"
        if [[ -e "${candidate}" ]]; then
          idx=$((idx + 1))
        else
          mkdir -p "${candidate}"
          printf '%s\n' "${candidate}"
          return 0
        fi
      done
      ;;
    version_*)
      candidate="${root}/${version}"
      mkdir -p "${candidate}"
      printf '%s\n' "${candidate}"
      ;;
    *)
      candidate="${root}/${version}"
      mkdir -p "${candidate}"
      printf '%s\n' "${candidate}"
      ;;
  esac
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
  local -a skill_gamma_args=()
  local -a spectral_modes=()
  local -a spectral_eta_factors=()
  local mode
  local eta_factor
  local first_forecast=true
  if [[ -n "${skill_horizon_gammas}" ]]; then
    read -r -a skill_gamma_args <<< "${skill_horizon_gammas}"
  fi

  if [[ "${plot_spectral_horizon}" == "true" ]]; then
    read -r -a spectral_modes <<< "${spectral_horizon_modes_text}"
    read -r -a spectral_eta_factors <<< "${spectral_eta_factors_text}"
  fi
  if [[ "${#spectral_modes[@]}" -eq 0 ]]; then
    spectral_modes=("")
  fi
  if [[ "${#spectral_eta_factors[@]}" -eq 0 ]]; then
    spectral_eta_factors=("2.0")
  fi

  if [[ "${plot_spectral_horizon}" == "true" ]]; then
    echo "Note: each spectral mode / eta-factor pass reruns visualize forecast."
    echo "Common forecast files keep the same names and are overwritten in place."
    echo "Only spectral_horizon_* files accumulate as separate outputs."
  fi

  for eta_factor in "${spectral_eta_factors[@]}"; do
    for mode in "${spectral_modes[@]}"; do
      forecast_cmd=(
        python -m visualize forecast
        --labels "${labels_ref[@]}" \
        --config "${config_path}" \
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
      if [[ "${plot_skill_horizon}" == "true" ]]; then
        forecast_cmd+=(--skill_horizon)
        if [[ "${#skill_gamma_args[@]}" -gt 0 ]]; then
          forecast_cmd+=(--skill_horizon_gammas "${skill_gamma_args[@]}")
        fi
      fi
      if [[ "${plot_spectral_horizon}" == "true" ]]; then
        forecast_cmd+=(--spectral_horizon --spectral_horizon_mode "${mode}" --spectral_eta_factor "${eta_factor}")
      fi
      if [[ "${reuse_rollouts}" == "true" || "${first_forecast}" == "false" ]]; then
        forecast_cmd+=(--reuse_rollouts)
      else
        forecast_cmd+=(--runs "${runs_ref[@]}")
      fi
      "${forecast_cmd[@]}"
      first_forecast=false
    done
  done

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

has_group_rollouts() {
  local -n labels_ref="$1"
  local label
  local run_dir

  for label in "${labels_ref[@]}"; do
    run_dir="${shared_rollout_dir}/${label}"
    if [[ ! -f "${run_dir}/metrics.csv" || ! -f "${run_dir}/per_step_metrics.csv" ]]; then
      return 1
    fi
    if ! compgen -G "${run_dir}/ic*_prediction_*.pt" > /dev/null && ! compgen -G "${run_dir}/prediction_*.pt" > /dev/null; then
      return 1
    fi
    if ! compgen -G "${run_dir}/ic*_truth_*.pt" > /dev/null && ! compgen -G "${run_dir}/truth_*.pt" > /dev/null; then
      return 1
    fi
  done
  return 0
}

# ---------------------------------------------------------------------------
# Output folders
# ---------------------------------------------------------------------------
# The script writes into one versioned root by default:
# forecast_group_visualizations/version_N/{figures_history,figures_forecast,...}
if [[ -z "${FIGURES_HISTORY_ROOT:-}" && -z "${FIGURES_FORECAST_ROOT:-}" && -z "${ROLLOUT_ROOT:-}" ]]; then
  version_dir="$(prepare_visualization_version "${visualization_root}" "${visualization_version}")"
  history_root="${version_dir}/figures_history"
  forecast_root="${version_dir}/figures_forecast"
  rollout_root="${version_dir}/rollout_results"
  echo "Visualization version directory: ${version_dir}"
else
  history_root="${FIGURES_HISTORY_ROOT:-./figures_history}"
  forecast_root="${FIGURES_FORECAST_ROOT:-./figures_forecast}"
  rollout_root="${ROLLOUT_ROOT:-./rollout_results}"
fi
all_history_dir="${history_root}/all_optimizers"
all_forecast_dir="${forecast_root}/all_optimizers"

without_spectral_blow_up_history_dir="${history_root}/without_spectral_blow_up"
without_spectral_blow_up_forecast_dir="${forecast_root}/without_spectral_blow_up"

without_spatial_blow_up_history_dir="${history_root}/without_spatial_blow_up"
without_spatial_blow_up_forecast_dir="${forecast_root}/without_spatial_blow_up"

stable_core_history_dir="${history_root}/stable_core"
stable_core_forecast_dir="${forecast_root}/stable_core"

custom_history_dir="${history_root}/custom"
custom_forecast_dir="${forecast_root}/custom"

# Every optimizer is rolled out once into this shared folder. The second group
# reuses the matching per-optimizer subfolders, so grouped comparisons stay
# accurate without repeating expensive model rollouts.
shared_rollout_dir="${rollout_root}/shared_rollouts"
legacy_shared_rollout_dir="./rollout_results/shared_rollouts"

if [[ "${reuse_legacy_rollouts}" == "true" && ! -d "${shared_rollout_dir}" && -d "${legacy_shared_rollout_dir}" ]]; then
  echo "Found existing legacy shared rollouts: ${legacy_shared_rollout_dir}"
  echo "Reusing them while writing figures to: ${forecast_root}"
  shared_rollout_dir="${legacy_shared_rollout_dir}"
fi

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
# Build every selected group before plotting. Keeping these arrays explicit
# makes each output folder correspond to exactly one comparison scope.
validate_group "all optimizers" all_optimizers all_labels
build_runs all_optimizers all_runs

validate_group "without spectral blow-up" without_spectral_blow_up_optimizers without_spectral_blow_up_labels
build_runs without_spectral_blow_up_optimizers without_spectral_blow_up_runs

validate_group "without spatial blow-up" without_spatial_blow_up_optimizers without_spatial_blow_up_labels
build_runs without_spatial_blow_up_optimizers without_spatial_blow_up_runs

validate_group "stable core" stable_core_optimizers stable_core_labels
build_runs stable_core_optimizers stable_core_runs

if [[ "${#custom_optimizers[@]}" -gt 0 || "${#custom_labels[@]}" -gt 0 ]]; then
  validate_group "custom" custom_optimizers custom_labels
  build_runs custom_optimizers custom_runs
fi

validate_bool "RUN_HISTORY_PLOTS" "${run_history_plots}"
validate_bool "DELETE_ROLLOUTS_AFTER_PLOTTING" "${delete_rollouts_after_plotting}"
validate_bool "REUSE_LEGACY_ROLLOUTS" "${reuse_legacy_rollouts}"
validate_bool "PLOT_SKILL_HORIZON" "${plot_skill_horizon}"
validate_bool "PLOT_SPECTRAL_HORIZON" "${plot_spectral_horizon}"

if [[ "${plot_spectral_horizon}" == "true" ]]; then
  read -r -a _spectral_modes_check <<< "${spectral_horizon_modes_text}"
  read -r -a _spectral_eta_factors_check <<< "${spectral_eta_factors_text}"
  if [[ "${#_spectral_modes_check[@]}" -eq 0 ]]; then
    echo "Error: SPECTRAL_HORIZON_MODES must contain at least one mode when PLOT_SPECTRAL_HORIZON=true." >&2
    exit 1
  fi
  if [[ "${#_spectral_eta_factors_check[@]}" -eq 0 ]]; then
    echo "Error: SPECTRAL_ETA_FACTORS must contain at least one factor when PLOT_SPECTRAL_HORIZON=true." >&2
    exit 1
  fi
  for _mode in "${_spectral_modes_check[@]}"; do
    case "${_mode}" in
      abs|positive) ;;
      *)
        echo "Error: invalid spectral horizon mode '${_mode}'. Allowed: abs positive" >&2
        exit 1
        ;;
    esac
  done
  for _eta_factor in "${_spectral_eta_factors_check[@]}"; do
    if ! awk "BEGIN { exit !(${_eta_factor} > 1.0) }"; then
      echo "Error: each SPECTRAL_ETA_FACTORS entry must be greater than 1; got '${_eta_factor}'." >&2
      exit 1
    fi
  done
fi

echo "=== Input paths ==="
echo "Runs root: ${runs_root}"
echo "Config: ${config_path}"
check_inputs

run_selected_history_group() {
  local group="$1"

  case "${group}" in
    all_optimizers)
      run_history_group "all optimizers" "${all_history_dir}" all_runs all_labels
      ;;
    without_spectral_blow_up)
      run_history_group "optimizers excluding spectral blow-ups" "${without_spectral_blow_up_history_dir}" without_spectral_blow_up_runs without_spectral_blow_up_labels
      ;;
    without_spatial_blow_up)
      run_history_group "optimizers excluding spatial blow-ups" "${without_spatial_blow_up_history_dir}" without_spatial_blow_up_runs without_spatial_blow_up_labels
      ;;
    stable_core)
      run_history_group "stable core optimizers" "${stable_core_history_dir}" stable_core_runs stable_core_labels
      ;;
    custom)
      run_history_group "custom optimizers" "${custom_history_dir}" custom_runs custom_labels
      ;;
    *)
      echo "Error: unknown visualization group '${group}'." >&2
      echo "Allowed groups: all_optimizers without_spectral_blow_up without_spatial_blow_up stable_core custom" >&2
      exit 1
      ;;
  esac
}

run_selected_forecast_group() {
  local group="$1"

  case "${group}" in
    all_optimizers)
      if has_group_rollouts all_labels; then
        echo "Found existing shared rollouts; reusing them for all optimizer plots."
        run_forecast_group "all optimizers" "${all_forecast_dir}" "${shared_rollout_dir}" true all_runs all_labels
      else
        run_forecast_group "all optimizers" "${all_forecast_dir}" "${shared_rollout_dir}" false all_runs all_labels
      fi
      ;;
    without_spectral_blow_up)
      if has_group_rollouts without_spectral_blow_up_labels; then
        run_forecast_group "optimizers excluding spectral blow-ups" "${without_spectral_blow_up_forecast_dir}" "${shared_rollout_dir}" true without_spectral_blow_up_runs without_spectral_blow_up_labels
      else
        run_forecast_group "optimizers excluding spectral blow-ups" "${without_spectral_blow_up_forecast_dir}" "${shared_rollout_dir}" false without_spectral_blow_up_runs without_spectral_blow_up_labels
      fi
      ;;
    without_spatial_blow_up)
      if has_group_rollouts without_spatial_blow_up_labels; then
        run_forecast_group "optimizers excluding spatial blow-ups" "${without_spatial_blow_up_forecast_dir}" "${shared_rollout_dir}" true without_spatial_blow_up_runs without_spatial_blow_up_labels
      else
        run_forecast_group "optimizers excluding spatial blow-ups" "${without_spatial_blow_up_forecast_dir}" "${shared_rollout_dir}" false without_spatial_blow_up_runs without_spatial_blow_up_labels
      fi
      ;;
    stable_core)
      if has_group_rollouts stable_core_labels; then
        run_forecast_group "stable core optimizers" "${stable_core_forecast_dir}" "${shared_rollout_dir}" true stable_core_runs stable_core_labels
      else
        run_forecast_group "stable core optimizers" "${stable_core_forecast_dir}" "${shared_rollout_dir}" false stable_core_runs stable_core_labels
      fi
      ;;
    custom)
      if has_group_rollouts custom_labels; then
        run_forecast_group "custom optimizers" "${custom_forecast_dir}" "${shared_rollout_dir}" true custom_runs custom_labels
      else
        run_forecast_group "custom optimizers" "${custom_forecast_dir}" "${shared_rollout_dir}" false custom_runs custom_labels
      fi
      ;;
    *)
      echo "Error: unknown visualization group '${group}'." >&2
      echo "Allowed groups: all_optimizers without_spectral_blow_up without_spatial_blow_up stable_core custom" >&2
      exit 1
      ;;
  esac
}

case "${run_history_plots}" in
  true)
    echo "=== History plot phase ==="
    for group in "${vis_groups[@]}"; do
      run_selected_history_group "${group}"
    done
    ;;
  false)
    echo "Skipping history plot phase because swan_checkpoints contains checkpoints but no TensorBoard/CSV history files."
    ;;
  *)
    echo "Error: RUN_HISTORY_PLOTS must be true or false, got '${run_history_plots}'." >&2
    exit 1
    ;;
esac

echo "=== Forecast and animation phase ==="

for group in "${vis_groups[@]}"; do
  run_selected_forecast_group "${group}"
done

if [[ "${delete_rollouts_after_plotting}" == "true" ]]; then
  echo "=== Removing saved rollout tensors after plotting ==="
  echo "Deleting: ${rollout_root}"
  rm -rf "${rollout_root}"
else
  echo "Keeping saved rollout tensors in: ${rollout_root}"
fi
