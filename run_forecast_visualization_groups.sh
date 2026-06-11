#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Error at ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND}" >&2' ERR

# ---------------------------------------------------------------------------
# Cluster pipeline wrapper
# ---------------------------------------------------------------------------
# Submit or run this file on a cluster when you want a non-interactive workflow.
# It keeps trained checkpoints by default and only auto-cleans visualization
# outputs, because training can be expensive while figures and rollout
# intermediates are easy to regenerate.
#
# If your cluster uses Slurm, edit the resource lines below for your queue.
# They are comments for normal bash execution, but Slurm reads #SBATCH lines
# when the file is submitted with sbatch.
#
#SBATCH --job-name=swan_visualize_2groups
#SBATCH --output=cluster_logs/%x_%j.out
#SBATCH --error=cluster_logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
# This grouped wrapper only visualizes saved swan_checkpoints models.
run_visualization=true

resolution_nlat=128
resolution_nlon=256

# overwrite_visualization_outputs=false preserves existing visualization runs by
# creating the next version_N folder. When true, the wrapper rewrites the latest
# existing version_N folder, or creates version_0 if no version exists yet.
overwrite_visualization_outputs=true

# Visualization settings passed to visualize_two_group_optimizers.sh for this wrapper
# run. Direct runs of visualize_two_group_optimizers.sh keep that file's defaults.
visualization_runs_root="./swan_checkpoints/logs"
visualization_config="./swan_checkpoints/config_paradis.yaml"
visualization_channel="vorticity"
visualization_history_scale="log"
visualization_forecast_error_scale="log"
visualization_skill_horizon=true
visualization_skill_horizon_gammas=""
visualization_autoreg_steps=250
visualization_output_freq=10
run_history_plots=false
delete_rollouts_after_plotting=true

# Available visualization groups:
#   all_optimizers:
#     Adam, AdamW, MUD, MUD-new, Muon, Muon-new, SGD
#   without_spectral_blow_up:
#     Adam, AdamW, MUD-new, Muon-new, SGD
#     excludes MUD and Muon
#   without_spatial_blow_up:
#     Adam, AdamW, MUD, Muon-new, SGD
#     excludes MUD-new and Muon
#   stable_core:
#     Adam, AdamW, Muon-new, SGD
#     excludes MUD, MUD-new, and Muon
#   custom:
#     uses CUSTOM_OPTIMIZERS and CUSTOM_LABELS below
VIS_GROUPS=(all_optimizers stable_core)
CUSTOM_OPTIMIZERS=(adam adamw)
CUSTOM_LABELS=(Adam AdamW)

# Animation playback mode: standard, smooth, or slow. Smooth uses higher FPS;
# slow chooses FPS from autoreg_steps/output_freq for audience-friendly playback.
visualization_animation_pacing="slow"

# Legacy ./rollout_results reuse is disabled by default so cluster outputs stay
# self-contained under visualization_root/version_N.
reuse_legacy_rollouts=false

# Versioned visualization outputs. With visualization_version="auto",
# overwrite_visualization_outputs controls whether the wrapper creates the next
# version_N or rewrites the latest existing version_N.
visualization_root="./visualization_runs_two_groups"
visualization_version="auto"

# Optional environment setup. Put module/conda commands here if needed, e.g.:
#
# module purge
# module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate swan

mkdir -p cluster_logs

print_settings() {
  echo "=== SWAN cluster wrapper ==="
  echo "Working directory: $(pwd)"
  echo "run_visualization=${run_visualization}"
  echo "resolution_nlat=${resolution_nlat}"
  echo "resolution_nlon=${resolution_nlon}"
  echo "overwrite_visualization_outputs=${overwrite_visualization_outputs}"
  echo "visualization_runs_root=${visualization_runs_root}"
  echo "visualization_config=${visualization_config}"
  echo "visualization_channel=${visualization_channel}"
  echo "visualization_history_scale=${visualization_history_scale}"
  echo "visualization_forecast_error_scale=${visualization_forecast_error_scale}"
  echo "visualization_skill_horizon=${visualization_skill_horizon}"
  echo "visualization_skill_horizon_gammas=${visualization_skill_horizon_gammas:-<default>}"
  echo "visualization_autoreg_steps=${visualization_autoreg_steps}"
  echo "visualization_output_freq=${visualization_output_freq}"
  echo "run_history_plots=${run_history_plots}"
  echo "delete_rollouts_after_plotting=${delete_rollouts_after_plotting}"
  echo "VIS_GROUPS=${VIS_GROUPS[*]}"
  echo "CUSTOM_OPTIMIZERS=${CUSTOM_OPTIMIZERS[*]}"
  echo "CUSTOM_LABELS=${CUSTOM_LABELS[*]}"
  echo "visualization_animation_pacing=${visualization_animation_pacing}"
  echo "reuse_legacy_rollouts=${reuse_legacy_rollouts}"
  echo "visualization_root=${visualization_root}"
  echo "visualization_version=${visualization_version}"
}

validate_bool() {
  case "$2" in
    true|false) ;;
    *) echo "Error: $1 must be true or false, got '$2'." >&2; exit 1 ;;
  esac
}

run_visualization_phase() {
  local clear_answer="n"
  local version_dir
  local groups

  case "${overwrite_visualization_outputs}" in
    true)
      clear_answer="y"
      echo "Latest/selected visualization version will be overwritten."
      ;;
    false)
      clear_answer="n"
      echo "Existing visualization versions will be preserved; a new version will be created when using auto."
      ;;
  esac

  version_dir="$(prepare_visualization_version)"
  groups="$(visualization_groups_env)"
  write_visualization_settings "${version_dir}"

  echo "=== Visualization phase ==="
  echo "Visualization version directory: ${version_dir}"
  printf '%s\n' "${clear_answer}" | env RUNS_ROOT="${visualization_runs_root}" VIS_CONFIG="${visualization_config}" VIS_CHANNEL="${visualization_channel}" HISTORY_SCALE="${visualization_history_scale}" FORECAST_ERROR_SCALE="${visualization_forecast_error_scale}" PLOT_SKILL_HORIZON="${visualization_skill_horizon}" SKILL_HORIZON_GAMMAS="${visualization_skill_horizon_gammas}" AUTOREG_STEPS="${visualization_autoreg_steps}" OUTPUT_FREQ="${visualization_output_freq}" RUN_HISTORY_PLOTS="${run_history_plots}" DELETE_ROLLOUTS_AFTER_PLOTTING="${delete_rollouts_after_plotting}" VIS_GROUPS="${groups}" CUSTOM_OPTIMIZERS="${CUSTOM_OPTIMIZERS[*]}" CUSTOM_LABELS="${CUSTOM_LABELS[*]}" ANIMATION_PACING="${visualization_animation_pacing}" REUSE_LEGACY_ROLLOUTS="${reuse_legacy_rollouts}" VIS_NLAT="${resolution_nlat}" VIS_NLON="${resolution_nlon}" FIGURES_HISTORY_ROOT="${version_dir}/figures_history" FIGURES_FORECAST_ROOT="${version_dir}/figures_forecast" ROLLOUT_ROOT="${version_dir}/rollout_results" bash visualize_two_group_optimizers.sh
}

prepare_visualization_version() {
  local root="${visualization_root}"
  local version="${visualization_version}"
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

validate_visualization_group() {
  case "$1" in
    all_optimizers|without_spectral_blow_up|without_spatial_blow_up|stable_core|custom) ;;
    *)
      echo "Error: unknown visualization group '$1'." >&2
      echo "Allowed groups: all_optimizers without_spectral_blow_up without_spatial_blow_up stable_core custom" >&2
      exit 1
      ;;
  esac
}

visualization_groups_env() {
  local group
  if [[ "${#VIS_GROUPS[@]}" -eq 0 ]]; then
    echo "Error: VIS_GROUPS must contain at least one group." >&2
    exit 1
  fi
  for group in "${VIS_GROUPS[@]}"; do
    validate_visualization_group "${group}"
    if [[ "${group}" == "custom" ]]; then
      if [[ "${#CUSTOM_OPTIMIZERS[@]}" -eq 0 ]]; then
        echo "Error: CUSTOM_OPTIMIZERS must contain at least one optimizer when VIS_GROUPS includes custom." >&2
        exit 1
      fi
      if [[ "${#CUSTOM_OPTIMIZERS[@]}" -ne "${#CUSTOM_LABELS[@]}" ]]; then
        echo "Error: CUSTOM_OPTIMIZERS and CUSTOM_LABELS must have the same length." >&2
        exit 1
      fi
    fi
  done
  printf '%s\n' "${VIS_GROUPS[*]}"
}

write_visualization_settings() {
  local version_dir="$1"

  cat > "${version_dir}/settings.txt" <<EOF
run_visualization=${run_visualization}
resolution_nlat=${resolution_nlat}
resolution_nlon=${resolution_nlon}
overwrite_visualization_outputs=${overwrite_visualization_outputs}
visualization_runs_root=${visualization_runs_root}
visualization_config=${visualization_config}
visualization_channel=${visualization_channel}
visualization_history_scale=${visualization_history_scale}
visualization_forecast_error_scale=${visualization_forecast_error_scale}
visualization_skill_horizon=${visualization_skill_horizon}
visualization_skill_horizon_gammas=${visualization_skill_horizon_gammas:-<default>}
visualization_autoreg_steps=${visualization_autoreg_steps}
visualization_output_freq=${visualization_output_freq}
run_history_plots=${run_history_plots}
delete_rollouts_after_plotting=${delete_rollouts_after_plotting}
VIS_GROUPS=${VIS_GROUPS[*]}
CUSTOM_OPTIMIZERS=${CUSTOM_OPTIMIZERS[*]}
CUSTOM_LABELS=${CUSTOM_LABELS[*]}
visualization_animation_pacing=${visualization_animation_pacing}
reuse_legacy_rollouts=${reuse_legacy_rollouts}
visualization_root=${visualization_root}
visualization_version=${visualization_version}
version_dir=${version_dir}
EOF
}

validate_bool "run_visualization" "${run_visualization}"
validate_bool "overwrite_visualization_outputs" "${overwrite_visualization_outputs}"
validate_bool "visualization_skill_horizon" "${visualization_skill_horizon}"
validate_bool "run_history_plots" "${run_history_plots}"
validate_bool "delete_rollouts_after_plotting" "${delete_rollouts_after_plotting}"
validate_bool "reuse_legacy_rollouts" "${reuse_legacy_rollouts}"
case "${visualization_animation_pacing}" in
  standard|smooth|slow) ;;
  *) echo "Error: visualization_animation_pacing must be standard, smooth, or slow." >&2; exit 1 ;;
esac

print_settings

case "${run_visualization}" in
  true) run_visualization_phase ;;
  false) echo "Skipping visualization phase." ;;
esac

echo "=== Cluster wrapper finished ==="
exit 0
