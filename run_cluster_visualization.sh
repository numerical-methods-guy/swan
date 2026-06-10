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
#SBATCH --job-name=swan_visualize
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
# Set run_training=true when this job should launch training before
# visualization. For figure tweaking, keep this false and reuse checkpoints
# already in ./logs/<optimizer>/version_0.
run_training=false
run_visualization=true

# clear_training_logs=true answers "y" to the training cleanup prompt and
# deletes ./logs before training. Use it for a clean fresh run only.
clear_training_logs=false

# One Gauss-Newton switch for this wrapper run. When false, the wrapper skips
# Gauss-Newton during training and produces only the base visualization group.
include_gauss_newton=true

# Epoch overrides used only when this wrapper calls train_all_optimizers.sh.
training_pretrain_epochs=25
training_gauss_newton_epochs=2
resolution_nlat=128
resolution_nlon=256

# overwrite_visualization_outputs=false preserves existing visualization runs by
# creating the next version_N folder. When true, the wrapper rewrites the latest
# existing version_N folder, or creates version_0 if no version exists yet.
overwrite_visualization_outputs=false

# Visualization settings passed to visualize_all_optimizers.sh for this wrapper
# run. Direct runs of visualize_all_optimizers.sh keep that file's defaults.
visualization_channel="vorticity"
visualization_history_scale="log"
visualization_forecast_error_scale="log"
visualization_autoreg_steps=100
visualization_output_freq=5
# Animation playback mode: standard, smooth, or slow. Smooth uses higher FPS;
# slow chooses FPS from autoreg_steps/output_freq for audience-friendly playback.
visualization_animation_pacing="slow"
enable_focused_animations=true
# User-facing focused animation selection. Use optimizer IDs from this set:
# adam, adamw, gauss_newton, mud, muon, sgd.
# Display labels and output folder names are derived internally.
focused_animation_optimizers=(adam adamw)

# Versioned visualization outputs. With visualization_version="auto",
# overwrite_visualization_outputs controls whether the wrapper creates the next
# version_N or rewrites the latest existing version_N.
visualization_root="./visualization_runs"
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
  echo "run_training=${run_training}"
  echo "run_visualization=${run_visualization}"
  echo "clear_training_logs=${clear_training_logs}"
  echo "include_gauss_newton=${include_gauss_newton}"
  echo "training_pretrain_epochs=${training_pretrain_epochs}"
  echo "training_gauss_newton_epochs=${training_gauss_newton_epochs}"
  echo "resolution_nlat=${resolution_nlat}"
  echo "resolution_nlon=${resolution_nlon}"
  echo "overwrite_visualization_outputs=${overwrite_visualization_outputs}"
  echo "visualization_channel=${visualization_channel}"
  echo "visualization_history_scale=${visualization_history_scale}"
  echo "visualization_forecast_error_scale=${visualization_forecast_error_scale}"
  echo "visualization_autoreg_steps=${visualization_autoreg_steps}"
  echo "visualization_output_freq=${visualization_output_freq}"
  echo "visualization_animation_pacing=${visualization_animation_pacing}"
  echo "enable_focused_animations=${enable_focused_animations}"
  echo "focused_animation_optimizers=${focused_animation_optimizers[*]}"
  echo "visualization_root=${visualization_root}"
  echo "visualization_version=${visualization_version}"
}

validate_bool() {
  case "$2" in
    true|false) ;;
    *) echo "Error: $1 must be true or false, got '$2'." >&2; exit 1 ;;
  esac
}

run_training_phase() {
  local clear_answer="n"

  case "${clear_training_logs}" in
    true)
      clear_answer="y"
      echo "Training logs will be cleared by train_all_optimizers.sh."
      ;;
    false)
      clear_answer="n"
      echo "Training logs will be kept. Existing logs may cause new Lightning version_* folders."
      ;;
  esac

  echo "=== Training phase ==="
  printf '%s\n' "${clear_answer}" | env PRETRAIN_EPOCHS="${training_pretrain_epochs}" GAUSS_NEWTON_EPOCHS="${training_gauss_newton_epochs}" INCLUDE_GAUSS_NEWTON="${include_gauss_newton}" TRAIN_NLAT="${resolution_nlat}" TRAIN_NLON="${resolution_nlon}" bash train_all_optimizers.sh
}

run_visualization_phase() {
  local clear_answer="n"
  local version_dir
  local export_suffix
  local focused_group_name
  local focused_optimizers
  local focused_labels

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
  export_suffix="$(animation_export_suffix "${version_dir}")"
  focused_group_name="$(focused_animation_group_name_env)"
  focused_optimizers="$(focused_animation_optimizers_env)"
  focused_labels="$(focused_animation_labels_env)"
  write_visualization_settings "${version_dir}"

  echo "=== Visualization phase ==="
  echo "Visualization version directory: ${version_dir}"
  echo "Animation export suffix: ${export_suffix}"
  printf '%s\n' "${clear_answer}" | env INCLUDE_GAUSS_NEWTON="${include_gauss_newton}" VIS_CHANNEL="${visualization_channel}" HISTORY_SCALE="${visualization_history_scale}" FORECAST_ERROR_SCALE="${visualization_forecast_error_scale}" AUTOREG_STEPS="${visualization_autoreg_steps}" OUTPUT_FREQ="${visualization_output_freq}" ANIMATION_PACING="${visualization_animation_pacing}" VIS_NLAT="${resolution_nlat}" VIS_NLON="${resolution_nlon}" ENABLE_FOCUSED_ANIMATIONS="${enable_focused_animations}" FOCUSED_ANIMATION_GROUP_NAME="${focused_group_name}" FOCUSED_ANIMATION_OPTIMIZERS="${focused_optimizers}" FOCUSED_ANIMATION_LABELS="${focused_labels}" VIS_EXPORT_SUFFIX="${export_suffix}" FIGURES_HISTORY_ROOT="${version_dir}/figures_history" FIGURES_FORECAST_ROOT="${version_dir}/figures_forecast" FOCUSED_ANIMATION_ROOT="${version_dir}/focused_animations" ROLLOUT_ROOT="${version_dir}/rollout_results" bash visualize_all_optimizers.sh
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

animation_export_suffix() {
  local version_dir="$1"
  local version_name
  version_name="$(basename "${version_dir}")"

  if [[ "${version_name}" =~ ^version_([0-9]+)$ ]]; then
    printf '_v%s\n' "${BASH_REMATCH[1]}"
  else
    version_name="${version_name//[^A-Za-z0-9]/_}"
    printf '_%s\n' "${version_name}"
  fi
}

focused_animation_label_for_optimizer() {
  case "$1" in
    adam) printf 'Adam\n' ;;
    adamw) printf 'AdamW\n' ;;
    gauss_newton) printf 'Gauss-Newton\n' ;;
    mud) printf 'MUD\n' ;;
    muon) printf 'Muon\n' ;;
    sgd) printf 'SGD\n' ;;
    *)
      echo "Error: unknown focused optimizer '$1'. Use one of: adam adamw gauss_newton mud muon sgd." >&2
      exit 1
      ;;
  esac
}

focused_animation_group_name_env() {
  if [[ "${#focused_animation_optimizers[@]}" -eq 0 ]]; then
    echo "Error: focused_animation_optimizers must contain at least one optimizer." >&2
    exit 1
  fi
  printf 'focused_%s\n' "$(IFS=_; printf '%s' "${focused_animation_optimizers[*]}")"
}

focused_animation_optimizers_env() {
  local optimizer
  for optimizer in "${focused_animation_optimizers[@]}"; do
    focused_animation_label_for_optimizer "${optimizer}" >/dev/null
  done
  printf '%s\n' "${focused_animation_optimizers[*]}"
}

focused_animation_labels_env() {
  local optimizer
  local label
  local values=()

  for optimizer in "${focused_animation_optimizers[@]}"; do
    label="$(focused_animation_label_for_optimizer "${optimizer}")"
    values+=("${label}")
  done
  printf '%s\n' "${values[*]}"
}

write_visualization_settings() {
  local version_dir="$1"

  cat > "${version_dir}/settings.txt" <<EOF
run_training=${run_training}
run_visualization=${run_visualization}
clear_training_logs=${clear_training_logs}
include_gauss_newton=${include_gauss_newton}
training_pretrain_epochs=${training_pretrain_epochs}
training_gauss_newton_epochs=${training_gauss_newton_epochs}
resolution_nlat=${resolution_nlat}
resolution_nlon=${resolution_nlon}
overwrite_visualization_outputs=${overwrite_visualization_outputs}
visualization_channel=${visualization_channel}
visualization_history_scale=${visualization_history_scale}
visualization_forecast_error_scale=${visualization_forecast_error_scale}
visualization_autoreg_steps=${visualization_autoreg_steps}
visualization_output_freq=${visualization_output_freq}
visualization_animation_pacing=${visualization_animation_pacing}
enable_focused_animations=${enable_focused_animations}
focused_animation_optimizers=${focused_animation_optimizers[*]}
visualization_root=${visualization_root}
visualization_version=${visualization_version}
version_dir=${version_dir}
EOF
}

validate_bool "run_training" "${run_training}"
validate_bool "run_visualization" "${run_visualization}"
validate_bool "clear_training_logs" "${clear_training_logs}"
validate_bool "include_gauss_newton" "${include_gauss_newton}"
validate_bool "overwrite_visualization_outputs" "${overwrite_visualization_outputs}"
validate_bool "enable_focused_animations" "${enable_focused_animations}"
case "${visualization_animation_pacing}" in
  standard|smooth|slow) ;;
  *) echo "Error: visualization_animation_pacing must be standard, smooth, or slow." >&2; exit 1 ;;
esac

print_settings

case "${run_training}" in
  true) run_training_phase ;;
  false) echo "Skipping training phase." ;;
esac

case "${run_visualization}" in
  true) run_visualization_phase ;;
  false) echo "Skipping visualization phase." ;;
esac

echo "=== Cluster wrapper finished ==="
exit 0
