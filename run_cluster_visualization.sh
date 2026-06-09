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

# clear_visualization_outputs=true answers "y" to the visualization cleanup
# prompt and clears ./figures_history, ./figures_forecast, and ./rollout_results.
clear_visualization_outputs=true

# Visualization settings passed to visualize_all_optimizers.sh for this wrapper
# run. Direct runs of visualize_all_optimizers.sh keep that file's defaults.
visualization_channel="vorticity"
visualization_history_scale="log"
visualization_forecast_error_scale="log"
visualization_autoreg_steps=100
visualization_output_freq=5

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
  echo "clear_visualization_outputs=${clear_visualization_outputs}"
  echo "visualization_channel=${visualization_channel}"
  echo "visualization_history_scale=${visualization_history_scale}"
  echo "visualization_forecast_error_scale=${visualization_forecast_error_scale}"
  echo "visualization_autoreg_steps=${visualization_autoreg_steps}"
  echo "visualization_output_freq=${visualization_output_freq}"
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
  printf '%s\n' "${clear_answer}" | env PRETRAIN_EPOCHS="${training_pretrain_epochs}" GAUSS_NEWTON_EPOCHS="${training_gauss_newton_epochs}" INCLUDE_GAUSS_NEWTON="${include_gauss_newton}" bash train_all_optimizers.sh
}

run_visualization_phase() {
  local clear_answer="n"

  case "${clear_visualization_outputs}" in
    true)
      clear_answer="y"
      echo "Visualization outputs will be cleared before plotting."
      ;;
    false)
      clear_answer="n"
      echo "Visualization outputs will be kept; new plots may overwrite files with the same names."
      ;;
  esac

  echo "=== Visualization phase ==="
  printf '%s\n' "${clear_answer}" | env INCLUDE_GAUSS_NEWTON="${include_gauss_newton}" VIS_CHANNEL="${visualization_channel}" HISTORY_SCALE="${visualization_history_scale}" FORECAST_ERROR_SCALE="${visualization_forecast_error_scale}" AUTOREG_STEPS="${visualization_autoreg_steps}" OUTPUT_FREQ="${visualization_output_freq}" bash visualize_all_optimizers.sh
}

validate_bool "run_training" "${run_training}"
validate_bool "run_visualization" "${run_visualization}"
validate_bool "clear_training_logs" "${clear_training_logs}"
validate_bool "include_gauss_newton" "${include_gauss_newton}"
validate_bool "clear_visualization_outputs" "${clear_visualization_outputs}"

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
