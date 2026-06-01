#!/bin/bash -l
#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#SBATCH --job-name=swan_save_data
#SBATCH --account=eccc_pegasus_mrd__gpu_a100
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=109G
#SBATCH --time=12:00:00
#SBATCH --output=/fs/hestia_Heccc/rpnatm/avg000/work/logs/%j.log
#SBATCH --comment="image=registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu22.04"

set -euo pipefail

REPO="/home/avg000/swan"
PY="/home/avg000/miniconda3/envs/swan/bin/python"
TRAIN_DIR="/fs/hestia_Heccc/rpnatm/avg000/work/datasets/train"
VAL_DIR="/fs/hestia_Heccc/rpnatm/avg000/work/datasets/val"
LOGDIR="/fs/hestia_Heccc/rpnatm/avg000/work/logs"

mkdir -p "$TRAIN_DIR" "$VAL_DIR" "$LOGDIR"

cd "$REPO"

TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOGDIR/save_datasets_${TS}.log"

cp "$0" "$LOGDIR/save_datasets_${TS}.slurm"
mkdir -p "$REPO/old_scripts/${DATE}"
cp "$0" "$REPO/old_scripts/${DATE}/save_datasets_${TS}.slurm"

echo "Host: $(hostname)"      | tee -a "$LOGFILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOGFILE"
echo "Time: $(date)"         | tee -a "$LOGFILE"
nvidia-smi 2>&1              | tee -a "$LOGFILE" || true

COMMON="--dt 900 --dt_solver 15 --nlat 128 --nlon 256 --n_steps_per_trajectory 10 --device cuda \
  --n_stability_samples 20 --n_stability_steps 8 --stability_threshold 0.01"

# ── TRAINING DATA ──────────────────────────────────────────────────────────────

echo "" | tee -a "$LOGFILE"
echo "=== Train: gbells_h (1024 samples, ref=random) ===" | tee -a "$LOGFILE"
PYTHONPATH="$REPO" $PY dataset/dataset_saver.py $COMMON \
  --ictype gbells_h \
  --n_samples 1024 \
  --gbells_ref_ictype random \
  --save_dir "$TRAIN_DIR" \
  2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Train: williamson_case2 (128 samples) ===" | tee -a "$LOGFILE"
PYTHONPATH="$REPO" $PY dataset/dataset_saver.py $COMMON \
  --ictype williamson_case2 \
  --n_samples 128 \
  --save_dir "$TRAIN_DIR" \
  2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Train: williamson_case6 (128 samples) ===" | tee -a "$LOGFILE"
PYTHONPATH="$REPO" $PY dataset/dataset_saver.py $COMMON \
  --ictype williamson_case6 \
  --n_samples 128 \
  --save_dir "$TRAIN_DIR" \
  2>&1 | tee -a "$LOGFILE"

# ── VALIDATION DATA ────────────────────────────────────────────────────────────
# Saved to a separate directory — different random draws from training data

echo "" | tee -a "$LOGFILE"
echo "=== Val: gbells_h (128 samples, ref=random) ===" | tee -a "$LOGFILE"
PYTHONPATH="$REPO" $PY dataset/dataset_saver.py $COMMON \
  --ictype gbells_h \
  --n_samples 128 \
  --gbells_ref_ictype random \
  --save_dir "$VAL_DIR" \
  2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "DONE: $(date)" | tee -a "$LOGFILE"
exit 0
