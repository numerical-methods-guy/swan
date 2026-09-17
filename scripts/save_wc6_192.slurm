#!/bin/bash -l
#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#SBATCH --job-name=swan_save_wc6_192
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
TRAIN_DIR="/fs/hestia_Heccc/rpnatm/avg000/work/datasets_192/train"
VAL_DIR="/fs/hestia_Heccc/rpnatm/avg000/work/datasets_192/val"
LOGDIR="/fs/hestia_Heccc/rpnatm/avg000/work/logs"

mkdir -p "$TRAIN_DIR" "$VAL_DIR" "$LOGDIR"

cd "$REPO"

DATE="$(date +%Y%m%d)"
TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOGDIR/save_wc6_192_${TS}.log"

cp "$0" "$LOGDIR/save_wc6_192_${TS}.slurm"
mkdir -p "$REPO/old_scripts/${DATE}"
cp "$0" "$REPO/old_scripts/${DATE}/save_wc6_192_${TS}.slurm"

echo "Host: $(hostname)"      | tee -a "$LOGFILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOGFILE"
echo "Time: $(date)"         | tee -a "$LOGFILE"
nvidia-smi 2>&1              | tee -a "$LOGFILE" || true

COMMON="--dt 3600 --dt_solver 60 --dt_solver_ref 4 --nlat 192 --nlon 384 --n_steps_per_trajectory 40 --device cuda \
  --n_stability_samples 0 --n_stability_steps 0 --stability_threshold 0.001"

# ── TRAINING DATA ──────────────────────────────────────────────────────────────

echo "" | tee -a "$LOGFILE"
echo "=== Train: williamson_case6_r4 matched ranges (512 samples) ===" | tee -a "$LOGFILE"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" $PY dataset/dataset_saver.py $COMMON \
  --ictype williamson_case6_r4 \
  --n_samples 512 \
  --wc6_omega_min 3.0e-7 --wc6_omega_max 3.5e-7 \
  --wc6_h0_min 9700 --wc6_h0_max 10200 \
  --save_dir "$TRAIN_DIR/wc6/matched" \
  2>&1 | tee -a "$LOGFILE"

# ── VALIDATION DATA ────────────────────────────────────────────────────────────

echo "" | tee -a "$LOGFILE"
echo "=== Val: williamson_case6_r4 matched ranges (64 samples) ===" | tee -a "$LOGFILE"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" $PY dataset/dataset_saver.py $COMMON \
  --ictype williamson_case6_r4 \
  --n_samples 64 \
  --wc6_omega_min 3.0e-7 --wc6_omega_max 3.5e-7 \
  --wc6_h0_min 9700 --wc6_h0_max 10200 \
  --save_dir "$VAL_DIR/wc6/matched" \
  2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Train: williamson_case6_r4 rv_matched ranges (512 samples) ===" | tee -a "$LOGFILE"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" $PY dataset/dataset_saver.py $COMMON \
  --ictype williamson_case6_r4 \
  --n_samples 512 \
  --wc6_omega_min 4.282e-7 --wc6_omega_max 6.423e-7 \
  --wc6_h0_min 9622 --wc6_h0_max 10217 \
  --save_dir "$TRAIN_DIR/wc6/rv_matched" \
  2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Val: williamson_case6_r4 rv_matched ranges (64 samples) ===" | tee -a "$LOGFILE"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" $PY dataset/dataset_saver.py $COMMON \
  --ictype williamson_case6_r4 \
  --n_samples 64 \
  --wc6_omega_min 4.282e-7 --wc6_omega_max 6.423e-7 \
  --wc6_h0_min 9622 --wc6_h0_max 10217 \
  --save_dir "$VAL_DIR/wc6/rv_matched" \
  2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "DONE: $(date)" | tee -a "$LOGFILE"
exit 0
