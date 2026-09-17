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
TRAIN_DIR="/fs/hestia_Heccc/rpnatm/avg000/work/datasets/train/wc6"
VAL_DIR="/fs/hestia_Heccc/rpnatm/avg000/work/datasets/val/wc6"
LOGDIR="/fs/hestia_Heccc/rpnatm/avg000/work/logs"

mkdir -p "$TRAIN_DIR" "$VAL_DIR" "$LOGDIR"

cd "$REPO"

DATE="$(date +%Y%m%d)"
TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOGDIR/save_datasets_${TS}.log"

cp "$0" "$LOGDIR/save_datasets_${TS}.slurm"
mkdir -p "$REPO/old_scripts/${DATE}"
cp "$0" "$REPO/old_scripts/${DATE}/save_datasets_${TS}.slurm"

echo "Host: $(hostname)"      | tee -a "$LOGFILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOGFILE"
echo "Time: $(date)"         | tee -a "$LOGFILE"
nvidia-smi 2>&1              | tee -a "$LOGFILE" || true

COMMON="--dt 3600 --dt_solver 60 --dt_solver_ref 4 --nlat 128 --nlon 256 --n_steps_per_trajectory 40 --device cuda \
  --n_stability_samples 0 --n_stability_steps 0 --stability_threshold 0.001"

# ── TRAINING DATA ──────────────────────────────────────────────────────────────

#echo "" | tee -a "$LOGFILE"
#echo "=== Train: gbells_h_rv (2048 samples, ref=random) ===" | tee -a "$LOGFILE"
#CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" $PY dataset/dataset_saver.py $COMMON \
#  --ictype gbells_h_rv \
#  --n_samples 2048 \
#  --gbells_ref_ictype random \
#  --gbells_ref_n_samples 30 \
#  --save_dir "$TRAIN_DIR" \
#  2>&1 | tee -a "$LOGFILE"

#echo "" | tee -a "$LOGFILE"
#echo "=== Train: williamson_case6_standard (128 samples) ===" | tee -a "$LOGFILE"
#CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" $PY dataset/dataset_saver.py $COMMON \
#  --ictype williamson_case6_standard \
#  --n_samples 128 \
#  --save_dir "$TRAIN_DIR/standard" \
#  2>&1 | tee -a "$LOGFILE"

#echo "" | tee -a "$LOGFILE"
#echo "=== Train: williamson_case6_r4 default ranges (512 samples) ===" | tee -a "$LOGFILE"
#CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" $PY dataset/dataset_saver.py $COMMON \
#  --ictype williamson_case6_r4 \
#  --n_samples 512 \
#  --save_dir "$TRAIN_DIR/default" \
#  2>&1 | tee -a "$LOGFILE"

#echo "" | tee -a "$LOGFILE"
#echo "=== Train: williamson_case6_r4 matched ranges (512 samples) ===" | tee -a "$LOGFILE"
#CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" $PY dataset/dataset_saver.py $COMMON \
#  --ictype williamson_case6_r4 \
#  --n_samples 512 \
#  --wc6_omega_min 3.0e-7 --wc6_omega_max 3.5e-7 \
#  --wc6_h0_min 9700 --wc6_h0_max 10200 \
#  --save_dir "$TRAIN_DIR/matched" \
#  2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Train: williamson_case6_r4 rv_matched ranges (512 samples) ===" | tee -a "$LOGFILE"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" $PY dataset/dataset_saver.py $COMMON \
  --ictype williamson_case6_r4 \
  --n_samples 512 \
  --wc6_omega_min 4.282e-7 --wc6_omega_max 6.423e-7 \
  --wc6_h0_min 9622 --wc6_h0_max 10217 \
  --save_dir "$TRAIN_DIR/rv_matched" \
  2>&1 | tee -a "$LOGFILE"

# ── VALIDATION DATA ────────────────────────────────────────────────────────────

#echo "" | tee -a "$LOGFILE"
#echo "=== Val: gbells_h_rv (128 samples, ref=random) ===" | tee -a "$LOGFILE"
#CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" $PY dataset/dataset_saver.py $COMMON \
#  --ictype gbells_h_rv \
#  --n_samples 128 \
#  --gbells_ref_ictype random \
#  --gbells_ref_n_samples 30 \
#  --save_dir "$VAL_DIR" \
#  2>&1 | tee -a "$LOGFILE"

#echo "" | tee -a "$LOGFILE"
#echo "=== Val: williamson_case6_standard (32 samples) ===" | tee -a "$LOGFILE"
#CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" $PY dataset/dataset_saver.py $COMMON \
#  --ictype williamson_case6_standard \
#  --n_samples 32 \
#  --save_dir "$VAL_DIR/standard" \
#  2>&1 | tee -a "$LOGFILE"

#echo "" | tee -a "$LOGFILE"
#echo "=== Val: williamson_case6_r4 default ranges (64 samples) ===" | tee -a "$LOGFILE"
#CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" $PY dataset/dataset_saver.py $COMMON \
#  --ictype williamson_case6_r4 \
#  --n_samples 64 \
#  --save_dir "$VAL_DIR/default" \
#  2>&1 | tee -a "$LOGFILE"

#echo "" | tee -a "$LOGFILE"
#echo "=== Val: williamson_case6_r4 matched ranges (64 samples) ===" | tee -a "$LOGFILE"
#CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" $PY dataset/dataset_saver.py $COMMON \
#  --ictype williamson_case6_r4 \
#  --n_samples 64 \
#  --wc6_omega_min 3.0e-7 --wc6_omega_max 3.5e-7 \
#  --wc6_h0_min 9700 --wc6_h0_max 10200 \
#  --save_dir "$VAL_DIR/matched" \
#  2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Val: williamson_case6_r4 rv_matched ranges (64 samples) ===" | tee -a "$LOGFILE"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" $PY dataset/dataset_saver.py $COMMON \
  --ictype williamson_case6_r4 \
  --n_samples 64 \
  --wc6_omega_min 4.282e-7 --wc6_omega_max 6.423e-7 \
  --wc6_h0_min 9622 --wc6_h0_max 10217 \
  --save_dir "$VAL_DIR/rv_matched" \
  2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "DONE: $(date)" | tee -a "$LOGFILE"
exit 0
