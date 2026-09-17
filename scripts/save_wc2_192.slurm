#!/bin/bash -l
#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#SBATCH --job-name=save_wc2_192
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
LOGDIR="/fs/hestia_Heccc/rpnatm/avg000/work/logs"

mkdir -p "$TRAIN_DIR" "$LOGDIR"

cd "$REPO"

DATE="$(date +%Y%m%d)"
TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOGDIR/save_wc2_192_${TS}.log"

cp "$0" "$LOGDIR/save_wc2_192_${TS}.slurm"
mkdir -p "$REPO/old_scripts/${DATE}"
cp "$0" "$REPO/old_scripts/${DATE}/save_wc2_192_${TS}.slurm"

echo "Host: $(hostname)"      | tee -a "$LOGFILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOGFILE"
echo "Time: $(date)"         | tee -a "$LOGFILE"
nvidia-smi 2>&1              | tee -a "$LOGFILE" || true

COMMON="--dt 3600 --dt_solver 60 --dt_solver_ref 4 --nlat 192 --nlon 384 --n_steps_per_trajectory 40 --device cuda \
  --n_stability_samples 20 --n_stability_steps 15 --stability_threshold 0.01"

echo "" | tee -a "$LOGFILE"
echo "=== Train: williamson_case2 (128 samples) ===" | tee -a "$LOGFILE"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" $PY dataset/dataset_saver.py $COMMON \
  --ictype williamson_case2 \
  --n_samples 128 \
  --wc2_gh0_min 98450 --wc2_gh0_max 98460 \
  --wc2_u0_min 1.269 --wc2_u0_max 3.807 \
  --save_dir "$TRAIN_DIR" \
  2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "DONE: $(date)" | tee -a "$LOGFILE"
exit 0
