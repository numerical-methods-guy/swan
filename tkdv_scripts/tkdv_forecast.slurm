#!/bin/bash -l
#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#SBATCH --job-name=tkdv_forecast
#SBATCH --account=eccc_pegasus_mrd__gpu_a100
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/fs/hestia_Heccc/rpnatm/avg000/work/logs/%j.log
#SBATCH --comment="image=registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu22.04"

set -euo pipefail

REPO="/home/avg000/swan"
PY="/home/avg000/miniconda3/envs/swan/bin/python"

CKPT="/fs/hestia_Heccc/rpnatm/avg000/work/tkdv_datasets/upstream/train_weights/20261116_run2/tkdv_train_upstream_20261116_070010/version_0/checkpoints/pretrain-epoch=883-val_loss=0.4356.ckpt"
STATS="/fs/hestia_Heccc/rpnatm/avg000/work/tkdv_datasets/upstream/train_weights/20261116_run2/stats.pt"
IC_FOLDER="/fs/hestia_Heccc/rpnatm/avg000/work/tkdv_datasets/samples/seed_7/tkdv_gibbs_dt1.0_dtsolver0.01_20261115"
OUT_DIR="/fs/hestia_Heccc/rpnatm/avg000/work/tkdv_forecast_results"

LOGDIR="/fs/hestia_Heccc/rpnatm/avg000/work/logs"
mkdir -p "$LOGDIR" "$OUT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOGDIR/tkdv_forecast_${TS}.log"

echo "Host: $(hostname)"      | tee -a "$LOGFILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOGFILE"
echo "Time: $(date)"         | tee -a "$LOGFILE"
nvidia-smi 2>&1              | tee -a "$LOGFILE" || true

export PYTHONPATH="$REPO"
cd "$REPO"

$PY tkdv_forecast.py \
  --config config_tkdv.yaml \
  --checkpoint "$CKPT" \
  --stats_path "$STATS" \
  --ic_folder  "$IC_FOLDER" \
  --output_dir "$OUT_DIR" \
  --autoreg_steps 40 \
  --num_ics 8 \
  --device cuda \
  2>&1 | tee -a "$LOGFILE"

echo "DONE: $(date)" | tee -a "$LOGFILE"
exit 0
