#!/bin/bash -l
#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#SBATCH --job-name=tkdv_save_dataset
#SBATCH --account=eccc_pegasus_mrd__gpu_a100
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/fs/hestia_Heccc/rpnatm/avg000/work/logs/%j.log
#SBATCH --comment="image=registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu22.04"

set -euo pipefail

REPO="/home/avg000/swan"
PY="/home/avg000/miniconda3/envs/swan/bin/python"
SAVE_DIR="/fs/hestia_Heccc/rpnatm/avg000/work/tkdv_datasets"

LOGDIR="/fs/hestia_Heccc/rpnatm/avg000/work/logs"
mkdir -p "$LOGDIR"
mkdir -p "$SAVE_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOGDIR/tkdv_save_dataset_${TS}.log"

echo "Host: $(hostname)"      | tee -a "$LOGFILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOGFILE"
echo "Time: $(date)"         | tee -a "$LOGFILE"

$PY -c "import torch; print('torch', torch.__version__)" 2>&1 | tee -a "$LOGFILE"

export PYTHONPATH="$REPO"
cd "$REPO"

# --- Upstream case: D=1.0, T=200 (Qi & Majda 2019, Fig 1B) ---
echo "" | tee -a "$LOGFILE"
echo "=== Upstream case (D=1.0, n_steps=200) ===" | tee -a "$LOGFILE"
$PY -m kdv.tkdv_dataset_saver \
    --ictype gibbs \
    --J 32 \
    --dt 0.01 \
    --dt_solver 0.01 \
    --dt_solver_ref 0.001 \
    --D 1.0 \
    --E0 100.0 \
    --L0 6.0 \
    --theta -0.1 \
    --gibbs_burnin 1000 \
    --gibbs_thin 10 \
    --n_samples 10000 \
    --n_steps_per_trajectory 200 \
    --n_stability_samples 10 \
    --n_stability_steps 5 \
    --stability_threshold 0.01 \
    --device cpu \
    --save_dir "$SAVE_DIR" \
    2>&1 | tee -a "$LOGFILE"

# --- Downstream case: D=0.24, T=50 (Qi & Majda 2019, Fig 1A) ---
echo "" | tee -a "$LOGFILE"
echo "=== Downstream case (D=0.24, n_steps=50) ===" | tee -a "$LOGFILE"
$PY -m kdv.tkdv_dataset_saver \
    --ictype gibbs \
    --J 32 \
    --dt 0.01 \
    --dt_solver 0.01 \
    --dt_solver_ref 0.001 \
    --D 0.24 \
    --E0 100.0 \
    --L0 6.0 \
    --theta -0.1 \
    --gibbs_burnin 1000 \
    --gibbs_thin 10 \
    --n_samples 10000 \
    --n_steps_per_trajectory 50 \
    --n_stability_samples 10 \
    --n_stability_steps 5 \
    --stability_threshold 0.01 \
    --device cpu \
    --save_dir "$SAVE_DIR" \
    2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "DONE: $(date)" | tee -a "$LOGFILE"
exit 0
