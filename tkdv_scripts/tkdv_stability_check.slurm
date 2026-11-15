#!/bin/bash -l
#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#SBATCH --job-name=tkdv_stability
#SBATCH --account=eccc_pegasus_mrd__gpu_a100
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/fs/hestia_Heccc/rpnatm/avg000/work/logs/%j.log
#SBATCH --comment="image=registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu22.04"

set -euo pipefail

REPO="/home/avg000/swan"
PY="/home/avg000/miniconda3/envs/swan/bin/python"
SAVE_DIR="/fs/hestia_Heccc/rpnatm/avg000/work/tkdv_stability"

LOGDIR="/fs/hestia_Heccc/rpnatm/avg000/work/logs"
mkdir -p "$LOGDIR"
mkdir -p "$SAVE_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOGDIR/tkdv_stability_${TS}.log"

echo "Host: $(hostname)"      | tee -a "$LOGFILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOGFILE"
echo "Time: $(date)"         | tee -a "$LOGFILE"

export PYTHONPATH="$REPO"
cd "$REPO"

# Compare dt_solver=1.0 (coarse) vs dt_solver_ref=0.001 (fine reference)
# 20 samples, all stability-checked, 40 steps, downstream case D=0.24
echo "=== Stability check: dt=1.0 vs dt=0.001 (downstream D=0.24) ===" | tee -a "$LOGFILE"
$PY -m kdv.tkdv_dataset_saver \
    --ictype gibbs \
    --J 32 \
    --dt 1.0 \
    --dt_solver 1.0 \
    --dt_solver_ref 0.001 \
    --D 0.24 \
    --E0 100.0 \
    --L0 6.0 \
    --theta -0.5 \
    --gibbs_burnin 1000 \
    --gibbs_thin 10 \
    --n_samples 20 \
    --n_steps_per_trajectory 40 \
    --n_stability_samples 20 \
    --n_stability_steps 40 \
    --stability_threshold 0.01 \
    --device cpu \
    --save_dir "$SAVE_DIR" \
    2>&1 | tee -a "$LOGFILE"

echo "DONE: $(date)" | tee -a "$LOGFILE"
exit 0
