#!/bin/bash -l
#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#SBATCH --job-name=tkdv_long_traj
#SBATCH --account=eccc_pegasus_mrd__gpu_a100
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/fs/hestia_Heccc/rpnatm/avg000/work/logs/%j.log
#SBATCH --comment="image=registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu22.04"

set -euo pipefail

REPO="/home/avg000/swan"
PY="/home/avg000/miniconda3/envs/swan/bin/python"

LOGDIR="/fs/hestia_Heccc/rpnatm/avg000/work/logs"
mkdir -p "$LOGDIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOGDIR/tkdv_long_traj_${TS}.log"

echo "Host: $(hostname)"      | tee -a "$LOGFILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOGFILE"
echo "Time: $(date)"         | tee -a "$LOGFILE"

export PYTHONPATH="$REPO"
cd "$REPO"

$PY -u tkdv_scripts/plot_long_trajectory.py 2>&1 | tee -a "$LOGFILE"

echo "DONE: $(date)" | tee -a "$LOGFILE"
