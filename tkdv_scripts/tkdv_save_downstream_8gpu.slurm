#!/bin/bash -l
#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#SBATCH --job-name=tkdv_save_downstream
#SBATCH --account=eccc_pegasus_mrd__gpu_a100
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=109G
#SBATCH --time=06:00:00
#SBATCH --array=0-7
#SBATCH --output=/fs/hestia_Heccc/rpnatm/avg000/work/logs/%j_%a.log
#SBATCH --comment="image=registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu22.04"

set -euo pipefail

REPO="/home/avg000/swan"
PY="/home/avg000/miniconda3/envs/swan/bin/python"
SAVE_DIR="/fs/hestia_Heccc/rpnatm/avg000/work/tkdv_datasets/downstream/seed_${SLURM_ARRAY_TASK_ID}"
LOGDIR="/fs/hestia_Heccc/rpnatm/avg000/work/logs"

mkdir -p "$SAVE_DIR" "$LOGDIR"

DATE="$(date +%Y%m%d)"
TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOGDIR/tkdv_save_downstream_seed${SLURM_ARRAY_TASK_ID}_${TS}.log"

cp "$0" "$LOGDIR/tkdv_save_downstream_seed${SLURM_ARRAY_TASK_ID}_${TS}.slurm"
mkdir -p "$REPO/old_scripts/${DATE}"
cp "$0" "$REPO/old_scripts/${DATE}/tkdv_save_downstream_seed${SLURM_ARRAY_TASK_ID}_${TS}.slurm"

echo "Host: $(hostname)"                        | tee -a "$LOGFILE"
echo "Job ID: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID" | tee -a "$LOGFILE"
echo "Time: $(date)"                            | tee -a "$LOGFILE"
nvidia-smi 2>&1                                 | tee -a "$LOGFILE" || true

COMMON="--ictype gibbs --J 32 --dt 1.0 --dt_solver 0.01 --dt_solver_ref 0.01 \
  --D 0.24 --E0 100.0 --L0 6.0 --theta -0.5 \
  --gibbs_burnin 2500 --gibbs_thin 10 \
  --n_steps_per_trajectory 40 \
  --n_stability_samples 5 --n_stability_steps 5 --stability_threshold 0.01 \
  --device cuda"

export PYTHONPATH="$REPO"
cd "$REPO"

echo "" | tee -a "$LOGFILE"
echo "=== Downstream (D=0.24, theta=-0.5): seed=${SLURM_ARRAY_TASK_ID}, 256 samples ===" | tee -a "$LOGFILE"
$PY -m kdv.tkdv_dataset_saver $COMMON \
  --n_samples 256 \
  --seed "${SLURM_ARRAY_TASK_ID}" \
  --save_dir "$SAVE_DIR" \
  2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "DONE: $(date)" | tee -a "$LOGFILE"
exit 0
