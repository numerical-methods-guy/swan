#!/bin/bash -l
#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#SBATCH --job-name=swan_debug
#SBATCH --account=eccc_pegasus_mrd__gpu_a100
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=109G
#SBATCH --time=00:30:00
#SBATCH --output=/fs/hestia_Heccc/rpnatm/avg000/work/logs/%j.log
#SBATCH --comment="image=registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu22.04"

set -euo pipefail

REPO="/home/avg000/swan"
PY="/home/avg000/miniconda3/envs/swan/bin/python"
SAVEDIR="/space/hall0/work/eccc/mrd/rpnatm/avg000/test_runs_for_debug"
DATE="$(date +%Y%m%d)"
mkdir -p "$SAVEDIR"

LOGDIR="/fs/hestia_Heccc/rpnatm/avg000/work/logs"
mkdir -p "$LOGDIR"

cd "$REPO"

TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOGDIR/swan_debug_${TS}.log"
GPU_LOGFILE="$LOGDIR/swan_debug_${TS}_gpu.log"
FAIL_SUMMARY="$LOGDIR/failures_swan_debug_${TS}.txt"
: > "$FAIL_SUMMARY"
: > "$GPU_LOGFILE"

echo "Host: $(hostname)"        | tee -a "$LOGFILE"
echo "Job ID: $SLURM_JOB_ID"   | tee -a "$LOGFILE"
echo "Time: $(date)"           | tee -a "$LOGFILE"
echo "GPU:"                    | tee -a "$LOGFILE"
nvidia-smi 2>&1                | tee -a "$LOGFILE" || true

$PY -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())" 2>&1 | tee -a "$LOGFILE"
$PY -c "import torch_harmonics; print('torch_harmonics OK')" 2>&1 | tee -a "$LOGFILE"

# GPU monitor: logs every 30s in background
log_gpu_usage() {
  while true; do
    echo "=== GPU SNAPSHOT: $(date) ===" >> "$GPU_LOGFILE"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory \
      --format=csv,noheader,nounits >> "$GPU_LOGFILE" 2>/dev/null || true
    sleep 30
  done
}
log_gpu_usage &
GPU_MONITOR_PID=$!
echo "GPU monitor PID: $GPU_MONITOR_PID  log: $GPU_LOGFILE" | tee -a "$LOGFILE"

mkdir -p "$REPO/old_scripts/${DATE}"
cp "$0" "$SAVEDIR/train_debug_${TS}.slurm"
cp "$0" "$REPO/old_scripts/${DATE}/train_debug_${TS}.slurm"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" $PY Training/train.py \
  --config config_paradis.yaml \
  --training.save_dir "$SAVEDIR" \
  --experiment.name swan_debug_${TS} \
  --training.pretrain_epochs 2 \
  --training.finetune_epochs 0 \
  --training.learning_rate 0.00001 \
  --data.batch_size 4 \
  --data.dt_solver 15 \
  --model.paradis.hidden_dim 4 \
  --model.paradis.num_layers 2 \
  --model.paradis.num_encoder_layers 1 \
  --model.paradis.num_vels 2 \
  --model.paradis.diffusion_size 4 \
  --model.paradis.reaction_size 4 \
  --model.paradis.bias_channels 1 \
  --n_rollout_steps 1 \
  --input_step_idx 4 \
  --train_ic_dict '{"gbells_h": [8, {"gbells_ref_ictype": "williamson_case2"}], "williamson_case2": 4, "williamson_case6": 4}' \
  --val_ic_dict '{"gbells_h": [4, {"gbells_ref_ictype": "williamson_case2"}]}' \
  2>&1 | tee -a "$LOGFILE"

RC=$?
kill "$GPU_MONITOR_PID" 2>/dev/null || true
wait "$GPU_MONITOR_PID" 2>/dev/null || true

if [[ $RC -ne 0 ]]; then
  echo "[FAIL] $(date) rc=$RC" | tee -a "$LOGFILE" >> "$FAIL_SUMMARY"
  echo "Training failed but job exiting cleanly." | tee -a "$LOGFILE"
else
  echo "[OK] $(date) rc=0" | tee -a "$LOGFILE"
fi

echo "DONE: $(date)" | tee -a "$LOGFILE"
exit 0
