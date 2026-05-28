#!/bin/bash
set -euo pipefail

REPO="/home/ubuntu/swan"
PY="python3"
SAVEDIR="/home/ubuntu/debug_runs"
mkdir -p "$SAVEDIR"

LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"

cd "$REPO"

TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOGDIR/swan_muon_debug_${TS}.log"
GPU_LOGFILE="$LOGDIR/swan_muon_debug_${TS}_gpu.log"
FAIL_SUMMARY="$LOGDIR/failures_swan_muon_debug_${TS}.txt"
: > "$FAIL_SUMMARY"
: > "$GPU_LOGFILE"

echo "Host: $(hostname)"        | tee -a "$LOGFILE"
echo "PID: $$"                  | tee -a "$LOGFILE"
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

cp "$0" "$SAVEDIR/train_muon_debug_${TS}.sh"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" $PY Training/train_muon.py \
  --config config_paradis.yaml \
  --training.save_dir "$SAVEDIR" \
  --experiment.name swan_muon_debug_${TS} \
  --training.pretrain_epochs 2 \
  --training.finetune_epochs 0 \
  --training.learning_rate 0.00015 \
  --data.batch_size 4 \
  --data.dt_solver 15 \
  --model.paradis.hidden_dim 48 \
  --model.paradis.num_layers 8 \
  --model.paradis.num_encoder_layers 3 \
  --model.paradis.num_vels 12 \
  --model.paradis.diffusion_size 24 \
  --model.paradis.reaction_size 12 \
  --model.paradis.bias_channels 3 \
  --n_rollout_steps 1 \
  --input_step_idx 4 \
  --train_ic_dict '{"gbells_h": 8, "williamson_case2": 4, "williamson_case6": 4}' \
  --val_ic_dict '{"gbells_h": 4}' \
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
