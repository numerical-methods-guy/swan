#!/bin/bash -l
#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#SBATCH --job-name=swan_muon
#SBATCH --account=eccc_pegasus_mrd__gpu_a100
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=109G
#SBATCH --time=12:00:00
#SBATCH --output=/home/avg000/swan/slurm_logs/%j.log
#SBATCH --comment="image=registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu22.04"

set -euo pipefail

export PYTHONPATH="/fs/hestia_Heccc/rpnatm/avg000/work/site-packages:${PYTHONPATH:-}"

REPO="/home/avg000/swan"
PY="/home/avg000/miniconda3/envs/swan/bin/python"
BASE_SAVEDIR="/space/hall0/work/eccc/mrd/rpnatm/avg000/Trained_weights_and_Graphs"

# Create a dated, auto-numbered run folder
DATE="$(date +%Y%m%d)"
RUN_IDX=1
while [[ -d "$BASE_SAVEDIR/${DATE}_run${RUN_IDX}" ]]; do
  RUN_IDX=$((RUN_IDX + 1))
done
SAVEDIR="$BASE_SAVEDIR/${DATE}_run${RUN_IDX}"
mkdir -p "$SAVEDIR"

LOGDIR="$REPO/slurm_logs"
mkdir -p "$LOGDIR"

cd "$REPO"

TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOGDIR/swan_muon_${TS}.log"
GPU_LOGFILE="$LOGDIR/swan_muon_${TS}_gpu.log"
FAIL_SUMMARY="$LOGDIR/failures_swan_muon_${TS}.txt"
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

cp "$0" "$SAVEDIR/train_muon_${TS}.slurm"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" $PY Training/train_muon.py \
  --config config_paradis.yaml \
  --training.save_dir "$SAVEDIR" \
  --experiment.name swan_muon_${TS} \
  --training.pretrain_epochs 100 \
  --training.finetune_epochs 0 \
  --training.learning_rate 0.0009 \
  --data.batch_size 32 \
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
  --train_ic_dict '{"gbells_h": [768, {"gbells_ref_ictype": "williamson_case2"}], "williamson_case2": 128, "williamson_case6": 128}' \
  --val_ic_dict '{"gbells_h": [128, {"gbells_ref_ictype": "williamson_case2"}]}' \
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
