#!/bin/bash -l
#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#SBATCH --job-name=gbls_h_multi_trial
#SBATCH --account=eccc_pegasus_mrd__gpu_a100
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=01:00:00
#SBATCH --output=/fs/hestia_Heccc/rpnatm/avg000/work/logs/%j.log
#SBATCH --comment="image=registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu22.04"

set -euo pipefail

REPO="/home/avg000/swan"
PY="/home/avg000/miniconda3/envs/swan/bin/python"
BASE_SAVEDIR="/space/hall0/work/eccc/mrd/rpnatm/avg000/Trained_weights_and_Graphs"

DATE="$(date +%Y%m%d)"
RUN_IDX=1
while [[ -d "$BASE_SAVEDIR/${DATE}_run${RUN_IDX}" ]]; do
  RUN_IDX=$((RUN_IDX + 1))
done
SAVEDIR="$BASE_SAVEDIR/${DATE}_run${RUN_IDX}"
mkdir -p "$SAVEDIR"

LOGDIR="/fs/hestia_Heccc/rpnatm/avg000/work/logs"
mkdir -p "$LOGDIR"

TRAIN_DIR="/fs/hestia_Heccc/rpnatm/avg000/work/datasets/train"
VAL_DIR="/fs/hestia_Heccc/rpnatm/avg000/work/datasets/val"

cd "$REPO"

TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOGDIR/gbls_h_multi_trial_${TS}.log"
GPU_LOGFILE="$LOGDIR/gbls_h_multi_trial_${TS}_gpu.log"
FAIL_SUMMARY="$LOGDIR/failures_gbls_h_multi_trial_${TS}.txt"
: > "$FAIL_SUMMARY"
: > "$GPU_LOGFILE"

echo "Host: $(hostname)"        | tee -a "$LOGFILE"
echo "Job ID: $SLURM_JOB_ID"   | tee -a "$LOGFILE"
echo "Time: $(date)"           | tee -a "$LOGFILE"
echo "GPU:"                    | tee -a "$LOGFILE"
nvidia-smi 2>&1                | tee -a "$LOGFILE" || true

$PY -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())" 2>&1 | tee -a "$LOGFILE"
$PY -c "import torch_harmonics; print('torch_harmonics OK')" 2>&1 | tee -a "$LOGFILE"

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

cp "$0" "$SAVEDIR/gbls_h_multi_trial_${TS}.slurm"
mkdir -p "$REPO/old_scripts/${DATE}"
cp "$0" "$REPO/old_scripts/${DATE}/gbls_h_multi_trial_${TS}.slurm"

export PYTHONPATH="$REPO"
srun --cpus-per-task=2 $PY Training/train.py \
  --config config_paradis.yaml \
  --training.save_dir "$SAVEDIR" \
  --experiment.name gbls_h_multi_trial_${TS} \
  --training.pretrain_epochs 2 \
  --training.finetune_epochs 0 \
  --training.learning_rate 0.00001 \
  --data.batch_size 8 \
  --data.dt 3600 \
  --data.dt_solver 60 \
  --model.paradis.hidden_dim 96 \
  --model.paradis.num_layers 8 \
  --model.paradis.num_encoder_layers 4 \
  --model.paradis.num_vels 24 \
  --model.paradis.diffusion_size 48 \
  --model.paradis.reaction_size 24 \
  --model.paradis.bias_channels 6 \
  --n_rollout_steps 8 \
  --input_step_idx 14 \
  --start_ckpt "/space/hall0/work/eccc/mrd/rpnatm/avg000/Trained_weights_and_Graphs/20260803_run1/swan_precomp_20260803_231540/version_0/checkpoints/pretrain-epoch=97-val_loss=0.0754.ckpt" \
  --train_ic_dict '{"precomputed_gbells_h": [2048, "'"$TRAIN_DIR"'/gbells_h_60.0_20260803"]}' \
  --val_ic_dict   '{"precomputed_gbells_h": [128,  "'"$VAL_DIR"'/gbells_h_60.0_20260803"]}' \
  --devices 8 \
  --data.num_workers 2 \
  --training.gradient_checkpointing true \
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
