#!/bin/bash
BASE="/space/hall0/work/eccc/mrd/rpnatm/avg000/Trained_weights_and_Graphs"
AMSE_DIR="$BASE/multi_6h_$(date +%Y%m%d)"
VAL_DIR_GBELLS="/fs/hestia_Heccc/rpnatm/avg000/work/datasets_6h_4/val"
VAL_DIR_WC6="/fs/hestia_Heccc/rpnatm/avg000/work/datasets_6h_2/val"
REPO="/home/avg000/swan"
PY="/home/avg000/miniconda3/envs/swan/bin/python"

declare -A CKPTS
CKPTS[multi_6h_h_wc6]="$BASE/20261005_run1/gbls_h+wc6_6h_multi_2_20261005_050957/version_0/checkpoints/pretrain-epoch=100-val_loss=0.4867.ckpt"

declare -A STATS
STATS[multi_6h_h_wc6]="$BASE/20261005_run1/stats.pt"

declare -A DATASETS
DATASETS[gbells_h]="$VAL_DIR_GBELLS/gbells_h_60.0_20261003"
DATASETS[wc6_matched]="$VAL_DIR_WC6/wc6/matched/williamson_case6_r4_60.0_20261001"

mkdir -p "$AMSE_DIR"

for RUN in multi_6h_h_wc6; do
  CKPT="${CKPTS[$RUN]}"
  STAT="${STATS[$RUN]}"
  if [[ "$CKPT" == "None" || "$STAT" == "None" ]]; then
    echo "Skipping $RUN (checkpoint not ready)"
    continue
  fi

  for DS in gbells_h wc6_matched; do
    FOLDER="${DATASETS[$DS]}"
    OUT="$AMSE_DIR/$RUN/forecasts/$DS/ic47"

    for CH in 0 1 2; do
      case $CH in
        0) CH_NAME="geopotential" ;;
        1) CH_NAME="vorticity" ;;
        2) CH_NAME="divergence" ;;
      esac

      sbatch --job-name="fc_${RUN}_${DS}_ch${CH}" \
        --account=eccc_pegasus_mrd__gpu_a100 \
        --partition=gpu_a100 \
        --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 \
        --mem=40G --time=00:10:00 \
        --output="/fs/hestia_Heccc/rpnatm/avg000/work/logs/fc_${RUN}_${DS}_ch${CH}_%j.log" \
        --comment="image=registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu22.04" \
        --wrap="cd $REPO && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$REPO $PY forecast.py \
          --config config_paradis.yaml \
          --checkpoint '$CKPT' \
          --stats_path '$STAT' \
          --ic_type precomputed \
          --precomputed_folder '$FOLDER' \
          --num_ics 1 \
          --ic_start_index 47 \
          --autoreg_steps 40 \
          --output_dir '$OUT/$CH_NAME' \
          --device cuda \
          --plot_channel $CH \
          --model.paradis.hidden_dim 96 \
          --model.paradis.num_layers 8 \
          --model.paradis.num_encoder_layers 4 \
          --model.paradis.num_vels 24 \
          --model.paradis.diffusion_size 48 \
          --model.paradis.reaction_size 24 \
          --model.paradis.bias_channels 6"
      echo "Submitted $RUN x $DS x $CH_NAME"
    done
  done
done
