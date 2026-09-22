#!/bin/bash
BASE="/space/hall0/work/eccc/mrd/rpnatm/avg000/Trained_weights_and_Graphs"
AMSE_DIR="$BASE/amse_$(date +%Y%m%d)_2"
VAL_DIR="/fs/hestia_Heccc/rpnatm/avg000/work/datasets/val"
REPO="/home/avg000/swan"
PY="/home/avg000/miniconda3/envs/swan/bin/python"

# AMSE-trained checkpoints — set to "None" until a run completes
declare -A CKPTS
CKPTS[amse_h_wc6]="$BASE/20260921_run8/gbls_h+wc6_multi_amse_20260921_202608/version_0/checkpoints/pretrain-epoch=99-val_loss=0.0069.ckpt"
CKPTS[amse_h]="$BASE/20260922_run1/gbls_h_multi_amse_20260922_045509/version_0/checkpoints/pretrain-epoch=99-val_loss=0.0991.ckpt"
CKPTS[amse_rv]="None"
CKPTS[amse_rv_wc6]="None"

declare -A STATS
STATS[amse_h_wc6]="$BASE/20260921_run8/stats.pt"
STATS[amse_h]="$BASE/20260922_run1/stats.pt"
STATS[amse_rv]="None"
STATS[amse_rv_wc6]="None"

declare -A DATASETS
DATASETS[gbells_h]="$VAL_DIR/gbells_h_60.0_20260803"
DATASETS[gbells_h_rv]="$VAL_DIR/ref_random/gbells_h_rv_60.0_20260804"
DATASETS[wc6_matched]="$VAL_DIR/wc6/matched/williamson_case6_r4_60.0_20260804"
DATASETS[wc6_rv_matched]="$VAL_DIR/wc6/rv_matched/williamson_case6_r4_60.0_20260804"
DATASETS[wc2]="/fs/hestia_Heccc/rpnatm/avg000/work/datasets/train/wc2_alpha0_forecast/williamson_case2_60.0_20260814"

declare -A IC_IDX
IC_IDX[amse_h_wc6_gbells_h]=47
IC_IDX[amse_h_wc6_gbells_h_rv]=47
IC_IDX[amse_h_wc6_wc6_matched]=47
IC_IDX[amse_h_wc6_wc6_rv_matched]=47
IC_IDX[amse_h_wc6_wc2]=0
IC_IDX[amse_h_gbells_h]=47
IC_IDX[amse_h_gbells_h_rv]=47
IC_IDX[amse_h_wc6_matched]=47
IC_IDX[amse_h_wc6_rv_matched]=47
IC_IDX[amse_h_wc2]=0
IC_IDX[amse_rv_gbells_h]=47
IC_IDX[amse_rv_gbells_h_rv]=47
IC_IDX[amse_rv_wc6_matched]=47
IC_IDX[amse_rv_wc6_rv_matched]=47
IC_IDX[amse_rv_wc2]=0
IC_IDX[amse_rv_wc6_gbells_h]=47
IC_IDX[amse_rv_wc6_gbells_h_rv]=47
IC_IDX[amse_rv_wc6_wc6_matched]=47
IC_IDX[amse_rv_wc6_wc6_rv_matched]=47
IC_IDX[amse_rv_wc6_wc2]=0

mkdir -p "$AMSE_DIR"

for RUN in amse_h_wc6 amse_h amse_rv amse_rv_wc6; do
  CKPT="${CKPTS[$RUN]}"
  STAT="${STATS[$RUN]}"
  if [[ "$CKPT" == "None" || "$STAT" == "None" ]]; then
    echo "Skipping $RUN (checkpoint not ready)"
    continue
  fi

  for DS in gbells_h gbells_h_rv wc6_matched wc6_rv_matched wc2; do
    IC="${IC_IDX[${RUN}_${DS}]}"
    FOLDER="${DATASETS[$DS]}"
    OUT="$AMSE_DIR/$RUN/forecasts/$DS/ic${IC}"

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
          --ic_start_index $IC \
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
