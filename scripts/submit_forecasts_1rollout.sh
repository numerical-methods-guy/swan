#!/bin/bash
BASE="/space/hall0/work/eccc/mrd/rpnatm/avg000/Trained_weights_and_Graphs"
VAL_DIR="/fs/hestia_Heccc/rpnatm/avg000/work/datasets/val"
REPO="/home/avg000/swan"
PY="/home/avg000/miniconda3/envs/swan/bin/python"

declare -A CKPTS
CKPTS[run1]="$BASE/20260803_run1/swan_precomp_20260803_231540/version_0/checkpoints/pretrain-epoch=97-val_loss=0.0754.ckpt"
CKPTS[run2]="$BASE/20260804_run2/swan_precomp_20260804_052318/version_0/checkpoints/pretrain-epoch=96-val_loss=0.0837.ckpt"
CKPTS[run3]="$BASE/20260804_run3/swan_precomp_20260804_053322/version_0/checkpoints/pretrain-epoch=88-val_loss=0.0773.ckpt"
CKPTS[run4]="$BASE/20260804_run4/swan_precomp_20260804_053408/version_0/checkpoints/pretrain-epoch=99-val_loss=0.0694.ckpt"

declare -A STATS
STATS[run1]="$BASE/20260803_run1/stats.pt"
STATS[run2]="$BASE/20260804_run2/stats.pt"
STATS[run3]="$BASE/20260804_run3/stats.pt"
STATS[run4]="$BASE/20260804_run4/stats.pt"

declare -A DATASETS
DATASETS[gbells_h]="$VAL_DIR/gbells_h_60.0_20260803"
DATASETS[gbells_h_rv]="$VAL_DIR/ref_random/gbells_h_rv_60.0_20260804"
DATASETS[wc6_matched]="$VAL_DIR/wc6/matched/williamson_case6_r4_60.0_20260804"
DATASETS[wc6_rv_matched]="$VAL_DIR/wc6/rv_matched/williamson_case6_r4_60.0_20260804"

declare -A IC_IDX
IC_IDX[run1_gbells_h]=12
IC_IDX[run1_gbells_h_rv]=12
IC_IDX[run1_wc6_matched]=12
IC_IDX[run1_wc6_rv_matched]=12
IC_IDX[run2_gbells_h]=12
IC_IDX[run2_gbells_h_rv]=12
IC_IDX[run2_wc6_matched]=12
IC_IDX[run2_wc6_rv_matched]=12
IC_IDX[run3_gbells_h]=12
IC_IDX[run3_gbells_h_rv]=12
IC_IDX[run3_wc6_matched]=12
IC_IDX[run3_wc6_rv_matched]=12
IC_IDX[run4_gbells_h]=12
IC_IDX[run4_gbells_h_rv]=12
IC_IDX[run4_wc6_matched]=12
IC_IDX[run4_wc6_rv_matched]=12

for RUN in run1 run2 run3 run4; do
  for DS in gbells_h gbells_h_rv wc6_matched wc6_rv_matched; do
    sbatch --job-name="fc_${RUN}_${DS}" \
      --account=eccc_pegasus_mrd__gpu_a100 \
      --partition=gpu_a100 \
      --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 \
      --mem=40G --time=01:00:00 \
      --output="/fs/hestia_Heccc/rpnatm/avg000/work/logs/fc_${RUN}_${DS}_%j.log" \
      --comment="image=registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu22.04" \
      --wrap="cd $REPO && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$REPO $PY forecast.py \
        --config config_paradis.yaml \
        --checkpoint '${CKPTS[$RUN]}' \
        --stats_path '${STATS[$RUN]}' \
        --ic_type precomputed \
        --precomputed_folder '${DATASETS[$DS]}' \
        --num_ics 1 \
        --ic_start_index ${IC_IDX[${RUN}_${DS}]} \
        --autoreg_steps 40 \
        --output_dir '$BASE/${RUN}/forecasts/${DS}/ic${IC_IDX[${RUN}_${DS}]}' \
        --device cuda \
        --model.paradis.hidden_dim 96 \
        --model.paradis.num_layers 8 \
        --model.paradis.num_encoder_layers 4 \
        --model.paradis.num_vels 24 \
        --model.paradis.diffusion_size 48 \
        --model.paradis.reaction_size 24 \
        --model.paradis.bias_channels 6"
    echo "Submitted $RUN x $DS"
  done
done
