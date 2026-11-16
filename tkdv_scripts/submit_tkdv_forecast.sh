#!/bin/bash
REPO="/home/avg000/swan"
PY="/home/avg000/miniconda3/envs/swan/bin/python"

CKPT="/fs/hestia_Heccc/rpnatm/avg000/work/tkdv_datasets/upstream/train_weights/20261116_run2/tkdv_train_upstream_20261116_070010/version_0/checkpoints/pretrain-epoch=883-val_loss=0.4356.ckpt"
STATS="/fs/hestia_Heccc/rpnatm/avg000/work/tkdv_datasets/upstream/train_weights/20261116_run2/stats.pt"
IC_FOLDER="/fs/hestia_Heccc/rpnatm/avg000/work/tkdv_datasets/samples/seed_7/tkdv_gibbs_dt1.0_dtsolver0.01_20261115"
OUT="/fs/hestia_Heccc/rpnatm/avg000/work/tkdv_forecast_results"

mkdir -p "$OUT"

sbatch --job-name="tkdv_forecast" \
  --account=eccc_pegasus_mrd__gpu_a100 \
  --partition=gpu_a100 \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 \
  --mem=32G --time=00:30:00 \
  --output="/fs/hestia_Heccc/rpnatm/avg000/work/logs/tkdv_forecast_%j.log" \
  --comment="image=registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu22.04" \
  --wrap="cd $REPO && PYTHONPATH=$REPO $PY tkdv_forecast.py \
    --config config_tkdv.yaml \
    --checkpoint '$CKPT' \
    --stats_path '$STATS' \
    --ic_type precomputed \
    --precomputed_folder '$IC_FOLDER' \
    --num_ics 1 \
    --ic_start_index 0 \
    --autoreg_steps 40 \
    --output_dir '$OUT' \
    --device cuda \
    --model.paradis.hidden_dim 24 \
    --model.paradis.num_layers 4 \
    --model.paradis.num_encoder_layers 4 \
    --model.paradis.num_vels 8 \
    --model.paradis.diffusion_size 6 \
    --model.paradis.reaction_size 2 \
    --model.paradis.bias_channels 3"
echo "Submitted tkdv forecast"
