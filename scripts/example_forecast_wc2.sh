#!/bin/bash
# Example forecast run using 20260623_run1 weights with Williamson case 2 ICs.
#
# Run from the repo root: bash scripts/example_forecast_wc2.sh

PYTHONPATH=/home/avg000/swan \
/home/avg000/miniconda3/envs/swan/bin/python forecast.py \
  --config config_paradis.yaml \
  --checkpoint "/home/avg000/swan/weights/20260623_run1/swan_precomp_20260623_170312/version_0/checkpoints/pretrain-epoch=92-val_loss=0.0728.ckpt" \
  --stats_path /home/avg000/swan/weights/20260623_run1/stats.pt \
  --ic_type precomputed \
  --precomputed_folder /home/avg000/swan/datasets/train/williamson_case2_60_20260618 \
  --num_ics 1 \
  --autoreg_steps 40 \
  --output_dir /home/avg000/swan/weights/20260623_run1 \
  --device cuda \
  --model.paradis.hidden_dim 384 \
  --model.paradis.num_layers 8 \
  --model.paradis.num_encoder_layers 6 \
  --model.paradis.num_vels 48 \
  --model.paradis.diffusion_size 96 \
  --model.paradis.reaction_size 48 \
  --model.paradis.bias_channels 6
