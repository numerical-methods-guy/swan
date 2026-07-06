#!/bin/bash
# Example forecast run using 20260623_run1 weights (hidden_dim=384, 8 layers)
# with a precomputed gbells_h_rv dataset as IC source.
#
# Run from the repo root: bash scripts/example_forecast.sh

PYTHONPATH=/home/avg000/swan \
/home/avg000/miniconda3/envs/swan/bin/python forecast.py \
  --config config_paradis.yaml \
  --checkpoint "/home/avg000/swan/weights/20260623_run1/swan_precomp_20260623_170312/version_0/checkpoints/pretrain-epoch=92-val_loss=0.0728.ckpt" \
  --stats_path /home/avg000/swan/weights/20260623_run1/stats.pt \
  --ic_type precomputed \
  --precomputed_folder /home/avg000/swan/datasets/train/gbells_h_rv_60_20260621 \
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
