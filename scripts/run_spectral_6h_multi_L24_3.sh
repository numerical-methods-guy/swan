#!/bin/bash
BASE="/space/hall0/work/eccc/mrd/rpnatm/avg000/Trained_weights_and_Graphs"
FORECAST_BASE="$BASE/forecasts_6h_multi_L24_3_20261020/forecasts"
STATS="$BASE/20261019_run1/stats.pt"
REPO="/home/avg000/swan"
PY="/home/avg000/miniconda3/envs/swan/bin/python"

for DS in gbells_h wc6_matched; do
  for CH in geopotential vorticity divergence; do
    FORECAST_DIR="$FORECAST_BASE/$DS/ic47/$CH/20261020_forecast1"
    OUT_DIR="$BASE/spectral_6h_multi_L24_3_per_step/$DS/$CH"
    mkdir -p "$OUT_DIR"
    sbatch --job-name="spec_L24_3_${DS}_${CH}" \
      --account=eccc_pegasus_mrd__gpu_a100 \
      --partition=gpu_a100 \
      --nodes=1 --ntasks=1 --cpus-per-task=2 --gres=gpu:0 \
      --mem=16G --time=00:10:00 \
      --output="/fs/hestia_Heccc/rpnatm/avg000/work/logs/spec_L24_3_${DS}_${CH}_%j.log" \
      --comment="image=registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu22.04" \
      --wrap="cd $REPO && PYTHONPATH=$REPO $PY eval_spectral.py \
        --forecast_dir '$FORECAST_DIR' \
        --stats_path '$STATS' \
        --output_dir '$OUT_DIR' \
        --steps 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 \
        --per_step \
        --tag 6h_multi_L24_3"
    echo "Submitted spectral eval: $DS x $CH"
  done
done
