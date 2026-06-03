#!/usr/bin/env bash
set -euo pipefail

optimizers=(
  adam
  adamw
#  gauss_newton
  mud
  muon
  sgd
)

logs_dir="./logs"

read -r -p "Clear ${logs_dir} before starting training? [y/N] " clear_logs
case "${clear_logs}" in
  [yY]|[yY][eE][sS])
    echo "Clearing ${logs_dir}"
    rm -rf "${logs_dir}"
    ;;
  *)
    echo "Keeping existing ${logs_dir}"
    ;;
esac

for opt in "${optimizers[@]}"; do
  echo "=== Training with optimizer: ${opt} ==="
  python train.py \
    --config config_paradis.yaml \
    --optimizer "${opt}" \
    --experiment.name "${opt}"
done
