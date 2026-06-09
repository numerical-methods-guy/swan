#!/usr/bin/env bash
set -euo pipefail

# Defaults can be overridden by the cluster wrapper through environment
# variables, e.g. PRETRAIN_EPOCHS=50 bash train_all_optimizers.sh. Keeping this
# fallback here preserves the old direct-run behavior.
pretrain_epochs="${PRETRAIN_EPOCHS:-25}"
# Can be overridden by the cluster wrapper with INCLUDE_GAUSS_NEWTON=false when
# a job should train only the standard optimizer set.
include_gauss_newton="${INCLUDE_GAUSS_NEWTON:-true}"
# Gauss-Newton is much more expensive per epoch than the first-order methods, so
# keep its epoch budget explicit and usually smaller than pretrain_epochs.
gauss_newton_epochs="${GAUSS_NEWTON_EPOCHS:-2}"

optimizers=(
  adam
  adamw
  gauss_newton
  mud
  muon
  sgd
)

if [[ "${include_gauss_newton}" != "true" ]]; then
  filtered_optimizers=()
  for opt in "${optimizers[@]}"; do
    if [[ "${opt}" != "gauss_newton" ]]; then
      filtered_optimizers+=("${opt}")
    fi
  done
  optimizers=("${filtered_optimizers[@]}")
fi

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
  opt_epochs="${pretrain_epochs}"
  if [[ "${opt}" == "gauss_newton" ]]; then
    opt_epochs="${gauss_newton_epochs}"
  fi
  python train.py \
    --config config_paradis.yaml \
    --optimizer "${opt}" \
    --experiment.name "${opt}" \
    --training.pretrain_epochs "${opt_epochs}"
done
