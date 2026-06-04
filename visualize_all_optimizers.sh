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

labels=(
  Adam
  AdamW
#  Gauss-Newton
  MUD
  Muon
  SGD
)

if [[ "${#optimizers[@]}" -ne "${#labels[@]}" ]]; then
  echo "Error: optimizers and labels must have the same length." >&2
  exit 1
fi

runs=()
for opt in "${optimizers[@]}"; do
  runs+=("./logs/${opt}/version_0")
done

figures_history_dir="./figures_history"
figures_forecast_dir="./figures_forecast"
rollout_dir="./rollout_results"

read -r -p "Clear ${figures_history_dir}, ${figures_forecast_dir}, and ${rollout_dir} before plotting? [y/N] " clear_outputs
case "${clear_outputs}" in
  [yY]|[yY][eE][sS])
    echo "Clearing visualization outputs"
    rm -rf "${figures_history_dir}" "${figures_forecast_dir}" "${rollout_dir}"
    ;;
  *)
    echo "Keeping existing visualization outputs"
    ;;
esac

echo "=== Plotting training history ==="
python -m visualize plot_history \
  --runs "${runs[@]}" \
  --labels "${labels[@]}" \
  --stage both \
  --plot both \
  --error_metric l2 \
  --efficiency_metric both \
  --outdir "${figures_history_dir}"

echo "=== Plotting forecast comparison ==="
python -m visualize forecast \
  --runs "${runs[@]}" \
  --labels "${labels[@]}" \
  --config config_paradis.yaml \
  --autoreg_steps 100 \
  --spherical_method spherical\
  --summary_step final\
  --output_freq 10 \
  --channel vorticity \
  --rollout_dir "${rollout_dir}" \
  --outdir "${figures_forecast_dir}"
