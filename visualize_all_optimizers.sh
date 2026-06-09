#!/usr/bin/env bash
set -euo pipefail

channel="vorticity"
include_sgd=true
history_scale="log"
forecast_error_scale="log"

optimizers=(
  adam
  adamw
#  gauss_newton
  mud
  muon
)

labels=(
  Adam
  AdamW
#  Gauss-Newton
  MUD
  Muon
)

if [[ "${include_sgd}" == "true" ]]; then
  optimizers+=(sgd)
  labels+=(SGD)
fi

if [[ "${#optimizers[@]}" -ne "${#labels[@]}" ]]; then
  echo "Error: optimizers and labels must have the same length." >&2
  exit 1
fi

runs=()
for opt in "${optimizers[@]}"; do
  runs+=("./logs/${opt}/version_0")
done

output_suffix=""
if [[ "${include_sgd}" != "true" ]]; then
  output_suffix="_no_sgd"
fi

figures_history_dir="./figures_history${output_suffix}"
figures_forecast_dir="./figures_forecast${output_suffix}"
rollout_dir="./rollout_results${output_suffix}"

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
  --history_scale "${history_scale}" \
  --outdir "${figures_history_dir}"

echo "=== Plotting forecast comparison ==="
python -m visualize forecast \
  --runs "${runs[@]}" \
  --labels "${labels[@]}" \
  --config config_paradis.yaml \
  --autoreg_steps 100 \
  --spherical_method spherical \
  --summary_step final \
  --output_freq 5 \
  --channel "${channel}" \
  --rollout_dir "${rollout_dir}" \
  --forecast_error_scale "${forecast_error_scale}" \
  --outdir "${figures_forecast_dir}"

echo "=== Animating forecast comparison ==="
python -m visualize animate \
  --rollout_dir "${rollout_dir}" \
  --labels "${labels[@]}" \
  --channel "${channel}" \
  --show_error \
  --output "${figures_forecast_dir}/rollout_fields.gif" \
  --spectral_output "${figures_forecast_dir}/rollout_spectra.gif"
