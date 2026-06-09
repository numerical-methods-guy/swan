# Bash Script Guide

This project provides three bash entry points for training, visualization, and
cluster execution:

```text
train_all_optimizers.sh
visualize_all_optimizers.sh
run_cluster_visualization.sh
```

Use the first two directly for local/manual work. Use
`run_cluster_visualization.sh` for non-interactive cluster jobs.

## `train_all_optimizers.sh`

Runs training for the optimizer list defined near the top of the file.

```bash
bash train_all_optimizers.sh
```

Important settings:

| Setting | Default | Meaning |
|---|---:|---|
| `pretrain_epochs` | `25` | Epoch count for the standard optimizers. Passed to `--training.pretrain_epochs`. |
| `include_gauss_newton` | `true` | Whether to include `gauss_newton` in the training loop. |
| `gauss_newton_epochs` | `2` | Epoch count for Gauss-Newton. Kept separate because it is usually more expensive per epoch. |
| `optimizers` | `adam adamw gauss_newton mud muon sgd` | Optimizers attempted by the loop. Gauss-Newton is removed when `include_gauss_newton=false`. |

If you run this script directly and want to change the epoch counts, edit the
fallback numbers near the top of `train_all_optimizers.sh`:

```bash
pretrain_epochs="${PRETRAIN_EPOCHS:-25}"
gauss_newton_epochs="${GAUSS_NEWTON_EPOCHS:-2}"
```

Here, `25` is the default epoch count for Adam, AdamW, MUD, Muon, and SGD, and
`2` is the default epoch count for Gauss-Newton. For example, changing them to:

```bash
pretrain_epochs="${PRETRAIN_EPOCHS:-50}"
gauss_newton_epochs="${GAUSS_NEWTON_EPOCHS:-5}"
```

makes a direct `bash train_all_optimizers.sh` run use 50 epochs for the standard
optimizers and 5 epochs for Gauss-Newton.

The script asks:

```text
Clear ./logs before starting training? [y/N]
```

Answering `y` deletes `./logs`, including trained checkpoints. Answering `n`
keeps checkpoints, but new training runs may create Lightning folders such as
`version_1`, `version_2`, etc. The current visualization script reads
`./logs/<optimizer>/version_0`.

## `visualize_all_optimizers.sh`

Generates history plots, forecast plots, rollout field GIFs, and rollout
spectral GIFs from trained checkpoints.

```bash
bash visualize_all_optimizers.sh
```

Important settings:

| Setting | Default | Meaning |
|---|---:|---|
| `include_gauss_newton` | `true` | Adds all-with-Gauss-Newton and Gauss-Newton-only visualization groups. |
| `channel` | `vorticity` | Field channel used for forecast grids and animations. |
| `history_scale` | `log` | Scale used by history plots. Learning curves use it on the metric y-axis; hitting curves use it on the threshold x-axis. |
| `forecast_error_scale` | `log` | Y-axis scale for forecast L2 error curves. |
| `autoreg_steps` | `100` | Number of autoregressive forecast steps. |
| `output_freq` | `5` | Saves rollout tensors/plots every N forecast steps. |
| `delete_rollouts_after_plotting` | `true` | Deletes `./rollout_results` after figures and animations are complete. |

The script asks:

```text
Clear visualization output folders before plotting? [y/N]
```

This only cleans visualization outputs:

```text
./figures_history
./figures_forecast
./rollout_results
```

It does not delete `./logs` or trained checkpoints.

With `include_gauss_newton=false`, outputs are:

```text
figures_history/base_optimizers/
figures_forecast/base_optimizers/
rollout_results/base_optimizers/
```

With `include_gauss_newton=true`, history and forecast figures are separated
by comparison scope:

```text
figures_history/base_optimizers/
figures_history/with_gauss_newton/
figures_history/gauss_newton_only/

figures_forecast/base_optimizers/
figures_forecast/with_gauss_newton/
figures_forecast/gauss_newton_only/
```

These three versions mean:

| Subfolder | Optimizers included | Purpose |
|---|---|---|
| `base_optimizers` | Adam, AdamW, MUD, Muon, SGD | Main comparison among the standard optimizers. |
| `with_gauss_newton` | Adam, AdamW, Gauss-Newton, MUD, Muon, SGD | Single combined graph including Gauss-Newton. Use this when Gauss-Newton should be shown beside every other optimizer. |
| `gauss_newton_only` | Gauss-Newton only | Diagnostic view for Gauss-Newton by itself, useful when it was trained with a different epoch budget or behaves very differently. |

Forecast rollouts are generated once for all selected optimizers into:

```text
rollout_results/shared_rollouts/
```

Then the script reuses those saved rollouts to build the base-only,
with-Gauss-Newton, and Gauss-Newton-only forecast figures. If
`delete_rollouts_after_plotting=true`, `./rollout_results` is removed at the
end to save storage.

## `run_cluster_visualization.sh`

Non-interactive wrapper for cluster use. It can call the training script, the
visualization script, or both.

For Slurm, edit the `#SBATCH` resource lines near the top, then submit:

```bash
sbatch run_cluster_visualization.sh
```

For local or interactive testing:

```bash
bash run_cluster_visualization.sh
```

Important settings:

| Setting | Default | Meaning |
|---|---:|---|
| `run_training` | `false` | Whether to call `train_all_optimizers.sh`. |
| `run_visualization` | `true` | Whether to call `visualize_all_optimizers.sh`. |
| `clear_training_logs` | `false` | Automatically answers the training cleanup prompt. `true` deletes `./logs`; `false` keeps it. |
| `include_gauss_newton` | `true` | Single wrapper switch for Gauss-Newton. It controls both training and visualization for this wrapper run. |
| `training_pretrain_epochs` | `25` | Overrides `pretrain_epochs` when this wrapper calls `train_all_optimizers.sh`. |
| `training_gauss_newton_epochs` | `2` | Overrides `gauss_newton_epochs` when this wrapper calls `train_all_optimizers.sh`. |
| `clear_visualization_outputs` | `true` | Automatically answers the visualization cleanup prompt. `true` clears old figures/rollouts; `false` keeps them. |
| `visualization_channel` | `vorticity` | Overrides the forecast/animation channel for this wrapper run. |
| `visualization_history_scale` | `log` | Overrides the history plot scale for this wrapper run. |
| `visualization_forecast_error_scale` | `log` | Overrides the forecast error curve scale for this wrapper run. |
| `visualization_autoreg_steps` | `100` | Overrides the forecast rollout horizon for this wrapper run. |
| `visualization_output_freq` | `5` | Overrides how often rollout tensors/plots are saved for this wrapper run. |

The cluster wrapper is non-interactive. It pipes `y` or `n` into the cleanup
prompts based on the settings above.

The two training epoch settings are passed through environment variables only
for this wrapper run. Running `train_all_optimizers.sh` directly still uses the
fallback numbers inside that file, meaning the `25` in
`pretrain_epochs="${PRETRAIN_EPOCHS:-25}"` and the `2` in
`gauss_newton_epochs="${GAUSS_NEWTON_EPOCHS:-2}"`, unless you manually set
`PRETRAIN_EPOCHS` or `GAUSS_NEWTON_EPOCHS`.

`include_gauss_newton` is passed to both child scripts as
`INCLUDE_GAUSS_NEWTON`. When it is `false`, the wrapper skips Gauss-Newton
training and asks visualization to produce only the base optimizer comparison.
When it is `true`, visualization expects a Gauss-Newton checkpoint, either from
the same wrapper run or from an existing `logs/gauss_newton/version_0` folder.

The visualization settings are also passed through environment variables only
for this wrapper run. Running `visualize_all_optimizers.sh` directly still uses
the fallback values inside that file unless you manually set `VIS_CHANNEL`,
`HISTORY_SCALE`, `FORECAST_ERROR_SCALE`, `AUTOREG_STEPS`, or `OUTPUT_FREQ`.

Recommended cluster patterns:

```bash
# Visualize existing trained models. Safest default.
run_training=false
run_visualization=true
include_gauss_newton=true
clear_training_logs=false
clear_visualization_outputs=true
```

```bash
# Fresh full training and visualization run.
# Use only when it is okay to delete ./logs.
run_training=true
run_visualization=true
include_gauss_newton=true
clear_training_logs=true
clear_visualization_outputs=true
```

```bash
# Figure tweaking after models are already trained.
run_training=false
run_visualization=true
include_gauss_newton=true
clear_training_logs=false
clear_visualization_outputs=true
```

## Quick Checks

Before submitting to a cluster, run:

```bash
bash -n train_all_optimizers.sh
bash -n visualize_all_optimizers.sh
bash -n run_cluster_visualization.sh
```

The `-n` option tells bash to check the script syntax without executing the
script, so it catches quoting or control-flow errors without starting training
or visualization.

If `visualize_all_optimizers.sh` has `include_gauss_newton=true`, make sure a
Gauss-Newton checkpoint exists at:

```text
logs/gauss_newton/version_0/checkpoints/
```

Otherwise set `include_gauss_newton=false` before running visualization.
