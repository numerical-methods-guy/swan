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
| `train_nlat` | `128` | Training grid latitude count. Passed as `--data.nlat`. |
| `train_nlon` | `256` | Training grid longitude count. Passed as `--data.nlon`. |
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
| `include_gauss_newton` | `true` | Controls whether Gauss-Newton is included and which comparison groups are generated. |
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
figures_history/all_optimizers/
figures_history/without_sgd/

figures_forecast/all_optimizers/
figures_forecast/without_sgd/
```

These two versions mean:

| Subfolder | Optimizers included | Purpose |
|---|---|---|
| `all_optimizers` | Adam, AdamW, MUD, Muon, SGD | Full comparison when Gauss-Newton is not part of the run. |
| `without_sgd` | Adam, AdamW, MUD, Muon | Main comparison without the SGD baseline. |

With `include_gauss_newton=true`, history and forecast figures are separated
by comparison scope:

```text
figures_history/all_optimizers/
figures_history/sgd_and_gauss_newton/
figures_history/without_sgd_gauss_newton/

figures_forecast/all_optimizers/
figures_forecast/sgd_and_gauss_newton/
figures_forecast/without_sgd_gauss_newton/
```

These three versions mean:

| Subfolder | Optimizers included | Purpose |
|---|---|---|
| `all_optimizers` | Adam, AdamW, Gauss-Newton, MUD, Muon, SGD | Full comparison including every optimizer. |
| `sgd_and_gauss_newton` | Gauss-Newton, SGD | Diagnostic comparison of the two weak/unstable baselines. |
| `without_sgd_gauss_newton` | Adam, AdamW, MUD, Muon | Main comparison among the stronger non-SGD, non-Gauss-Newton optimizers. |

Forecast rollouts are generated once for all selected optimizers into:

```text
rollout_results/shared_rollouts/
```

Then the script reuses those saved rollouts to build the subset forecast
figures for each comparison scope. If
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
| `resolution_nlat` | `128` | Shared latitude count for training and visualization in this wrapper run. |
| `resolution_nlon` | `256` | Shared longitude count for training and visualization in this wrapper run. |
| `overwrite_visualization_outputs` | `false` | With `visualization_version=auto`, `false` creates the next new `version_N`; `true` clears and rewrites the latest existing `version_N`. |
| `visualization_channel` | `vorticity` | Overrides the forecast/animation channel for this wrapper run. |
| `visualization_history_scale` | `log` | Overrides the history plot scale for this wrapper run. |
| `visualization_forecast_error_scale` | `log` | Overrides the forecast error curve scale for this wrapper run. |
| `visualization_autoreg_steps` | `100` | Overrides the forecast rollout horizon for this wrapper run. |
| `visualization_output_freq` | `5` | Overrides how often rollout tensors/plots are saved for this wrapper run. |
| `visualization_root` | `./visualization_runs` | Top-level folder for versioned visualization outputs from the wrapper. |
| `visualization_version` | `auto` | Output version folder. With `auto`, `overwrite_visualization_outputs` controls whether to append a new version or rewrite the latest one. Any other value is used as a folder name under `visualization_root`. |

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

`resolution_nlat` and `resolution_nlon` are shared by training and
visualization in the cluster wrapper, which avoids accidentally training at one
resolution and evaluating at another. Direct script users can still override
training with `TRAIN_NLAT/TRAIN_NLON` and visualization with
`VIS_NLAT/VIS_NLON` if they intentionally want a resolution-generalization test.

The visualization settings are also passed through environment variables only
for this wrapper run. Running `visualize_all_optimizers.sh` directly still uses
the fallback values inside that file unless you manually set `VIS_CHANNEL`,
`HISTORY_SCALE`, `FORECAST_ERROR_SCALE`, `AUTOREG_STEPS`, `OUTPUT_FREQ`,
`VIS_NLAT`, or `VIS_NLON`.

The default `resolution_nlat=128` and `resolution_nlon=256` match
`config_paradis.yaml`. If you change these values, use a new visualization
version folder so outputs from different resolutions are not mixed.

The cluster wrapper writes visualization outputs into a versioned folder so
different rollout horizons or plotting settings do not overwrite each other.
With the default:

```bash
visualization_version="auto"
overwrite_visualization_outputs=false
```

each run creates the next available directory:

```text
visualization_runs/
  version_0/
    figures_history/
    figures_forecast/
    rollout_results/
    settings.txt
  version_1/
    ...
```

`settings.txt` records the wrapper settings used for that version. If
`delete_rollouts_after_plotting=true` in `visualize_all_optimizers.sh`,
`rollout_results/` is deleted after figures and animations are generated, but
the version's `figures_history/`, `figures_forecast/`, and `settings.txt`
remain.

To rewrite the latest existing auto version instead of creating a new one, set:

```bash
visualization_version="auto"
overwrite_visualization_outputs=true
```

For example, if `version_0`, `version_1`, and `version_2` already exist, this
clears and regenerates `visualization_runs/version_2/`.

To force a specific output folder, set for example:

```bash
visualization_version="rollout_200_steps"
```

This writes to:

```text
visualization_runs/rollout_200_steps/
```

### Preferred Settings When Changing Rollout Horizon

If models are already trained and you only want to compare a different rollout
horizon, do not retrain. Use a new visualization version so the new horizon does
not overwrite the previous figures:

```bash
run_training=false
run_visualization=true
overwrite_visualization_outputs=false
visualization_version="auto"
visualization_autoreg_steps=200
visualization_output_freq=5
```

This creates a new folder such as:

```text
visualization_runs/version_3/
```

with its own `settings.txt`, `figures_history/`, and `figures_forecast/`. If you
want to intentionally regenerate the latest visualization version instead, set:

```bash
overwrite_visualization_outputs=true
```

### Finding the Rollout Instability Point

To estimate when a model becomes unstable, first run a long rollout with a
moderate save frequency:

```bash
run_training=false
run_visualization=true
overwrite_visualization_outputs=false
visualization_version="auto"
visualization_autoreg_steps=300
visualization_output_freq=5
```

Inspect the field animation, spectral animation, and forecast L2 curve. Look
for the first window where pointwise error grows sharply, spatial artifacts
appear, or spectral energy starts to diverge from the ground truth.

After estimating the rough failure window, rerun with a focused horizon and
finer output frequency:

```bash
run_training=false
run_visualization=true
overwrite_visualization_outputs=false
visualization_version="auto"
visualization_autoreg_steps=200
visualization_output_freq=1
```

Then report the instability as a range rather than a single exact step. For
example:

```text
The rollout remains visually stable through roughly step 140-150, begins to
degrade around step 160, and is clearly unstable by step 180.
```

Recommended cluster patterns:

```bash
# Visualize existing trained models. Safest default.
run_training=false
run_visualization=true
include_gauss_newton=true
clear_training_logs=false
overwrite_visualization_outputs=false
```

```bash
# Fresh full training and visualization run.
# Use only when it is okay to delete ./logs.
run_training=true
run_visualization=true
include_gauss_newton=true
clear_training_logs=true
overwrite_visualization_outputs=false
```

```bash
# Figure tweaking after models are already trained.
run_training=false
run_visualization=true
include_gauss_newton=true
clear_training_logs=false
overwrite_visualization_outputs=false
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
