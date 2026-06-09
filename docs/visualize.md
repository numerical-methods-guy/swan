# Optimizer Visualization Tools for SWAN

This toolkit compares different optimizer runs in the SWAN project.

All commands are exposed through the `visualize` Python package:

```bash
python -m visualize <command> [options]
```

It also includes two optional bash convenience scripts:

```text
train_all_optimizers.sh
visualize_all_optimizers.sh
```

---

## 1. Package Overview

```text
visualize/
  history.py   — load and prepare training/validation scalar histories
  rollout.py   — load, generate, and prepare forecast rollout data
  plots.py     — all matplotlib figure-building functions
  _cli.py      — argument parsing and command entry points
```

Users do not import these sub-modules directly. All interaction goes through
the three commands below.

### Commands

```bash
python -m visualize plot_history ...   # training/validation history curves
python -m visualize forecast ...       # post-training rollout comparison
python -m visualize animate ...        # animated rollout comparison (GIF/MP4)
```

---

### `train_all_optimizers.sh`

Bash helper for training the optimizer list used in the examples.

It runs:

```bash
python train.py --config config_paradis.yaml --optimizer <optimizer> --experiment.name <optimizer>
```

for each optimizer in the script. With the default `training.save_dir: ./logs`, this creates folders such as `./logs/adam/version_0`.

The script asks whether to clear `./logs` before starting a new training sweep.

---

### `visualize_all_optimizers.sh`

Bash helper for running both visualization commands against the folders created by `train_all_optimizers.sh`.

It runs `plot_history` and `forecast` with matching `--runs` and `--labels` arrays, using paths such as:

```text
./logs/adam/version_0
./logs/adamw/version_0
./logs/mud/version_0
./logs/muon/version_0
./logs/sgd/version_0
```

The script asks whether to clear `./figures_history`, `./figures_forecast`, and `./rollout_results` before plotting.

---

## 2. Requirements

### Python packages

The toolkit expects the usual SWAN project dependencies plus:

```bash
pip install numpy pandas matplotlib
```

For reading TensorBoard event files, install TensorBoard:

```bash
pip install tensorboard
```

If TensorBoard is not installed, `plot_history` can still read CSV fallback files, such as `history.csv` or `scalars.csv`, if they are present in each run folder.

---

## 3. Required File Placement

Place these toolkit files in the SWAN repository root, or somewhere from which Python can import the original SWAN modules:

```text
swan/
  visualize.py
  history_utils.py
  rollout_utils.py
  forecast.py              # original SWAN forecast script; needed by the forecast command
  config_paradis.yaml      # or another config file passed with --config
  ...                      # other original SWAN files/modules
```

Your optimizer training scripts are **not required** to have any particular names.

For example, these are all fine:

```text
train_adam.py
train_muon.py
experiments/train_mud_optimizer.py
scripts/my_custom_optimizer_training.py
```

The visualization toolkit does **not** require specific training script names. It only needs the run folders produced by your training scripts.

The `forecast` command uses functionality from the original `forecast.py`, so `forecast.py` should be importable from the same folder or from your Python path.

---

## 4. Expected Training Output Folder Structure

Your training scripts should save TensorBoard logs and checkpoints in Lightning-style folders. This is exactly the folder structure what the original `train.py` outputs.

Example:

```text
logs/
  adam/
    version_0/
      events.out.tfevents...
      checkpoints/
        last.ckpt
        pretrain-epoch=...-val_loss=....ckpt

  adamw/
    version_0/
      events.out.tfevents...
      checkpoints/
        last.ckpt
        pretrain-epoch=...-val_loss=....ckpt

  mud/
    version_0/
      events.out.tfevents...
      checkpoints/
        last.ckpt
        pretrain-epoch=...-val_loss=....ckpt

  muon/
    version_0/
      events.out.tfevents...
      checkpoints/
        last.ckpt
        pretrain-epoch=...-val_loss=....ckpt
```

The paths passed to `--runs` should be the `version_0` folders, for example:

```text
./logs/adam/version_0
./logs/adamw/version_0
./logs/mud/version_0
./logs/muon/version_0
./logs/sgd/version_0
```

The labels passed to `--labels` should be in the same order:

```text
Adam AdamW MUD Muon SGD
```

The number of paths passed to `--runs` must match the number of labels passed to `--labels`.

The tools support any number of optimizers, not just three.

For example, this is valid:

```text
--runs ./logs/adam/version_0 ./logs/adamw/version_0 ./logs/mud/version_0 ./logs/muon/version_0 ./logs/sgd/version_0
--labels Adam AdamW MUD Muon SGD
```

---

## 5. Recommended Fair Training Setup

> **Terminal line-continuation note:** The examples below use Unix shell line continuation with `\`.  
> This works in macOS/Linux/Git Bash/WSL.  
> If you use **PowerShell**, replace each trailing `\` with a backtick character `` ` ``.  
> If you use **Windows Command Prompt (`cmd.exe`)**, replace each trailing `\` with `^`.  
> Or use the one-line command version, which works in all terminals.

For a fair optimizer comparison, train all optimizers with the same:

- config file;
- random seed;
- model architecture;
- loss function;
- batch size;
- number of training examples;
- number of validation examples;
- number of epochs;
- learning-rate schedule;
- `num_workers=0`, if possible.

The only intended difference should be the optimizer.

Example bash commands:

```bash
python train.py \
  --config config_paradis.yaml \
  --optimizer adam \
  --experiment.seed 42 \
  --experiment.name adam \
  --data.num_workers 0

python train.py \
  --config config_paradis.yaml \
  --optimizer adamw \
  --experiment.seed 42 \
  --experiment.name adamw \
  --data.num_workers 0

python train.py \
  --config config_paradis.yaml \
  --optimizer mud \
  --experiment.seed 42 \
  --experiment.name mud \
  --data.num_workers 0
```

For more optimizers, repeat the same pattern with a unique `--experiment.name` for each run. This repository also includes `train_all_optimizers.sh`, which follows this convention and writes runs under `./logs/<optimizer>/version_0`.

In Unix shells, the backslash `\` is the line-continuation character. It must be the final character on the line.

---



# Command 1: `plot_history`

Use this command to compare TensorBoard training/validation histories.

---

## Basic Example

```bash
python -m visualize plot_history \
  --runs ./logs/adam/version_0 ./logs/adamw/version_0 ./logs/mud/version_0 ./logs/muon/version_0 ./logs/sgd/version_0 \
  --labels Adam AdamW MUD Muon SGD \
  --stage validation \
  --plot both \
  --error_metric l2 \
  --efficiency_metric both \
  --outdir ./figures_history
```

This generates validation L2 learning curves and first-hitting curves against both training step and relative wall-clock time.

Typical outputs:

```text
figures_history/
  learning_curve_validation_step_l2.png
  learning_curve_validation_time_l2.png
  hitting_curve_validation_step_l2.png
  hitting_curve_validation_time_l2.png
```

---

## `plot_history` Flags

| Flag | Required? | Default | Choices | Meaning |
|---|---:|---|---|---|
| `--runs` | yes | none | paths | TensorBoard run folders, usually `version_0` folders. |
| `--labels` | yes | none | strings | Optimizer names shown in legends. Must match `--runs` order. |
| `--stage` | no | `validation` | `training`, `validation`, `both` | Which stage history to plot. |
| `--plot` | no | `learning_curve` | `learning_curve`, `hitting_curve`, `both` | Which history plot type to generate. |
| `--error_metric` | no | `loss` | `loss`, `l1`, `l2`, `sq_l2`, `w11` | Error/loss metric. |
| `--efficiency_metric` | no | `both` | `step`, `time`, `both` | X-axis resource for history plots. |
| `--history_scale` | no | `linear` | `linear`, `log` | Scale for history plots. For learning curves, this controls the metric y-axis; for hitting curves, this controls the target-threshold x-axis. |
| `--outdir` | no | `./figures` | path | Folder where history figures are saved. |

---

## Metric Mapping

For validation:

```text
--stage validation --error_metric loss  -> val_loss
--stage validation --error_metric l1    -> val_l1
--stage validation --error_metric l2    -> val_l2
--stage validation --error_metric sq_l2 -> val_sq_l2
--stage validation --error_metric w11   -> val_w11
```

For training:

```text
--stage training --error_metric loss -> train_loss_epoch
```

The original SWAN `train.py` logs training loss, but not training L1/L2/W11 by default.

Therefore, when using:

```bash
--stage both --error_metric l2
```

the toolkit uses:

```text
training   -> train_loss_epoch
validation -> val_l2
```


---

## Learning Curves

A learning curve plots:

```text
x-axis = training step or relative wall-clock time
y-axis = selected training/validation metric
one curve per optimizer
```

The legend is placed outside the plot area so it does not cover overlapping
curves. Validation learning curves show markers at each logged validation point;
training curves use lines without markers to avoid clutter when many training
points are logged. Use `--history_scale log` when relative loss/error
differences are more important than absolute differences. For learning curves,
this makes the metric y-axis logarithmic; for hitting curves, this makes the
target-threshold x-axis logarithmic while keeping first-hit step/time linear.

Example:

```bash
python -m visualize plot_history \
  --runs ./logs/adam/version_0 ./logs/adamw/version_0 ./logs/mud/version_0 ./logs/muon/version_0 ./logs/sgd/version_0 \
  --labels Adam AdamW MUD Muon SGD \
  --stage validation \
  --plot learning_curve \
  --error_metric l2 \
  --efficiency_metric both \
  --outdir ./figures_history
```

Outputs:

```text
learning_curve_validation_step_l2.png
learning_curve_validation_time_l2.png
```

---

## Hitting Curves

A hitting curve plots:

```text
x-axis = target error threshold
y-axis = first step/time at which best-so-far error reaches that threshold
one curve per optimizer
```
If one observes that a curve does not continue beyond a threshold in the graph, this means that the optimizer is never able to reach it during the training phase.

The code uses cumulative best error, so if validation error temporarily increases later, the first hitting result remains stable.

Example:

```bash
python -m visualize plot_history \
  --runs ./logs/adam/version_0 ./logs/adamw/version_0 ./logs/mud/version_0 ./logs/muon/version_0 ./logs/sgd/version_0 \
  --labels Adam AdamW MUD Muon SGD \
  --stage validation \
  --plot hitting_curve \
  --error_metric l2 \
  --efficiency_metric both \
  --outdir ./figures_history
```

Outputs:

```text
hitting_curve_validation_step_l2.png
hitting_curve_validation_time_l2.png
```

---

## Full History Comparison

To generate both training and validation plots, and both learning/hitting curves:

```bash
python -m visualize plot_history \
  --runs ./logs/adam/version_0 ./logs/adamw/version_0 ./logs/mud/version_0 ./logs/muon/version_0 ./logs/sgd/version_0 \
  --labels Adam AdamW MUD Muon SGD \
  --stage both \
  --plot both \
  --error_metric l2 \
  --efficiency_metric both \
  --outdir ./figures_history
```

Expected outputs:

```text
learning_curve_training_step_loss.png
learning_curve_training_time_loss.png
hitting_curve_training_step_loss.png
hitting_curve_training_time_loss.png
learning_curve_validation_step_l2.png
learning_curve_validation_time_l2.png
hitting_curve_validation_step_l2.png
hitting_curve_validation_time_l2.png
```

---

# Command 2: `forecast`

Use this command to compare post-training rollout behavior from trained checkpoints.

The command finds checkpoints inside each run folder, runs autoregressive rollout evaluation, saves per-optimizer rollout data, and generates cross-optimizer comparison figures.

---

## Basic Example

```bash
python -m visualize forecast \
  --runs ./logs/adam/version_0 ./logs/adamw/version_0 ./logs/mud/version_0 ./logs/muon/version_0 ./logs/sgd/version_0 \
  --labels Adam AdamW MUD Muon SGD \
  --config config_paradis.yaml \
  --autoreg_steps 100 \
  --output_freq 10 \
  --channel vorticity \
  --rollout_dir ./rollout_results \
  --outdir ./figures_forecast
```

Typical outputs:

```text
figures_forecast/
  forecast_error_curve_l2.png
  forecast_accuracy_bar_l2.png
  forecast_runtime_ratio_bar.png
  forecast_prediction_grid_vorticity_final.png
  forecast_error_grid_vorticity_final_signed.png
  forecast_spectra_final.png
```

---

## `forecast` Flags

| Flag | Required? | Default | Choices | Meaning |
|---|---:|---|---|---|
| `--runs` | yes, unless `--synthetic_demo` | none | paths | Training run folders containing `checkpoints/`. |
| `--labels` | yes, unless `--synthetic_demo` | synthetic defaults to `Adam MUD Muon` | strings | Optimizer names shown in plots. Must match `--runs` order. |
| `--config` | no | `config_paradis.yaml` | path | SWAN config file used for rollout evaluation. |
| `--checkpoint_choice` | no | `best` | `best`, `last` | Which checkpoint to use from each run. |
| `--autoreg_steps` | no | `100` | integer | Number of autoregressive rollout steps. |
| `--output_freq` | no | `10` | integer | Save rollout tensors/plots every N steps. |
| `--num_ics` | no | `1` | integer | Number of forecast initial conditions. |
| `--ic_type` | no | `random` | `random`, `galewsky` | Forecast initial-condition type. |
| `--seed` | no | `42` | integer | Forecast-time random seed. |
| `--channel` | no | `vorticity` | `h`, `vorticity`, `divergence` | Field channel for spatial plots. |
| `--error_metric` | no | `l2` | `loss`, `l1`, `l2`, `w11` | Scalar forecast metric for curves/bars (aggregated). |
| `--forecast_error_scale` | no | `linear` | `linear`, `log` | Y-axis scale for `forecast_error_curve_<metric>.png`. |
| `--error_mode` | no | `signed` | `signed`, `abs`, `squared` | Pointwise error map mode. |
| `--summary_step` | no | `final` | `final`, `latest`, or integer | Rollout step loaded for grid and spectra plots. |
| `--spherical_method` | no | `spherical` | `spherical`, `fft` | Spectral method for `forecast_spectra_final.png`. |
| `--grid_cols` | no | `3` | integer | Maximum number of columns in spatial grid figures. |
| `--rollout_dir` | no | `./rollout_results` | path | Folder for per-optimizer rollout outputs. |
| `--outdir` | no | `./figures_forecast` | path | Folder for final comparison figures. |
| `--device` | no | auto | `cpu`, `cuda`, etc. | Optional device for real rollout. |
| `--synthetic_demo` | no | off | flag | Generate artificial rollout data for testing only. |
| `--reuse_rollouts` | no | off | flag | Skip model rollout execution and plot from existing per-optimizer folders in `--rollout_dir`. |

---

## Important Forecast Concepts

### `--reuse_rollouts`

Use `--reuse_rollouts` when the rollout tensors and metrics already exist and
you only want to make another grouped set of forecast figures. This is accurate
when the saved rollouts were generated with the same checkpoint, config,
`--autoreg_steps`, `--output_freq`, seed, initial-condition setup, and channel
settings. The flag is useful for comparing different subsets of optimizers
without rerunning the expensive forecast step.

### `--autoreg_steps`

Controls how far the model is rolled out autoregressively.

For example:

```bash
--autoreg_steps 100
```

means the model is applied 100 times. Each prediction is fed back into the model as the next input.

### `--output_freq`

Controls how often rollout snapshots are saved.

For example:

```bash
--autoreg_steps 100 --output_freq 10
```

saves steps:

```text
0, 10, 20, 30, ..., 100
```

### `--channel`

Controls which physical field is shown in spatial plots.

```text
h          -> channel 0
vorticity  -> channel 1
divergence -> channel 2
```

### `--seed`

The forecast seed controls forecast-time initial conditions. This is separate from the training seed.

Training seed controls training-time randomness.

Forecast seed controls rollout-evaluation randomness.

The toolkit resets the forecast seed for each optimizer, so each optimizer is evaluated on the same forecast initial-condition sequence when `--ic_type random` is used.

### `--summary_step`

Controls which saved rollout step is loaded for the spatial grids and spectra.

```bash
--summary_step final
--summary_step latest
--summary_step 50
```

`final` and `latest` both select the latest common saved step across all rollout folders. An integer selects that exact saved step.

The generated filenames currently keep the suffix `final`, even when `--summary_step` is an integer. The figure titles report the actual selected step.

### `--spherical_method`

Controls how `forecast_spectra_final.png` computes spectra:

```bash
--spherical_method spherical
--spherical_method fft
```

`spherical` is the default for real rollouts and uses the same spherical-harmonic diagnostic as `forecast.py`. `fft` uses the grid-based FFT fallback from `rollout_utils.py`. Synthetic demo rollouts always use the FFT fallback because they do not have a SWAN spherical-harmonic transform.

---

## Expected Rollout Output Folder Structure

The `forecast` command writes per-optimizer rollout results into `--rollout_dir`.

Example:

```text
rollout_results/
  Adam/
    ic000_prediction_000.pt
    ic000_truth_000.pt
    ic000_prediction_010.pt
    ic000_truth_010.pt
    ...
    metrics.csv
    per_step_metrics.csv

  MUD/
    ic000_prediction_000.pt
    ic000_truth_000.pt
    ...
    metrics.csv
    per_step_metrics.csv

  Muon/
    ...
```

The `.pt` files are saved PyTorch tensors containing full-grid fields at each saved rollout step.

Conceptually:

```text
checkpoint .ckpt      -> trained model weights
prediction_010.pt     -> model forecast at rollout step 10
truth_010.pt          -> numerical solver ground truth at rollout step 10
metrics.csv           -> aggregate rollout summary
per_step_metrics.csv  -> error history over rollout steps
```

---

## Forecast Figures

### 1. Forecast Error Curve

```text
forecast_error_curve_l2.png
```

Plots forecast error against autoregressive rollout step.

```text
x-axis = rollout step
y-axis = selected scalar error metric
one curve per optimizer
```

The selected metric is controlled by:

```bash
--error_metric l2
```

Use `--forecast_error_scale log` when relative error growth matters or when one
optimizer dominates the linear-scale curve. The default is `linear` for direct
absolute-error reading.

### 2. Aggregate Accuracy Bar Chart

```text
forecast_accuracy_bar_l2.png
```

Plots one aggregate forecast error value per optimizer, usually averaged over rollout steps and initial conditions.

This is a bar chart, not a histogram.

### 3. Runtime Ratio Bar Chart

```text
forecast_runtime_ratio_bar.png
```

Plots:

```text
runtime_ratio = ml_time_mean / solver_time_mean
```

Lower is better. A value of `1.0` means the ML rollout and non-ML solver took
the same time; values above `1.0` mean the ML rollout was slower than the
non-ML solver, and values below `1.0` mean it was faster.

### 4. Prediction Grid

```text
forecast_prediction_grid_vorticity_final.png
```

Shows ground truth and all optimizer predictions at the selected rollout step.

All panels use the same color scale.

### 5. Pointwise Error Grid

```text
forecast_error_grid_vorticity_final_signed.png
```

Shows optimizer errors against ground truth at the selected rollout step.

The pointwise error mode is controlled by:

```bash
--error_mode signed
--error_mode abs
--error_mode squared
```

Default:

```text
--error_mode signed
```

For signed error:

```text
error = prediction - truth
```

All pointwise error panels use one shared automatic color scale. Signed errors use a symmetric scale around zero; absolute and squared errors use the shared data min/max.

### 6. Combined Spectral Plot

```text
forecast_spectra_final.png
```

Contains one 2-by-2 spectral figure:

```text
rotational kinetic energy | divergent kinetic energy
potential energy          | total energy
```

Each subplot includes ground truth and all optimizers, with one shared legend.

By default this figure uses spherical-harmonic spectra for real rollouts. Pass `--spherical_method fft` to generate the same combined plot with the grid FFT fallback instead.

---

## Testing Without Real Checkpoints

Use `--synthetic_demo` to test the plotting pipeline without real SWAN checkpoints:

```bash
python -m visualize forecast \
  --synthetic_demo \
  --labels Adam MUD Muon SGD RMSProp AdamW \
  --autoreg_steps 20 \
  --output_freq 5 \
  --num_ics 2 \
  --channel vorticity \
  --error_metric l2 \
  --error_mode signed \
  --grid_cols 3 \
  --rollout_dir ./demo_rollout \
  --outdir ./demo_forecast_figures
```

This creates artificial rollout data and all forecast comparison plots. It is useful for checking installation and plotting behavior.

---

## Recommended Workflow

### Step 1: Train all optimizers fairly

Use `train_all_optimizers.sh` to train the optimizer list with a fixed config and one experiment name per optimizer.

```bash
bash train_all_optimizers.sh
```

The script asks whether to clear `./logs` before training. With the default config and `--experiment.name <optimizer>`, it writes runs such as:

```text
./logs/adam/version_0
./logs/adamw/version_0
./logs/mud/version_0
./logs/muon/version_0
./logs/sgd/version_0
```

### Step 2: Plot training/validation history

```bash
python -m visualize plot_history \
  --runs ./logs/adam/version_0 ./logs/adamw/version_0 ./logs/mud/version_0 ./logs/muon/version_0 ./logs/sgd/version_0 \
  --labels Adam AdamW MUD Muon SGD \
  --stage both \
  --plot both \
  --error_metric l2 \
  --efficiency_metric both \
  --outdir ./figures_history
```

### Step 3: Plot forecast comparison

```bash
python -m visualize forecast \
  --runs ./logs/adam/version_0 ./logs/adamw/version_0 ./logs/mud/version_0 ./logs/muon/version_0 ./logs/sgd/version_0 \
  --labels Adam AdamW MUD Muon SGD \
  --config config_paradis.yaml \
  --autoreg_steps 100 \
  --output_freq 10 \
  --channel vorticity \
  --error_metric l2 \
  --error_mode signed \
  --grid_cols 3 \
  --rollout_dir ./rollout_results \
  --outdir ./figures_forecast
```

You can also run both visualization commands with:

```bash
bash visualize_all_optimizers.sh
```

That script asks whether to clear `./figures_history`, `./figures_forecast`, and `./rollout_results` before plotting.

### Step 4: Animate the rollout comparison

Once `forecast` has written rollout data to `./rollout_results`, you can
animate how each optimizer's prediction evolves over time:

```bash
python -m visualize animate \
  --labels Adam AdamW MUD Muon SGD \
  --channel vorticity \
  --output ./figures_forecast/rollout_fields.gif
```

Add `--show_error` to include a second row of signed error maps
(prediction − truth) below the prediction row.

The command also writes an animated spectral-analysis comparison by default.
With the default output paths, the two GIFs are:

```text
figures_forecast/rollout_fields.gif
figures_forecast/rollout_spectra.gif
```

By default, the spectral animation overlays all optimizers in one animated
spherical-harmonic spectral comparison graph. Use `--split_spectral` to instead
show each optimizer's saved spectral-analysis image separately.

---

# Command 3: `animate`

Use this command to create an animated GIF or MP4 from pre-computed rollout
data generated by the `forecast` command.

---

## Basic Example

```bash
python -m visualize animate \
  --labels Adam MUD Muon \
  --channel vorticity \
  --output ./figures_forecast/rollout_fields.gif
```

---

## `animate` Flags

| Flag | Required? | Default | Choices | Meaning |
|---|---:|---|---|---|
| `--labels` | yes, unless `--synthetic_demo` | synthetic defaults to `Adam MUD Muon` | strings | Optimizer labels matching subfolders in `--rollout_dir`. |
| `--rollout_dir` | no | `./rollout_results` | path | Directory containing per-optimizer rollout folders. |
| `--channel` | no | `vorticity` | `h`, `vorticity`, `divergence` | Field channel to animate. |
| `--output` | no | `./figures_forecast/rollout_fields.gif` | path | Output file for field animation (.gif or .mp4). |
| `--spectral_output` | no | derived from `--output` | path | Output file for spectral-analysis animation (.gif or .mp4). |
| `--split_spectral` | no | off | flag | Show each optimizer's saved spectral-analysis image separately. By default, all optimizers are combined in one spherical-harmonic graph. |
| `--fps` | no | `8` | integer | Frames per second. |
| `--show_error` | no | off | flag | Add a second row of signed error maps (prediction − truth). |
| `--synthetic_demo` | no | off | flag | Generate synthetic rollout data before animating. For tests only. |
| `--autoreg_steps` | no | `20` | integer | Autoregressive steps for synthetic demo. |
| `--output_freq` | no | `4` | integer | Save-every-N-steps for synthetic demo. |
| `--seed` | no | `42` | integer | Random seed for synthetic demo. |

---

## Animation Layout

The top row always shows ground truth alongside every optimizer's prediction,
all on the same shared color scale.

```text
Ground Truth | Opt 1 prediction | Opt 2 prediction | ...
```

With `--show_error` a second row is added:

```text
Opt 1 error  | Opt 2 error      | ...
```

Each error panel shows `prediction − truth` on a symmetric shared color scale.

The spatial animation has a shared title and rollout-step subtitle. The default
spectral animation also has a shared title and combines all optimizers into one
animated spherical-harmonic spectral-analysis graph.

```bash
python -m visualize animate \
  --labels Adam MUD Muon \
  --output ./figures_forecast/rollout_fields.gif \
  --spectral_output ./figures_forecast/rollout_spectra.gif
```

To show each optimizer's already-saved `spectra_*.png` image separately:

```bash
python -m visualize animate \
  --labels Adam MUD Muon \
  --split_spectral \
  --output ./figures_forecast/rollout_fields.gif \
  --spectral_output ./figures_forecast/rollout_spectra_split.gif
```

---

## Testing Without Real Rollouts

```bash
python -m visualize animate \
  --synthetic_demo \
  --labels Adam MUD Muon \
  --channel vorticity \
  --show_error \
  --output ./figures_forecast/rollout_fields.gif
```

This also writes `./figures_forecast/rollout_spectra.gif`.

---

## Troubleshooting

### `bash` is not recognized on Windows

Use Git Bash, WSL, or convert the examples to PowerShell by replacing each trailing `\` with a backtick character `` ` ``. You can also write any example as a single line.

### A run goes into `version_1` instead of a new folder

Make sure each training command uses a different experiment name:

```bash
--experiment.name adam
--experiment.name adamw
--experiment.name mud
--experiment.name muon
--experiment.name sgd
```

Lightning saves logs to:

```text
training.save_dir / experiment.name / version_k
```

### `plot_history` cannot find training curves

The original SWAN script logs training loss, not training L1/L2/W11. The toolkit searches for training loss tags such as:

```text
train_loss_epoch
train_loss
train_loss_step
```

If none exists, training plots cannot be generated.

### `forecast` cannot find checkpoints

Each run folder should contain a `checkpoints/` folder:

```text
logs/
  adam/
    version_0/
      checkpoints/
        last.ckpt
        pretrain-epoch=...-val_loss=....ckpt
```

By default, the toolkit uses:

```bash
--checkpoint_choice best
```

which selects the non-`last.ckpt` checkpoint if available. To use `last.ckpt`, pass:

```bash
--checkpoint_choice last
```
