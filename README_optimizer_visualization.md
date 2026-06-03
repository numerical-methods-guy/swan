# Optimizer Visualization Tools for SWAN

This toolkit compares different optimizer runs in the SWAN project.

It provides one public user-facing script:

```text
visualize.py
```

and two backend helper modules:

```text
history_utils.py
rollout_utils.py
```

Users should normally interact only with `visualize.py`.

---

## 1. File Overview

### `visualize.py`

Public command-line interface and plotting code.

It provides two commands:

```bash
python visualize.py plot_history ...
python visualize.py forecast ...
```

Use `plot_history` for TensorBoard training/validation curves.

Use `forecast` for post-training rollout comparison from trained checkpoints.

---

### `history_utils.py`

Backend for `plot_history`.

It handles:

- finding TensorBoard event files;
- reading TensorBoard scalar histories;
- reading CSV fallback histories if TensorBoard is unavailable;
- mapping user-friendly metric names such as `l2` to real TensorBoard tags such as `val_l2`;
- preparing learning-curve and hitting-curve data.

Users do not run this file directly.

---

### `rollout_utils.py`

Backend for `forecast`.

It handles:

- finding checkpoints inside training run folders;
- running or preparing rollout evaluation;
- saving per-optimizer rollout outputs;
- saving per-step forecast metrics;
- loading saved forecast fields;
- preparing spectra and grid-wise comparison data.

Users do not run this file directly.

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
results/
  adam/
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
./results/adam/version_0
./results/mud/version_0
./results/muon/version_0
```

The labels passed to `--labels` should be in the same order:

```text
Adam MUD Muon
```

The number of paths passed to `--runs` must match the number of labels passed to `--labels`.

The tools support any number of optimizers, not just three.

For example, this is valid:

```text
--runs ./results/adam/version_0 ./results/adamw/version_0 ./results/mud/version_0 ./results/muon/version_0 ./results/sgd/version_0
--labels Adam AdamW MUD Muon SGD
```

---

## 5. Recommended Fair Training Setup

> **Terminal line-continuation note:** The examples below use PowerShell line continuation with the backtick character `` ` ``.  
> If you use **macOS/Linux/Git Bash/WSL**, replace each trailing `` ` `` with `\`.  
> If you use **Windows Command Prompt (`cmd.exe`)**, replace each trailing `` ` `` with `^`.  
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

Example PowerShell commands:

```powershell
python path/to/your_train_script_for_adam.py `
  --config config_paradis.yaml `
  --experiment.seed 42 `
  --experiment.name adam `
  --training.save_dir ./results `
  --data.num_workers 0

python path/to/your_train_script_for_mud.py `
  --config config_paradis.yaml `
  --experiment.seed 42 `
  --experiment.name mud `
  --training.save_dir ./results `
  --data.num_workers 0

python path/to/your_train_script_for_muon.py `
  --config config_paradis.yaml `
  --experiment.seed 42 `
  --experiment.name muon `
  --training.save_dir ./results `
  --data.num_workers 0
```

For more optimizers, repeat the same pattern with a unique `--experiment.name` for each run.

In PowerShell, the backtick character `` ` `` is the line-continuation character. It must be the final character on the line.

---



# Command 1: `plot_history`

Use this command to compare TensorBoard training/validation histories.

---

## Basic Example

```powershell
python visualize.py plot_history `
  --runs ./results/adam/version_0 ./results/mud/version_0 ./results/muon/version_0 `
  --labels Adam MUD Muon `
  --stage validation `
  --plot both `
  --error_metric l2 `
  --efficiency_metric both `
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

```powershell
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

Example:

```powershell
python visualize.py plot_history `
  --runs ./results/adam/version_0 ./results/mud/version_0 ./results/muon/version_0 `
  --labels Adam MUD Muon `
  --stage validation `
  --plot learning_curve `
  --error_metric l2 `
  --efficiency_metric both `
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

```powershell
python visualize.py plot_history `
  --runs ./results/adam/version_0 ./results/mud/version_0 ./results/muon/version_0 `
  --labels Adam MUD Muon `
  --stage validation `
  --plot hitting_curve `
  --error_metric l2 `
  --efficiency_metric both `
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

```powershell
python visualize.py plot_history `
  --runs ./results/adam/version_0 ./results/mud/version_0 ./results/muon/version_0 `
  --labels Adam MUD Muon `
  --stage both `
  --plot both `
  --error_metric l2 `
  --efficiency_metric both `
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

```powershell
python visualize.py forecast `
  --runs ./results/adam/version_0 ./results/mud/version_0 ./results/muon/version_0 `
  --labels Adam MUD Muon `
  --config config_paradis.yaml `
  --autoreg_steps 100 `
  --output_freq 10 `
  --channel vorticity `
  --rollout_dir ./rollout_results `
  --outdir ./figures_forecast
```

Typical outputs:

```text
figures_forecast/
  forecast_error_curve_l2.png
  forecast_accuracy_bar_l2.png
  forecast_speedup_bar.png
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
| `--error_mode` | no | `signed` | `signed`, `abs`, `squared` | Pointwise error map mode. |
| `--summary_step` | no | `final` | `final`, `latest`, or integer | Rollout step loaded for grid and spectra plots. |
| `--grid_cols` | no | `3` | integer | Maximum number of columns in spatial grid figures. |
| `--rollout_dir` | no | `./rollout_results` | path | Folder for per-optimizer rollout outputs. |
| `--outdir` | no | `./figures_forecast` | path | Folder for final comparison figures. |
| `--device` | no | auto | `cpu`, `cuda`, etc. | Optional device for real rollout. |
| `--synthetic_demo` | no | off | flag | Generate artificial rollout data for testing only. |

---

## Important Forecast Concepts

### `--autoreg_steps`

Controls how far the model is rolled out autoregressively.

For example:

```powershell
--autoreg_steps 100
```

means the model is applied 100 times. Each prediction is fed back into the model as the next input.

### `--output_freq`

Controls how often rollout snapshots are saved.

For example:

```powershell
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

```powershell
--summary_step final
--summary_step latest
--summary_step 50
```

`final` and `latest` both select the latest common saved step across all rollout folders. An integer selects that exact saved step.

The generated filenames currently keep the suffix `final`, even when `--summary_step` is an integer. The figure titles report the actual selected step.

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

```powershell
--error_metric l2
```

### 2. Aggregate Accuracy Bar Chart

```text
forecast_accuracy_bar_l2.png
```

Plots one aggregate forecast error value per optimizer, usually averaged over rollout steps and initial conditions.

This is a bar chart, not a histogram.

### 3. Speedup Bar Chart

```text
forecast_speedup_bar.png
```

Plots:

```text
speedup_mean = solver_time_mean / ml_time_mean
```

Higher is better.

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

```powershell
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

---

## Testing Without Real Checkpoints

Use `--synthetic_demo` to test the plotting pipeline without real SWAN checkpoints:

```powershell
python visualize.py forecast `
  --synthetic_demo `
  --labels Adam MUD Muon SGD RMSProp AdamW `
  --autoreg_steps 20 `
  --output_freq 5 `
  --num_ics 2 `
  --channel vorticity `
  --error_metric l2 `
  --error_mode signed `
  --grid_cols 3 `
  --rollout_dir ./demo_rollout `
  --outdir ./demo_forecast_figures
```

This creates artificial rollout data and all forecast comparison plots. It is useful for checking installation and plotting behavior.

---

## Recommended Workflow

### Step 1: Train all optimizers fairly

Use your own optimizer training scripts. The script names do not matter. The important part is to keep the configuration fixed and give each run a unique experiment name.

```powershell
python path/to/your_train_script_for_adam.py `
  --config config_paradis.yaml `
  --experiment.seed 42 `
  --experiment.name adam `
  --training.save_dir ./results `
  --data.num_workers 0

python path/to/your_train_script_for_mud.py `
  --config config_paradis.yaml `
  --experiment.seed 42 `
  --experiment.name mud `
  --training.save_dir ./results `
  --data.num_workers 0

python path/to/your_train_script_for_muon.py `
  --config config_paradis.yaml `
  --experiment.seed 42 `
  --experiment.name muon `
  --training.save_dir ./results `
  --data.num_workers 0
```

For five or six optimizers, simply add more training commands with different experiment names, such as `adamw`, `sgd`, `rmsprop`, or any custom optimizer name.

### Step 2: Plot training/validation history

```powershell
python visualize.py plot_history `
  --runs ./results/adam/version_0 ./results/mud/version_0 ./results/muon/version_0 `
  --labels Adam MUD Muon `
  --stage validation `
  --plot both `
  --error_metric l2 `
  --efficiency_metric both `
  --outdir ./figures_history
```

### Step 3: Plot forecast comparison

```powershell
python visualize.py forecast `
  --runs ./results/adam/version_0 ./results/mud/version_0 ./results/muon/version_0 `
  --labels Adam MUD Muon `
  --config config_paradis.yaml `
  --autoreg_steps 100 `
  --output_freq 10 `
  --channel vorticity `
  --error_metric l2 `
  --error_mode signed `
  --grid_cols 3 `
  --rollout_dir ./rollout_results `
  --outdir ./figures_forecast
```

---

## Troubleshooting

### `bash` is not recognized on Windows

You are using PowerShell. Run commands directly in PowerShell using backticks for line continuation, or use one-line commands.

### A run goes into `version_1` instead of a new folder

Make sure each training command uses a different experiment name:

```powershell
--experiment.name adam
--experiment.name mud
--experiment.name muon
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
results/
  adam/
    version_0/
      checkpoints/
        last.ckpt
        pretrain-epoch=...-val_loss=....ckpt
```

By default, the toolkit uses:

```powershell
--checkpoint_choice best
```

which selects the non-`last.ckpt` checkpoint if available. To use `last.ckpt`, pass:

```powershell
--checkpoint_choice last
```

