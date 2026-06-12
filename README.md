# SWAN — Shallow-Water Artificial Network

SWAN is a lightweight research platform for machine learning-based numerical weather prediction (MLNWP). It implements a shallow-water version of the [PARADIS](https://arxiv.org/abs/2601.21151) architecture, providing a compact environment for fast experimentation with modern ML forecasting methods.

The [shallow water equations](https://en.wikipedia.org/wiki/Shallow_water_equations) are often called the "Swiss army knife" of meteorology: compact and useful for prototyping atmospheric dynamics. Modern MLNWP systems, by contrast, are large and costly to train. SWAN bridges this gap — retaining key characteristics of production-grade ML forecasting while remaining simple enough for rapid research iteration.

## Contents

- [Architecture](#architecture)
- [Project layout](#project-layout)
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [Hyperparameter tuning](#hyperparameter-tuning)
- [Visualization](#visualization)
- [Configuration](#configuration)
- [Loss functions](#loss-functions)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

---

## Architecture

SWAN adapts PARADIS to the shallow water setting. The model operates on three prognostic fields — geopotential height, vorticity, and divergence — on a global equiangular grid.

Each forward step applies `num_layers` physics-informed latent updates, each consisting of:

- **Neural semi-Lagrangian advection** — learned velocities drive a rotated-coordinate grid-sample interpolation
- **Learned diffusion** — separable convolutions acting on a (optionally downsampled) latent state
- **Pointwise reaction** — channel-wise nonlinearity as the primary forcing term

Physical winds (u, v) are extracted from vorticity/divergence via inverse vector SHT and fed as additional inputs alongside the three prognostic fields.

The spherical harmonic transform is provided by [torch-harmonics](https://github.com/NVIDIA/torch-harmonics).

---

## Project layout

```text
swan/
  model/                  # PARADIS model (advection, blocks, padding)
  training/               # Lightning module, datasets, optimizer strategies
  visualize/              # Post-training history, rollout, and animation tools
  utils/                  # Loss functions (reversed Huber, AMSE)
  train.py                # Main training entry point
  forecast.py             # Autoregressive inference and spectral analysis
  optuna_tune.py          # Hyperparameter search
  config_paradis.yaml     # Default experiment configuration
  optuna_search_spaces.yaml
  docs/                   # Detailed guides (see Documentation below)
```

---

## Installation

Requires **Python 3.10–3.12**.

**Linux / WSL**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
sudo apt install -y build-essential   # needed for torch-harmonics
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, compiling `torch-harmonics` from source requires a C++ build toolchain (Visual Studio Build Tools or WSL). WSL is the recommended path for GPU training.

`requirements.txt` pins `torch==2.9.1` and builds [torch-harmonics](https://github.com/NVIDIA/torch-harmonics) from source (`v0.9.1`). Do **not** install the PyPI wheel alone — prebuilt wheels are compiled against a different PyTorch ABI and fail with `undefined symbol` import errors.

For NVIDIA GPU, install a matching CUDA build of PyTorch first, then run `pip install -r requirements.txt` without the CPU `--extra-index-url` (edit or use a separate constraints file).

---

## Training

```bash
python train.py --config config_paradis.yaml
```

Choose an optimizer via config (`training.optimizer`) or CLI:

```bash
python train.py --config config_paradis.yaml --optimizer adamw
python train.py --config config_paradis.yaml --optimizer muon
python train.py --config config_paradis.yaml --optimizer mud_new
python train.py --config config_paradis.yaml --optimizer gauss_newton
```

Supported optimizer strategies are `adam`, `adamw`, `gauss_newton`, `mud`, `mud_new`, `muon`, `muon_new`, and `sgd`. Gauss-Newton supports `matrix_free` and `explicit` methods under `training.gauss_newton`; the explicit method is intended for tiny debugging runs.

Config values can be overridden from the command line using dot notation:

```bash
python train.py --config config_paradis.yaml --model.paradis.hidden_dim 64 --training.pretrain_epochs 50
```

### Pretraining and finetuning

Training runs in two optional phases controlled by `training.pretrain_epochs` and `training.finetune_epochs` in the config. During finetuning, the rollout horizon is extended and a separate learning rate is used.

To skip pretraining and resume from a checkpoint for finetuning:

```bash
python train.py --config config_paradis.yaml \
  --resume_from logs/spherical_swe_paradis/version_0/checkpoints/last.ckpt \
  --training.finetune_epochs 10
```

Checkpoints are saved under `logs/<experiment_name>/`. TensorBoard logging is enabled by default; optional MLflow logging is available via `logging.mlflow.enabled` in the config.

For the optimizer sweep and cluster wrapper bash scripts, see [docs/bash_scripts.md](docs/bash_scripts.md).

---

## Inference

Run autoregressive rollout from a trained checkpoint:

```bash
python forecast.py \
  --config config_paradis.yaml \
  --checkpoint logs/spherical_swe_paradis/version_0/checkpoints/last.ckpt \
  --autoreg_steps 100 \
  --ic_type random \
  --num_ics 4 \
  --output_dir ./results
```

Key flags:

| Flag | Description |
|------|-------------|
| `--autoreg_steps` | Number of autoregressive steps (default: 100 for random, 6 days for Galewsky) |
| `--num_ics` | Number of random ICs to average over |
| `--ic_type` | `random` or `galewsky` |
| `--output_freq` | Save plots/tensors every N steps |
| `--no_plots` | Disable plot generation |
| `--spectral_analysis` | Compute and save energy spectra (enabled by default) |

Outputs include per-step comparison plots, energy spectra, saved field tensors, and a `metrics.csv` with L1/L2/W11 errors and ML vs. solver timing/speedup.

---

## Hyperparameter tuning

SWAN includes an Optuna-based tuner that searches optimizer hyperparameters without modifying `config_paradis.yaml`:

```bash
python optuna_tune.py --config config_paradis.yaml --optimizer sgd \
  --search_space optuna_search_spaces.yaml \
  --n_trials 40 \
  --study_name sgd_tuning
```

Results are saved under `optuna_results/`. See [docs/optuna.md](docs/optuna.md) for search-space format, pruning, and study management.

---

## Visualization

SWAN has two related visualization paths:

1. **`python -m visualize`** — compare multiple optimizer runs (TensorBoard histories, shared rollouts, grouped figures).
2. **`forecast.py`** — single-checkpoint inference with per-step plots, spectra, and `metrics.csv` (see [Inference](#inference)).

All comparison tooling lives in the `visualize` package:

```bash
python -m visualize <command> [options]
```

### Commands

| Command | Purpose | Typical outputs |
|---------|---------|-----------------|
| `plot_history` | Training/validation curves from TensorBoard or CSV fallbacks | `learning_curve_*`, `hitting_curve_*` under `--outdir` (default `./figures_history`) |
| `forecast` | Roll out checkpoints and build cross-optimizer comparison figures | Error curves/bars, spatial grids, combined spectra, optional skill/spectral horizon diagnostics under `--outdir` (default `./figures_forecast`); per-optimizer tensors under `--rollout_dir` |
| `animate` | GIF/MP4 from pre-computed rollouts | Field, error, and spectral animations (`--show_error`, `--split_spectral` for per-optimizer spectra) |

Minimal examples:

```bash
python -m visualize plot_history \
  --runs ./logs/adam/version_0 ./logs/muon/version_0 \
  --labels Adam Muon \
  --stage both --plot both --outdir ./figures_history

python -m visualize forecast \
  --runs ./logs/adam/version_0 ./logs/muon/version_0 \
  --labels Adam Muon \
  --config config_paradis.yaml \
  --rollout_dir ./rollout_results \
  --outdir ./figures_forecast

python -m visualize animate \
  --rollout_dir ./rollout_results \
  --labels Adam Muon \
  --output ./figures_forecast/rollout_fields.gif
```

Use `--reuse_rollouts` with `forecast` to replot different optimizer subsets from existing rollout folders without rerunning models. Optional diagnostics: `--skill_horizon`, `--spectral_horizon`.

### Bash wrappers

| Script | Role |
|--------|------|
| [`visualize_all_optimizers.sh`](visualize_all_optimizers.sh) | Local sweep: history + forecast + animations for predefined optimizer groups under `./logs/`; writes `figures_history/`, `figures_forecast/`, `rollout_results/`, and optional `focused_animations/` |
| [`run_cluster_visualization.sh`](run_cluster_visualization.sh) | SLURM/non-interactive wrapper around training + `visualize_all_optimizers.sh` |
| [`visualize_two_group_optimizers.sh`](visualize_two_group_optimizers.sh) | Grouped forecast/history plots from saved checkpoints (used by the group wrapper below) |
| [`run_forecast_visualization_groups.sh`](run_forecast_visualization_groups.sh) | Cluster/local wrapper for comparing saved models in `swan_checkpoints/` across groups such as `all_optimizers`, `without_spectral_blow_up`, `without_spatial_blow_up`, and `stable_core` |

`train_all_optimizers.sh` trains the optimizer list first; pair it with `visualize_all_optimizers.sh` for the full local workflow.

Full flag reference, output file names, and cluster settings: [docs/visualize.md](docs/visualize.md), [docs/bash_scripts.md](docs/bash_scripts.md), and [docs/forecast_group_visualization.md](docs/forecast_group_visualization.md).

---

## Configuration

The main config file is `config_paradis.yaml`. It defines experiment metadata, grid resolution, model architecture, loss function, optimizer hyperparameters, and logging options.

For a full reference of all default values — optimizer, loss function, scheduler, model architecture, and Optuna search spaces — see [docs/training_defaults.md](docs/training_defaults.md).

---

## Loss functions

| Name | Description |
|------|-------------|
| `reversed_huber` | Linear for small errors, quadratic for large (default) |
| `mse` | Mean squared error |
| `mae` | Mean absolute error |
| `amse` | Adjusted MSE via spherical harmonic decomposition ([Subich et al., 2025](https://arxiv.org/abs/2501.19374)) |

All losses support latitude weighting and per-variable weighting.

---

## Documentation

| Guide | Description |
|-------|-------------|
| [docs/training_defaults.md](docs/training_defaults.md) | Default config values and optimizer parameters |
| [docs/bash_scripts.md](docs/bash_scripts.md) | Optimizer sweep and cluster bash scripts |
| [docs/optuna.md](docs/optuna.md) | Hyperparameter tuning with Optuna |
| [docs/visualize.md](docs/visualize.md) | History, rollout, and animation tools |
| [docs/forecast_group_visualization.md](docs/forecast_group_visualization.md) | Grouped forecast comparison on saved checkpoints |

---

## Citation

If you use SWAN or build on the PARADIS architecture, please cite:

```
@article{pereira2026learning,
  title={Learning to Advect: A Neural Semi-Lagrangian Architecture for Weather Forecasting},
  author={Pereira, Carlos A and Gaudreault, St{\'e}phane and Dallerit, Valentin and Subich, Christopher and Panday, Shoyon and Wei, Siqi and Zhang, Sasa and Rout, Siddharth and Haber, Eldad and Spiteri, Raymond J and others},
  journal={arXiv preprint arXiv:2601.21151},
  year={2026}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
This product bundles modified code from torch-harmonics, which is available under a BSD-3-Clause license.
