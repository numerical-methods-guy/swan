# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About

SWAN (Shallow-Water Artificial Networks) implements the **Paradis** neural architecture for learning the shallow water equations (SWEs) on the sphere. The model runs on HPC clusters (ECCC's `ppp5`) via PBS job scripts.

- GitHub: `numerical_methods_guy/swan`
- Main branch: `main` | Active development: `avi`
- HPC repo path: `/space/hall5/sitestore/eccc/mrd/rpnatm/avg000/Toy_Dataset_Shallow_Water/swan`
- HPC Python: `/home/csu001/data/ppp5/conda_env/paradis/bin/python`

## Running training and forecasting

### Training — unified script (current)

`train_every_option.py` is the current unified training script. It supports IC mixing, rollout loss with burn-in, precomputed datasets, and multiple loss functions:

```bash
python train_every_option.py \
  --config config_paradis.yaml \
  --training.save_dir trial_test \
  --training.pretrain_epochs 10 \
  --which_loss SquaredL2 \
  --n_rollout 2 \
  --burn_in 0
```

Key flags beyond dot-notation config overrides:
- `--ic_mix <spec>` — repeatable; each spec is either `b0 b1 b2 n [scaling_scheme]` (Gaussian bells per channel) or `williamson_case2 n [scaling_scheme]` / `williamson_case6 n [scaling_scheme]`
- `--scale_all SCHEME` — global post-scale target: `unit`, `zscore`, `gbells`, `default`
- `--normalize_scheme SCHEME` — normalization source: `None` (dataset stats), `gbells`, `unit`
- `--rollout_dataset_dir PATH` — enables precomputed rollout mode; set `--rollout_dataset_mode` to `build_dataset_only` or `precomputed_training`
- `--burn_in N` — skip first N rollout steps from loss
- `--n_rollout K` — total autoregressive steps (default: `1 + config.training.nfuture`)
- `--which_loss` — `SquaredL2`, `berhu`, `amse`
- `--pretrain_ckpt PATH` — warm-start from a checkpoint

### Training — older scripts (legacy)

```bash
python avi_train_gbell_train_gbell_val_5_all_channel.py \
  --config config_paradis.yaml \
  --training.save_dir trial_test \
  --training.pretrain_epochs 10
```

### Forecasting

```bash
python avi_forecast_gbell_withfooargs_steadystate_and_Rossby_Hurwitz_final.py \
  --config config_paradis.yaml \
  --checkpoint <path/to/checkpoint.ckpt> \
  --output_dir <output_dir> \
  --autoreg_steps 40 \
  --device cuda \
  --dt_solver 25 \
  --model.paradis.hidden_dim 48 \
  --model.paradis.num_layers 8 \
  --model.paradis.num_encoder_layers 3 \
  --model.paradis.num_vels 6 \
  --model.paradis.diffusion_size 24 \
  --model.paradis.reaction_size 12 \
  --model.paradis.bias_channels 3 \
  --williamson_case6   # or --williamson_case2 or --gbells
```

### HPC submission

```bash
qsub <job_script>.pbs
```

Logs go to `pbs_logs/`. Checkpoints are saved under `trial*/` directories.

### Hyperparameter search

```bash
python optuna_paradis_sweep_v5.py  # latest version
```

## Architecture

### Paradis model (`paradis/`)

The `ParadisModel` (`paradis/paradis.py`) is a neural PDE solver with `num_layers` repeated ADR (Advection–Diffusion–Reaction) blocks:

1. **Input projection** — 5-channel input (3 SWE fields + 2 wind channels) → `hidden_dim` latent via 1×1 convolutions
2. **Per-layer ADR loop**:
   - `velocity_nets[i]` (GMBlock/SepConv) → raw velocities `(batch, 2, num_vels, nlat, nlon)`
   - `advection[i]` (NeuralSemiLagrangian) — semi-Lagrangian advection on the sphere using bicubic/bilinear interpolation with `GeoCyclicPadding`
   - `diffusion[i]` (GMBlock/SepConv) — diffusion operator
   - `reaction[i]` (GMBlock/CLinear×2) — pointwise reaction operator
   - All three contribute via residual addition
3. **Output projection** (GMBlock) → 3 physical fields

The timestep is scaled by Earth's rotation rate (`7.29212e-5`) and divided across layers.

### Key building blocks

- `paradis/blocks.py` — `GMBlock`: composable block from `CLinear` (1×1 conv) and `SepConv` (separable conv); all ops on `(batch, channels, lat, lon)` tensors
- `paradis/advection.py` — `NeuralSemiLagrangian`: projects latent to `num_vels` channels, advects via grid sampling, projects back
- `paradis/padding.py` — `GeoCyclicPadding`: periodic in longitude, symmetric at poles

### Dataset pipeline

Datasets are built by composing three wrappers:

```
PdeDatasetWithWinds          — base; random/Galewsky ICs + u,v wind channels
    ↓
GaussianBellsAllFieldsWrapperWithWinds   — IC mix scheduling; per-component scaling
    ↓
TrajectoryFromSolverWithWinds            — normalization; global scale-all
    ↓
DataLoader → SWERolloutLightningModule
```

Alternatively, `PrecomputedRolloutDataset` replaces the two inner wrappers when `--rollout_dataset_mode precomputed_training` is set (loads pre-materialized `(inp, tar_rollout)` tensors from disk).

- `pde_dataset_with_winds.py` — `PdeDatasetWithWinds`; grid must be `equiangular`
- `ic_datasets.py` — `GaussianBellsAllFieldsWrapperWithWinds`, `TrajectoryFromSolverWithWinds`, `PrecomputedRolloutDataset`
- `ic_utils.py` — CLI IC spec parsing (`parse_ic_mix_components`), Williamson IC makers, IC scaling registry (`IC_SCALING_REGISTRY`)
- `stats_utils.py` — Welford online mean/var, normalization scheme resolution
- `rollout_dataset.py` — materialize, persist, and load precomputed multi-step trajectory datasets; SHA256 config fingerprinting for integrity
- `metadata_io.py` — writes JSON audit trail (`{experiment}_ic_scaling_metadata_{timestamp}.json`) capturing CLI args, component stats, and normalization config

### IC mixing

The `--ic_mix` flag composes multiple IC sources into a flat sampling schedule. Each component is one of:
- `b0 b1 b2 n [scaling_scheme]` — Gaussian bells with per-channel modes (0 = zero, 1 = single bell, 2 = multi-bell), `n` samples, optional scaling
- `williamson_case2 n [scaling_scheme]` — steady-state geostrophic flow ICs
- `williamson_case6 n [scaling_scheme]` — Rossby–Haurwitz wave ICs

Components are deduplicated by `(kind, spec, scaling_scheme)`. Per-component scaling is applied before global `--scale_all` post-processing.

### Loss functions

- `amse_loss.py` — `AMSELoss` (Phase 3): spectral loss in spherical harmonic space separating amplitude error from decorrelation. Reference: Subich et al. 2025.
- `reverse_huber_loss.py` — Reverse Huber / Berhu loss (has known bugs, see README2)
- `torch_harmonics.examples.losses` — `SquaredL2LossS2`, `L2LossS2`, `L1LossS2`, `W11LossS2` (Phases 1–2)

### Training phases

| Phase | Loss | Script |
|-------|------|--------|
| 1 | L2/Berhu, 1-step | `avi_train_gbell_train_gbell_val_*.py` (legacy) |
| 2 | Rollout loss with burn-in | `train_every_option.py` with `--n_rollout >1 --burn_in N` |
| 3 | AMSE, longer lead times | `train_every_option.py` with `--which_loss amse` |

"Burn-in" discards the first N autoregressive steps from the loss, keeping only the later (harder) steps.

### Lightning module (`train_every_option.py`)

`SWERolloutLightningModule` handles the training loop:
- `_rollout_and_collect()` — K-step autoregressive rollout using the PDE solver, computes loss + wind MSE + optional eval metrics
- `_rollout_and_collect_from_saved()` — same but reads pre-materialized targets
- `WilliamsonRolloutCallback` — runs Williamson case 2/6 autoreg eval at the end of each validation epoch, logs `wc2_rollout_loss` and `wc6_rollout_loss`
- Optimizer: Adam + ReduceLROnPlateau (patience=5, factor=0.5)
- ModelCheckpoint monitors `val_loss`, `wc2_rollout_loss`, `wc6_rollout_loss`

### Config

`config_paradis.yaml` is the canonical config. Any field can be overridden at the command line using dot-notation: `--data.nlat 64`, `--model.paradis.hidden_dim 32`, etc. The `update_config_from_args` helper handles this in all training/forecast scripts.

### Checkpoint directories

Active trials:
- `trial2_24feb`, `trial3_3march`, `trial3_3march_berhu`, `trial4_4march`, `trial_optuna`

Checkpoint filenames encode hyperparameters, e.g. `rolloutloss-epoch=20-val_loss=0.0242.ckpt`.

## Git

```bash
git pull origin avi   # pull latest dev branch
```

Token is already configured in the remote.
