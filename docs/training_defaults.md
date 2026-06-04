# SWAN Training Defaults

This document describes the default configuration used when running the PARADIS model on the shallow water equations (SWE) benchmark. All values are sourced from [`config_paradis.yaml`](../config_paradis.yaml) and the strategy files in [`training/strategies/`](../training/strategies/).

---

## Loss Function

**Default:** `reversed_huber` (δ = 1.0)

Defined in [`training/loss.py`](../training/loss.py) and implemented in [`utils/loss.py`](../utils/loss.py).

The loss applies three layers of weighting before reducing to a scalar:

1. **Variable weights** — equal weight (1.0) for all three SWE output channels: geopotential height (`h`), vorticity, and divergence.
2. **Pressure weights** — for SWE there is only one pseudo-level (1000 hPa), so this is effectively 1.0.
3. **Latitude weights** — GraphCast-style cosine weighting so that grid cells near the equator (which cover more physical area) contribute more to the loss than polar cells.

The reversed Huber loss itself applies:
- **Linear** penalty when `|error| ≤ δ` → `δ · |error|`
- **Quadratic** penalty when `|error| > δ` → `(error² + δ²) / (2δ)`

This is the inverse of the standard Huber loss: it is lenient on large errors and strict on small ones, which tends to prevent the model from ignoring outliers.

**Other available options** (set via `loss.loss_function` in config):

| Value | Function |
|---|---|
| `reversed_huber` | **(default)** smooth L1/L2 blend as described above |
| `mse` | Mean Squared Error (`torch.nn.MSELoss`) |
| `mae` | Mean Absolute Error (`torch.nn.L1Loss`) |
| `amse` | Area-weighted MSE via `utils/amse_loss.py` |

---

## Optimizer

**Default:** `Adam`

Set via `training.optimizer: adam` in [`config_paradis.yaml`](../config_paradis.yaml).  
Implemented in [`training/strategies/adam.py`](../training/strategies/adam.py).

| Parameter | Default | Config key |
|---|---|---|
| Learning rate (pretrain) | `1e-2` | `training.learning_rate` |
| Learning rate (finetune) | `1e-3` | `training.finetune_learning_rate` |
| β₁ | `0.9` | `training.beta1` |
| β₂ | `0.999` | `training.beta2` |
| ε | `1e-8` | `training.epsilon` |
| Weight decay | `0.0` | `training.adam_weight_decay` |

The finetune learning rate is used automatically when `nfuture > 0` (multi-step rollout fine-tuning phase).

**Other available optimizers** (set via `training.optimizer`):

| Value | File | Notes |
|---|---|---|
| `adam` | `strategies/adam.py` | **(default)** |
| `adamw` | `strategies/adamW.py` | Adam with decoupled weight decay (default `0.01`) |
| `sgd` | `strategies/sgd.py` | SGD with momentum (default `0.9`) |
| `muon` | `strategies/muon.py` | Muon optimizer with internal AdamW for non-matrix params |
| `mud` | `strategies/mud.py` | MUD optimizer (momentum-based adaptive) |
| `gauss_newton` | `strategies/gauss_newton.py` | Second-order method via conjugate gradients |

---

## Learning Rate Scheduler

The Adam strategy selects a scheduler based on which keys are present in the config. The priority order is:

1. **`CosineAnnealingLR`** — used when `cosine_eta_min` is set *(this is the current default)*.
2. **`MultiStepLR`** — used when `lr_milestones` is set and `cosine_eta_min` is absent.
3. **`ReduceLROnPlateau`** — fallback when neither key is present.

**Current default config activates CosineAnnealingLR:**

| Parameter | Value | Config key |
|---|---|---|
| Schedule type | `CosineAnnealingLR` | `training.cosine_eta_min` present |
| `T_max` (period in epochs) | `5` (= `pretrain_epochs`) | `training.pretrain_epochs` |
| Minimum LR (`eta_min`) | `1e-6` | `training.cosine_eta_min` |

The config also has `lr_milestones: [10, 16]` and `lr_gamma: 0.5` set, but these are **not active** as long as `cosine_eta_min` is present — `CosineAnnealingLR` takes priority.

To switch schedulers, remove or comment out `cosine_eta_min` (and optionally `lr_milestones`) in the config.

All strategies (Adam, AdamW, SGD, Muon, MUD) share the same scheduler priority logic. The `ReduceLROnPlateau` fallback uses `factor=0.5, patience=5, monitor=val_loss` across all strategies.

---

## Model Architecture

Defined under `model.paradis` in [`config_paradis.yaml`](../config_paradis.yaml).

| Parameter | Default | Description |
|---|---|---|
| `hidden_dim` | `32` | Channel width throughout the network |
| `num_layers` | `4` | Number of ADR (Advection-Diffusion-Reaction) blocks |
| `num_vels` | `6` | Number of learned velocity vector fields for advection |
| `base_dt` | `60` s | Must match `data.dt`; sets the physics timestep |
| `interpolation` | `bicubic` | Advection interpolation method (`bilinear` or `bicubic`) |
| `bias_channels` | `2` | Channels for the global bias correction term |
| `activation` | `SiLU` | Nonlinearity (`SiLU` or `GELU`) |

**Sub-block structure** (the physblock layers within each ADR layer):

| Block | Layers | Hidden dim |
|---|---|---|
| `input_proj` | SepConv → CLinear | 32 |
| `velocity_net` | SepConv | 32 |
| `diffusion` | SepConv | 16 |
| `reaction` | CLinear → CLinear | 16 |
| `output_proj` | SepConv → CLinear | 32 |
| `advection.down_projection` | CLinear | — (pass-through) |
| `advection.up_projection` | SepConv | — (pass-through) |

- `SepConv`: depthwise-separable convolution, default kernel size 3
- `CLinear`: channel-wise linear (1×1 convolution)

Weight initialization: Conv2D layers use **Kaiming normal** (He initialization with ReLU nonlinearity). GlobalBias layers use **Normal(0, 1e-3)**.

---

## Data & Training Setup

| Parameter | Default | Config key |
|---|---|---|
| Grid | `equiangular` 32×64 | `data.grid`, `data.nlat`, `data.nlon` |
| Timestep | `60` s | `data.dt` |
| Batch size | `4` | `data.batch_size` |
| Training examples | `512` | `data.num_train_examples` |
| Validation examples | `64` | `data.num_val_examples` |
| Pretrain epochs | `5` | `training.pretrain_epochs` |
| Finetune epochs | `0` | `training.finetune_epochs` |
| Rollout steps (`nfuture`) | `1` | `training.nfuture` |
| Random seed | `42` | `experiment.seed` |
| Mixed precision | `none` (fp32) | `training.amp_mode` |
| Gradient checkpointing | `false` | `training.gradient_checkpointing` |
| Checkpoint metric | `val_loss` (min) | hardcoded in `train.py` |

---

## Hyperparameter Optimization with Optuna

Search spaces are defined in [`optuna_search_spaces.yaml`](../optuna_search_spaces.yaml). These are the bounds Optuna explores — not the defaults above.

**Architecture parameters (tuned for all optimizers):**

| Parameter | Type | Search space |
|---|---|---|
| `hidden_dim` | categorical | {16, 32, 64} |
| `num_layers` | int | [1, 4] |
| `num_vels` | categorical | {2, 4, 6, 8} |
| `bias_channels` | categorical | {0, 1, 2, 4, 8} |

**Optimizer-specific search spaces:**

| Optimizer | Parameter | Range |
|---|---|---|
| Adam | LR | [1e-5, 1e-2] log |
| Adam | β₁ | [0.8, 0.99] |
| Adam | β₂ | [0.9, 0.9999] |
| Adam | weight decay | {0, 1e-6, 1e-5, 1e-4, 1e-3} |
| AdamW | LR | [1e-5, 1e-2] log |
| AdamW | weight decay | {0, 1e-6, …, 1e-2} |
| SGD | LR | [1e-4, 1e-1] log |
| SGD | momentum | [0.0, 0.99] |
| Muon | Muon LR | [1e-5, 1e-1] log |
| Muon | AdamW LR | [1e-5, 1e-2] log |
| Muon | momentum | [0.8, 0.99] |
| MUD | LR | [1e-5, 1e-2] log |
| MUD | β (MUD) | [0.8, 0.99] |
| MUD | passes | [1, 3] |
| Gauss-Newton | damping | [1e-5, 1e-1] log |
| Gauss-Newton | step size | [0.1, 1.0] |
| Gauss-Newton | CG iterations | [1, 5] |
