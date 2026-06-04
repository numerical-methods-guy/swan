# SWAN — Shallow-Water Artificial Network

SWAN is a lightweight research platform for machine learning-based numerical weather prediction (MLNWP). It implements a shallow-water version of the [PARADIS](https://arxiv.org/abs/2601.21151) architecture, providing a compact environment for fast experimentation with modern ML forecasting methods.

The [shallow water equations](https://en.wikipedia.org/wiki/Shallow_water_equations) are often called the "Swiss army knife" of meteorology: compact and useful for prototyping atmospheric dynamics. Modern MLNWP systems, by contrast, are large and costly to train. SWAN bridges this gap — retaining key characteristics of production-grade ML forecasting while remaining simple enough for rapid research iteration.

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

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Linux, `build-essential` is required so [torch-harmonics](https://github.com/NVIDIA/torch-harmonics) can compile against your PyTorch install. Pin `torch==2.9.1` and build torch-harmonics from source (see `requirements.txt`); the PyPI wheel alone often breaks with `undefined symbol` import errors on newer PyTorch versions.

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
python train.py --config config_paradis.yaml --optimizer gauss_newton
```

Supported optimizer strategies are `adam`, `adamw`, `gauss_newton`, `mud`, `muon`, and `sgd`. Muon requires a PyTorch build that provides `torch.optim.Muon`. Gauss-Newton supports `matrix_free` and `explicit` methods under `training.gauss_newton`; the explicit method is intended for tiny debugging runs.

Config values can be overridden from the command line using dot notation:

```bash
python train.py --config config_paradis.yaml --model.paradis.hidden_dim 64 --training.pretrain_epochs 50
```

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
| `--spectral_analysis` | Compute and save energy spectra |

Outputs include per-step comparison plots, energy spectra, saved field tensors, and a `metrics.csv` with L1/L2/W11 errors and ML vs. solver timing/speedup.

## Configuration

Example in `config_paradis.yaml`. For a full reference of all default values — optimizer, loss function, scheduler, model architecture, and Optuna search spaces — see [docs/training_defaults.md](docs/training_defaults.md).

## Loss Functions

| Name | Description |
|------|-------------|
| `reversed_huber` | Linear for small errors, quadratic for large (default) |
| `mse` | Mean squared error |
| `mae` | Mean absolute error |
| `amse` | Adjusted MSE via spherical harmonic decomposition ([Subich et al., 2025](https://arxiv.org/abs/2501.19374)) |

All losses support latitude weighting and per-variable weighting.

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
