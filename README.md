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
pip install torch pytorch-lightning torch-harmonics pyyaml numpy pandas matplotlib
```

---

## Training

```bash
python train.py --config config_paradis.yaml
```

To use the Muon optimizer variant:

```bash
python train_muon.py --config config_paradis.yaml
```

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

Example in `config_paradis.yaml`:

## Loss Functions

| Name | Description |
|------|-------------|
| `reversed_huber` | Linear for small errors, quadratic for large (default) |
| `mse` | Mean squared error |
| `mae` | Mean absolute error |
| `amse` | Adjusted MSE via spherical harmonic decomposition ([Subich et al., 2025](https://doi.org/10.1002/qj.4884)) |

All losses support latitude weighting and per-variable weighting.

---

## Citation

If you use SWAN or build on the PARADIS architecture, please cite:

```
@article{paradis2025,
  title  = {PARADIS: A Physics-Informed Neural Weather Prediction Model},
  url    = {https://arxiv.org/abs/2601.21151},
  year   = {2025}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
This product bundles modified code from torch-harmonics, which is available under a BSD-3-Clause license.
