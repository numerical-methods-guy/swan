# swan
Shallow-Water Artificial Network (SWAN)

## Repository Structure

### `dataset/`

- `shallow_water_solver.py` — spectral shallow water equations solver on the sphere (`ShallowWaterSolver(nn.Module)`)
  - **Variables:** `dt`, `nlat`, `nlon`, `grid`, `lmax`, `mmax`, `sht`, `isht`, `vsht`, `ivsht`
  - **Buffers:** `lats`, `lons`, `l`, `lap`, `invlap`, `coriolis`, `hyperdiff`, `quad_weights`, `radius`, `omega`, `gravity`, `havg`, `hamp`
  - **Module-level:** `_great_circle_distance(lat, lon, lat0, lon0)` — great-circle angular distance (radians) between a grid and a centre point, used for Gaussian bell placement
  - **Methods:** `grid2spec`, `spec2grid`, `vrtdivspec`, `getuv`, `gethuv`, `dudtspec`, `timestep`, `random_initial_condition`, `galewsky_initial_condition`, `williamson_case2_initial_condition`, `gaussian_bells_initial_condition`, `precomputed_initial_condition`, `potential_vorticity`, `dimensionless`, `integrate_grid`, `plot_griddata`, `plot_specdata`
  - `williamson_case2_initial_condition(alpha=None, gh0=29400, u0=None)` — steady-state balanced geostrophic flow tilted at angle `alpha`; if `alpha=None`, samples from U(0, π/2) each call so each sample has a different tilt; returns spectral state (3, lmax, mmax)
  - `gaussian_bells_initial_condition(ref_mean, ref_std, k_min, k_max, sigma_min_deg, sigma_max_deg, signed, mean_scale, std_scale)` — places K random Gaussian bumps per channel on the sphere via great-circle distance; normalizes each channel to zero mean/unit std, then scales by `mean_scale * ref_mean + std_scale * ref_std * bump`; returns spectral state (3, lmax, mmax)

- `shallow_water_pde_dataset.py` — on-the-fly dataset generating (input, target) pairs using the solver (`ShallowWaterPDEDataset(Dataset)`)
  - **Variables:** `dtype`, `num_examples`, `device`, `stream`, `rank`, `nlat`, `nlon`, `nsteps`, `normalize`, `solver`, `inp_shape`, `tar_shape`, `inp_mean`, `inp_var`, `ictype`, `precomputed_folder`, `gbells_kwargs`, `gbells_ref_mean`, `gbells_ref_std`
  - **Methods:** `__len__`, `__getitem__`, `set_initial_condition`, `set_num_examples`, `_compute_gbells_ref_stats`, `_get_sample`
  - `_compute_gbells_ref_stats(n_samples)` — draws `n_samples` random ICs and computes per-channel mean and std for Gaussian bell scaling
  - `_get_sample` — supports `random`, `galewsky`, `williamson_case2`, `gbells`, and `precomputed` ictypes; for precomputed, loads `{index}_0.pt` as input and `{index}_1.pt` as target (falls back to `solver.timestep` if target file is missing)

- `pde_dataset_with_winds.py` — wraps base dataset to also return physical wind components (u, v) (`PdeDatasetWithWinds(Dataset)`)
  - **Variables:** `base_dataset`, `solver`, `nlat`, `nlon`, `grid`, `nsteps`, `normalize`, `device`, `ictype`, `precomputed_folder`, `inp_mean`, `inp_var`, `wind_mean`, `wind_var`
  - **Methods:** `__len__`, `__getitem__`, `set_initial_condition`, `set_num_examples`, `_compute_inp_statistics`, `_compute_wind_statistics`, `_get_sample_with_winds`
  - All methods support `random`, `galewsky`, `williamson_case2`, `gbells`, and `precomputed` ictypes; for precomputed, loads `{index}_0.pt` and `{index}_1.pt` with fallback to `solver.timestep`

- `multistep_pde_dataset_with_winds.py` — child of `PdeDatasetWithWinds`; generates multi-step rollout targets (`MultiStepPdeDatasetWithWinds`)
  - **Variables:** (inherits all from parent) + `n_rollout_steps`, `input_step_idx`
  - **Methods:** `__getitem__`, `_compute_inp_statistics`, `_compute_wind_statistics`, `_get_sample_with_winds`, `_get_sample_precomputed`
  - `_get_sample_with_winds` — dispatches to `_get_sample_precomputed` for precomputed ictype; supports `random`, `galewsky`, `williamson_case2`, and `gbells` for on-the-fly generation
  - `_get_sample_precomputed` — loads `{index}_0.pt` then advances step-by-step loading `{index}_{s}.pt` where available and falling back to `solver.timestep` for missing files; same fallback logic for each rollout target step

- `dataset_saver.py` — script for generating and saving precomputed trajectory datasets to disk (`ShallowWaterSolver` → spectral states saved as `{index}_{step}.pt`); runs a stability check inline using a finer reference solver; saves normalization stats and metadata
  - `build_solver(nlat, nlon, dt, dt_solver, device)` — constructs a `ShallowWaterSolver` and returns it with `nsteps`
  - `make_output_folder(ictype, dt_solver)` — creates `Saved_Datasets/{ictype}_{dt_solver}_{YYYYMMDD}/` and `stability_check/` subfolder
  - `generate_ic(solver, ictype, gbells_ref_mean, gbells_ref_std, gbells_kwargs)` — dispatches to `random_initial_condition`, `galewsky_initial_condition`, `williamson_case2_initial_condition`, or `gaussian_bells_initial_condition`
  - `_compute_gbells_ref_stats(solver, n_samples)` — computes per-channel mean/std from random ICs for Gaussian bell scaling
  - `welford_update(count, mean, M2, new_value)` — one step of Welford's online mean/variance algorithm
  - `save_trajectories(..., gbells_kwargs)` — main generation loop; saves `{i}_{step}.pt` for each trajectory; accumulates normalization stats online via Welford; for the first `n_stability_samples` trajectories runs a reference solver with `dt_solver_ref` in lockstep, saves `stability_check/{i}_{step}_ref.pt`, computes relative L2 error in grid space; returns stats dict and stability summary
  - `get_git_hash()` — returns current git commit hash for reproducibility
  - `save_metadata(output_folder, args, nsteps, stats, stability_summary)` — writes `metadata.json` (machine-readable) and `metadata.txt` (human-readable) containing ictype, dt, dt_solver, nsteps, grid dims, n_samples, n_steps_per_trajectory, normalization stats, git hash, timestamp, and stability check results
  - `visualize(output_folder, index, step, solver, compare_ref)` — loads `{index}_{step}.pt`, converts to grid space, plots h/vorticity/divergence as heatmaps; if `compare_ref=True` also loads `stability_check/{index}_{step}_ref.pt` and shows reference fields and pointwise difference as additional rows
  - `parse_args()` — CLI arguments: `--ictype` (choices: `random`, `galewsky`, `gbells`, `williamson_case2`), `--dt`, `--dt_solver`, `--dt_solver_ref`, `--nlat`, `--nlon`, `--n_samples`, `--n_steps_per_trajectory`, `--n_stability_samples`, `--n_stability_steps`, `--stability_threshold`, `--device`, `--visualize_index`, `--visualize_step`, `--compare_ref`; gbells options: `--gbells_k_min`, `--gbells_k_max`, `--gbells_sigma_min_deg`, `--gbells_sigma_max_deg`, `--gbells_mean_scale`, `--gbells_std_scale`, `--gbells_unsigned`
  - `main()` — entry point; builds solver, builds gbells_kwargs if needed, generates dataset, saves metadata, optionally visualizes

- `Saved_Datasets/` — directory storing precomputed datasets; each subdirectory holds `.pt` files named `{index}_{step}.pt` (spectral state tensors)

### `model/`
- `paradis.py` — Paradis neural architecture (ADR: Advection-Diffusion-Reaction)
- `blocks.py` — building blocks: GMBlock, CLinear, SepConv
- `advection.py` — NeuralSemiLagrangian semi-Lagrangian advection on the sphere
- `padding.py` — GeoCyclicPadding (periodic longitude, symmetric poles)

### `Training/`
- `train.py` — training script using Adam optimizer
  - `load_config` — loads YAML config file
  - `update_config_from_args` — applies dot-notation CLI overrides to config
  - `build_paradis_loss` — constructs ParadisLoss from config
  - `parse_ic_dict` — parses a JSON string into an ic_dict; converts list values to tuples for precomputed entries
  - `create_datasets` — builds train/val datasets using `build_mixed_dataset`; accepts `train_ic_dict`, `val_ic_dict`, `n_rollout_steps`, `input_step_idx`
  - `main` — entry point; CLI args include `--n_rollout_steps`, `--input_step_idx`, `--train_ic_dict`, `--val_ic_dict`, `--should_detach`
  - `SWELightningModule` — PyTorch Lightning training module
    - **Variables:** `solver`, `inp_mean`, `inp_var`, `wind_mean`, `wind_var`, `should_detach`
    - `__init__` — accepts `solver`, `inp_mean`, `inp_var`, `wind_mean`, `wind_var`, `should_detach`
    - `forward`
    - `_fields_to_winds` — unnormalizes predicted fields, extracts winds via `solver.grid2spec` + `solver.getuv`, renormalizes; optionally detaches from computation graph
    - `training_step` — autoregressive rollout over `n_rollout_steps`, recomputing winds between steps via `_fields_to_winds`; loss averaged across steps
    - `validation_step` — same rollout as training; metrics computed against final rollout step
    - `configure_optimizers` — Adam with MultiStepLR or ReduceLROnPlateau
    - `optimizer_zero_grad`
    - `on_load_checkpoint`

- `train_muon.py` — training script using Muon + AdamW optimizers
  - `load_config` — loads YAML config file
  - `update_config_from_args` — applies dot-notation CLI overrides to config
  - `build_paradis_loss` — constructs ParadisLoss from config
  - `parse_ic_dict` — parses a JSON string into an ic_dict; converts list values to tuples for precomputed entries
  - `split_params_for_muon` — separates 2D parameters (Muon) from all others (AdamW)
  - `create_datasets` — builds train/val datasets using `build_mixed_dataset`; accepts `train_ic_dict`, `val_ic_dict`, `n_rollout_steps`, `input_step_idx`
  - `main` — entry point; CLI args include `--n_rollout_steps`, `--input_step_idx`, `--train_ic_dict`, `--val_ic_dict`, `--should_detach`
  - `SWELightningModule` — PyTorch Lightning training module
    - **Variables:** `solver`, `inp_mean`, `inp_var`, `wind_mean`, `wind_var`, `should_detach`
    - `__init__` — accepts `solver`, `inp_mean`, `inp_var`, `wind_mean`, `wind_var`, `should_detach`
    - `forward`
    - `_fields_to_winds` — unnormalizes predicted fields, extracts winds via `solver.grid2spec` + `solver.getuv`, renormalizes; optionally detaches from computation graph
    - `training_step` — manual optimization; autoregressive rollout over `n_rollout_steps`, recomputing winds between steps; loss averaged across steps
    - `validation_step` — same rollout as training; metrics computed against final rollout step
    - `configure_optimizers` — Muon (2D params) + AdamW (everything else)
    - `_get_schedulers` — builds LR schedulers for both optimizers
    - `on_train_epoch_end` — manually steps schedulers each epoch
    - `on_validation_epoch_end` — steps validation-based schedulers
    - `on_load_checkpoint`

### `utils/`
- `loss.py` — ParadisLoss: weighted loss supporting MSE, MAE, RMSE, reversed Huber, and AMSE
- `amse_loss.py` — AMSELoss: spectral loss in spherical harmonic space (Subich et al. 2025)
- `dataset_utils.py` — `build_mixed_dataset`: builds a ConcatDataset from multiple IC types with shared overall normalization stats

### Root
- `forecast.py` — inference and forecasting script
- `config_paradis.yaml` — canonical model and training configuration
