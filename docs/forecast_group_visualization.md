# Grouped Forecast Visualization

Use `run_forecast_visualization_groups.sh` when you want the same cluster
workflow as `run_cluster_visualization.sh`, but reading saved models from
`swan_checkpoints` and making selected comparison groups:

1. `all_optimizers`
2. `without_spectral_blow_up`
3. `without_spatial_blow_up`
4. `stable_core`

The wrapper calls `visualize_two_group_optimizers.sh`, which is a minimal variant
of `visualize_all_optimizers.sh`.

## What It Reads

By default this wrapper does not train. It reads:

```text
swan_checkpoints/
  config_paradis.yaml
  logs/
    Adam/
    AdamW/
    MUD_finetuned/
    MUD_old/
    Muon_flattened/
    Muon_old/
    SGD/
```

The run root and config are set near the top of
`run_forecast_visualization_groups.sh`:

```bash
visualization_runs_root="./swan_checkpoints/logs"
visualization_config="./swan_checkpoints/config_paradis.yaml"
```

For the config file, the wrapper first tries the exact
`visualization_config` path. If that file is missing, it then falls back to:

```text
dirname(visualization_runs_root)/config_paradis.yaml
./config_paradis.yaml
```

So with the default `visualization_runs_root="./swan_checkpoints/logs"`, the
helper will accept either:

```text
./swan_checkpoints/config_paradis.yaml
./config_paradis.yaml
```

## Optimizer Groups

The script always uses these groups:

| Output subfolder | Optimizers |
| --- | --- |
| `all_optimizers` | Adam, AdamW, MUD-finetuned, MUD-old, Muon-flattened, Muon-old, SGD |
| `without_spectral_blow_up` | Adam, AdamW, MUD-finetuned, Muon-flattened, SGD |
| `without_spatial_blow_up` | Adam, AdamW, MUD-old, Muon-flattened, SGD |
| `stable_core` | Adam, AdamW, Muon-flattened, SGD |
| `custom` | User-defined with `CUSTOM_OPTIMIZERS` and `CUSTOM_LABELS` |

Control which groups are rendered with:

```bash
VIS_GROUPS=(all_optimizers without_spectral_blow_up without_spatial_blow_up stable_core)
```

The built-in groups are now defined explicitly near the top of
`run_forecast_visualization_groups.sh` with paired arrays such as:

```bash
ALL_OPTIMIZERS=(Adam AdamW MUD_finetuned MUD_old Muon_flattened Muon_old SGD)
ALL_LABELS=(Adam AdamW MUD-finetuned MUD-old Muon-flattened Muon-old SGD)
```

Each `*_OPTIMIZERS` entry must use the exact subfolder names under
`swan_checkpoints/logs`, while each `*_LABELS` entry is only the human-readable
plot label.

The helper accepts either of these checkpoint layouts:

```text
swan_checkpoints/logs/<optimizer>/checkpoints/
```

or

```text
swan_checkpoints/logs/<optimizer>/version_0/checkpoints/
```

To render a custom group:

```bash
VIS_GROUPS=(all_optimizers custom)
CUSTOM_OPTIMIZERS=(Adam AdamW Muon_flattened)
CUSTOM_LABELS=(Adam AdamW Muon-flattened)
```

The full group runs forecast rollouts first. The other groups reuse the saved
rollouts from the full group, so they should not repeat the expensive rollout
generation.

`without_spectral_blow_up` excludes `MUD` and `Muon`, which are the two
optimizers that make the existing `figures_forecast/all_optimizers` spectral
comparison hard to read. The L2 curve identifies a different bad pair because it
measures a different failure mode.

`without_spatial_blow_up` excludes `MUD-new` and `Muon`, based on the physical
L2 curve. `stable_core` excludes `MUD`, `MUD-new`, and `Muon` so the remaining
methods can be compared without either major scale distortion.

By default, the wrapper keeps rollout intermediates inside the selected
`forecast_group_visualizations/version_N/` folder. If an older flat-output run
already produced `./rollout_results/shared_rollouts` and you explicitly want to
reuse those tensors, set:

```bash
reuse_legacy_rollouts=true
```

The checked-in `swan_checkpoints` folders contain checkpoints but not
TensorBoard event files or CSV scalar histories. Therefore history plots are
disabled by default:

```bash
run_history_plots=false
```

Set it to `true` only if each run folder contains `events.out.tfevents*`,
`scalars.csv`, `history.csv`, or `metrics_history.csv`.

## Basic Usage

On a login node or interactive cluster shell:

```bash
bash run_forecast_visualization_groups.sh
```

With Slurm:

```bash
sbatch run_forecast_visualization_groups.sh
```

The default settings match the original cluster wrapper style:

```bash
run_visualization=true
visualization_runs_root="./swan_checkpoints/logs"
visualization_config="./swan_checkpoints/config_paradis.yaml"
visualization_channel="vorticity"
visualization_autoreg_steps=250
visualization_output_freq=10
visualization_skill_horizon=true
visualization_spectral_horizon=true
visualization_spectral_horizon_modes=(abs positive)
visualization_spectral_eta_factors=(1.05 1.1 1.25 1.5 2 5)
run_history_plots=false
delete_rollouts_after_plotting=true
VIS_GROUPS=(all_optimizers custom)
ALL_OPTIMIZERS=(Adam AdamW MUD_finetuned MUD_old Muon_flattened Muon_old SGD)
ALL_LABELS=(Adam AdamW MUD-finetuned MUD-old Muon-flattened Muon-old SGD)
WITHOUT_SPECTRAL_BLOW_UP_OPTIMIZERS=(Adam AdamW MUD_finetuned Muon_flattened SGD)
WITHOUT_SPECTRAL_BLOW_UP_LABELS=(Adam AdamW MUD-finetuned Muon-flattened SGD)
WITHOUT_SPATIAL_BLOW_UP_OPTIMIZERS=(Adam AdamW MUD_old Muon_flattened SGD)
WITHOUT_SPATIAL_BLOW_UP_LABELS=(Adam AdamW MUD-old Muon-flattened SGD)
STABLE_CORE_OPTIMIZERS=(Adam AdamW Muon_flattened SGD)
STABLE_CORE_LABELS=(Adam AdamW Muon-flattened SGD)
CUSTOM_OPTIMIZERS=(Adam AdamW)
CUSTOM_LABELS=(Adam AdamW)
visualization_animation_pacing="slow"
reuse_legacy_rollouts=false
```

Edit these near the top of `run_forecast_visualization_groups.sh` the same
way you would edit `run_cluster_visualization.sh`.

## Outputs

With `visualization_root="./forecast_group_visualizations"` and
`visualization_version="auto"`, each run writes a versioned folder such as:

```text
forecast_group_visualizations/version_0/
  figures_history/
    all_optimizers/
    without_spectral_blow_up/
    without_spatial_blow_up/
    stable_core/
    custom/
  figures_forecast/
    all_optimizers/
    without_spectral_blow_up/
    without_spatial_blow_up/
    stable_core/
    custom/
  settings.txt
```

Rollout tensors are written under the version folder while plotting runs. By
default they are deleted after figures and animations are created, matching the
existing visualization script behavior.

Each selected forecast group still writes its normal animations, for example:

```text
figures_forecast/stable_core/rollout_fields.gif
figures_forecast/stable_core/rollout_spectra.gif
```

With `visualization_skill_horizon=true`, each selected forecast group also
writes one shared reliability plot containing all optimizers in that group:

```text
figures_forecast/stable_core/skill_horizon_vs_gamma.png
```

This plot shows the forecast skill horizon as a function of relative-error
threshold:

```math
T_{\mathrm{skill}}(\gamma)=\min\{t:e(t)>\gamma\}.
```

Here the rollout relative error is

```math
e(t)=
\frac{
\left(
\|\widetilde h(t)-h(t)\|_{\ell^2(X_h)}^2+
\|\widetilde\zeta(t)-\zeta(t)\|_{\ell^2(X_h)}^2+
\|\widetilde\delta(t)-\delta(t)\|_{\ell^2(X_h)}^2
\right)^{1/2}
}{
\left(
\|h(t)\|_{\ell^2(X_h)}^2+
\|\zeta(t)\|_{\ell^2(X_h)}^2+
\|\delta(t)\|_{\ell^2(X_h)}^2
\right)^{1/2}
}.
```

So, for each threshold `\gamma`, the curve records the first rollout step where
the relative forecast error exceeds that threshold. Lower curves lose skill
earlier; higher curves remain accurate for more rollout steps. If a threshold is
never crossed within the saved rollout horizon, the plotted value stays at the
maximum saved rollout step.

With `visualization_spectral_horizon=true`, each selected forecast group also
writes two shared spectral-horizon plots, one for each mode:

```text
figures_forecast/stable_core/spectral_horizon_abs_factor2.png
figures_forecast/stable_core/spectral_horizon_positive_factor2.png
```

The grouped spectral-horizon plots use the same saved rollout tensors and the
same spherical-harmonic spectral diagnostic as `visualize forecast` when
`--spherical_method spherical` is active.

For each spectral component

```math
s\in\{\mathrm{rot},\mathrm{div},\mathrm{pot},\mathrm{total}\},
```

the code computes

```math
L_s(t,k)=\log\frac{E_{s,\mathrm{pred}}(t,k)+\varepsilon}{E_{s,\mathrm{true}}(t,k)+\varepsilon},
```

Here:

```math
E_{s,\mathrm{pred}}(t,k)
```

is the predicted spectral energy of component `s` at rollout step `t` and
wavenumber `k`, and

```math
E_{s,\mathrm{true}}(t,k)
```

is the corresponding ground-truth spectral energy from the solver at the same
step and wavenumber.

where `\varepsilon` is a small positive numerical-stability floor, and each
entry of `visualization_spectral_eta_factors` is converted into

```math
\eta=\log(\texttt{visualization\_spectral\_eta\_factor}).
```

The two modes are:

```math
T_{s,\mathrm{spec}}^{\mathrm{abs}}(k;\eta)=\min\{t:|L_s(t,k)|>\eta\},
```

and

```math
T_{s,\mathrm{spec}}^{+}(k;\eta)=\min\{t:L_s(t,k)>\eta\}.
```

Because the wrapper uses saved rollout snapshots, these horizons are resolved
at the saved rollout steps determined by `visualization_output_freq`.

Running `visualize_two_group_optimizers.sh` directly also defaults to the same
`forecast_group_visualizations/version_N/` layout.

## Syntax Check

Before submitting:

```bash
bash -n visualize_two_group_optimizers.sh
bash -n run_forecast_visualization_groups.sh
```
