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
    adam/version_0/
    adamw/version_0/
    mud/version_0/
    mud_new/version_0/
    muon/version_0/
    muon_new/version_0/
    sgd/version_0/
```

The run root and config are set near the top of
`run_forecast_visualization_groups.sh`:

```bash
visualization_runs_root="./swan_checkpoints/logs"
visualization_config="./swan_checkpoints/config_paradis.yaml"
```

## Optimizer Groups

The script always uses these groups:

| Output subfolder | Optimizers |
| --- | --- |
| `all_optimizers` | Adam, AdamW, MUD, MUD-new, Muon, Muon-new, SGD |
| `without_spectral_blow_up` | Adam, AdamW, MUD-new, Muon-new, SGD |
| `without_spatial_blow_up` | Adam, AdamW, MUD, Muon-new, SGD |
| `stable_core` | Adam, AdamW, Muon-new, SGD |
| `custom` | User-defined with `CUSTOM_OPTIMIZERS` and `CUSTOM_LABELS` |

Control which groups are rendered with:

```bash
VIS_GROUPS=(all_optimizers without_spectral_blow_up without_spatial_blow_up stable_core)
```

To render a custom group:

```bash
VIS_GROUPS=(all_optimizers custom)
CUSTOM_OPTIMIZERS=(adam adamw muon_new)
CUSTOM_LABELS=(Adam AdamW Muon-new)
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
`visualization_runs_two_groups/version_N/` folder. If an older flat-output run
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

Set it to `true` only if each `swan_checkpoints/logs/<optimizer>/version_0`
folder contains `events.out.tfevents*`, `scalars.csv`, `history.csv`, or
`metrics_history.csv`.

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
visualization_autoreg_steps=100
visualization_output_freq=5
visualization_skill_horizon=true
run_history_plots=false
delete_rollouts_after_plotting=true
VIS_GROUPS=(all_optimizers without_spectral_blow_up without_spatial_blow_up stable_core)
CUSTOM_OPTIMIZERS=(adam adamw)
CUSTOM_LABELS=(Adam AdamW)
visualization_animation_pacing="slow"
reuse_legacy_rollouts=false
```

Edit these near the top of `run_forecast_visualization_groups.sh` the same
way you would edit `run_cluster_visualization.sh`.

## Outputs

With `visualization_root="./visualization_runs_two_groups"` and
`visualization_version="auto"`, each run writes a versioned folder such as:

```text
visualization_runs_two_groups/version_0/
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

Running `visualize_two_group_optimizers.sh` directly also defaults to the same
`visualization_runs_two_groups/version_N/` layout.

## Syntax Check

Before submitting:

```bash
bash -n visualize_two_group_optimizers.sh
bash -n run_forecast_visualization_groups.sh
```
