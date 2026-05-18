import torch
from torch.utils.data import ConcatDataset

from dataset.multistep_pde_dataset_with_winds import MultiStepPdeDatasetWithWinds


def build_mixed_dataset(
    ic_dict,
    dt,
    nsteps,
    n_rollout_steps=1,
    input_step_idx=0,
    dims=(384, 768),
    device=torch.device("cpu"),
    normalize=True,
):
    """Build a dataset mixing multiple initial condition types with shared normalization stats.

    Args:
        ic_dict: dict mapping IC type string to either:
                 - a positive int (number of examples), for "random" and "galewsky"
                 - a (n_examples, folder) tuple for "precomputed"
                 e.g. {"random": 100, "galewsky": 50, "precomputed": (30, "path/to/folder")}
        dt: model timestep in seconds
        nsteps: number of solver sub-steps per dt
        n_rollout_steps: number of target steps to generate per sample
        input_step_idx: which solver step to use as NN input (steps before this are discarded)
        dims: (nlat, nlon) grid dimensions
        device: torch device
        normalize: whether to normalize samples

    Returns:
        dataset: ConcatDataset with shared overall normalization stats attached
        stats: dict with per-IC and overall inp_mean, inp_var, wind_mean, wind_var
    """
    for ic_type, val in ic_dict.items():
        if ic_type == "precomputed":
            if not (isinstance(val, tuple) and len(val) == 2
                    and isinstance(val[0], int) and val[0] > 0
                    and isinstance(val[1], str)):
                raise ValueError(
                    f"Value for 'precomputed' must be a (n_examples, folder) tuple, got {val}"
                )
        elif isinstance(val, tuple):
            if not (len(val) == 2 and isinstance(val[0], int) and val[0] > 0 and isinstance(val[1], dict)):
                raise ValueError(
                    f"Value for '{ic_type}' must be a positive int or (n_examples, kwargs_dict) tuple, got {val}"
                )
        else:
            if not isinstance(val, int) or val <= 0:
                raise ValueError(
                    f"Number of examples for '{ic_type}' must be a positive integer, got {val}"
                )

    sub_datasets = []
    for ic_type, val in ic_dict.items():
        if ic_type == "precomputed":
            n_examples, folder = val
            ic_kwargs = None
            d = MultiStepPdeDatasetWithWinds(
                dt=dt,
                nsteps=nsteps,
                n_rollout_steps=n_rollout_steps,
                input_step_idx=input_step_idx,
                dims=dims,
                initial_condition="precomputed",
                num_examples=n_examples,
                device=device,
                normalize=normalize,
                precomputed_folder=folder,
            )
        elif isinstance(val, tuple):
            n_examples, ic_kwargs = val
            # for gbells/gbells_h, "gbells_ref_ictype" may be embedded in ic_kwargs
            gbells_ref_ictype = "random"
            if ic_type in ("gbells", "gbells_h") and ic_kwargs and "gbells_ref_ictype" in ic_kwargs:
                ic_kwargs = dict(ic_kwargs)
                gbells_ref_ictype = ic_kwargs.pop("gbells_ref_ictype")
            d = MultiStepPdeDatasetWithWinds(
                dt=dt,
                nsteps=nsteps,
                n_rollout_steps=n_rollout_steps,
                input_step_idx=input_step_idx,
                dims=dims,
                initial_condition=ic_type,
                num_examples=n_examples,
                device=device,
                normalize=normalize,
                ic_kwargs=ic_kwargs,
                gbells_ref_ictype=gbells_ref_ictype,
            )
        else:
            n_examples = val
            d = MultiStepPdeDatasetWithWinds(
                dt=dt,
                nsteps=nsteps,
                n_rollout_steps=n_rollout_steps,
                input_step_idx=input_step_idx,
                dims=dims,
                initial_condition=ic_type,
                num_examples=n_examples,
                device=device,
                normalize=normalize,
            )
        d.sht = d.solver.sht
        sub_datasets.append(d)

    # per-IC stats (stored before overwriting)
    per_ic_stats = [
        {
            "ic_type": ic_type,
            "inp_mean": d.inp_mean,
            "inp_var": d.inp_var,
            "wind_mean": d.wind_mean,
            "wind_var": d.wind_var,
        }
        for ic_type, d in zip(ic_dict.keys(), sub_datasets)
    ]

    # overall stats: weighted average by number of examples
    total = sum(len(d) for d in sub_datasets)
    inp_mean  = sum(d.inp_mean  * len(d) for d in sub_datasets) / total
    inp_var   = sum(d.inp_var   * len(d) for d in sub_datasets) / total
    wind_mean = sum(d.wind_mean * len(d) for d in sub_datasets) / total
    wind_var  = sum(d.wind_var  * len(d) for d in sub_datasets) / total

    # overwrite each sub-dataset's stats with overall stats
    for d in sub_datasets:
        d.inp_mean  = inp_mean
        d.inp_var   = inp_var
        d.wind_mean = wind_mean
        d.wind_var  = wind_var

    dataset = ConcatDataset(sub_datasets)

    # attach config and overall stats to the ConcatDataset for external access
    dataset.dt              = dt
    dataset.nsteps          = nsteps
    dataset.n_rollout_steps = n_rollout_steps
    dataset.input_step_idx  = input_step_idx
    dataset.dims            = dims
    dataset.device          = device
    dataset.normalize       = normalize
    dataset.ic_dict         = ic_dict
    dataset.sht       = sub_datasets[0].solver.sht
    dataset.inp_mean  = inp_mean
    dataset.inp_var   = inp_var
    dataset.wind_mean = wind_mean
    dataset.wind_var  = wind_var

    stats = {
        "per_ic": per_ic_stats,
        "overall": {
            "inp_mean": inp_mean,
            "inp_var": inp_var,
            "wind_mean": wind_mean,
            "wind_var": wind_var,
        },
    }

    return dataset, stats
