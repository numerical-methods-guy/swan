import json
import os
import warnings

import torch
from torch.utils.data import ConcatDataset

from kdv.multistep_tkdv_dataset import MultiStepTKdVDataset


def _load_metadata(folder):
    path = os.path.join(folder, "metadata.json")
    with open(path) as f:
        return json.load(f)


def build_tkdv_dataset(
    ic_dict,
    dt,
    nsteps,
    n_rollout_steps=1,
    input_step_idx=0,
    device=torch.device("cpu"),
    normalize=True,
    use_advection_vel=False,
):
    """Build a tKdV dataset from precomputed folders with shared normalization stats.

    Args:
        ic_dict: dict mapping a label string to a (n_examples, folder) tuple.
                 e.g. {"seed_0": (256, "path/to/seed_0/tkdv_folder"),
                       "seed_1": (256, "path/to/seed_1/tkdv_folder")}
        dt: model timestep
        nsteps: number of solver sub-steps per dt
        n_rollout_steps: number of target steps per sample
        input_step_idx: which solver step to use as NN input
        device: torch device
        normalize: whether to normalize samples
        use_advection_vel: whether to return advection velocity alongside field

    Returns:
        dataset: ConcatDataset with shared overall normalization stats attached
        stats: dict with per-IC and overall inp_mean, inp_var
    """
    for label, val in ic_dict.items():
        if not (isinstance(val, tuple) and len(val) == 2
                and isinstance(val[0], int) and val[0] > 0
                and isinstance(val[1], str)):
            raise ValueError(
                f"Value for '{label}' must be a (n_examples, folder) tuple, got {val}"
            )

    # Read per-folder physics params from metadata.json
    metas = {label: _load_metadata(val[1]) for label, val in ic_dict.items()}

    # Warn if J, E0, or L0 differ across folders (D and theta may differ freely)
    for key in ("J", "E0", "L0"):
        vals = [m[key] for m in metas.values()]
        if len(set(vals)) > 1:
            warnings.warn(f"'{key}' differs across ic_dict folders: {dict(zip(metas.keys(), vals))}")

    sub_datasets = []
    for label, val in ic_dict.items():
        n_examples, folder = val
        m = metas[label]
        d = MultiStepTKdVDataset(
            dt=dt,
            nsteps=nsteps,
            n_rollout_steps=n_rollout_steps,
            input_step_idx=input_step_idx,
            J=int(m["J"]),
            D=float(m["D"]),
            E0=float(m["E0"]),
            L0=float(m["L0"]),
            theta=float(m["theta"]),
            initial_condition="precomputed",
            num_examples=n_examples,
            device=device,
            normalize=normalize,
            precomputed_folder=folder,
            use_advection_vel=use_advection_vel,
        )
        sub_datasets.append(d)

    # per-IC stats (stored before overwriting)
    per_ic_stats = [
        {
            "label":    label,
            "inp_mean": d.inp_mean,
            "inp_var":  d.inp_var,
        }
        for label, d in zip(ic_dict.keys(), sub_datasets)
    ]

    # overall stats: weighted average by number of examples
    total    = sum(len(d) for d in sub_datasets)
    inp_mean = sum(d.inp_mean * len(d) for d in sub_datasets) / total
    inp_var  = sum(d.inp_var  * len(d) for d in sub_datasets) / total

    # overwrite each sub-dataset's stats with overall stats
    for d in sub_datasets:
        d.inp_mean = inp_mean
        d.inp_var  = inp_var

    dataset = ConcatDataset(sub_datasets)

    # attach config and overall stats to the ConcatDataset for external access
    dataset.dt              = dt
    dataset.nsteps          = nsteps
    dataset.n_rollout_steps = n_rollout_steps
    dataset.input_step_idx  = input_step_idx
    dataset.dims            = int(list(metas.values())[0]["J"])
    dataset.device          = device
    dataset.normalize       = normalize
    dataset.ic_dict         = ic_dict
    dataset.inp_mean        = inp_mean
    dataset.inp_var         = inp_var

    stats = {
        "per_ic": per_ic_stats,
        "overall": {
            "inp_mean": inp_mean,
            "inp_var":  inp_var,
        },
    }

    return dataset, stats
