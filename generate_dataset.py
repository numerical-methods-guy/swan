#!/usr/bin/env python3
"""
generate_dataset.py

Offline paired LR/HR shallow water dataset generator.

Native pairing path
-------------------
For each seed:
1. Generate the IC on the HR grid using random_initial_condition(mach=0.2)
   or galewsky_initial_condition().
2. Optionally rotate the HR scalar IC once onto a rotated lat-lon grid,
   then project it back to spectral space.
3. Run the HR solver forward nsteps -> save hr/t0, hr/t1, hr/w0, hr/w1.
4. Spectrally truncate the (possibly rotated) HR IC to the LR spectral
   resolution by slicing hr_ic[:, :lmax_lr, :lmax_lr].
5. Run the LR solver forward nsteps -> save lr/t0, lr/t1, lr/w0, lr/w1.

This guarantees the LR and HR runs start from the same physical field
(the HR IC band-limited to lmax_lr after any optional rotation), so the
only difference between the two solutions is the sub-grid-scale content
added by the HR dynamics.

All arrays are stored in RAW (un-normalised) physical units.
Normalisation statistics are stored in /attrs so that lam_patch_dataset.py
can normalise on the fly without recomputing them.
"""

import argparse
import os
from math import ceil

import h5py
import torch
import yaml
from tqdm import tqdm

from rotation_utils import describe_latitudes, rotate_spectral_scalar_state
from shallow_water_solver import ShallowWaterSolver


# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------

def _make_solver(
    nlat: int,
    nlon: int,
    dt_solver: float,
    device: torch.device,
) -> ShallowWaterSolver:
    lmax = ceil(nlat / 3)
    return (
        ShallowWaterSolver(
            nlat,
            nlon,
            dt_solver,
            lmax=lmax,
            mmax=lmax,
            grid="equiangular",
        )
        .to(device)
        .float()
    )


# ---------------------------------------------------------------------------
# Spectral truncation
# ---------------------------------------------------------------------------

def _truncate_spectral(hr_ic: torch.Tensor, lmax_lr: int) -> torch.Tensor:
    """Truncate HR spectral IC to LR spectral resolution.

    torch_harmonics stores coefficients with shape [C, lmax, mmax] where
    lmax and mmax are the sizes passed to ShallowWaterSolver (i.e. counts,
    not maximum degree indices). The slice [:, :lmax_lr, :lmax_lr] retains
    all wavenumbers the LR solver can represent.
    """
    return hr_ic[:, :lmax_lr, :lmax_lr].clone()


# ---------------------------------------------------------------------------
# Rotation config
# ---------------------------------------------------------------------------

def _parse_rotation_config(cfg: dict) -> dict:
    rc = cfg.get("rotation", {}) or {}

    enabled = bool(rc.get("enabled", False))
    interpolation = str(rc.get("interpolation", "bilinear")).lower()

    if interpolation != "bilinear":
        raise ValueError(
            f"Phase 1 only supports interpolation='bilinear'; got '{interpolation}'"
        )

    return {
        "enabled": enabled,
        "pole_target_lat_deg": float(rc.get("pole_target_lat_deg", 90.0)),
        "pole_target_lon_deg": float(rc.get("pole_target_lon_deg", 0.0)),
        "interpolation": interpolation,
        "output_suffix": str(rc.get("output_suffix", "_rotated")),
    }


# ---------------------------------------------------------------------------
# Paired IC construction
# ---------------------------------------------------------------------------

def _build_paired_ics(
    hr_solver: ShallowWaterSolver,
    lr_solver: ShallowWaterSolver,
    ic_type: str,
    lmax_lr: int,
    rotation_cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct paired HR/LR ICs following the dataset policy.

    If rotation is enabled:
      native HR IC -> rotate HR scalar state -> grid2spec -> truncate to LR

    If rotation is disabled:
      native HR IC -> truncate to LR

    For Galewsky, when rotation is disabled, preserve the current native path:
      HR and LR are generated directly by their respective solvers.
    """
    rotation_enabled = rotation_cfg["enabled"]

    if ic_type == "random":
        hr_ic_native = hr_solver.random_initial_condition(mach=0.2)

        if rotation_enabled:
            hr_ic = rotate_spectral_scalar_state(
                solver=hr_solver,
                uspec_native=hr_ic_native,
                pole_target_lat_deg=rotation_cfg["pole_target_lat_deg"],
                pole_target_lon_deg=rotation_cfg["pole_target_lon_deg"],
                interpolation=rotation_cfg["interpolation"],
            )
        else:
            hr_ic = hr_ic_native

        lr_ic = _truncate_spectral(hr_ic, lmax_lr)
        return hr_ic, lr_ic

    if ic_type == "galewsky":
        if rotation_enabled:
            hr_ic_native = hr_solver.galewsky_initial_condition()
            hr_ic = rotate_spectral_scalar_state(
                solver=hr_solver,
                uspec_native=hr_ic_native,
                pole_target_lat_deg=rotation_cfg["pole_target_lat_deg"],
                pole_target_lon_deg=rotation_cfg["pole_target_lon_deg"],
                interpolation=rotation_cfg["interpolation"],
            )
            lr_ic = _truncate_spectral(hr_ic, lmax_lr)
            return hr_ic, lr_ic

        hr_ic = hr_solver.galewsky_initial_condition()
        lr_ic = lr_solver.galewsky_initial_condition()
        return hr_ic, lr_ic

    raise ValueError(f"Unsupported ic_type='{ic_type}'")


# ---------------------------------------------------------------------------
# Run a single IC through a solver
# ---------------------------------------------------------------------------

def _run_ic(
    solver: ShallowWaterSolver,
    ic_spec: torch.Tensor,
    nsteps: int,
):
    """Time-step one IC and return (t0_fields, t0_winds, t1_fields, t1_winds)
    as float32 CPU tensors in raw physical units.
    """
    with torch.no_grad():
        tar_spec = solver.timestep(ic_spec, nsteps)
        t0_f = solver.spec2grid(ic_spec)
        t1_f = solver.spec2grid(tar_spec)
        t0_w = solver.getuv(ic_spec[1:])
        t1_w = solver.getuv(tar_spec[1:])
    return t0_f.cpu(), t0_w.cpu(), t1_f.cpu(), t1_w.cpu()


# ---------------------------------------------------------------------------
# Data for autoregressive rollout
# ---------------------------------------------------------------------------

def _run_trajectory(
    solver: ShallowWaterSolver,
    ic_spec: torch.Tensor,
    nsteps: int,
    rollout_steps: int,
):
    """Return full trajectory fields/winds as CPU float32 tensors.

    fields: [rollout_steps + 1, 3, nlat, nlon]
    winds : [rollout_steps + 1, 2, nlat, nlon]
    """
    fields = []
    winds = []
    spec = ic_spec.clone()

    with torch.no_grad():
        for _ in range(rollout_steps + 1):
            fields.append(solver.spec2grid(spec).cpu())
            winds.append(solver.getuv(spec[1:]).cpu())
            spec = solver.timestep(spec, nsteps)

    return torch.stack(fields, dim=0), torch.stack(winds, dim=0)


# ---------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------

def _channel_stats(samples: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.stack(samples)
    mean = xs.mean(dim=(0, 2, 3))
    var = xs.var(dim=(0, 2, 3)).clamp(min=1e-12)
    return mean, var


def _compute_norm_stats(
    hr_solver: ShallowWaterSolver,
    lr_solver: ShallowWaterSolver,
    ic_type: str,
    lmax_lr: int,
    rotation_cfg: dict,
    num_samples: int = 200,
) -> tuple:
    """Compute per-channel mean/var for HR and LR fields/winds.

    Statistics are computed from fresh ICs drawn from the same construction
    path as the dataset itself. When rotation is enabled, the HR IC is rotated
    first and then truncated to LR so stats match the generated data exactly.
    """
    hr_field_samples = []
    hr_wind_samples = []
    lr_field_samples = []
    lr_wind_samples = []

    with torch.no_grad():
        for _ in range(num_samples):
            hr_ic, lr_ic = _build_paired_ics(
                hr_solver=hr_solver,
                lr_solver=lr_solver,
                ic_type=ic_type,
                lmax_lr=lmax_lr,
                rotation_cfg=rotation_cfg,
            )

            hr_field_samples.append(hr_solver.spec2grid(hr_ic).cpu())
            hr_wind_samples.append(hr_solver.getuv(hr_ic[1:]).cpu())
            lr_field_samples.append(lr_solver.spec2grid(lr_ic).cpu())
            lr_wind_samples.append(lr_solver.getuv(lr_ic[1:]).cpu())

    hr_f_mean, hr_f_var = _channel_stats(hr_field_samples)
    hr_w_mean, hr_w_var = _channel_stats(hr_wind_samples)
    lr_f_mean, lr_f_var = _channel_stats(lr_field_samples)
    lr_w_mean, lr_w_var = _channel_stats(lr_wind_samples)

    return (
        hr_f_mean,
        hr_f_var,
        hr_w_mean,
        hr_w_var,
        lr_f_mean,
        lr_f_var,
        lr_w_mean,
        lr_w_var,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate paired LR/HR SWE dataset")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--num_ics", type=int, default=None, help="Override total ICs")
    parser.add_argument("--out_path", default="data/swe_paired.h5", help="Output HDF5 path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--seed_base",
        type=int,
        default=None,
        help="Override lam.ic_seed_base",
    )
    parser.add_argument(
        "--norm_samples",
        type=int,
        default=200,
        help="ICs used for norm stats",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=10,
        help="Number of dt-sized rollout steps to store as trajectories",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dc = cfg["data"]
    lamc = cfg["lam"]
    rotation_cfg = _parse_rotation_config(cfg)

    dt = float(dc["dt"])
    dt_solver = float(dc["dt_solver"])
    nsteps = int(round(dt / dt_solver))
    if nsteps < 1:
        raise ValueError(
            f"Minimum 1 solver substep per data step; "
            f"dt={dt}, dt_solver={dt_solver}, nsteps={nsteps}"
        )

    ic_type = str(dc.get("initial_condition", "random")).lower()
    device = torch.device(args.device)

    lr_nlat = int(dc["nlat"])
    lr_nlon = int(dc["nlon"])
    s_lat = int(lamc["refinement_factor_lat"])
    s_lon = int(lamc["refinement_factor_lon"])
    assert s_lat == s_lon, "Non-isotropic refinement factors are not supported."

    hr_nlat = lr_nlat * s_lat
    hr_nlon = lr_nlon * s_lon

    lmax_lr = ceil(lr_nlat / 3)
    lmax_hr = ceil(hr_nlat / 3)

    seed_base = (
        args.seed_base
        if args.seed_base is not None
        else int(lamc.get("ic_seed_base", 0))
    )

    num_ics = (
        args.num_ics
        if args.num_ics is not None
        else int(dc["num_train_examples"])
        + int(dc["num_val_examples"])
        + int(dc["num_test_examples"])
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)

    print(f"LR grid : {lr_nlat} x {lr_nlon} (lmax={lmax_lr})")
    print(f"HR grid : {hr_nlat} x {hr_nlon} (lmax={lmax_hr})")
    print(f"Upscale : {s_lat}x")
    print(f"nsteps : {nsteps} (dt={dt}s, dt_solver={dt_solver}s)")
    print(f"num ICs : {num_ics} (seed_base={seed_base})")
    print(f"IC type : {ic_type}")
    print(f"Output : {args.out_path}")
    print(f"rollout steps stored : {args.rollout_steps}")

    if rotation_cfg["enabled"]:
        print("Rotation enabled")
        print(
            "  original North Pole -> "
            f"({rotation_cfg['pole_target_lat_deg']:.3f} deg, "
            f"{rotation_cfg['pole_target_lon_deg']:.3f} deg)"
        )
        print(f"  interpolation : {rotation_cfg['interpolation']}")
    else:
        print("Rotation disabled")

    # --- build solvers ----------------------------------------------------
    hr_solver = _make_solver(hr_nlat, hr_nlon, dt_solver, device)
    lr_solver = _make_solver(lr_nlat, lr_nlon, dt_solver, device)

    if rotation_cfg["enabled"]:
        print("HR latitude axis:", describe_latitudes(hr_solver.lats))
        print("LR latitude axis:", describe_latitudes(lr_solver.lats))

    # --- normalisation statistics -----------------------------------------
    print(f"\nComputing paired normalisation statistics ({args.norm_samples} samples) ...")
    (
        hr_f_mean,
        hr_f_var,
        hr_w_mean,
        hr_w_var,
        lr_f_mean,
        lr_f_var,
        lr_w_mean,
        lr_w_var,
    ) = _compute_norm_stats(
        hr_solver=hr_solver,
        lr_solver=lr_solver,
        ic_type=ic_type,
        lmax_lr=lmax_lr,
        rotation_cfg=rotation_cfg,
        num_samples=args.norm_samples,
    )

    # --- allocate HDF5 ----------------------------------------------------
    with h5py.File(args.out_path, "w") as hf:
        # metadata
        hf.attrs["lr_nlat"] = lr_nlat
        hf.attrs["lr_nlon"] = lr_nlon
        hf.attrs["hr_nlat"] = hr_nlat
        hf.attrs["hr_nlon"] = hr_nlon
        hf.attrs["dt"] = dt
        hf.attrs["dt_solver"] = dt_solver
        hf.attrs["nsteps"] = nsteps
        hf.attrs["num_ics"] = num_ics
        hf.attrs["rollout_steps"] = args.rollout_steps
        hf.attrs["upscale_factor_lat"] = s_lat
        hf.attrs["upscale_factor_lon"] = s_lon
        hf.attrs["ic_type"] = ic_type
        hf.attrs["seed_base"] = seed_base

        hf.attrs["rotation_enabled"] = int(rotation_cfg["enabled"])
        hf.attrs["rotation_pole_target_lat_deg"] = rotation_cfg["pole_target_lat_deg"]
        hf.attrs["rotation_pole_target_lon_deg"] = rotation_cfg["pole_target_lon_deg"]
        hf.attrs["rotation_method"] = (
            "quasi_spectral_remap" if rotation_cfg["enabled"] else "none"
        )
        hf.attrs["rotation_interpolation"] = (
            rotation_cfg["interpolation"] if rotation_cfg["enabled"] else "none"
        )

        # norm stats
        hf.attrs["lr_inp_mean"] = lr_f_mean.numpy()
        hf.attrs["lr_inp_var"] = lr_f_var.numpy()
        hf.attrs["lr_wind_mean"] = lr_w_mean.numpy()
        hf.attrs["lr_wind_var"] = lr_w_var.numpy()

        hf.attrs["hr_inp_mean"] = hr_f_mean.numpy()
        hf.attrs["hr_inp_var"] = hr_f_var.numpy()
        hf.attrs["hr_wind_mean"] = hr_w_mean.numpy()
        hf.attrs["hr_wind_var"] = hr_w_var.numpy()

        # datasets
        lr_grp = hf.create_group("lr")
        hr_grp = hf.create_group("hr")
        ds = {}

        for grp, nlat, nlon, prefix in [
            (lr_grp, lr_nlat, lr_nlon, "lr"),
            (hr_grp, hr_nlat, hr_nlon, "hr"),
        ]:
            ds[f"{prefix}_t0"] = grp.create_dataset(
                "t0", shape=(num_ics, 3, nlat, nlon), dtype="f4"
            )
            ds[f"{prefix}_t1"] = grp.create_dataset(
                "t1", shape=(num_ics, 3, nlat, nlon), dtype="f4"
            )
            ds[f"{prefix}_w0"] = grp.create_dataset(
                "w0", shape=(num_ics, 2, nlat, nlon), dtype="f4"
            )
            ds[f"{prefix}_w1"] = grp.create_dataset(
                "w1", shape=(num_ics, 2, nlat, nlon), dtype="f4"
            )
            ds[f"{prefix}_fields"] = grp.create_dataset(
                "fields",
                shape=(num_ics, args.rollout_steps + 1, 3, nlat, nlon),
                dtype="f4",
                chunks=(1, 1, 3, nlat, nlon),
            )
            ds[f"{prefix}_winds"] = grp.create_dataset(
                "winds",
                shape=(num_ics, args.rollout_steps + 1, 2, nlat, nlon),
                dtype="f4",
                chunks=(1, 1, 2, nlat, nlon),
            )

        # --- generate ICs -------------------------------------------------
        print()
        for i in tqdm(range(num_ics), desc="Generating ICs"):
            seed = seed_base + i
            torch.manual_seed(seed)

            hr_ic, lr_ic = _build_paired_ics(
                hr_solver=hr_solver,
                lr_solver=lr_solver,
                ic_type=ic_type,
                lmax_lr=lmax_lr,
                rotation_cfg=rotation_cfg,
            )

            hr_fields, hr_winds = _run_trajectory(
                hr_solver, hr_ic, nsteps, args.rollout_steps
            )
            lr_fields, lr_winds = _run_trajectory(
                lr_solver, lr_ic, nsteps, args.rollout_steps
            )

            # Backward-compatible one-step keys
            ds["hr_t0"][i] = hr_fields[0].numpy()
            ds["hr_t1"][i] = hr_fields[1].numpy()
            ds["hr_w0"][i] = hr_winds[0].numpy()
            ds["hr_w1"][i] = hr_winds[1].numpy()

            ds["lr_t0"][i] = lr_fields[0].numpy()
            ds["lr_t1"][i] = lr_fields[1].numpy()
            ds["lr_w0"][i] = lr_winds[0].numpy()
            ds["lr_w1"][i] = lr_winds[1].numpy()

            # Full trajectories
            ds["hr_fields"][i] = hr_fields.numpy()
            ds["hr_winds"][i] = hr_winds.numpy()
            ds["lr_fields"][i] = lr_fields.numpy()
            ds["lr_winds"][i] = lr_winds.numpy()

    print(f"\nDataset written to {args.out_path}")
    print(f"LR snapshots    : {num_ics} x [3, {lr_nlat}, {lr_nlon}]")
    print(f"HR snapshots    : {num_ics} x [3, {hr_nlat}, {hr_nlon}]")
    print(
        f"LR trajectories : {num_ics} x "
        f"[{args.rollout_steps + 1}, 3, {lr_nlat}, {lr_nlon}]"
    )
    print(
        f"HR trajectories : {num_ics} x "
        f"[{args.rollout_steps + 1}, 3, {hr_nlat}, {hr_nlon}]"
    )


if __name__ == "__main__":
    main()