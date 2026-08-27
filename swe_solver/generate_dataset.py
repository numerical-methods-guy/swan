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
import json

from utils.rotation_utils import (
    describe_latitudes,
    build_pole_target_rotation,
    rotate_scalar_state_on_grid,
    is_identity_rotation,
)
from swe_solver.shallow_water_solver import ShallowWaterSolver
from swe_solver.localized_perturbation import (
    LocalizedPerturbationConfig,
    apply_localized_perturbation,
    perturbation_metadata,
)


# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------

def _make_solver(
    nlat: int,
    nlon: int,
    dt_solver: float,
    device: torch.device,
) -> ShallowWaterSolver:
    lmax = ceil(nlat / 2)
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
    """Construct paired native-frame HR/LR initial conditions.

    The solvers evolve native spectral states. When rotation is enabled,
    rotation is applied once during output rendering, not to the initial
    condition before solver evolution.
    """
    if ic_type == "random":
        hr_ic = hr_solver.random_initial_condition(mach=0.2)
        lr_ic = _truncate_spectral(hr_ic, lmax_lr)
        return hr_ic, lr_ic

    if ic_type == "galewsky":
        hr_ic = hr_solver.galewsky_initial_condition()

        # Preserve the paired/truncated IC policy used in the rotation-enabled
        # path, while allowing the existing native Galewsky baseline otherwise.
        if rotation_cfg["enabled"]:
            lr_ic = _truncate_spectral(hr_ic, lmax_lr)
        else:
            lr_ic = lr_solver.galewsky_initial_condition()

        return hr_ic, lr_ic

    raise ValueError(f"Unsupported ic_type='{ic_type}'")

# --------------------------------------------------------------------------------
# Run a single IC through a solver (OLD, KEPT IN CASE LATER REQUIRED IN ROLLBACK)
# --------------------------------------------------------------------------------

# def _run_ic(
#     solver: ShallowWaterSolver,
#     ic_spec: torch.Tensor,
#     nsteps: int,
#     *,
#     rotation_matrix: torch.Tensor | None = None,
#     rotation_interpolation: str = "bilinear",
# ):
#     with torch.no_grad():
#         tar_spec = solver.timestep(ic_spec, nsteps)

#         t0_f, t0_w = _render_output_frame(
#             solver,
#             ic_spec,
#             rotation_matrix,
#             rotation_interpolation,
#         )
#         t1_f, t1_w = _render_output_frame(
#             solver,
#             tar_spec,
#             rotation_matrix,
#             rotation_interpolation,
#         )

#     return t0_f.cpu(), t0_w.cpu(), t1_f.cpu(), t1_w.cpu()


# ---------------------------------------------------------------------------
# Data for autoregressive rollout
# ---------------------------------------------------------------------------

def _render_output_frame(
    solver: ShallowWaterSolver,
    spec: torch.Tensor,
    rotation_matrix: torch.Tensor | None,
    rotation_interpolation: str,
):
    fields_native = solver.spec2grid(spec)

    if rotation_matrix is None:
        fields_out = fields_native
        spec_out = torch.tril(spec.clone())
    else:
        fields_out = rotate_scalar_state_on_grid(
            ugrid_native=fields_native,
            lats=solver.lats,
            lons=solver.lons,
            rotation_matrix=rotation_matrix,
            interpolation=rotation_interpolation,
        )
        spec_out = torch.tril(solver.grid2spec(fields_out))

    winds_out = solver.getuv(spec_out[1:])
    return fields_out, winds_out

def _run_trajectory(
    solver: ShallowWaterSolver,
    ic_spec: torch.Tensor,
    nsteps: int,
    rollout_steps: int,
    *,
    rotation_matrix: torch.Tensor | None = None,
    rotation_interpolation: str = "bilinear",
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
            fields_out, winds_out = _render_output_frame(
                solver,
                spec,
                rotation_matrix,
                rotation_interpolation,
            )
            fields.append(fields_out.cpu())
            winds.append(winds_out.cpu())
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
    solver: ShallowWaterSolver,
    ic_type: str,
    num_samples: int = 200,
    *,
    rotation_matrix: torch.Tensor | None = None,
    rotation_interpolation: str = "bilinear",
) -> tuple:
    field_samples, wind_samples = [], []

    with torch.no_grad():
        for _ in range(num_samples):
            if ic_type == "random":
                ic = solver.random_initial_condition(mach=0.2)
            else:
                ic = solver.galewsky_initial_condition()

            fields_out, winds_out = _render_output_frame(
                solver,
                ic,
                rotation_matrix,
                rotation_interpolation,
            )
            field_samples.append(fields_out.cpu())
            wind_samples.append(winds_out.cpu())

    fs = torch.stack(field_samples)
    ws = torch.stack(wind_samples)

    f_mean = fs.mean(dim=(0, 2, 3))
    f_var = fs.var(dim=(0, 2, 3)).clamp(min=1e-12)
    w_mean = ws.mean(dim=(0, 2, 3))
    w_var = ws.var(dim=(0, 2, 3)).clamp(min=1e-12)
    return f_mean, f_var, w_mean, w_var


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
    rotc = cfg.get("rotation", {})
    rotation_enabled = bool(rotc.get("enabled", False))
    pole_target_lat_deg = float(rotc.get("pole_target_lat_deg", 90.0))
    pole_target_lon_deg = float(rotc.get("pole_target_lon_deg", 0.0))
    rotation_interpolation = str(rotc.get("interpolation", "bilinear"))
    rotation_cfg = _parse_rotation_config(cfg)
    perturbation_cfg = LocalizedPerturbationConfig.from_mapping(
        cfg.get("perturbation")
    )
    if perturbation_cfg.enabled:
        perturbation_cfg.validate()

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

    lmax_lr = ceil(lr_nlat / 2)
    lmax_hr = ceil(hr_nlat / 2)

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

    if perturbation_cfg.enabled:
        print("Localized perturbation enabled")
        print(
            f" kind={perturbation_cfg.kind}, "
            f"center=({perturbation_cfg.center_lat_deg:.2f}, "
            f"{perturbation_cfg.center_lon_deg:.2f}) deg, "
            f"radius={perturbation_cfg.radius_km:.1f} km, "
            f"amplitude={perturbation_cfg.geopotential_amplitude:.2f} m^2/s^2"
        )
    else:
        print("Localized perturbation disabled")

    # --- build solvers ----------------------------------------------------
    hr_solver = _make_solver(hr_nlat, hr_nlon, dt_solver, device)
    lr_solver = _make_solver(lr_nlat, lr_nlon, dt_solver, device)

    if rotation_enabled and not is_identity_rotation(pole_target_lat_deg, pole_target_lon_deg):
        hr_rotation_matrix = build_pole_target_rotation(
            pole_target_lat_deg,
            pole_target_lon_deg,
            dtype=hr_solver.lats.dtype,
            device=hr_solver.lats.device,
        )
        lr_rotation_matrix = build_pole_target_rotation(
            pole_target_lat_deg,
            pole_target_lon_deg,
            dtype=lr_solver.lats.dtype,
            device=lr_solver.lats.device,
        )
    else:
        hr_rotation_matrix = None
        lr_rotation_matrix = None

    if rotation_cfg["enabled"]:
        print("HR latitude axis:", describe_latitudes(hr_solver.lats))
        print("LR latitude axis:", describe_latitudes(lr_solver.lats))

    # --- normalisation statistics -----------------------------------------
    print(f"\nComputing paired normalisation statistics ({args.norm_samples} samples) ...")
    hr_f_mean, hr_f_var, hr_w_mean, hr_w_var = _compute_norm_stats(
        hr_solver,
        ic_type,
        args.norm_samples,
        rotation_matrix=hr_rotation_matrix,
        rotation_interpolation=rotation_interpolation,
    )

    lr_f_mean, lr_f_var, lr_w_mean, lr_w_var = _compute_norm_stats(
        lr_solver,
        ic_type,
        args.norm_samples,
        rotation_matrix=lr_rotation_matrix,
        rotation_interpolation=rotation_interpolation,
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

        hf.attrs["rotation_enabled"] = rotation_enabled
        hf.attrs["rotation_pole_target_lat_deg"] = pole_target_lat_deg
        hf.attrs["rotation_pole_target_lon_deg"] = pole_target_lon_deg
        hf.attrs["rotation_interpolation"] = rotation_interpolation
        hf.attrs["rotation_workflow"] = "native_spectral_evolution_rotated_output"

        perturbation_meta = perturbation_metadata(perturbation_cfg)
        hf.attrs["perturbation_enabled"] = perturbation_cfg.enabled
        hf.attrs["perturbation_config_json"] = json.dumps(perturbation_meta)
        for key, value in perturbation_meta.items():
            hf.attrs[f"perturbation_{key}"] = value

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

            if perturbation_cfg.enabled:
                hr_ic = apply_localized_perturbation(
                    solver=hr_solver,
                    ic_spec=hr_ic,
                    config=perturbation_cfg,
                )

                # Derive LR from the perturbed HR state, rather than applying the
                # perturbation independently at low resolution. This preserves the
                # same resolved physical initial anomaly at both resolutions.
                lr_ic = _truncate_spectral(hr_ic, lmax_lr)

            hr_fields, hr_winds = _run_trajectory(
                hr_solver,
                hr_ic,
                nsteps,
                args.rollout_steps,
                rotation_matrix=hr_rotation_matrix,
                rotation_interpolation=rotation_interpolation,
            )

            lr_fields, lr_winds = _run_trajectory(
                lr_solver,
                lr_ic,
                nsteps,
                args.rollout_steps,
                rotation_matrix=lr_rotation_matrix,
                rotation_interpolation=rotation_interpolation,
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