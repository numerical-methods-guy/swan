import os
import argparse
import sys
import warnings
import json
import hashlib
from datetime import datetime, timezone
import yaml
import torch
from torch.utils.data import DataLoader
import math
import torch.multiprocessing as mp

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from torch_harmonics import RealSHT
from torch_harmonics.examples import PdeDataset
from torch_harmonics.examples.losses import (
    SquaredL2LossS2,
    L1LossS2,
    L2LossS2,
    W11LossS2,
)
from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperator
from torch_harmonics.examples.models.s2transformer import SphericalTransformer
from amse_loss import AMSELoss

from paradis import ParadisModel
from pde_dataset_with_winds import PdeDatasetWithWinds


# -----------------------
# config helpers
# -----------------------
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def update_config_from_args(config, unknown_args):
    for i in range(0, len(unknown_args), 2):
        if i + 1 >= len(unknown_args):
            break
        key = unknown_args[i].lstrip("-")
        val = unknown_args[i + 1]
        keys = key.split(".")
        cur = config
        for k in keys[:-1]:
            if k not in cur:
                cur[k] = {}
            cur = cur[k]
        try:
            if "." in val:
                val = float(val)
            else:
                val = int(val)
        except ValueError:
            pass
        cur[keys[-1]] = val
    return config


def _online_update_mean_var(sum_, sumsq_, count_, x):
    # x: (C,H,W)
    sum_ = sum_ + x.sum(dim=(-1, -2))
    sumsq_ = sumsq_ + (x * x).sum(dim=(-1, -2))
    count_ = count_ + x.shape[-1] * x.shape[-2]
    return sum_, sumsq_, count_


def _finalize_mean_var(sum_, sumsq_, count_, eps=1e-12):
    mean = (sum_ / count_).reshape(-1, 1, 1)
    var = (sumsq_ / count_ - (sum_ / count_)**2).clamp_min(eps).reshape(-1, 1, 1)
    return mean, var


def _great_circle_distance(lat, lon, lat0, lon0):
    sin1, cos1 = torch.sin(lat), torch.cos(lat)
    sin0, cos0 = torch.sin(lat0), torch.cos(lat0)
    dlon = lon - lon0
    cosgamma = sin1 * sin0 + cos1 * cos0 * torch.cos(dlon)
    cosgamma = torch.clamp(cosgamma, -1.0, 1.0)
    return torch.acos(cosgamma)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected: true/false")


def _parse_nonnegative_int(token, field_name):
    try:
        value = int(token)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"Invalid {field_name} '{token}': expected a nonnegative integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError(f"Invalid {field_name} '{token}': expected a nonnegative integer")
    return value


def parse_ic_mix_components(raw_ic_mix_groups):
    """
    Parse repeatable --ic_mix components into canonical dicts.

    Accepted forms per --ic_mix occurrence:
      1) b0 b1 b2 n [scaling_scheme], where each b is in {0,1,2}
      2) williamson_case2 n [scaling_scheme]
      3) williamson_case6 n [scaling_scheme]
    """
    default_components = [
        {
            "kind": "triplet",
            "channels": (0, 0, 0),
            "n": 1024,
            "scaling_scheme": None,
        }
    ]

    if not raw_ic_mix_groups:
        return default_components

    components = []
    valid_williamson = {"williamson_case2", "williamson_case6"}
    for group in raw_ic_mix_groups:
        if len(group) not in {2, 3, 4, 5}:
            raise argparse.ArgumentTypeError(
                f"Invalid --ic_mix arity {len(group)} for tokens {group}. "
                "Expected: b0 b1 b2 n [scaling_scheme] or williamson_case2/6 n [scaling_scheme]."
            )

        first = str(group[0]).strip().lower()
        if first in valid_williamson:
            if len(group) not in {2, 3}:
                raise argparse.ArgumentTypeError(
                    f"Invalid --ic_mix form for {group}. "
                    "Williamson form requires: williamson_case2|williamson_case6 n [scaling_scheme]."
                )
            n_value = _parse_nonnegative_int(group[1], "n")
            scaling_scheme = str(group[2]).strip() if len(group) == 3 else None
            if n_value == 0 and scaling_scheme is not None:
                warnings.warn(
                    f"--ic_mix {' '.join(group)} provides scaling_scheme with n=0; scaling_scheme is ignored for zero samples.",
                    stacklevel=2,
                )
            components.append(
                {
                    "kind": first,
                    "channels": None,
                    "n": n_value,
                    "scaling_scheme": scaling_scheme,
                }
            )
            continue

        if len(group) not in {4, 5}:
            raise argparse.ArgumentTypeError(
                f"Invalid --ic_mix form for {group}. "
                "Triplet form requires: b0 b1 b2 n [scaling_scheme]."
            )

        try:
            channels = tuple(int(group[i]) for i in range(3))
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid --ic_mix channels {group[:3]}: expected integers in {{0,1,2}}"
            ) from exc

        for channel_value in channels:
            if channel_value not in {0, 1, 2}:
                raise argparse.ArgumentTypeError(
                    f"Invalid channel value {channel_value} in --ic_mix {group}. Allowed values are {{0,1,2}}."
                )

        n_value = _parse_nonnegative_int(group[3], "n")
        scaling_scheme = str(group[4]).strip() if len(group) == 5 else None
        if n_value == 0 and scaling_scheme is not None:
            warnings.warn(
                f"--ic_mix {' '.join(group)} provides scaling_scheme with n=0; scaling_scheme is ignored for zero samples.",
                stacklevel=2,
            )

        components.append(
            {
                "kind": "triplet",
                "channels": channels,
                "n": n_value,
                "scaling_scheme": scaling_scheme,
            }
        )

    return components


def build_ic_mix_component_index_and_schedule(parsed_components):
    """
    Build unique internal components and a flat sample schedule.

    Uniqueness key: (kind, spec, scaling_scheme)
    - spec is channel triplet for triplet kind
    - spec is case-name string for williamson kinds
    """
    unique_components = {}
    component_order = []

    for component in parsed_components:
        kind = component["kind"]
        if kind == "triplet":
            spec = tuple(component["channels"])
        else:
            spec = kind
        scaling_scheme = component["scaling_scheme"]
        key = (kind, spec, scaling_scheme)

        if key not in unique_components:
            unique_components[key] = {
                "component_id": len(component_order),
                "kind": kind,
                "spec": spec,
                "scaling_scheme": scaling_scheme,
                "count": 0,
            }
            component_order.append(key)

        unique_components[key]["count"] += int(component["n"])

    components = [unique_components[key] for key in component_order]

    flat_sample_schedule = []
    sample_index = 0
    for component in components:
        for local_index in range(int(component["count"])):
            flat_sample_schedule.append(
                {
                    "sample_index": sample_index,
                    "component_id": int(component["component_id"]),
                    "component_local_index": local_index,
                }
            )
            sample_index += 1

    total_count = sample_index
    return components, flat_sample_schedule, total_count


def _ic_scaling_default_apply(unnormalized_sample, context=None):
    return unnormalized_sample


def _ic_scaling_default_stats_info(context=None):
    return {
        "scheme": "default",
        "kind": "identity",
        "requires": [],
    }


def _make_ic_scaling_stub_apply(scheme_name):
    def _stub_apply(unnormalized_sample, context=None):
        raise NotImplementedError(
            f"IC scaling scheme '{scheme_name}' is registered but not implemented yet."
        )

    return _stub_apply


def _make_ic_scaling_stub_stats_info(scheme_name):
    def _stub_stats_info(context=None):
        raise NotImplementedError(
            f"IC scaling stats for scheme '{scheme_name}' are not implemented yet."
        )

    return _stub_stats_info


IC_SCALING_DEFAULT_SCHEME = "default"
IC_SCALING_REGISTRY = {
    "default": {
        "name": "default",
        "apply": _ic_scaling_default_apply,
        "stats_info": _ic_scaling_default_stats_info,
    },
    "global_zscore": {
        "name": "global_zscore",
        "apply": _make_ic_scaling_stub_apply("global_zscore"),
        "stats_info": _make_ic_scaling_stub_stats_info("global_zscore"),
    },
    "component_zscore": {
        "name": "component_zscore",
        "apply": _make_ic_scaling_stub_apply("component_zscore"),
        "stats_info": _make_ic_scaling_stub_stats_info("component_zscore"),
    },
}


def resolve_ic_scaling_scheme_name(scaling_scheme):
    if scaling_scheme is None:
        return IC_SCALING_DEFAULT_SCHEME
    name = str(scaling_scheme).strip()
    if name == "":
        return IC_SCALING_DEFAULT_SCHEME
    return name


def get_ic_scaling_scheme(scaling_scheme):
    scheme_name = resolve_ic_scaling_scheme_name(scaling_scheme)
    if scheme_name not in IC_SCALING_REGISTRY:
        available = ", ".join(sorted(IC_SCALING_REGISTRY.keys()))
        raise ValueError(f"Unknown scaling scheme '{scheme_name}'. Available: {available}")
    return IC_SCALING_REGISTRY[scheme_name]


def apply_ic_scaling(scaling_scheme, unnormalized_sample, context=None):
    scheme = get_ic_scaling_scheme(scaling_scheme)
    return scheme["apply"](unnormalized_sample, context=context)


def get_ic_scaling_stats_info(scaling_scheme, context=None):
    scheme = get_ic_scaling_scheme(scaling_scheme)
    return scheme["stats_info"](context=context)


def make_williamson_case2_ic_spec_from_winds(
    solver,
    gh0=29400.0,
    u0=None,
    alpha=None,
    flip_vort=False,
):
    device = solver.lap.device
    dtype = solver.lap.dtype

    lat = solver.lats.to(device=device, dtype=dtype).reshape(-1, 1)
    lon = solver.lons.to(device=device, dtype=dtype).reshape(1, -1)

    a = solver.radius.to(dtype=dtype)
    Omega = solver.omega.to(dtype=dtype)

    if u0 is None:
        day = torch.as_tensor(86400.0, device=device, dtype=dtype)
        u0 = 2.0 * torch.pi * a / (12.0 * day)
    else:
        u0 = torch.as_tensor(float(u0), device=device, dtype=dtype)

    sinlat = torch.sin(lat)
    coslat = torch.cos(lat)
    sinlon = torch.sin(lon)
    coslon = torch.cos(lon)

    if alpha is None:
        alpha_t = torch.as_tensor(0.0, device=device, dtype=dtype)
    else:
        alpha_t = torch.as_tensor(float(alpha), device=device, dtype=dtype)

    sinalpha = torch.sin(alpha_t)
    cosalpha = torch.cos(alpha_t)

    u = u0 * (coslat * cosalpha + sinlat * coslon * sinalpha)
    v = -u0 * sinlon * sinalpha

    u_grid = u
    v_grid = v + 0.0 * lat

    cterm = (-coslon * coslat * sinalpha + sinlat * cosalpha)
    phi_grid = torch.as_tensor(float(gh0), device=device, dtype=dtype) - (a * Omega * u0 + 0.5 * u0 * u0) * (cterm ** 2)

    uv_grid = torch.stack([u_grid, v_grid], dim=0)
    vrtdiv_spec = solver.vrtdivspec(uv_grid)
    if flip_vort:
        vrtdiv_spec[0] = -vrtdiv_spec[0]

    phi_spec = solver.grid2spec(torch.stack([phi_grid], dim=0))[0]

    ctype = torch.complex128 if solver.lap.dtype == torch.float64 else torch.complex64
    uspec = torch.zeros(3, solver.lmax, solver.mmax, dtype=ctype, device=device)
    uspec[0] = phi_spec
    uspec[1:] = vrtdiv_spec.to(dtype=ctype)

    return torch.tril(uspec)


def make_williamson_case6_ic_spec_from_winds(
    solver,
    R=4,
    omega=7.848e-6,
    K=None,
    h0=8000.0,
    flip_vort=False,
    eps_cos=1e-6,
):
    device = solver.lap.device
    dtype = solver.lap.dtype

    lat = solver.lats.to(device=device, dtype=dtype).reshape(-1, 1)
    lon = solver.lons.to(device=device, dtype=dtype).reshape(1, -1)

    a = solver.radius.to(dtype=dtype)
    Omega = solver.omega.to(dtype=dtype)
    g = solver.gravity.to(dtype=dtype)

    R_int = int(R)
    omega_t = torch.as_tensor(float(omega), device=device, dtype=dtype)
    if K is None:
        K_t = omega_t
    else:
        K_t = torch.as_tensor(float(K), device=device, dtype=dtype)

    h0_t = torch.as_tensor(float(h0), device=device, dtype=dtype)

    sinlat = torch.sin(lat)
    coslat = torch.cos(lat)
    cos_safe = torch.clamp(torch.abs(coslat), min=eps_cos) * torch.sign(coslat + 0.0)

    A = (omega_t / 2.0) * (2.0 * Omega + omega_t) * (coslat ** 2) + (K_t ** 2) / 4.0 * (coslat ** (2 * R_int)) * (
        (R_int + 1) * (coslat ** 2)
        + (2.0 * (R_int ** 2) - R_int - 2.0)
        - 2.0 * (R_int ** 2) * (cos_safe ** (-2))
    )

    B = 2.0 * (Omega + omega_t) * K_t / ((R_int + 1) * (R_int + 2)) * (coslat ** R_int) * (
        (R_int ** 2 + 2 * R_int + 2) - ((R_int + 1) ** 2) * (coslat ** 2)
    )

    C = (K_t ** 2) / 4.0 * (coslat ** (2 * R_int)) * (
        (R_int + 1) * (coslat ** 2) - (R_int + 2.0)
    )

    h = h0_t + (a ** 2) * (A + B * torch.cos(R_int * lon) + C * torch.cos(2.0 * R_int * lon)) / g

    cos_pow = coslat ** (R_int - 1)
    u = a * omega_t * coslat + a * K_t * cos_pow * (R_int * (sinlat ** 2) - (coslat ** 2)) * torch.cos(R_int * lon)
    v = -a * K_t * R_int * cos_pow * sinlat * torch.sin(R_int * lon)

    phi_grid = g * h

    uv_grid = torch.stack([u, v], dim=0)
    vrtdiv_spec = solver.vrtdivspec(uv_grid)
    if flip_vort:
        vrtdiv_spec[0] = -vrtdiv_spec[0]

    phi_spec = solver.grid2spec(torch.stack([phi_grid], dim=0))[0]

    ctype = torch.complex128 if solver.lap.dtype == torch.float64 else torch.complex64
    uspec = torch.zeros(3, solver.lmax, solver.mmax, dtype=ctype, device=device)
    uspec[0] = phi_spec
    uspec[1:] = vrtdiv_spec.to(dtype=ctype)

    return torch.tril(uspec)


def compute_random_ic_field_stats(
    solver,
    num_samples=200,
    mach=0.2,
):
    """
    Compute per-channel mean/std of solver.random_initial_condition(mach)
    using num_samples random ICs.
    Returns:
      field_mean: (3,1,1)
      field_std:  (3,1,1)
    """
    device = solver.lap.device
    dtype = solver.lap.dtype

    sum_f = torch.zeros(3, device=device, dtype=torch.float64)
    sumsq_f = torch.zeros(3, device=device, dtype=torch.float64)
    count_f = torch.tensor(0.0, device=device, dtype=torch.float64)

    with torch.no_grad():
        for _ in range(int(num_samples)):
            ref_spec = solver.random_initial_condition(mach=mach)
            ref_grid = solver.spec2grid(ref_spec).to(torch.float64)
            sum_f, sumsq_f, count_f = _online_update_mean_var(sum_f, sumsq_f, count_f, ref_grid)

    field_mean, field_var = _finalize_mean_var(sum_f, sumsq_f, count_f)
    field_mean = field_mean.to(dtype)
    field_std = torch.sqrt(field_var).to(dtype)
    return field_mean, field_std


def compute_stats_for_gbells_allfields_ic(
    solver,
    seed,
    num_samples=200,
    mach=0.2,
    k_min=1,
    k_max=8,
    sigma_min_deg=5.0,
    sigma_max_deg=20.0,
    signed=True,
):
    """
    Compute mean/var for the Gaussian-bells ALL-FIELDS IC distribution (t=0),
    for both fields (3 channels) and winds (2 channels), where the bells are
    scaled using per-channel mean/std estimated from 200 random initial conditions.
    """
    device = solver.lap.device
    dtype = solver.lap.dtype

    lat = solver.lats.reshape(-1, 1).to(device)
    lon = solver.lons.reshape(1, -1).to(device)

    sigma_min = math.radians(sigma_min_deg)
    sigma_max = math.radians(sigma_max_deg)

    def _rand(shape, idx, offset):
        g = torch.Generator(device=device)
        g.manual_seed(int(seed) + int(idx) + int(offset))
        return torch.rand(*shape, device=device, dtype=dtype, generator=g)

    # --- ref means/stds from 200 random initial conditions ---
    with torch.no_grad():
        ref_mean, ref_std = compute_random_ic_field_stats(
            solver,
            num_samples=200,
            mach=mach,
        )

        ref_mean0 = ref_mean[0, 0, 0]
        ref_mean1 = ref_mean[1, 0, 0]
        ref_mean2 = ref_mean[2, 0, 0]

        ref_std0 = ref_std[0, 0, 0].clamp_min(1e-12)
        ref_std1 = ref_std[1, 0, 0].clamp_min(1e-12)
        ref_std2 = ref_std[2, 0, 0].clamp_min(1e-12)

    def _sample_bells_grid(idx, ref_std, mean, offset_base):
        uK = _rand((1,), idx, offset_base + 5).item()
        K = int(k_min + math.floor(uK * (k_max - k_min + 1)))
        K = max(k_min, min(k_max, K))

        u = 2.0 * _rand((K,), idx, offset_base + 10) - 1.0
        lat0 = torch.asin(u)
        lon0 = 2.0 * math.pi * _rand((K,), idx, offset_base + 20)

        sigma = sigma_min + (sigma_max - sigma_min) * _rand((K,), idx, offset_base + 30)

        amp = _rand((K,), idx, offset_base + 40)
        if signed:
            signs = torch.where(
                _rand((K,), idx, offset_base + 50) < 0.5,
                -torch.ones_like(amp),
                torch.ones_like(amp),
            )
            amp = amp * signs

        bump = torch.zeros(solver.nlat, solver.nlon, device=device, dtype=dtype)
        for i in range(K):
            gamma = _great_circle_distance(lat, lon, lat0[i].view(1, 1), lon0[i].view(1, 1))
            bump = bump + amp[i] * torch.exp(-(gamma * gamma) / (2.0 * sigma[i] * sigma[i]))

        bump = bump - bump.mean()
        bump = bump / (bump.std() + 1e-6)

        return (mean + ref_std * bump).to(dtype)

    sum_f = torch.zeros(3, device=device, dtype=torch.float64)
    sumsq_f = torch.zeros(3, device=device, dtype=torch.float64)
    count_f = torch.tensor(0.0, device=device, dtype=torch.float64)

    sum_w = torch.zeros(2, device=device, dtype=torch.float64)
    sumsq_w = torch.zeros(2, device=device, dtype=torch.float64)
    count_w = torch.tensor(0.0, device=device, dtype=torch.float64)

    with torch.no_grad():
        for i in range(int(num_samples)):
            phi_grid = _sample_bells_grid(i, ref_std0, ref_mean0, offset_base=0)
            vort_grid = _sample_bells_grid(i, ref_std1, ref_mean1, offset_base=1000)
            div_grid = _sample_bells_grid(i, ref_std2, ref_mean2, offset_base=2000)

            inp_grid = torch.stack([phi_grid, vort_grid, div_grid], dim=0)
            inp_spec = solver.grid2spec(inp_grid)
            inp_spec = torch.tril(inp_spec)

            grid = solver.spec2grid(inp_spec).to(torch.float64)
            uv = solver.getuv(inp_spec[1:]).to(torch.float64)

            sum_f, sumsq_f, count_f = _online_update_mean_var(sum_f, sumsq_f, count_f, grid)
            sum_w, sumsq_w, count_w = _online_update_mean_var(sum_w, sumsq_w, count_w, uv)

    field_mean, field_var = _finalize_mean_var(sum_f, sumsq_f, count_f)
    wind_mean, wind_var = _finalize_mean_var(sum_w, sumsq_w, count_w)

    field_mean = field_mean.to(dtype)
    field_var = field_var.to(dtype)
    wind_mean = wind_mean.to(dtype)
    wind_var = wind_var.to(dtype)

    return field_mean, field_var, wind_mean, wind_var


def compute_component_scaled_ic_stats(wrapper):
    """
    Compute per-component stats AFTER component-level scaling.
    Returns dict keyed by component identity string.
    """
    if getattr(wrapper, "ic_mix_components", None) is None or getattr(wrapper, "ic_mix_flat_schedule", None) is None:
        return {}

    component_by_id = {int(comp["component_id"]): comp for comp in wrapper.ic_mix_components}
    accum = {}
    for comp in wrapper.ic_mix_components:
        component_id = int(comp["component_id"])
        accum[component_id] = {
            "sum_f": torch.zeros(3, device=wrapper.device, dtype=torch.float64),
            "sumsq_f": torch.zeros(3, device=wrapper.device, dtype=torch.float64),
            "count_f": torch.tensor(0.0, device=wrapper.device, dtype=torch.float64),
            "sum_w": torch.zeros(2, device=wrapper.device, dtype=torch.float64),
            "sumsq_w": torch.zeros(2, device=wrapper.device, dtype=torch.float64),
            "count_w": torch.tensor(0.0, device=wrapper.device, dtype=torch.float64),
        }

    with torch.no_grad():
        for sample_index, schedule_entry in enumerate(wrapper.ic_mix_flat_schedule):
            component_id = int(schedule_entry["component_id"])
            inp_spec = wrapper.build_inp_spec(sample_index)
            fields = wrapper.solver.spec2grid(inp_spec).to(torch.float64)
            winds = wrapper.solver.getuv(inp_spec[1:]).to(torch.float64)

            accum[component_id]["sum_f"], accum[component_id]["sumsq_f"], accum[component_id]["count_f"] = _online_update_mean_var(
                accum[component_id]["sum_f"],
                accum[component_id]["sumsq_f"],
                accum[component_id]["count_f"],
                fields,
            )
            accum[component_id]["sum_w"], accum[component_id]["sumsq_w"], accum[component_id]["count_w"] = _online_update_mean_var(
                accum[component_id]["sum_w"],
                accum[component_id]["sumsq_w"],
                accum[component_id]["count_w"],
                winds,
            )

    stats_by_identity = {}
    for comp in wrapper.ic_mix_components:
        component_id = int(comp["component_id"])
        component_identity = (
            f"kind={comp['kind']}|spec={comp['spec']}|scaling={comp.get('scaling_scheme', None)}"
        )

        field_mean, field_var = _finalize_mean_var(
            accum[component_id]["sum_f"],
            accum[component_id]["sumsq_f"],
            accum[component_id]["count_f"],
        )
        wind_mean, wind_var = _finalize_mean_var(
            accum[component_id]["sum_w"],
            accum[component_id]["sumsq_w"],
            accum[component_id]["count_w"],
        )

        field_mean = field_mean.to(wrapper.solver.lap.dtype)
        field_var = field_var.to(wrapper.solver.lap.dtype)
        wind_mean = wind_mean.to(wrapper.solver.lap.dtype)
        wind_var = wind_var.to(wrapper.solver.lap.dtype)

        stats_by_identity[component_identity] = {
            "component_id": component_id,
            "kind": comp["kind"],
            "spec": comp["spec"],
            "scaling_scheme": comp.get("scaling_scheme", None),
            "n": int(comp["count"]),
            "field_mean": field_mean,
            "field_var": field_var,
            "field_std": torch.sqrt(field_var),
            "wind_mean": wind_mean,
            "wind_var": wind_var,
            "wind_std": torch.sqrt(wind_var),
        }

    return stats_by_identity


def _resolve_scale_all_target_stats(scale_all_scheme, default_field_mean, default_field_var):
    scheme = str(scale_all_scheme).strip().lower()
    if scheme in {"unit", "zscore", "standard"}:
        return {
            "scheme": scheme,
            "field_mean": torch.zeros_like(default_field_mean),
            "field_var": torch.ones_like(default_field_var),
        }
    if scheme in {"gbells", "default"}:
        return {
            "scheme": scheme,
            "field_mean": default_field_mean,
            "field_var": default_field_var,
        }
    raise ValueError(
        f"Unknown --scale_all scheme '{scale_all_scheme}'. "
        "Supported schemes: unit, zscore, standard, gbells, default"
    )


def _component_identity_for_sample(wrapper, sample_index):
    schedule_entry = wrapper.ic_mix_flat_schedule[int(sample_index)]
    component_id = int(schedule_entry["component_id"])
    component = wrapper._ic_mix_component_by_id[component_id]
    return f"kind={component['kind']}|spec={component['spec']}|scaling={component.get('scaling_scheme', None)}"


def apply_scale_all_to_fields_for_sample(wrapper, inp_fields, sample_index):
    if getattr(wrapper, "scale_all_scheme", None) is None:
        return inp_fields

    component_identity = _component_identity_for_sample(wrapper, sample_index)
    component_stats = wrapper.component_scaled_stats_by_identity[component_identity]
    target_stats = wrapper.scale_all_target_stats

    component_std = component_stats["field_std"].clamp_min(1e-12)
    z_fields = (inp_fields - component_stats["field_mean"]) / component_std
    out_fields = z_fields * torch.sqrt(target_stats["field_var"]).clamp_min(1e-12) + target_stats["field_mean"]
    return out_fields


def compute_global_post_scale_all_stats(wrapper):
    sum_f = torch.zeros(3, device=wrapper.device, dtype=torch.float64)
    sumsq_f = torch.zeros(3, device=wrapper.device, dtype=torch.float64)
    count_f = torch.tensor(0.0, device=wrapper.device, dtype=torch.float64)

    sum_w = torch.zeros(2, device=wrapper.device, dtype=torch.float64)
    sumsq_w = torch.zeros(2, device=wrapper.device, dtype=torch.float64)
    count_w = torch.tensor(0.0, device=wrapper.device, dtype=torch.float64)

    with torch.no_grad():
        for idx in range(len(wrapper)):
            inp_spec = wrapper.build_inp_spec(idx)
            inp_fields = wrapper.solver.spec2grid(inp_spec)
            inp_fields = apply_scale_all_to_fields_for_sample(wrapper, inp_fields, idx)
            scaled_spec = torch.tril(wrapper.solver.grid2spec(inp_fields))
            inp_winds = wrapper.solver.getuv(scaled_spec[1:])

            sum_f, sumsq_f, count_f = _online_update_mean_var(sum_f, sumsq_f, count_f, inp_fields.to(torch.float64))
            sum_w, sumsq_w, count_w = _online_update_mean_var(sum_w, sumsq_w, count_w, inp_winds.to(torch.float64))

    field_mean, field_var = _finalize_mean_var(sum_f, sumsq_f, count_f)
    wind_mean, wind_var = _finalize_mean_var(sum_w, sumsq_w, count_w)
    dtype = wrapper.solver.lap.dtype
    field_mean = field_mean.to(dtype)
    field_var = field_var.to(dtype)
    wind_mean = wind_mean.to(dtype)
    wind_var = wind_var.to(dtype)

    return {
        "n": int(len(wrapper)),
        "scheme": wrapper.scale_all_scheme,
        "field_mean": field_mean,
        "field_var": field_var,
        "field_std": torch.sqrt(field_var),
        "wind_mean": wind_mean,
        "wind_var": wind_var,
        "wind_std": torch.sqrt(wind_var),
    }


def _resolve_normalize_scheme_field_stats(normalize_scheme, dataset_field_mean, dataset_field_var, gbells_field_mean, gbells_field_var):
    if normalize_scheme is None:
        return dataset_field_mean, dataset_field_var, "dataset"

    scheme = str(normalize_scheme).strip().lower()
    if scheme in {"dataset", "default", "final_dataset"}:
        return dataset_field_mean, dataset_field_var, scheme
    if scheme in {"gbells"}:
        return gbells_field_mean, gbells_field_var, scheme
    if scheme in {"unit", "zscore", "standard"}:
        return torch.zeros_like(dataset_field_mean), torch.ones_like(dataset_field_var), scheme

    raise ValueError(
        f"Unknown --normalize_scheme '{normalize_scheme}'. "
        "Supported schemes: dataset, default, final_dataset, gbells, unit, zscore, standard"
    )


def _validate_scale_all_scheme_name(scale_all_scheme):
    if scale_all_scheme is None:
        return None
    scheme = str(scale_all_scheme).strip().lower()
    allowed = {"unit", "zscore", "standard", "gbells", "default"}
    if scheme not in allowed:
        raise ValueError(
            f"Unknown --scale_all scheme '{scale_all_scheme}'. "
            f"Supported schemes: {sorted(allowed)}"
        )
    return scheme


def _validate_normalize_scheme_name(normalize_scheme):
    if normalize_scheme is None:
        return None
    scheme = str(normalize_scheme).strip().lower()
    allowed = {"dataset", "default", "final_dataset", "gbells", "unit", "zscore", "standard"}
    if scheme not in allowed:
        raise ValueError(
            f"Unknown --normalize_scheme '{normalize_scheme}'. "
            f"Supported schemes: {sorted(allowed)}"
        )
    return scheme


def _to_serializable(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return [_to_serializable(x) for x in value]
    if isinstance(value, list):
        return [_to_serializable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _build_args_snapshot(known_args):
    args_dict = {}
    for key, value in vars(known_args).items():
        args_dict[key] = _to_serializable(value)
    return args_dict


def save_ic_scaling_metadata_json(config, known_args, train_ds, val_ds):
    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_name = str(config["experiment"]["name"])
    out_path = os.path.join(save_dir, f"{experiment_name}_ic_scaling_metadata_{timestamp_utc}.json")

    train_base = train_ds.base
    val_base = val_ds.base

    payload = {
        "timestamp_utc": timestamp_utc,
        "seed": int(config["experiment"]["seed"]),
        "parsed_args": _build_args_snapshot(known_args),
        "resolved_components": _to_serializable(known_args.ic_mix_components),
        "component_stats": {
            "train": _to_serializable(getattr(train_base, "component_scaled_stats_by_identity", {})),
            "val": _to_serializable(getattr(val_base, "component_scaled_stats_by_identity", {})),
        },
        "scale_all": {
            "scheme": getattr(train_base, "scale_all_scheme", None),
            "global_post_scale_all_stats": {
                "train": _to_serializable(getattr(train_base, "global_post_scale_all_stats", None)),
                "val": _to_serializable(getattr(val_base, "global_post_scale_all_stats", None)),
            },
        },
        "normalization": {
            "source": {
                "train": getattr(train_base, "normalization_source", None),
                "val": getattr(val_base, "normalization_source", None),
            },
            "final": {
                "train": {
                    "inp_mean": _to_serializable(train_ds.inp_mean),
                    "inp_var": _to_serializable(train_ds.inp_var),
                    "wind_mean": _to_serializable(train_ds.wind_mean),
                    "wind_var": _to_serializable(train_ds.wind_var),
                },
                "val": {
                    "inp_mean": _to_serializable(val_ds.inp_mean),
                    "inp_var": _to_serializable(val_ds.inp_var),
                    "wind_mean": _to_serializable(val_ds.wind_mean),
                    "wind_var": _to_serializable(val_ds.wind_var),
                },
            },
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved IC/scaling metadata JSON: {out_path}")
    print(
        "Metadata summary: "
        f"components={len(known_args.ic_mix_components)}, "
        f"scale_all={getattr(train_base, 'scale_all_scheme', None)}, "
        f"norm_train={getattr(train_base, 'normalization_source', None)}, "
        f"norm_val={getattr(val_base, 'normalization_source', None)}"
    )
    return out_path


ROLLOUT_DATASET_SCHEMA_VERSION = "1.0"
ROLLOUT_DATASET_REQUIRED_SPLIT_KEYS = {
    "inp_fields_0",
    "inp_winds_0",
    "inp_spec_0",
    "tar_fields_rollout",
    "tar_winds_rollout",
}


def build_rollout_dataset_config_snapshot(config, K, burn_in):
    return {
        "dt": int(config["data"]["dt"]),
        "dt_solver": int(config["data"]["dt_solver"]),
        "burn_in": int(burn_in),
        "K": int(K),
    }


def build_rollout_dataset_config_fingerprint(config_snapshot):
    canonical_json = json.dumps(config_snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def build_rollout_dataset_split_payload(inp_fields_0, inp_winds_0, inp_spec_0, tar_fields_rollout, tar_winds_rollout):
    return {
        "inp_fields_0": inp_fields_0,
        "inp_winds_0": inp_winds_0,
        "inp_spec_0": inp_spec_0,
        "tar_fields_rollout": tar_fields_rollout,
        "tar_winds_rollout": tar_winds_rollout,
    }


def validate_rollout_dataset_split_payload(payload, split_name):
    missing_keys = [key for key in sorted(ROLLOUT_DATASET_REQUIRED_SPLIT_KEYS) if key not in payload]
    if missing_keys:
        raise ValueError(
            f"Invalid rollout dataset split payload for '{split_name}': "
            f"missing required keys {missing_keys}"
        )


def _payload_tensors_to_cpu(payload):
    cpu_payload = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            cpu_payload[key] = value.detach().to("cpu")
        else:
            cpu_payload[key] = value
    return cpu_payload


def load_rollout_dataset_split_cpu(path):
    payload = torch.load(path, map_location="cpu")
    return _payload_tensors_to_cpu(payload)


def load_rollout_dataset_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_tensor_cpu(value, dtype=torch.float32):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device="cpu")


def _validate_saved_config_mismatch_policy(saved_config, current_config):
    for field_name in ["dt", "dt_solver", "burn_in", "K"]:
        saved_value = saved_config.get(field_name, None)
        current_value = current_config.get(field_name, None)
        if saved_value == current_value:
            continue

        msg = (
            f"Precomputed rollout config mismatch for '{field_name}': "
            f"saved value={saved_value}, received value={current_value}; values do not match."
        )
        if field_name == "dt":
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)


def load_and_validate_precomputed_rollout_artifacts(rollout_dataset_dir, config, K, burn_in):
    train_out_path = os.path.join(rollout_dataset_dir, "train_rollout_dataset.pt")
    val_out_path = os.path.join(rollout_dataset_dir, "val_rollout_dataset.pt")
    metadata_out_path = os.path.join(rollout_dataset_dir, "rollout_dataset_metadata.json")

    missing_paths = [p for p in [train_out_path, val_out_path, metadata_out_path] if not os.path.isfile(p)]
    if missing_paths:
        raise FileNotFoundError(
            "Precomputed training requested but required rollout dataset artifacts are missing: "
            + ", ".join(missing_paths)
        )

    train_payload = load_rollout_dataset_split_cpu(train_out_path)
    val_payload = load_rollout_dataset_split_cpu(val_out_path)
    metadata_payload = load_rollout_dataset_metadata(metadata_out_path)

    validate_rollout_dataset_split_payload(train_payload, split_name="train")
    validate_rollout_dataset_split_payload(val_payload, split_name="val")

    schema_version = metadata_payload.get("schema_version", None)
    if schema_version != ROLLOUT_DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported rollout dataset schema version '{schema_version}'. "
            f"Expected '{ROLLOUT_DATASET_SCHEMA_VERSION}'."
        )

    for required_key in ["config", "post_burnin_rollout_stats"]:
        if required_key not in metadata_payload:
            raise ValueError(f"Invalid rollout metadata: missing required key '{required_key}'.")

    current_config_snapshot = build_rollout_dataset_config_snapshot(config=config, K=K, burn_in=burn_in)
    saved_config_snapshot = metadata_payload["config"]
    _validate_saved_config_mismatch_policy(saved_config_snapshot, current_config_snapshot)

    print(f"[rollout dataset] loaded precomputed train split: {train_out_path}")
    print(f"[rollout dataset] loaded precomputed val split: {val_out_path}")
    print(f"[rollout dataset] loaded precomputed metadata: {metadata_out_path}")

    return train_payload, val_payload, metadata_payload


def build_rollout_dataset_metadata(config_snapshot, post_burnin_rollout_stats):
    return {
        "schema_version": ROLLOUT_DATASET_SCHEMA_VERSION,
        "config": config_snapshot,
        "config_fingerprint": build_rollout_dataset_config_fingerprint(config_snapshot),
        "post_burnin_rollout_stats": post_burnin_rollout_stats,
    }


def _materialize_full_rollout_split(ds, K, split_name):
    print(f"[rollout dataset] materializing split={split_name}, n={len(ds)}, K={K}")

    inp_fields_0_list = []
    inp_winds_0_list = []
    inp_spec_0_list = []
    tar_fields_rollout_list = []
    tar_winds_rollout_list = []

    sqrt_inp_var = torch.sqrt(ds.inp_var)
    sqrt_wind_var = torch.sqrt(ds.wind_var)

    with torch.inference_mode():
        for idx in range(len(ds)):
            inp_fields_norm, inp_winds_norm, inp_spec_0 = ds[idx]

            inp_fields_0 = inp_fields_norm * sqrt_inp_var + ds.inp_mean
            inp_winds_0 = inp_winds_norm * sqrt_wind_var + ds.wind_mean

            cur_spec = inp_spec_0.clone()
            sample_tar_fields = []
            sample_tar_winds = []

            for _ in range(int(K)):
                cur_spec = ds.solver.timestep(cur_spec, ds.step_nsteps)
                sample_tar_fields.append(ds.solver.spec2grid(cur_spec).detach().to("cpu"))
                sample_tar_winds.append(ds.solver.getuv(cur_spec[1:]).detach().to("cpu"))

            inp_fields_0_list.append(inp_fields_0.detach().to("cpu"))
            inp_winds_0_list.append(inp_winds_0.detach().to("cpu"))
            inp_spec_0_list.append(inp_spec_0.detach().to("cpu"))
            tar_fields_rollout_list.append(torch.stack(sample_tar_fields, dim=0))
            tar_winds_rollout_list.append(torch.stack(sample_tar_winds, dim=0))

    split_payload = build_rollout_dataset_split_payload(
        inp_fields_0=torch.stack(inp_fields_0_list, dim=0),
        inp_winds_0=torch.stack(inp_winds_0_list, dim=0),
        inp_spec_0=torch.stack(inp_spec_0_list, dim=0),
        tar_fields_rollout=torch.stack(tar_fields_rollout_list, dim=0),
        tar_winds_rollout=torch.stack(tar_winds_rollout_list, dim=0),
    )
    validate_rollout_dataset_split_payload(split_payload, split_name=split_name)
    return split_payload


def save_full_rollout_dataset_artifacts(rollout_dataset_dir, train_ds, val_ds, config, K, burn_in):
    os.makedirs(rollout_dataset_dir, exist_ok=True)

    train_payload = _payload_tensors_to_cpu(_materialize_full_rollout_split(train_ds, K=K, split_name="train"))
    val_payload = _payload_tensors_to_cpu(_materialize_full_rollout_split(val_ds, K=K, split_name="val"))

    train_out_path = os.path.join(rollout_dataset_dir, "train_rollout_dataset.pt")
    val_out_path = os.path.join(rollout_dataset_dir, "val_rollout_dataset.pt")
    metadata_out_path = os.path.join(rollout_dataset_dir, "rollout_dataset_metadata.json")

    torch.save(train_payload, train_out_path)
    torch.save(val_payload, val_out_path)

    post_burnin_train_fields = train_payload["tar_fields_rollout"][:, int(burn_in):]
    post_burnin_val_fields = val_payload["tar_fields_rollout"][:, int(burn_in):]
    post_burnin_train_winds = train_payload["tar_winds_rollout"][:, int(burn_in):]
    post_burnin_val_winds = val_payload["tar_winds_rollout"][:, int(burn_in):]

    post_burnin_fields = torch.cat([post_burnin_train_fields, post_burnin_val_fields], dim=0).to(torch.float64)
    post_burnin_winds = torch.cat([post_burnin_train_winds, post_burnin_val_winds], dim=0).to(torch.float64)

    field_count = int(
        post_burnin_fields.shape[0]
        * post_burnin_fields.shape[1]
        * post_burnin_fields.shape[3]
        * post_burnin_fields.shape[4]
    )
    wind_count = int(
        post_burnin_winds.shape[0]
        * post_burnin_winds.shape[1]
        * post_burnin_winds.shape[3]
        * post_burnin_winds.shape[4]
    )

    field_mean = (post_burnin_fields.sum(dim=(0, 1, 3, 4)) / field_count).to(torch.float32).reshape(-1, 1, 1)
    field_var = (
        post_burnin_fields.square().sum(dim=(0, 1, 3, 4)) / field_count
        - field_mean.reshape(-1).to(torch.float64).square()
    ).clamp_min(1e-12)
    field_std = torch.sqrt(field_var).to(torch.float32).reshape(-1, 1, 1)

    wind_mean = (post_burnin_winds.sum(dim=(0, 1, 3, 4)) / wind_count).to(torch.float32).reshape(-1, 1, 1)
    wind_var = (
        post_burnin_winds.square().sum(dim=(0, 1, 3, 4)) / wind_count
        - wind_mean.reshape(-1).to(torch.float64).square()
    ).clamp_min(1e-12)
    wind_std = torch.sqrt(wind_var).to(torch.float32).reshape(-1, 1, 1)

    post_burnin_rollout_stats = {
        "fields_mean": field_mean.to("cpu"),
        "fields_std": field_std.to("cpu"),
        "winds_mean": wind_mean.to("cpu"),
        "winds_std": wind_std.to("cpu"),
        "num_samples": int(train_payload["tar_fields_rollout"].shape[0] + val_payload["tar_fields_rollout"].shape[0]),
        "num_steps_used": int(K - burn_in),
        "total_elements_accumulated": {
            "fields": field_count,
            "winds": wind_count,
        },
    }

    config_snapshot = build_rollout_dataset_config_snapshot(config, K=K, burn_in=burn_in)
    metadata_payload = build_rollout_dataset_metadata(
        config_snapshot=config_snapshot,
        post_burnin_rollout_stats=post_burnin_rollout_stats,
    )
    metadata_payload["shapes"] = {
        "train": {k: list(v.shape) for k, v in train_payload.items() if isinstance(v, torch.Tensor)},
        "val": {k: list(v.shape) for k, v in val_payload.items() if isinstance(v, torch.Tensor)},
    }
    metadata_payload["generated_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(metadata_out_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(metadata_payload), f, indent=2)

    print(f"[rollout dataset] saved train split: {train_out_path}")
    print(f"[rollout dataset] saved val split: {val_out_path}")
    print(f"[rollout dataset] saved metadata: {metadata_out_path}")


class GaussianBellsPhiWrapperWithWinds(torch.utils.data.Dataset):
    """
    Unused in the current rollout-loss training path, kept here for completeness.
    """
    def __init__(
        self,
        base_dataset,
        mach=0.2,
        k_min=1,
        k_max=8,
        sigma_min_deg=5.0,
        sigma_max_deg=20.0,
        signed=True,
        seed=None,
    ):
        self.base = base_dataset
        self.solver = base_dataset.solver
        self.device = base_dataset.device
        self.mach = mach

        self.k_min = k_min
        self.k_max = k_max
        self.sigma_min = math.radians(sigma_min_deg)
        self.sigma_max = math.radians(sigma_max_deg)
        self.signed = signed
        self.seed = seed

        self.lat = self.solver.lats.reshape(-1, 1).to(self.device)
        self.lon = self.solver.lons.reshape(1, -1).to(self.device)

        self.use_base_normalization = getattr(self.base, "normalize", False)
        if self.use_base_normalization:
            self.inp_mean = self.base.inp_mean
            self.inp_var = self.base.inp_var
            self.wind_mean = self.base.wind_mean
            self.wind_var = self.base.wind_var

        with torch.inference_mode():
            ref_mean, ref_std = compute_random_ic_field_stats(
                self.solver,
                num_samples=200,
                mach=self.mach,
            )
            self.ref_phi_mean = ref_mean[0, 0, 0]
            self.ref_phi_std = ref_std[0, 0, 0].clamp_min(1e-12)

    def __len__(self):
        return len(self.base)

    def _rand(self, shape, idx, offset=0):
        dtype = self.solver.lap.dtype
        if self.seed is None:
            return torch.rand(*shape, device=self.device, dtype=dtype)
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.seed) + int(idx) + int(offset))
        return torch.rand(*shape, device=self.device, dtype=dtype, generator=g)

    def _sample_phi_bells_grid(self, idx):
        dtype = self.solver.lap.dtype
        uK = self._rand((1,), idx, offset=5).item()
        K = int(self.k_min + math.floor(uK * (self.k_max - self.k_min + 1)))
        K = max(self.k_min, min(self.k_max, K))

        u = 2.0 * self._rand((K,), idx, 10) - 1.0
        lat0 = torch.asin(u)
        lon0 = 2.0 * math.pi * self._rand((K,), idx, 20)
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * self._rand((K,), idx, 30)

        amp = self._rand((K,), idx, 40)
        if self.signed:
            signs = torch.where(self._rand((K,), idx, 50) < 0.5, -torch.ones_like(amp), torch.ones_like(amp))
            amp = amp * signs

        bump = torch.zeros(self.solver.nlat, self.solver.nlon, device=self.device, dtype=dtype)
        for i in range(K):
            gamma = _great_circle_distance(self.lat, self.lon, lat0[i].view(1, 1), lon0[i].view(1, 1))
            bump = bump + amp[i] * torch.exp(-(gamma * gamma) / (2.0 * sigma[i] * sigma[i]))

        bump = bump - bump.mean()
        bump = bump / (bump.std() + 1e-6)

        return (self.ref_phi_mean + self.ref_phi_std * bump).to(dtype)

    def __getitem__(self, idx):
        with torch.inference_mode():
            inp_spec = self.solver.random_initial_condition(mach=self.mach)

            phi_grid = self._sample_phi_bells_grid(idx)
            zeros = torch.zeros_like(phi_grid)
            phi_spec0 = self.solver.grid2spec(torch.stack([phi_grid, zeros, zeros], dim=0))[0]

            inp_spec = inp_spec.clone()
            inp_spec[0] = phi_spec0
            inp_spec = torch.tril(inp_spec)

            tar_spec = self.solver.timestep(inp_spec, self.base.nsteps)

            inp_fields = self.solver.spec2grid(inp_spec)
            tar_fields = self.solver.spec2grid(tar_spec)

            inp_winds = self.solver.getuv(inp_spec[1:])
            tar_winds = self.solver.getuv(tar_spec[1:])

            if self.use_base_normalization:
                inp_fields = (inp_fields - self.inp_mean) / torch.sqrt(self.inp_var)
                tar_fields = (tar_fields - self.inp_mean) / torch.sqrt(self.inp_var)
                inp_winds = (inp_winds - self.wind_mean) / torch.sqrt(self.wind_var)
                tar_winds = (tar_winds - self.wind_mean) / torch.sqrt(self.wind_var)

            return inp_fields.clone(), inp_winds.clone(), tar_fields.clone(), tar_winds.clone()


class GaussianBellsAllFieldsWrapperWithWinds(torch.utils.data.Dataset):
    """
    Gaussian bells for ALL THREE fields (phi + vorticity + divergence),
    scaled using per-channel mean/std estimated from 200 random initial conditions.
    This wrapper does NOT normalize; trajectory dataset will normalize using gbells stats.
    """

    def __init__(
        self,
        base_dataset,
        mach=0.2,
        k_min=1,
        k_max=8,
        sigma_min_deg=5.0,
        sigma_max_deg=20.0,
        signed=True,
        seed=None,
        ic_mix_components=None,
        ic_mix_flat_schedule=None,
    ):
        self.base = base_dataset
        self.solver = base_dataset.solver
        self.device = base_dataset.device
        self.mach = mach

        self.k_min = k_min
        self.k_max = k_max
        self.sigma_min = math.radians(sigma_min_deg)
        self.sigma_max = math.radians(sigma_max_deg)
        self.signed = signed
        self.seed = seed
        self.ic_mix_components = ic_mix_components
        self.ic_mix_flat_schedule = ic_mix_flat_schedule
        self._ic_mix_total_count = len(ic_mix_flat_schedule) if ic_mix_flat_schedule is not None else None
        self._ic_mix_component_by_id = {}
        if self.ic_mix_components is not None:
            self._ic_mix_component_by_id = {int(comp["component_id"]): comp for comp in self.ic_mix_components}
            for comp in self.ic_mix_components:
                _ = get_ic_scaling_scheme(comp.get("scaling_scheme", None))

        self._coverage_logs_emitted = set()

        self.lat = self.solver.lats.reshape(-1, 1).to(self.device)
        self.lon = self.solver.lons.reshape(1, -1).to(self.device)

        self.inp_mean = None
        self.inp_var = None
        self.wind_mean = None
        self.wind_var = None

        # scale matching: 200 random initial-condition means/stds
        with torch.inference_mode():
            ref_mean, ref_std = compute_random_ic_field_stats(
                self.solver,
                num_samples=200,
                mach=self.mach,
            )

            # Per-channel random-IC statistics used to scale Gaussian bells:
            # channel 0 = geopotential, 1 = vorticity, 2 = divergence.
            self.ref_mean0 = ref_mean[0, 0, 0]
            self.ref_mean1 = ref_mean[1, 0, 0]
            self.ref_mean2 = ref_mean[2, 0, 0]

            self.ref_std0 = ref_std[0, 0, 0].clamp_min(1e-12)
            self.ref_std1 = ref_std[1, 0, 0].clamp_min(1e-12)
            self.ref_std2 = ref_std[2, 0, 0].clamp_min(1e-12)

    def __len__(self):
        if self._ic_mix_total_count is not None:
            return int(self._ic_mix_total_count)
        return len(self.base)

    def _rand(self, shape, idx, offset=0):
        dtype = self.solver.lap.dtype
        if self.seed is None:
            return torch.rand(*shape, device=self.device, dtype=dtype)
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.seed) + int(idx) + int(offset))
        return torch.rand(*shape, device=self.device, dtype=dtype, generator=g)

    def _sample_bells_grid(self, idx, ref_std, mean=0.0, offset_base=0):
        dtype = self.solver.lap.dtype

        uK = self._rand((1,), idx, offset=offset_base + 5).item()
        K = int(self.k_min + math.floor(uK * (self.k_max - self.k_min + 1)))
        K = max(self.k_min, min(self.k_max, K))

        u = 2.0 * self._rand((K,), idx, offset_base + 10) - 1.0
        lat0 = torch.asin(u)
        lon0 = 2.0 * math.pi * self._rand((K,), idx, offset_base + 20)

        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * self._rand((K,), idx, offset_base + 30)

        amp = self._rand((K,), idx, offset_base + 40)
        if self.signed:
            signs = torch.where(
                self._rand((K,), idx, offset_base + 50) < 0.5,
                -torch.ones_like(amp),
                torch.ones_like(amp),
            )
            amp = amp * signs

        bump = torch.zeros(self.solver.nlat, self.solver.nlon, device=self.device, dtype=dtype)
        for i in range(K):
            gamma = _great_circle_distance(self.lat, self.lon, lat0[i].view(1, 1), lon0[i].view(1, 1))
            bump = bump + amp[i] * torch.exp(-(gamma * gamma) / (2.0 * sigma[i] * sigma[i]))

        bump = bump - bump.mean()
        bump = bump / (bump.std() + 1e-6)

        # Unnormalized bells are scaled to random-IC distribution as:
        # field_channel = random_ic_mean_channel + random_ic_std_channel * normalized_bump
        return (mean + ref_std * bump).to(dtype)

    def build_inp_spec(self, idx):
        if self.ic_mix_flat_schedule is None:
            phi_grid = self._sample_bells_grid(idx, ref_std=self.ref_std0, mean=self.ref_mean0, offset_base=0)
            vort_grid = self._sample_bells_grid(idx, ref_std=self.ref_std1, mean=self.ref_mean1, offset_base=1000)
            div_grid = self._sample_bells_grid(idx, ref_std=self.ref_std2, mean=self.ref_mean2, offset_base=2000)

            inp_grid = torch.stack([phi_grid, vort_grid, div_grid], dim=0)
            inp_spec = self.solver.grid2spec(inp_grid)
            inp_spec = torch.tril(inp_spec)
            return inp_spec

        schedule_entry = self.ic_mix_flat_schedule[int(idx)]
        component = self._ic_mix_component_by_id[int(schedule_entry["component_id"])]
        component_local_index = int(schedule_entry["component_local_index"])
        component_id = int(component["component_id"])
        component_sample_index = int(component_id) * 1000003 + int(component_local_index)
        global_seed = int(self.seed) if self.seed is not None else 0
        sample_seed_base = global_seed + component_sample_index

        coverage_key = (
            component["kind"],
            str(component["spec"]),
            str(component.get("scaling_scheme", None)),
        )
        if coverage_key not in self._coverage_logs_emitted:
            print(
                "IC path coverage: "
                f"kind={component['kind']} spec={component['spec']} "
                f"scaling={component.get('scaling_scheme', None)}"
            )
            self._coverage_logs_emitted.add(coverage_key)

        if component["kind"] == "triplet":
            channels = tuple(component["spec"])
            channel_ref_std = [self.ref_std0, self.ref_std1, self.ref_std2]
            channel_ref_mean = [self.ref_mean0, self.ref_mean1, self.ref_mean2]

            torch.manual_seed(sample_seed_base + 1)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed_base + 1)
            random_spec = self.solver.random_initial_condition(mach=self.mach)
            random_grid = self.solver.spec2grid(random_spec)

            out_channels = []
            for channel_idx in range(3):
                mode = int(channels[channel_idx])
                if mode == 0:
                    out_channels.append(random_grid[channel_idx])
                elif mode == 1:
                    out_channels.append(
                        self._sample_bells_grid(
                            component_sample_index,
                            ref_std=channel_ref_std[channel_idx],
                            mean=channel_ref_mean[channel_idx],
                            offset_base=1000 * channel_idx,
                        )
                    )
                elif mode == 2:
                    out_channels.append(torch.zeros(self.solver.nlat, self.solver.nlon, device=self.device, dtype=self.solver.lap.dtype))
                else:
                    raise ValueError(f"Unsupported triplet channel mode {mode}")

            inp_grid = torch.stack(out_channels, dim=0)
            inp_grid = apply_ic_scaling(
                component.get("scaling_scheme", None),
                inp_grid,
                context={
                    "component": component,
                    "sample_seed_base": sample_seed_base,
                    "sample_index": int(idx),
                },
            )
            inp_spec = self.solver.grid2spec(inp_grid)
            inp_spec = torch.tril(inp_spec)
            return inp_spec

        if component["kind"] == "williamson_case2":
            alpha_g = torch.Generator(device=self.device)
            alpha_g.manual_seed(int(sample_seed_base) + 7001)
            alpha = (0.5 * torch.pi * torch.rand(1, device=self.device, dtype=self.solver.lap.dtype, generator=alpha_g)).item()

            inp_spec = make_williamson_case2_ic_spec_from_winds(self.solver, alpha=alpha)
            inp_grid = self.solver.spec2grid(inp_spec)
            inp_grid = apply_ic_scaling(
                component.get("scaling_scheme", None),
                inp_grid,
                context={
                    "component": component,
                    "sample_seed_base": sample_seed_base,
                    "sample_index": int(idx),
                },
            )
            inp_spec = self.solver.grid2spec(inp_grid)
            return torch.tril(inp_spec)

        if component["kind"] == "williamson_case6":
            inp_spec = make_williamson_case6_ic_spec_from_winds(self.solver)
            inp_grid = self.solver.spec2grid(inp_spec)
            inp_grid = apply_ic_scaling(
                component.get("scaling_scheme", None),
                inp_grid,
                context={
                    "component": component,
                    "sample_seed_base": sample_seed_base,
                    "sample_index": int(idx),
                },
            )
            inp_spec = self.solver.grid2spec(inp_grid)
            return torch.tril(inp_spec)

        raise ValueError(f"Unsupported IC component kind '{component['kind']}'")

    def __getitem__(self, idx):
        with torch.inference_mode():
            inp_spec = self.build_inp_spec(idx)
            tar_spec = self.solver.timestep(inp_spec, self.base.nsteps)

            inp_fields = self.solver.spec2grid(inp_spec)
            tar_fields = self.solver.spec2grid(tar_spec)

            inp_winds = self.solver.getuv(inp_spec[1:])
            tar_winds = self.solver.getuv(tar_spec[1:])

            return inp_fields.clone(), inp_winds.clone(), tar_fields.clone(), tar_winds.clone()


class TrajectoryFromSolverWithWinds(torch.utils.data.Dataset):
    """
    Converts an IC generator (in spectral space) into normalized inputs.
    Rollout targets are generated on-the-fly in training/validation using
    solver.timestep to avoid storing [K, C, H, W] targets per sample.
    Returns:
      inp_fields, inp_winds, inp_spec
    """
    def __init__(self, gbells_wrapper, K, step_nsteps=None):
        self.base = gbells_wrapper
        self.solver = self.base.solver
        self.device = self.base.device
        self.K = int(K)
        assert self.K >= 1

        self.step_nsteps = int(step_nsteps) if step_nsteps is not None else int(self.base.base.nsteps)

        self.inp_mean = self.base.inp_mean
        self.inp_var = self.base.inp_var
        self.wind_mean = self.base.wind_mean
        self.wind_var = self.base.wind_var

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        with torch.inference_mode():
            inp_spec = self.base.build_inp_spec(idx)

            inp_fields = self.solver.spec2grid(inp_spec)
            if getattr(self.base, "scale_all_scheme", None) is not None:
                inp_fields = apply_scale_all_to_fields_for_sample(self.base, inp_fields, idx)
                inp_spec = torch.tril(self.solver.grid2spec(inp_fields))

            inp_winds = self.solver.getuv(inp_spec[1:])

            assert self.inp_mean is not None and self.inp_var is not None, "Gaussian-bells field stats not set"
            assert self.wind_mean is not None and self.wind_var is not None, "Gaussian-bells wind stats not set"

            inp_fields = (inp_fields - self.inp_mean) / torch.sqrt(self.inp_var)
            inp_winds = (inp_winds - self.wind_mean) / torch.sqrt(self.wind_var)

            return inp_fields.clone(), inp_winds.clone(), inp_spec.clone()


class PrecomputedRolloutDataset(torch.utils.data.Dataset):
    """
    Dataset backed by persisted rollout artifacts.
    Returns normalized step-0 NN inputs plus saved unnormalized rollout truth.
    """
    def __init__(self, payload, solver, inp_mean, inp_var, wind_mean, wind_var):
        validate_rollout_dataset_split_payload(payload, split_name="precomputed")
        self.solver = solver

        self.inp_fields_0 = payload["inp_fields_0"].detach().to("cpu")
        self.inp_winds_0 = payload["inp_winds_0"].detach().to("cpu")
        self.inp_spec_0 = payload["inp_spec_0"].detach().to("cpu")
        self.tar_fields_rollout = payload["tar_fields_rollout"].detach().to("cpu")
        self.tar_winds_rollout = payload["tar_winds_rollout"].detach().to("cpu")

        self.inp_mean = inp_mean.detach().to("cpu")
        self.inp_var = inp_var.detach().to("cpu")
        self.wind_mean = wind_mean.detach().to("cpu")
        self.wind_var = wind_var.detach().to("cpu")

    def __len__(self):
        return int(self.inp_fields_0.shape[0])

    def __getitem__(self, idx):
        inp_fields_norm = (self.inp_fields_0[idx] - self.inp_mean) / torch.sqrt(self.inp_var)
        inp_winds_norm = (self.inp_winds_0[idx] - self.wind_mean) / torch.sqrt(self.wind_var)
        return (
            inp_fields_norm.clone(),
            inp_winds_norm.clone(),
            self.inp_spec_0[idx].clone(),
            self.tar_fields_rollout[idx].clone(),
            self.tar_winds_rollout[idx].clone(),
        )


class ReversedHuberLossS2(torch.nn.Module):
    """
    Reverse Huber (inverted Huber), continuous at delta:
        rho(a)=a                           if a<=delta
        rho(a)=(a^2+delta^2)/(2*delta)     if a>delta
    with GraphCast-consistent latitude weights (unit-mean), same as ParadisLoss.
    """

    def __init__(
        self,
        nlat: int,
        nlon: int,
        grid: str,
        lat_grid_deg: torch.Tensor,
        delta: float = 1.0,
        apply_latitude_weights: bool = True,
    ):
        super().__init__()
        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.delta = float(delta)
        self.apply_latitude_weights = bool(apply_latitude_weights)

        lat_w = self._compute_latitude_weights(lat_grid_deg)
        self.register_buffer("lat_weights", lat_w.view(1, 1, nlat, 1), persistent=False)

    def _compute_latitude_weights(self, grid_lat_deg: torch.Tensor) -> torch.Tensor:
        lat = grid_lat_deg.reshape(-1).to(dtype=torch.float64)
        weights = torch.cos(torch.deg2rad(lat)).clamp_min(0.0)

        if torch.all(weights <= 0):
            weights = torch.ones_like(weights)

        weights = weights / weights.mean()
        return weights.to(dtype=grid_lat_deg.dtype)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        e = (pred - target).abs()
        d = torch.as_tensor(self.delta, device=pred.device, dtype=pred.dtype)

        loss = torch.where(e <= d, e, (e * e + d * d) / (2.0 * d))

        if self.apply_latitude_weights:
            loss = loss * self.lat_weights.to(device=pred.device, dtype=pred.dtype)

        return loss.mean()


class SWERolloutLightningModule(pl.LightningModule):
    def __init__(
        self,
        config,
        n_rollout,
        burn_in,
        detach_after_burnin=True,
        which_loss="SquaredL2",
        loss_delta=1.0,
        apply_latitude_weights=True,
        lat_grid_deg=None,
        rollout_mode="disabled",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config

        self.nlat = config["data"]["nlat"]
        self.nlon = config["data"]["nlon"]
        self.grid = config["data"]["grid"]
        self.model_type = config["experiment"]["model_type"]
        self.use_winds = self.model_type == "paradis"

        if self.model_type == "sfno":
            self.model = self._create_sfno_model()
        elif self.model_type == "transformer":
            self.model = self._create_transformer_model()
        elif self.model_type == "paradis":
            self.model = self._create_paradis_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        which_loss_key = str(which_loss).strip().lower()
        if which_loss_key == "berhu":
            if lat_grid_deg is None:
                raise ValueError("lat_grid_deg must be provided when --which_loss berhu is used.")
            self.loss_fn = ReversedHuberLossS2(
                nlat=self.nlat,
                nlon=self.nlon,
                grid=self.grid,
                lat_grid_deg=lat_grid_deg,
                delta=float(loss_delta),
                apply_latitude_weights=bool(apply_latitude_weights),
            )
        elif which_loss_key == "squaredl2":
            self.loss_fn = SquaredL2LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        elif which_loss_key == "amse":
            self.loss_fn = AMSELoss(nlat=self.nlat, nlon=self.nlon, grid=self.grid, norm="backward")
        else:
            raise ValueError(f"Unknown --which_loss value: {which_loss}. Use SquaredL2, berhu, or amse.")

        self.metric_l1 = L1LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_l2 = L2LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)
        self.metric_w11 = W11LossS2(nlat=self.nlat, nlon=self.nlon, grid=self.grid)

        self.n_rollout = int(n_rollout)
        self.burn_in = int(burn_in)
        self.detach_after_burnin = bool(detach_after_burnin)
        self.rollout_mode = str(rollout_mode)

        assert self.n_rollout >= 1
        assert 0 <= self.burn_in < self.n_rollout, "burn_in must be < n_rollout"

        self.rollout_eval_bundle = None

    def _batch_to_device(self, batch):
        return tuple(x.to(self.device, non_blocking=True) if torch.is_tensor(x) else x for x in batch)

    def _create_sfno_model(self):
        mc = self.config["model"]["sfno"]
        return SphericalFourierNeuralOperator(
            img_size=(self.nlat, self.nlon),
            grid=self.grid,
            grid_internal=self.grid,
            scale_factor=mc["scale_factor"],
            in_chans=3,
            out_chans=3,
            embed_dim=mc["embed_dim"],
            num_layers=mc["num_layers"],
            normalization_layer=mc["normalization_layer"],
            use_mlp=mc["use_mlp"],
            mlp_ratio=mc["mlp_ratio"],
            drop_rate=mc["dropout"],
            hard_thresholding_fraction=mc["hard_thresholding_fraction"],
            residual_prediction=True,
        )

    def _create_transformer_model(self):
        mc = self.config["model"]["transformer"]
        return SphericalTransformer(
            img_size=(self.nlat, self.nlon),
            grid=self.grid,
            scale_factor=mc["scale_factor"],
            in_chans=3,
            out_chans=3,
            embed_dim=mc["embed_dim"],
            num_layers=mc["num_layers"],
            num_heads=mc["num_heads"],
            use_mlp=mc["use_mlp"],
            mlp_ratio=mc["mlp_ratio"],
            drop_rate=mc["dropout"],
            drop_path_rate=mc["drop_path"],
            pos_embed=mc["pos_embed"],
        )

    def _create_paradis_model(self):
        return ParadisModel(self.config)

    def forward(self, *args):
        return self.model(*args) if self.use_winds else self.model(args[0])

    def _bundle_on_device(self):
        if self.rollout_eval_bundle is None:
            raise RuntimeError("rollout_eval_bundle has not been set on the model.")
        bundle = self.rollout_eval_bundle
        solver = bundle["solver"].to(self.device)
        out = {
            "solver": solver,
            "inp_mean": bundle["inp_mean"].to(self.device),
            "inp_var": bundle["inp_var"].to(self.device),
            "wind_mean": bundle["wind_mean"].to(self.device) if bundle.get("wind_mean", None) is not None else None,
            "wind_var": bundle["wind_var"].to(self.device) if bundle.get("wind_var", None) is not None else None,
            "nsteps": bundle["nsteps"],
            "use_winds": bundle["use_winds"],
        }
        return out

    def _recompute_pred_winds(self, prd_fields_norm):
        bundle = self._bundle_on_device()
        solver = bundle["solver"]
        inp_mean = bundle["inp_mean"]
        inp_var = bundle["inp_var"]
        wind_mean = bundle["wind_mean"]
        wind_var = bundle["wind_var"]

        prd_fields_phys = prd_fields_norm * torch.sqrt(inp_var) + inp_mean
        prd_spec = solver.grid2spec(prd_fields_phys)
        prd_spec = torch.tril(prd_spec)
        prd_winds_phys = solver.getuv(prd_spec[:, 1:])
        prd_winds_norm = (prd_winds_phys - wind_mean) / torch.sqrt(wind_var)
        return prd_winds_norm

    def _step_truth_specs(self, cur_spec_batch):
        bundle = self._bundle_on_device()
        solver = bundle["solver"]
        nsteps = bundle["nsteps"]

        next_specs = []
        for b in range(cur_spec_batch.shape[0]):
            next_spec = solver.timestep(cur_spec_batch[b], nsteps)
            next_specs.append(next_spec)
        return torch.stack(next_specs, dim=0)

    def _log_channel_stats(self, prefix, tensor, channel_names, sync_dist=False, on_step=False, on_epoch=True):
        with torch.no_grad():
            for channel_idx, channel_name in enumerate(channel_names):
                chan = tensor[:, channel_idx]
                self.log(
                    f"{prefix}_{channel_name}_avg",
                    chan.mean(),
                    sync_dist=sync_dist,
                    on_step=on_step,
                    on_epoch=on_epoch,
                )
                self.log(
                    f"{prefix}_{channel_name}_min",
                    chan.min(),
                    sync_dist=sync_dist,
                    on_step=on_step,
                    on_epoch=on_epoch,
                )
                self.log(
                    f"{prefix}_{channel_name}_max",
                    chan.max(),
                    sync_dist=sync_dist,
                    on_step=on_step,
                    on_epoch=on_epoch,
                )

    def _rollout_and_collect(self, inp_fields, inp_winds, inp_spec, collect_eval_metrics=False):
        """
        inp_fields: (B,3,H,W) normalized
        inp_winds:  (B,2,H,W) normalized
        inp_spec:   (B,3,L,M) complex, unnormalized spectral IC
        Returns dict with differentiable train_loss and detached rollout statistics.
        """
        bundle = self._bundle_on_device()
        solver = bundle["solver"]
        inp_mean = bundle["inp_mean"]
        inp_var = bundle["inp_var"]
        wind_mean = bundle["wind_mean"]
        wind_var = bundle["wind_var"]

        sqrt_inp_var = torch.sqrt(inp_var)
        sqrt_wind_var = torch.sqrt(wind_var)

        prd = inp_fields
        cur_winds = inp_winds
        cur_truth_spec = inp_spec
        loss = 0.0
        denom = 0

        loss_step1 = None
        loss_final = None
        wind_err_step1 = None
        wind_err_final = None

        l1_sum = 0.0
        l2_sum = 0.0
        w11_sum = 0.0
        l1_step1 = None
        l2_step1 = None
        w11_step1 = None
        l1_final = None
        l2_final = None
        w11_final = None

        wind_err_sum = 0.0

        prd_fields_first_phys = None
        prd_fields_final_phys = None
        tar_fields_first_phys = None
        tar_fields_final_phys = None
        prd_winds_first_phys = None
        prd_winds_final_phys = None
        tar_winds_first_phys = None
        tar_winds_final_phys = None

        if self.burn_in > 0:
            with torch.no_grad():
                for _ in range(self.burn_in):
                    cur_truth_spec = self._step_truth_specs(cur_truth_spec)
                burnin_fields_phys = solver.spec2grid(cur_truth_spec)
                burnin_winds_phys = solver.getuv(cur_truth_spec[:, 1:])
                prd = (burnin_fields_phys - inp_mean) / sqrt_inp_var
                cur_winds = (burnin_winds_phys - wind_mean) / sqrt_wind_var

        for t in range(self.burn_in + 1, self.n_rollout + 1):
            if self.use_winds:
                prd = self.model(prd, cur_winds)
            else:
                prd = self.model(prd)

            with torch.no_grad():
                cur_truth_spec = self._step_truth_specs(cur_truth_spec)
                tar_fields_phys = solver.spec2grid(cur_truth_spec)
                tar_fields = (tar_fields_phys - inp_mean) / sqrt_inp_var

                tar_winds_phys = solver.getuv(cur_truth_spec[:, 1:])
                tar_winds = (tar_winds_phys - wind_mean) / sqrt_wind_var

            if self.use_winds and t < self.n_rollout:
                pred_winds = self._recompute_pred_winds(prd)
                cur_winds = pred_winds
            elif self.use_winds:
                with torch.no_grad():
                    pred_winds = self._recompute_pred_winds(prd)

            step_loss = self.loss_fn(prd, tar_fields)
            loss = loss + step_loss
            denom += 1

            with torch.no_grad():
                prd_fields_phys = prd * sqrt_inp_var + inp_mean
                prd_winds_phys = pred_winds * sqrt_wind_var + wind_mean

                step_wind_err = torch.mean((pred_winds - tar_winds) ** 2)
                wind_err_sum = wind_err_sum + step_wind_err

                if loss_step1 is None:
                    loss_step1 = step_loss.detach()
                    wind_err_step1 = step_wind_err.detach()
                    prd_fields_first_phys = prd_fields_phys.detach()
                    tar_fields_first_phys = tar_fields_phys.detach()
                    prd_winds_first_phys = prd_winds_phys.detach()
                    tar_winds_first_phys = tar_winds_phys.detach()

                loss_final = step_loss.detach()
                wind_err_final = step_wind_err.detach()
                prd_fields_final_phys = prd_fields_phys.detach()
                tar_fields_final_phys = tar_fields_phys.detach()
                prd_winds_final_phys = prd_winds_phys.detach()
                tar_winds_final_phys = tar_winds_phys.detach()

                if collect_eval_metrics:
                    step_l1 = self.metric_l1(prd, tar_fields)
                    step_l2 = self.metric_l2(prd, tar_fields)
                    step_w11 = self.metric_w11(prd, tar_fields)

                    l1_sum = l1_sum + step_l1
                    l2_sum = l2_sum + step_l2
                    w11_sum = w11_sum + step_w11

                    if l1_step1 is None:
                        l1_step1 = step_l1.detach()
                        l2_step1 = step_l2.detach()
                        w11_step1 = step_w11.detach()

                    l1_final = step_l1.detach()
                    l2_final = step_l2.detach()
                    w11_final = step_w11.detach()

        loss = loss / max(denom, 1)
        avg_wind_err = wind_err_sum / max(denom, 1)

        out = {
            "train_loss": loss,
            "loss_step1": loss_step1,
            "loss_final": loss_final,
            "loss_avg": loss.detach(),
            "wind_err_step1": wind_err_step1,
            "wind_err_final": wind_err_final,
            "wind_err_avg": avg_wind_err.detach() if torch.is_tensor(avg_wind_err) else avg_wind_err,
            "prd_fields_first_phys": prd_fields_first_phys,
            "tar_fields_first_phys": tar_fields_first_phys,
            "prd_fields_final_phys": prd_fields_final_phys,
            "tar_fields_final_phys": tar_fields_final_phys,
            "prd_winds_first_phys": prd_winds_first_phys,
            "tar_winds_first_phys": tar_winds_first_phys,
            "prd_winds_final_phys": prd_winds_final_phys,
            "tar_winds_final_phys": tar_winds_final_phys,
        }

        if collect_eval_metrics:
            out.update(
                {
                    "l1_step1": l1_step1,
                    "l1_final": l1_final,
                    "l1_avg": (l1_sum / max(denom, 1)).detach(),
                    "l2_step1": l2_step1,
                    "l2_final": l2_final,
                    "l2_avg": (l2_sum / max(denom, 1)).detach(),
                    "w11_step1": w11_step1,
                    "w11_final": w11_final,
                    "w11_avg": (w11_sum / max(denom, 1)).detach(),
                }
            )

        return out

    def _rollout_and_collect_from_saved(self, inp_fields, inp_winds, tar_fields_rollout_phys, tar_winds_rollout_phys, collect_eval_metrics=False):
        bundle = self._bundle_on_device()
        inp_mean = bundle["inp_mean"]
        inp_var = bundle["inp_var"]
        wind_mean = bundle["wind_mean"]
        wind_var = bundle["wind_var"]

        sqrt_inp_var = torch.sqrt(inp_var)
        sqrt_wind_var = torch.sqrt(wind_var)

        prd = inp_fields
        cur_winds = inp_winds
        loss = 0.0
        denom = 0

        loss_step1 = None
        loss_final = None
        wind_err_step1 = None
        wind_err_final = None

        l1_sum = 0.0
        l2_sum = 0.0
        w11_sum = 0.0
        l1_step1 = None
        l2_step1 = None
        w11_step1 = None
        l1_final = None
        l2_final = None
        w11_final = None

        wind_err_sum = 0.0

        prd_fields_first_phys = None
        prd_fields_final_phys = None
        tar_fields_first_phys = None
        tar_fields_final_phys = None
        prd_winds_first_phys = None
        prd_winds_final_phys = None
        tar_winds_first_phys = None
        tar_winds_final_phys = None

        if self.burn_in > 0:
            with torch.no_grad():
                burnin_fields_phys = tar_fields_rollout_phys[:, self.burn_in - 1]
                burnin_winds_phys = tar_winds_rollout_phys[:, self.burn_in - 1]
                prd = (burnin_fields_phys - inp_mean) / sqrt_inp_var
                cur_winds = (burnin_winds_phys - wind_mean) / sqrt_wind_var

        for t in range(self.burn_in + 1, self.n_rollout + 1):
            if self.use_winds:
                prd = self.model(prd, cur_winds)
            else:
                prd = self.model(prd)

            with torch.no_grad():
                tar_fields_phys = tar_fields_rollout_phys[:, t - 1]
                tar_winds_phys = tar_winds_rollout_phys[:, t - 1]
                tar_fields = (tar_fields_phys - inp_mean) / sqrt_inp_var
                tar_winds = (tar_winds_phys - wind_mean) / sqrt_wind_var

            if self.use_winds and t < self.n_rollout:
                pred_winds = self._recompute_pred_winds(prd)
                cur_winds = pred_winds
            elif self.use_winds:
                with torch.no_grad():
                    pred_winds = self._recompute_pred_winds(prd)

            step_loss = self.loss_fn(prd, tar_fields)
            loss = loss + step_loss
            denom += 1

            with torch.no_grad():
                prd_fields_phys = prd * sqrt_inp_var + inp_mean
                prd_winds_phys = pred_winds * sqrt_wind_var + wind_mean

                step_wind_err = torch.mean((pred_winds - tar_winds) ** 2)
                wind_err_sum = wind_err_sum + step_wind_err

                if loss_step1 is None:
                    loss_step1 = step_loss.detach()
                    wind_err_step1 = step_wind_err.detach()
                    prd_fields_first_phys = prd_fields_phys.detach()
                    tar_fields_first_phys = tar_fields_phys.detach()
                    prd_winds_first_phys = prd_winds_phys.detach()
                    tar_winds_first_phys = tar_winds_phys.detach()

                loss_final = step_loss.detach()
                wind_err_final = step_wind_err.detach()
                prd_fields_final_phys = prd_fields_phys.detach()
                tar_fields_final_phys = tar_fields_phys.detach()
                prd_winds_final_phys = prd_winds_phys.detach()
                tar_winds_final_phys = tar_winds_phys.detach()

                if collect_eval_metrics:
                    step_l1 = self.metric_l1(prd, tar_fields)
                    step_l2 = self.metric_l2(prd, tar_fields)
                    step_w11 = self.metric_w11(prd, tar_fields)

                    l1_sum = l1_sum + step_l1
                    l2_sum = l2_sum + step_l2
                    w11_sum = w11_sum + step_w11

                    if l1_step1 is None:
                        l1_step1 = step_l1.detach()
                        l2_step1 = step_l2.detach()
                        w11_step1 = step_w11.detach()

                    l1_final = step_l1.detach()
                    l2_final = step_l2.detach()
                    w11_final = step_w11.detach()

        loss = loss / max(denom, 1)
        avg_wind_err = wind_err_sum / max(denom, 1)

        out = {
            "train_loss": loss,
            "loss_step1": loss_step1,
            "loss_final": loss_final,
            "loss_avg": loss.detach(),
            "wind_err_step1": wind_err_step1,
            "wind_err_final": wind_err_final,
            "wind_err_avg": avg_wind_err.detach() if torch.is_tensor(avg_wind_err) else avg_wind_err,
            "prd_fields_first_phys": prd_fields_first_phys,
            "tar_fields_first_phys": tar_fields_first_phys,
            "prd_fields_final_phys": prd_fields_final_phys,
            "tar_fields_final_phys": tar_fields_final_phys,
            "prd_winds_first_phys": prd_winds_first_phys,
            "tar_winds_first_phys": tar_winds_first_phys,
            "prd_winds_final_phys": prd_winds_final_phys,
            "tar_winds_final_phys": tar_winds_final_phys,
        }

        if collect_eval_metrics:
            out.update(
                {
                    "l1_step1": l1_step1,
                    "l1_final": l1_final,
                    "l1_avg": (l1_sum / max(denom, 1)).detach(),
                    "l2_step1": l2_step1,
                    "l2_final": l2_final,
                    "l2_avg": (l2_sum / max(denom, 1)).detach(),
                    "w11_step1": w11_step1,
                    "w11_final": w11_final,
                    "w11_avg": (w11_sum / max(denom, 1)).detach(),
                }
            )

        return out

    def _rollout_final_prediction(self, inp_fields, inp_winds):
        prd = inp_fields
        cur_winds = inp_winds

        for t in range(1, self.n_rollout + 1):
            if self.use_winds:
                prd = self.model(prd, cur_winds)
                if t < self.n_rollout:
                    cur_winds = self._recompute_pred_winds(prd)
            else:
                prd = self.model(prd)

        return prd

    def training_step(self, batch, batch_idx):
        if self.use_winds:
            batch = self._batch_to_device(batch)
            if self.rollout_mode == "precomputed_training":
                inp_fields, inp_winds, _inp_spec, tar_fields_rollout_phys, tar_winds_rollout_phys = batch
                metrics = self._rollout_and_collect_from_saved(
                    inp_fields,
                    inp_winds,
                    tar_fields_rollout_phys,
                    tar_winds_rollout_phys,
                    collect_eval_metrics=False,
                )
            else:
                inp_fields, inp_winds, inp_spec = batch
                metrics = self._rollout_and_collect(inp_fields, inp_winds, inp_spec, collect_eval_metrics=False)
            loss = metrics["train_loss"]
        else:
            raise RuntimeError("This script currently expects winds (PARADIS) trajectory batches.")

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_rollout_step1_loss", metrics["loss_step1"], on_step=True, on_epoch=True)
        self.log("train_rollout_final_loss", metrics["loss_final"], on_step=True, on_epoch=True)
        self.log("train_winds_step1_error", metrics["wind_err_step1"], on_step=True, on_epoch=True)
        self.log("train_winds_final_error", metrics["wind_err_final"], on_step=True, on_epoch=True)
        self.log("train_winds_avg_error", metrics["wind_err_avg"], on_step=True, on_epoch=True)

        self._log_channel_stats(
            "train_nn_fields_step1", metrics["prd_fields_first_phys"], ["geopotential", "vorticity", "divergence"], on_step=True, on_epoch=False
        )
        self._log_channel_stats(
            "train_solver_fields_step1", metrics["tar_fields_first_phys"], ["geopotential", "vorticity", "divergence"], on_step=True, on_epoch=False
        )
        self._log_channel_stats(
            "train_nn_fields_final", metrics["prd_fields_final_phys"], ["geopotential", "vorticity", "divergence"], on_step=True, on_epoch=False
        )
        self._log_channel_stats(
            "train_solver_fields_final", metrics["tar_fields_final_phys"], ["geopotential", "vorticity", "divergence"], on_step=True, on_epoch=False
        )
        self._log_channel_stats("train_nn_winds_step1", metrics["prd_winds_first_phys"], ["u", "v"], on_step=True, on_epoch=False)
        self._log_channel_stats("train_solver_winds_step1", metrics["tar_winds_first_phys"], ["u", "v"], on_step=True, on_epoch=False)
        self._log_channel_stats("train_nn_winds_final", metrics["prd_winds_final_phys"], ["u", "v"], on_step=True, on_epoch=False)
        self._log_channel_stats("train_solver_winds_final", metrics["tar_winds_final_phys"], ["u", "v"], on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.use_winds:
            batch = self._batch_to_device(batch)
            if self.rollout_mode == "precomputed_training":
                inp_fields, inp_winds, _inp_spec, tar_fields_rollout_phys, tar_winds_rollout_phys = batch
                metrics = self._rollout_and_collect_from_saved(
                    inp_fields,
                    inp_winds,
                    tar_fields_rollout_phys,
                    tar_winds_rollout_phys,
                    collect_eval_metrics=True,
                )
            else:
                inp_fields, inp_winds, inp_spec = batch
                metrics = self._rollout_and_collect(inp_fields, inp_winds, inp_spec, collect_eval_metrics=True)
            loss = metrics["train_loss"]
        else:
            raise RuntimeError("This script currently expects winds (PARADIS) trajectory batches.")

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_step1_loss", metrics["loss_step1"], sync_dist=True)
        self.log("val_final_loss", metrics["loss_final"], sync_dist=True)
        self.log("val_avg_loss", metrics["loss_avg"], sync_dist=True)

        self.log("val_l1", metrics["l1_final"], sync_dist=True)
        self.log("val_l1_step1", metrics["l1_step1"], sync_dist=True)
        self.log("val_l1_avg", metrics["l1_avg"], sync_dist=True)

        self.log("val_l2", metrics["l2_final"], sync_dist=True)
        self.log("val_l2_step1", metrics["l2_step1"], sync_dist=True)
        self.log("val_l2_avg", metrics["l2_avg"], sync_dist=True)

        self.log("val_w11", metrics["w11_final"], sync_dist=True)
        self.log("val_w11_step1", metrics["w11_step1"], sync_dist=True)
        self.log("val_w11_avg", metrics["w11_avg"], sync_dist=True)

        self.log("val_winds_step1_error", metrics["wind_err_step1"], sync_dist=True)
        self.log("val_winds_final_error", metrics["wind_err_final"], sync_dist=True)
        self.log("val_winds_avg_error", metrics["wind_err_avg"], sync_dist=True)

        self._log_channel_stats(
            "val_nn_fields_step1", metrics["prd_fields_first_phys"], ["geopotential", "vorticity", "divergence"], sync_dist=True
        )
        self._log_channel_stats(
            "val_solver_fields_step1", metrics["tar_fields_first_phys"], ["geopotential", "vorticity", "divergence"], sync_dist=True
        )
        self._log_channel_stats(
            "val_nn_fields_final", metrics["prd_fields_final_phys"], ["geopotential", "vorticity", "divergence"], sync_dist=True
        )
        self._log_channel_stats(
            "val_solver_fields_final", metrics["tar_fields_final_phys"], ["geopotential", "vorticity", "divergence"], sync_dist=True
        )
        self._log_channel_stats("val_nn_winds_step1", metrics["prd_winds_first_phys"], ["u", "v"], sync_dist=True)
        self._log_channel_stats("val_solver_winds_step1", metrics["tar_winds_first_phys"], ["u", "v"], sync_dist=True)
        self._log_channel_stats("val_nn_winds_final", metrics["prd_winds_final_phys"], ["u", "v"], sync_dist=True)
        self._log_channel_stats("val_solver_winds_final", metrics["tar_winds_final_phys"], ["u", "v"], sync_dist=True)

        return loss

    def configure_optimizers(self):
        lr = self.config["training"]["finetune_learning_rate"]
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}


def _run_rollout_eval_case(pl_module, case_name, ic_spec, autoreg_steps=5):
    bundle = pl_module._bundle_on_device()
    solver = bundle["solver"]
    nsteps = bundle["nsteps"]
    use_winds = bundle["use_winds"]

    inp_mean = bundle["inp_mean"]
    inp_var = bundle["inp_var"]
    wind_mean = bundle["wind_mean"]
    wind_var = bundle["wind_var"]

    sqrt_inp_var = torch.sqrt(inp_var)
    sqrt_wind_var = torch.sqrt(wind_var) if wind_var is not None else None

    metrics_data = {
        "loss": [],
        "L1_error": [],
        "L2_error": [],
        "W11_error": [],
    }

    was_training = pl_module.training
    pl_module.eval()

    with torch.no_grad():
        prd_fields = (solver.spec2grid(ic_spec) - inp_mean) / sqrt_inp_var
        prd_fields = prd_fields.unsqueeze(0)

        if use_winds:
            prd_winds = solver.getuv(ic_spec[1:])
            prd_winds = (prd_winds - wind_mean) / sqrt_wind_var
            prd_winds = prd_winds.unsqueeze(0)
        else:
            prd_winds = None

        uspec = ic_spec.clone()

        print("-" * 70, flush=True)
        print(f"{case_name}: rollout evaluation for {autoreg_steps} steps", flush=True)

        for step in range(1, autoreg_steps + 1):
            if use_winds:
                prd_fields = pl_module(prd_fields, prd_winds)
                prd_unnorm = prd_fields * sqrt_inp_var + inp_mean
                prd_spec = solver.grid2spec(prd_unnorm.squeeze(0))
                prd_spec = torch.tril(prd_spec)
                prd_uv_grid = solver.getuv(prd_spec[1:])
                prd_winds = (prd_uv_grid - wind_mean) / sqrt_wind_var
                prd_winds = prd_winds.unsqueeze(0)
            else:
                prd_fields = pl_module(prd_fields)

            uspec = solver.timestep(uspec, nsteps)
            ref_grid = solver.spec2grid(uspec)
            ref_fields = (ref_grid - inp_mean) / sqrt_inp_var
            ref_fields = ref_fields.unsqueeze(0)

            l1 = pl_module.metric_l1(prd_fields, ref_fields).item()
            l2 = pl_module.metric_l2(prd_fields, ref_fields).item()
            w11 = pl_module.metric_w11(prd_fields, ref_fields).item()
            loss = pl_module.loss_fn(prd_fields, ref_fields).item()

            metrics_data["loss"].append(loss)
            metrics_data["L1_error"].append(l1)
            metrics_data["L2_error"].append(l2)
            metrics_data["W11_error"].append(w11)

            print(
                f"{case_name} Step {step}: "
                f"L1_error: {l1:.6f}, "
                f"L2_error: {l2:.6f}, "
                f"W11_error: {w11:.6f}, "
                f"loss: {loss:.6f}",
                flush=True,
            )

    summary = {}
    print(f"{case_name} SUMMARY", flush=True)
    for key, vals in metrics_data.items():
        vals_t = torch.tensor(vals, dtype=torch.float64, device=pl_module.device)
        mean_val = vals_t.mean()
        std_val = vals_t.std(unbiased=False)
        summary[f"{key}_mean"] = mean_val
        summary[f"{key}_std"] = std_val
        print(f"{case_name} {key:12s}: {mean_val.item():.6f} ± {std_val.item():.6f}", flush=True)

    if was_training:
        pl_module.train()

    return summary


class WilliamsonRolloutCallback(pl.Callback):
    def __init__(self, autoreg_steps=5):
        super().__init__()
        self.autoreg_steps = autoreg_steps

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if not hasattr(pl_module, "rollout_eval_bundle") or pl_module.rollout_eval_bundle is None:
            return

        solver = pl_module.rollout_eval_bundle["solver"].to(pl_module.device)

        print("\n" + "=" * 70, flush=True)
        print(f"WILLIAMSON ROLLOUT EVAL AFTER EPOCH {trainer.current_epoch}", flush=True)
        print("=" * 70, flush=True)

        wc2_ic = make_williamson_case2_ic_spec_from_winds(
            solver,
            gh0=29400.0,
            flip_vort=False,
        )
        wc2_summary = _run_rollout_eval_case(
            pl_module,
            "Williamson Case 2",
            wc2_ic,
            autoreg_steps=self.autoreg_steps,
        )

        wc6_ic = make_williamson_case6_ic_spec_from_winds(
            solver,
            R=4,
            omega=7.848e-6,
            K=None,
            h0=8000.0,
            flip_vort=False,
        )
        wc6_summary = _run_rollout_eval_case(
            pl_module,
            "Williamson Case 6",
            wc6_ic,
            autoreg_steps=self.autoreg_steps,
        )

        trainer.callback_metrics["wc2_rollout_loss"] = wc2_summary["loss_mean"].detach()
        trainer.callback_metrics["wc6_rollout_loss"] = wc6_summary["loss_mean"].detach()

        pl_module.log("wc2_rollout_loss", wc2_summary["loss_mean"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("wc6_rollout_loss", wc6_summary["loss_mean"], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        print("=" * 70 + "\n", flush=True)


def create_datasets_for_rollout(
    config,
    device,
    K,
    ic_mix_components=None,
    ic_mix_flat_schedule=None,
    scale_all_scheme=None,
    normalize_scheme=None,
):
    dt = config["data"]["dt"]
    dt_solver = config["data"]["dt_solver"]
    nsteps = dt // dt_solver
    nlat = config["data"]["nlat"]
    nlon = config["data"]["nlon"]
    grid = config["data"]["grid"]

    model_type = config["experiment"]["model_type"]
    use_winds = model_type == "paradis"
    if not use_winds:
        raise RuntimeError("This rollout-loss script is set up for PARADIS (winds) only.")

    base_train = PdeDatasetWithWinds(
        dt=dt, nsteps=nsteps, dims=(nlat, nlon), grid=grid, normalize=True, device=device
    )
    base_train.sht = RealSHT(nlat=nlat, nlon=nlon, grid=grid).to(device)
    base_train.set_initial_condition("random")
    base_train.set_num_examples(config["data"]["num_train_examples"])

    base_val = PdeDatasetWithWinds(
        dt=dt, nsteps=nsteps, dims=(nlat, nlon), grid=grid, normalize=True, device=device
    )
    base_val.sht = RealSHT(nlat=nlat, nlon=nlon, grid=grid).to(device)
    base_val.set_initial_condition("random")
    base_val.set_num_examples(config["data"]["num_val_examples"])

    bells_train = GaussianBellsAllFieldsWrapperWithWinds(
        base_train,
        mach=0.2, k_min=1, k_max=8,
        sigma_min_deg=5.0, sigma_max_deg=20.0,
        signed=True,
        seed=config["experiment"]["seed"],
        ic_mix_components=ic_mix_components,
        ic_mix_flat_schedule=ic_mix_flat_schedule,
    )
    bells_val = GaussianBellsAllFieldsWrapperWithWinds(
        base_val,
        mach=0.2, k_min=1, k_max=8,
        sigma_min_deg=5.0, sigma_max_deg=20.0,
        signed=True,
        seed=config["experiment"]["seed"] + 12345,
        ic_mix_components=ic_mix_components,
        ic_mix_flat_schedule=ic_mix_flat_schedule,
    )

    gb_field_mean, gb_field_var, gb_wind_mean, gb_wind_var = compute_stats_for_gbells_allfields_ic(
        solver=base_train.solver,
        seed=int(config["experiment"]["seed"]),
        num_samples=200,
        mach=0.2,
        k_min=1, k_max=8,
        sigma_min_deg=5.0, sigma_max_deg=20.0,
        signed=True,
    )

    for ds in [bells_train, bells_val]:
        ds.inp_mean = gb_field_mean
        ds.inp_var = gb_field_var
        ds.wind_mean = gb_wind_mean
        ds.wind_var = gb_wind_var
        ds.scale_all_scheme = scale_all_scheme
        ds.scale_all_target_stats = None
        ds.global_post_scale_all_stats = None
        ds.global_final_dataset_stats = None

    for name, ds in [("train", bells_train), ("val", bells_val)]:
        ds.component_scaled_stats_by_identity = compute_component_scaled_ic_stats(ds)
        if ds.component_scaled_stats_by_identity:
            print(f"Per-component scaled IC stats ({name}):")
            for identity, stats in ds.component_scaled_stats_by_identity.items():
                print(
                    "  "
                    f"id={stats['component_id']} "
                    f"n={stats['n']} "
                    f"{identity}"
                )
                print(f"    field_mean={stats['field_mean'].view(-1).tolist()}")
                print(f"    field_std ={stats['field_std'].view(-1).tolist()}")
                print(f"    wind_mean ={stats['wind_mean'].view(-1).tolist()}")
                print(f"    wind_std  ={stats['wind_std'].view(-1).tolist()}")

    if scale_all_scheme is not None:
        print(f"Scale-all selected scheme: {scale_all_scheme}")
        for ds in [bells_train, bells_val]:
            ds.scale_all_target_stats = _resolve_scale_all_target_stats(
                scale_all_scheme,
                default_field_mean=gb_field_mean,
                default_field_var=gb_field_var,
            )

    for name, ds in [("train", bells_train), ("val", bells_val)]:
        ds.global_final_dataset_stats = compute_global_post_scale_all_stats(ds)
        if scale_all_scheme is not None:
            ds.global_post_scale_all_stats = ds.global_final_dataset_stats

        if scale_all_scheme is not None and normalize_scheme is None:
            selected_field_mean = ds.global_final_dataset_stats["field_mean"]
            selected_field_var = ds.global_final_dataset_stats["field_var"]
            selected_norm_source = "scale_all_global"
        else:
            selected_field_mean, selected_field_var, selected_norm_source = _resolve_normalize_scheme_field_stats(
                normalize_scheme,
                dataset_field_mean=ds.global_final_dataset_stats["field_mean"],
                dataset_field_var=ds.global_final_dataset_stats["field_var"],
                gbells_field_mean=gb_field_mean,
                gbells_field_var=gb_field_var,
            )

        ds.inp_mean = selected_field_mean
        ds.inp_var = selected_field_var
        ds.wind_mean = ds.global_final_dataset_stats["wind_mean"]
        ds.wind_var = ds.global_final_dataset_stats["wind_var"]
        ds.normalization_source = selected_norm_source

        print(
            f"Normalization source ({name}): "
            f"inp={selected_norm_source}, winds=dataset_final, n={ds.global_final_dataset_stats['n']}"
        )
        print(f"  inp_mean ={ds.inp_mean.view(-1).tolist()}")
        print(f"  inp_std  ={torch.sqrt(ds.inp_var).view(-1).tolist()}")
        print(f"  wind_mean={ds.wind_mean.view(-1).tolist()}")
        print(f"  wind_std ={torch.sqrt(ds.wind_var).view(-1).tolist()}")

    print("Gaussian-bells normalization set:")
    print("  fields mean:", gb_field_mean.view(-1).tolist())
    print("  fields std :", torch.sqrt(gb_field_var).view(-1).tolist())
    print("  winds  mean:", gb_wind_mean.view(-1).tolist())
    print("  winds  std :", torch.sqrt(gb_wind_var).view(-1).tolist())

    train_ds = TrajectoryFromSolverWithWinds(bells_train, K=K, step_nsteps=nsteps)
    val_ds = TrajectoryFromSolverWithWinds(bells_val, K=K, step_nsteps=nsteps)

    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Rollout-loss training with configurable IC mixture/scaling.",
        epilog=(
            "Examples:\n"
            "  --ic_mix 0 1 2 512\n"
            "  --ic_mix williamson_case2 1024 my_scheme\n"
            "  --ic_mix 0 1 2 256 --ic_mix williamson_case6 128"
        ),
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--pretrain_ckpt",
        type=str,
        default=None,
        help="Optional path to BEST pretrain checkpoint (.ckpt) to load weights from.",
    )
    parser.add_argument(
        "--burn_in",
        type=int,
        default=5,
        help="Ignore loss for steps 1..burn_in. Start loss at burn_in+1.",
    )
    parser.add_argument(
        "--detach_after_burnin",
        action="store_true",
        default=True,
        help="If set, detach the prediction after burn_in to truncate gradients.",
    )
    parser.add_argument(
        "--n_rollout",
        type=int,
        default=None,
        help="Total rollout steps K. If not set, uses 1 + training.nfuture from config.",
    )
    parser.add_argument(
        "--which_loss",
        type=str,
        default="SquaredL2",
        help="Training loss to use: SquaredL2 (default), berhu, or amse.",
    )
    parser.add_argument(
        "--loss_delta",
        type=float,
        default=1.0,
        help="Delta for berhu loss (used only when --which_loss berhu).",
    )
    parser.add_argument(
        "--apply_latitude_weights",
        type=_str2bool,
        default=True,
        help="true/false. Applies latitude weights for berhu (used only when --which_loss berhu).",
    )
    parser.add_argument(
        "--ic_mix",
        nargs="+",
        action="append",
        default=None,
        help=(
            "Repeatable IC mix component.\n"
            "Forms: "
            "(b0 b1 b2 n [scaling_scheme]) or "
            "(williamson_case2 n [scaling_scheme]) or "
            "(williamson_case6 n [scaling_scheme]).\n"
            "Examples:\n"
            "  --ic_mix 0 1 2 512\n"
            "  --ic_mix williamson_case2 1024 my_scheme\n"
            "  --ic_mix 0 1 2 256 --ic_mix williamson_case6 128"
        ),
    )
    parser.add_argument(
        "--scale_all",
        type=str,
        default=None,
        help="Optional global scale_all scheme (unit|zscore|standard|gbells|default).",
    )
    parser.add_argument(
        "--normalize_scheme",
        type=str,
        default=None,
        help="Optional normalization scheme identifier.",
    )
    parser.add_argument(
        "--rollout_dataset_dir",
        type=str,
        default=None,
        help="Optional directory for persistent precomputed rollout dataset.",
    )
    known_args, unknown_args = parser.parse_known_args()

    rollout_dataset_dir = known_args.rollout_dataset_dir
    rollout_dataset_mode = "disabled"
    if rollout_dataset_dir is None:
        rollout_dataset_mode = "disabled"
    else:
        if os.path.exists(rollout_dataset_dir) and not os.path.isdir(rollout_dataset_dir):
            raise ValueError(
                f"--rollout_dataset_dir must point to a directory path. Got non-directory: {rollout_dataset_dir}"
            )
        if os.path.isdir(rollout_dataset_dir):
            rollout_dataset_mode = "precomputed_training"
        else:
            rollout_dataset_mode = "build_dataset_only"

    print(f"[rollout dataset] mode={rollout_dataset_mode}")
    print(f"[rollout dataset] directory={rollout_dataset_dir}")
    if rollout_dataset_mode == "disabled":
        print("[rollout dataset] branch=flag absent; using default training flow")
    elif rollout_dataset_mode == "build_dataset_only":
        print("[rollout dataset] branch=directory missing; build_dataset_only path selected")
    else:
        print("[rollout dataset] branch=directory exists; precomputed_training path selected")

    user_passed_scale_all = "--scale_all" in sys.argv
    user_passed_normalize_scheme = "--normalize_scheme" in sys.argv
    if user_passed_scale_all and user_passed_normalize_scheme:
        raise ValueError("--scale_all and --normalize_scheme are mutually exclusive; pass only one.")

    known_args.scale_all = _validate_scale_all_scheme_name(known_args.scale_all)
    known_args.normalize_scheme = _validate_normalize_scheme_name(known_args.normalize_scheme)

    try:
        parsed_ic_mix_components = parse_ic_mix_components(known_args.ic_mix)
    except argparse.ArgumentTypeError as exc:
        raise ValueError(f"Invalid --ic_mix: {exc}") from exc

    ic_mix_components, ic_mix_flat_schedule, ic_mix_total_count = build_ic_mix_component_index_and_schedule(
        parsed_ic_mix_components
    )
    known_args.ic_mix_components = ic_mix_components
    known_args.ic_mix_flat_schedule = ic_mix_flat_schedule
    known_args.ic_mix_total_count = ic_mix_total_count

    print("IC mix components:")
    for component in ic_mix_components:
        print(
            "  "
            f"id={component['component_id']} "
            f"kind={component['kind']} "
            f"spec={component['spec']} "
            f"scaling={component['scaling_scheme']} "
            f"count={component['count']}"
        )
    print(f"IC mix total samples: {ic_mix_total_count}")
    print(f"Parsed scaling flags: scale_all={known_args.scale_all}, normalize_scheme={known_args.normalize_scheme}")
    print(
        "IC scaling registry: "
        f"default={IC_SCALING_DEFAULT_SCHEME}, "
        f"available={sorted(IC_SCALING_REGISTRY.keys())}"
    )

    user_passed_loss_delta = "--loss_delta" in sys.argv
    user_passed_apply_lat_weights = "--apply_latitude_weights" in sys.argv
    which_loss_key = str(known_args.which_loss).strip().lower()
    if which_loss_key != "berhu" and (user_passed_loss_delta or user_passed_apply_lat_weights):
        warnings.warn(
            "--loss_delta and --apply_latitude_weights are ignored unless --which_loss berhu is used.",
            stacklevel=2,
        )

    mp.set_start_method("spawn", force=True)

    config = load_config(known_args.config)
    config = update_config_from_args(config, unknown_args)

    pl.seed_everything(config["experiment"]["seed"], workers=True)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if known_args.n_rollout is None:
        K = 1 + int(config["training"]["nfuture"])
    else:
        K = int(known_args.n_rollout)

    if not (0 <= known_args.burn_in < K):
        raise ValueError(f"burn_in must be in [0, K-1]. Got burn_in={known_args.burn_in}, K={K}")

    precomputed_train_payload = None
    precomputed_val_payload = None
    precomputed_metadata_payload = None
    if rollout_dataset_mode == "precomputed_training":
        precomputed_train_payload, precomputed_val_payload, precomputed_metadata_payload = load_and_validate_precomputed_rollout_artifacts(
            rollout_dataset_dir=rollout_dataset_dir,
            config=config,
            K=K,
            burn_in=known_args.burn_in,
        )

    if rollout_dataset_mode == "precomputed_training":
        dt = config["data"]["dt"]
        dt_solver = config["data"]["dt_solver"]
        nsteps = dt // dt_solver
        nlat = config["data"]["nlat"]
        nlon = config["data"]["nlon"]
        grid = config["data"]["grid"]

        solver = PdeDatasetWithWinds(
            dt=dt,
            nsteps=nsteps,
            dims=(nlat, nlon),
            grid=grid,
            normalize=True,
            device=device,
        ).solver

        stats = precomputed_metadata_payload["post_burnin_rollout_stats"]
        rollout_inp_mean = _to_tensor_cpu(stats["fields_mean"])
        rollout_inp_std = _to_tensor_cpu(stats["fields_std"])
        rollout_wind_mean = _to_tensor_cpu(stats["winds_mean"])
        rollout_wind_std = _to_tensor_cpu(stats["winds_std"])
        rollout_inp_var = rollout_inp_std.square().clamp_min(1e-12)
        rollout_wind_var = rollout_wind_std.square().clamp_min(1e-12)

        train_ds = PrecomputedRolloutDataset(
            payload=precomputed_train_payload,
            solver=solver,
            inp_mean=rollout_inp_mean,
            inp_var=rollout_inp_var,
            wind_mean=rollout_wind_mean,
            wind_var=rollout_wind_var,
        )
        val_ds = PrecomputedRolloutDataset(
            payload=precomputed_val_payload,
            solver=solver,
            inp_mean=rollout_inp_mean,
            inp_var=rollout_inp_var,
            wind_mean=rollout_wind_mean,
            wind_var=rollout_wind_var,
        )
    else:
        train_ds, val_ds = create_datasets_for_rollout(
            config,
            device,
            K=K,
            ic_mix_components=known_args.ic_mix_components,
            ic_mix_flat_schedule=known_args.ic_mix_flat_schedule,
            scale_all_scheme=known_args.scale_all,
            normalize_scheme=known_args.normalize_scheme,
        )

    if rollout_dataset_mode == "build_dataset_only":
        save_full_rollout_dataset_artifacts(
            rollout_dataset_dir=rollout_dataset_dir,
            train_ds=train_ds,
            val_ds=val_ds,
            config=config,
            K=K,
            burn_in=known_args.burn_in,
        )
        print("[rollout dataset] build_dataset_only complete; exiting before training.")
        return

    if hasattr(train_ds, "base") and hasattr(val_ds, "base"):
        save_ic_scaling_metadata_json(config, known_args, train_ds, val_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        persistent_workers=(config["data"]["num_workers"] > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        persistent_workers=(config["data"]["num_workers"] > 0),
    )

    model = SWERolloutLightningModule(
        config=config,
        n_rollout=K,
        burn_in=known_args.burn_in,
        detach_after_burnin=known_args.detach_after_burnin,
        which_loss=known_args.which_loss,
        loss_delta=known_args.loss_delta,
        apply_latitude_weights=known_args.apply_latitude_weights,
        lat_grid_deg=(train_ds.solver.lats * 180.0 / math.pi).to(torch.float32),
        rollout_mode=rollout_dataset_mode,
    )

    dt = config["data"]["dt"]
    dt_solver = config["data"]["dt_solver"]
    nsteps = dt // dt_solver

    model.rollout_eval_bundle = {
        "solver": train_ds.solver,
        "nsteps": nsteps,
        "use_winds": (config["experiment"]["model_type"] == "paradis"),
        "inp_mean": train_ds.inp_mean,
        "inp_var": train_ds.inp_var,
        "wind_mean": train_ds.wind_mean,
        "wind_var": train_ds.wind_var,
    }

    if known_args.pretrain_ckpt is not None:
        print(f"Loading pretrain checkpoint weights: {known_args.pretrain_ckpt}")
        ckpt = torch.load(known_args.pretrain_ckpt, map_location=device)
        state_dict = ckpt["state_dict"]

        for k in ["metric_w11.k_phi_mesh", "metric_w11.k_theta_mesh"]:
            if k in state_dict:
                del state_dict[k]

        model.load_state_dict(state_dict, strict=False)
        print("Loaded weights.\n")
    else:
        print("No pretrain checkpoint provided. Training from scratch.\n")

    logger = TensorBoardLogger(
        config["training"]["save_dir"],
        name=f"{config['experiment']['name']}_rolloutloss",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="rolloutloss-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    wc2_checkpoint_callback = ModelCheckpoint(
        monitor="wc2_rollout_loss",
        filename="rolloutloss-best-wc2-epoch{epoch:02d}-wc2_{wc2_rollout_loss:.6f}",
        save_top_k=1,
        mode="min",
        save_last=False,
        auto_insert_metric_name=False,
    )

    wc6_checkpoint_callback = ModelCheckpoint(
        monitor="wc6_rollout_loss",
        filename="rolloutloss-best-wc6-epoch{epoch:02d}-wc6_{wc6_rollout_loss:.6f}",
        save_top_k=1,
        mode="min",
        save_last=False,
        auto_insert_metric_name=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    precision = 32
    if config["training"]["amp_mode"] == "fp16":
        precision = 16
    elif config["training"]["amp_mode"] == "bf16":
        precision = "bf16"

    trainer = pl.Trainer(
        max_epochs=config["training"]["finetune_epochs"],
        logger=logger,
        callbacks=[
            WilliamsonRolloutCallback(autoreg_steps=5),
            checkpoint_callback,
            wc2_checkpoint_callback,
            wc6_checkpoint_callback,
            lr_monitor,
        ],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=precision,
        log_every_n_steps=config["training"]["log_every_n_steps"],
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )

    print(f"Rollout steps K={K}, burn_in={known_args.burn_in}, detach_after_burnin={known_args.detach_after_burnin}")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"\nBest rollout-loss checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best WC2 rollout checkpoint: {wc2_checkpoint_callback.best_model_path}")
    print(f"Best WC6 rollout checkpoint: {wc6_checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()