from __future__ import annotations

import math
from typing import Tuple

import torch


_TWO_PI = 2.0 * math.pi


def _meshgrid_ij(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        return torch.meshgrid(a, b, indexing="ij")
    except TypeError:
        return torch.meshgrid(a, b)


def latitude_is_ascending(lats: torch.Tensor) -> bool:
    if lats.ndim != 1:
        raise ValueError(f"Expected 1-D latitude array, got shape {tuple(lats.shape)}")
    if lats.numel() < 2:
        raise ValueError("Latitude array must contain at least two points")
    return bool((lats[-1] - lats[0]).item() > 0.0)


def validate_lat_lon_axes(lats: torch.Tensor, lons: torch.Tensor) -> None:
    if lats.ndim != 1 or lons.ndim != 1:
        raise ValueError(
            f"Expected 1-D axes; got lats.shape={tuple(lats.shape)}, lons.shape={tuple(lons.shape)}"
        )
    if lats.numel() < 2 or lons.numel() < 2:
        raise ValueError("Latitude and longitude axes must each have at least two points")

    dlat = torch.diff(lats)
    if not (torch.all(dlat > 0) or torch.all(dlat < 0)):
        raise ValueError("Latitude axis must be strictly monotone")

    dlon = torch.diff(lons)
    dlon_ref = dlon[0]
    tol = 1e-10 if lons.dtype == torch.float64 else 1e-6
    if not torch.all(torch.abs(dlon - dlon_ref) <= tol):
        raise ValueError("Longitude axis must be uniformly spaced")

    if torch.any(lons < 0) or torch.any(lons >= _TWO_PI + tol):
        raise ValueError("Expected longitude axis on [0, 2*pi)")


def describe_latitudes(lats: torch.Tensor, n: int = 4) -> str:
    lats_cpu = lats.detach().cpu()
    first_vals = ", ".join(f"{v.item(): .6f}" for v in lats_cpu[:n])
    last_vals = ", ".join(f"{v.item(): .6f}" for v in lats_cpu[-n:])
    order = "ascending" if latitude_is_ascending(lats_cpu) else "descending"
    return f"latitude order={order}; first[{n}]={first_vals}; last[{n}]={last_vals}"


def sph_to_cart(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    cos_lat = torch.cos(lat)
    x = cos_lat * torch.cos(lon)
    y = cos_lat * torch.sin(lon)
    z = torch.sin(lat)
    return torch.stack((x, y, z), dim=-1)


def cart_to_sph(xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2].clamp(-1.0, 1.0)
    lon = torch.atan2(y, x)
    lon = torch.remainder(lon, _TWO_PI)
    lat = torch.asin(z)
    return lat, lon


def _rotation_matrix_y(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    R = torch.zeros((3, 3), dtype=theta.dtype, device=theta.device)
    R[0, 0] = c
    R[0, 2] = s
    R[1, 1] = 1.0
    R[2, 0] = -s
    R[2, 2] = c
    return R


def _rotation_matrix_z(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    R = torch.zeros((3, 3), dtype=theta.dtype, device=theta.device)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    R[2, 2] = 1.0
    return R


def build_pole_target_rotation(
    pole_target_lat_deg: float,
    pole_target_lon_deg: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    lat = torch.tensor(math.radians(pole_target_lat_deg), dtype=dtype, device=device)
    lon = torch.tensor(math.radians(pole_target_lon_deg), dtype=dtype, device=device)

    theta = torch.tensor(0.5 * math.pi, dtype=dtype, device=device) - lat
    R = _rotation_matrix_z(lon) @ _rotation_matrix_y(theta)
    return R


def is_identity_rotation(
    pole_target_lat_deg: float,
    pole_target_lon_deg: float,
    *,
    atol_deg: float = 1e-12,
) -> bool:
    lat_ok = abs(pole_target_lat_deg - 90.0) <= atol_deg
    lon_ok = abs(((pole_target_lon_deg + 180.0) % 360.0) - 180.0) <= atol_deg
    return lat_ok and lon_ok


def compute_source_coords(
    dst_lats: torch.Tensor,
    dst_lons: torch.Tensor,
    rotation_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_lat_lon_axes(dst_lats, dst_lons)

    lat2d, lon2d = _meshgrid_ij(dst_lats, dst_lons)
    x_dst = sph_to_cart(lat2d, lon2d)

    Rinv = rotation_matrix.transpose(0, 1)
    x_src = torch.einsum("ij,...j->...i", Rinv, x_dst)

    src_lat, src_lon = cart_to_sph(x_src)
    return src_lat, src_lon


def _prepare_lat_axis_for_interp(
    field: torch.Tensor,
    src_lats_1d: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if latitude_is_ascending(src_lats_1d):
        return field, src_lats_1d
    return (
        torch.flip(field, dims=[0]).contiguous(),
        torch.flip(src_lats_1d, dims=[0]).contiguous(),
    )


def bilinear_remap_scalar(
    field: torch.Tensor,
    src_lats_1d: torch.Tensor,
    src_lons_1d: torch.Tensor,
    sample_lat: torch.Tensor,
    sample_lon: torch.Tensor,
) -> torch.Tensor:
    if field.ndim != 2:
        raise ValueError(f"Expected field shape [nlat, nlon], got {tuple(field.shape)}")
    if sample_lat.shape != sample_lon.shape:
        raise ValueError("sample_lat and sample_lon must have identical shape")
    if field.shape != (src_lats_1d.numel(), src_lons_1d.numel()):
        raise ValueError(
            f"Field shape {tuple(field.shape)} is inconsistent with axes "
            f"({src_lats_1d.numel()}, {src_lons_1d.numel()})"
        )

    validate_lat_lon_axes(src_lats_1d, src_lons_1d)

    work_field, work_lats = _prepare_lat_axis_for_interp(field, src_lats_1d)

    lat_min = work_lats[0]
    lat_max = work_lats[-1]
    sample_lat = sample_lat.clamp(min=lat_min, max=lat_max)

    idx_hi = torch.searchsorted(work_lats, sample_lat.contiguous(), right=False)
    idx_hi = idx_hi.clamp(1, work_lats.numel() - 1)
    idx_lo = idx_hi - 1

    lat0 = work_lats[idx_lo]
    lat1 = work_lats[idx_hi]
    lat_denom = (lat1 - lat0).clamp_min(torch.finfo(work_lats.dtype).eps)
    w_lat = ((sample_lat - lat0) / lat_denom).to(dtype=field.dtype)

    lon0 = src_lons_1d[0]
    dlon = src_lons_1d[1] - src_lons_1d[0]
    nlon = src_lons_1d.numel()

    sample_lon = torch.remainder(sample_lon - lon0, _TWO_PI)
    lon_pos = sample_lon / dlon
    j0 = torch.floor(lon_pos).to(torch.long) % nlon
    j1 = (j0 + 1) % nlon
    w_lon = (lon_pos - torch.floor(lon_pos)).to(dtype=field.dtype)

    f00 = work_field[idx_lo, j0]
    f01 = work_field[idx_lo, j1]
    f10 = work_field[idx_hi, j0]
    f11 = work_field[idx_hi, j1]

    one = torch.ones((), dtype=field.dtype, device=field.device)
    return (
        (one - w_lat) * (one - w_lon) * f00
        + (one - w_lat) * w_lon * f01
        + w_lat * (one - w_lon) * f10
        + w_lat * w_lon * f11
    )


def rotate_scalar_state_on_grid(
    ugrid_native: torch.Tensor,
    lats: torch.Tensor,
    lons: torch.Tensor,
    rotation_matrix: torch.Tensor,
    *,
    interpolation: str = "bilinear",
) -> torch.Tensor:
    if interpolation != "bilinear":
        raise NotImplementedError(f"Unsupported interpolation='{interpolation}'")
    if ugrid_native.ndim != 3:
        raise ValueError(
            f"Expected scalar state shape [channels, nlat, nlon], got {tuple(ugrid_native.shape)}"
        )
    if ugrid_native.shape[1] != lats.numel() or ugrid_native.shape[2] != lons.numel():
        raise ValueError(
            f"State shape {tuple(ugrid_native.shape)} does not match axes "
            f"({lats.numel()}, {lons.numel()})"
        )

    src_lat, src_lon = compute_source_coords(lats, lons, rotation_matrix)

    out = torch.empty_like(ugrid_native)
    for ch in range(ugrid_native.shape[0]):
        out[ch] = bilinear_remap_scalar(
            ugrid_native[ch],
            lats,
            lons,
            src_lat,
            src_lon,
        )
    return out


def rotate_spectral_scalar_state(
    solver,
    uspec_native: torch.Tensor,
    pole_target_lat_deg: float,
    pole_target_lon_deg: float,
    *,
    interpolation: str = "bilinear",
    enforce_tril: bool = True,
) -> torch.Tensor:
    if interpolation != "bilinear":
        raise NotImplementedError(f"Unsupported interpolation='{interpolation}'")

    if is_identity_rotation(pole_target_lat_deg, pole_target_lon_deg):
        return torch.tril(uspec_native.clone()) if enforce_tril else uspec_native.clone()

    ugrid_native = solver.spec2grid(uspec_native)
    R = build_pole_target_rotation(
        pole_target_lat_deg,
        pole_target_lon_deg,
        dtype=solver.lats.dtype,
        device=solver.lats.device,
    )

    ugrid_rot = rotate_scalar_state_on_grid(
        ugrid_native=ugrid_native,
        lats=solver.lats,
        lons=solver.lons,
        rotation_matrix=R,
        interpolation=interpolation,
    )

    uspec_rot = solver.grid2spec(ugrid_rot)
    return torch.tril(uspec_rot) if enforce_tril else uspec_rot