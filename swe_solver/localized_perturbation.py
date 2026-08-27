"""Localized, approximately geostrophically balanced SWE perturbations.

The ShallowWaterSolver state is [geopotential, vorticity, divergence] in
spherical-harmonic space. This module builds a smooth geopotential anomaly on
the solver grid, derives a local geostrophic wind perturbation, converts it to
vorticity/divergence with the solver's own operators, and adds it to an
existing spectral initial condition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch




@dataclass(frozen=True)
class LocalizedPerturbationConfig:
    enabled: bool = False
    kind: str = "balanced_vortex"
    center_lat_deg: float = 30.0
    center_lon_deg: float = 180.0
    radius_km: float = 500.0
    geopotential_amplitude: float = 500.0
    min_abs_lat_deg: float = 15.0
    coriolis_floor: float = 2.5e-5

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any] | None) -> "LocalizedPerturbationConfig":
        cfg = cfg or {}
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            kind=str(cfg.get("kind", "balanced_vortex")).lower(),
            center_lat_deg=float(cfg.get("center_lat_deg", 30.0)),
            center_lon_deg=float(cfg.get("center_lon_deg", 180.0)),
            radius_km=float(cfg.get("radius_km", 500.0)),
            geopotential_amplitude=float(cfg.get("geopotential_amplitude", 500.0)),
            min_abs_lat_deg=float(cfg.get("min_abs_lat_deg", 15.0)),
            coriolis_floor=float(cfg.get("coriolis_floor", 2.5e-5)),
        )

    def validate(self) -> None:
        if self.kind not in {"balanced_vortex", "height_only"}:
            raise ValueError("kind must be 'balanced_vortex' or 'height_only'")
        if not -90.0 < self.center_lat_deg < 90.0:
            raise ValueError("center_lat_deg must lie strictly between -90 and 90")
        if abs(self.center_lat_deg) < self.min_abs_lat_deg:
            raise ValueError(
                "Use a centre outside the equatorial band: "
                f"|center_lat_deg| must be >= {self.min_abs_lat_deg}."
            )
        if self.radius_km <= 0.0:
            raise ValueError("radius_km must be positive")
        if self.coriolis_floor <= 0.0:
            raise ValueError("coriolis_floor must be positive")


def _wrapped_longitude_delta(lon: torch.Tensor, lon0: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(lon - lon0), torch.cos(lon - lon0))


def build_localized_increment(solver, config: LocalizedPerturbationConfig) -> torch.Tensor:
    """Create a spectral [geopotential, vorticity, divergence] increment."""
    config.validate()
    device, dtype = solver.lats.device, solver.lats.dtype
    lat0 = torch.tensor(config.center_lat_deg * torch.pi / 180.0, device=device, dtype=dtype)
    lon0 = torch.tensor(config.center_lon_deg * torch.pi / 180.0, device=device, dtype=dtype)
    length = torch.tensor(config.radius_km * 1_000.0, device=device, dtype=dtype)
    amplitude = torch.tensor(config.geopotential_amplitude, device=device, dtype=dtype)

    lat, lon = torch.meshgrid(solver.lats, solver.lons, indexing="ij")
    dlon = _wrapped_longitude_delta(lon, lon0)
    x = solver.radius.to(dtype=dtype) * torch.cos(lat0) * dlon
    y = solver.radius.to(dtype=dtype) * (lat - lat0)
    phi = amplitude * torch.exp(-0.5 * (x.square() + y.square()) / length.square())

    increment = torch.zeros(3, solver.lmax, solver.mmax, dtype=solver.sht(phi).dtype, device=device)
    increment[0] = solver.grid2spec(phi.unsqueeze(0))[0]
    if config.kind == "height_only":
        return torch.tril(increment)

    f0 = 2.0 * solver.omega.to(dtype=dtype) * torch.sin(lat0)
    if torch.abs(f0) < config.coriolis_floor:
        raise ValueError("Coriolis parameter at centre is below coriolis_floor")

    # u is zonal/eastward and v is meridional/northward. The approximation is
    # local f-plane geostrophic balance: u=-Phi_y/f, v=Phi_x/f.
    u = y * phi / (f0 * length.square())
    v = -x * phi / (f0 * length.square())
    increment[1:] = solver.vrtdivspec(torch.stack((u, v), dim=0))
    return torch.tril(increment)


def apply_localized_perturbation(solver, ic_spec: torch.Tensor, config: LocalizedPerturbationConfig) -> torch.Tensor:
    """Return a perturbed clone of a solver spectral initial state."""
    if not config.enabled:
        return ic_spec.clone()
    expected = (3, solver.lmax, solver.mmax)
    if tuple(ic_spec.shape) != expected:
        raise ValueError(f"Expected IC shape {expected}, got {tuple(ic_spec.shape)}")
    return torch.tril(ic_spec + build_localized_increment(solver, config).to(ic_spec.dtype))


def perturbation_metadata(config: LocalizedPerturbationConfig) -> dict[str, Any]:
    return asdict(config)
