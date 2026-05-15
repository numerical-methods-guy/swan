# ignore_header_test
# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


import torch
import torch.nn as nn
import torch_harmonics as harmonics
from torch_harmonics.quadrature import *

import numpy as np


def _great_circle_distance(
    lat: torch.Tensor,
    lon: torch.Tensor,
    lat0: torch.Tensor,
    lon0: torch.Tensor,
) -> torch.Tensor:
    """Great-circle angular distance (radians) between grid point (lat, lon) and centre (lat0, lon0)."""
    sin1, cos1 = torch.sin(lat), torch.cos(lat)
    sin0, cos0 = torch.sin(lat0), torch.cos(lat0)
    dlon = lon - lon0
    cosgamma = sin1 * sin0 + cos1 * cos0 * torch.cos(dlon)
    cosgamma = torch.clamp(cosgamma, -1.0, 1.0)
    return torch.acos(cosgamma)


class ShallowWaterSolver(nn.Module):
    """
    SWE solver class. Interface inspired bu pyspharm and SHTns
    """

    def __init__(
        self,
        nlat,
        nlon,
        dt,
        lmax=None,
        mmax=None,
        grid="legendre-gauss",
        radius=6.37122e6,
        omega=7.292e-5,
        gravity=9.80616,
        havg=10.0e3,
        hamp=120.0,
    ):
        super().__init__()

        # time stepping param
        self.dt = dt

        # grid parameters
        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid

        # physical sonstants
        self.register_buffer("radius", torch.as_tensor(radius, dtype=torch.float64))
        self.register_buffer("omega", torch.as_tensor(omega, dtype=torch.float64))
        self.register_buffer("gravity", torch.as_tensor(gravity, dtype=torch.float64))
        self.register_buffer("havg", torch.as_tensor(havg, dtype=torch.float64))
        self.register_buffer("hamp", torch.as_tensor(hamp, dtype=torch.float64))

        # SHT
        self.sht = harmonics.RealSHT(
            nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False
        )
        self.isht = harmonics.InverseRealSHT(
            nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False
        )
        self.vsht = harmonics.RealVectorSHT(
            nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False
        )
        self.ivsht = harmonics.InverseRealVectorSHT(
            nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False
        )

        self.lmax = lmax or self.sht.lmax
        self.mmax = lmax or self.sht.mmax

        # compute gridpoints
        if self.grid == "legendre-gauss":
            cost, quad_weights = harmonics.quadrature.legendre_gauss_weights(
                self.nlat, -1, 1
            )
        elif self.grid == "lobatto":
            cost, quad_weights = harmonics.quadrature.lobatto_weights(self.nlat, -1, 1)
        elif self.grid == "equiangular":
            cost, quad_weights = harmonics.quadrature.clenshaw_curtiss_weights(
                self.nlat, -1, 1
            )

        quad_weights = torch.as_tensor(quad_weights).reshape(-1, 1)

        # apply cosine transform and flip them
        lats = -torch.as_tensor(np.arcsin(cost))
        lons = torch.linspace(0, 2 * np.pi, self.nlon + 1, dtype=torch.float64)[:nlon]

        self.lmax = self.sht.lmax
        self.mmax = self.sht.mmax

        # compute the laplace and inverse laplace operators
        l = torch.arange(0, self.lmax).reshape(self.lmax, 1).double()
        l = l.expand(self.lmax, self.mmax)
        # the laplace operator acting on the coefficients is given by - l (l + 1)
        lap = -l * (l + 1) / self.radius**2
        invlap = -(self.radius**2) / l / (l + 1)
        invlap[0] = 0.0

        # compute coriolis force
        coriolis = 2 * self.omega * torch.sin(lats).reshape(self.nlat, 1)

        # hyperdiffusion
        hyperdiff = torch.exp(
            torch.asarray((-self.dt / 2 / 3600.0) * (lap / lap[-1, 0]) ** 4)
        )

        # register all
        self.register_buffer("lats", lats)
        self.register_buffer("lons", lons)
        self.register_buffer("l", l)
        self.register_buffer("lap", lap)
        self.register_buffer("invlap", invlap)
        self.register_buffer("coriolis", coriolis)
        self.register_buffer("hyperdiff", hyperdiff)
        self.register_buffer("quad_weights", quad_weights)

    def grid2spec(self, ugrid):
        """
        spectral coefficients from spatial data
        """
        return self.sht(ugrid)

    def spec2grid(self, uspec):
        """
        spatial data from spectral coefficients
        """
        return self.isht(uspec)

    def vrtdivspec(self, ugrid):
        """spatial data from spectral coefficients"""
        vrtdivspec = self.lap * self.radius * self.vsht(ugrid)
        return vrtdivspec

    def getuv(self, vrtdivspec):
        """
        compute wind vector from spectral coeffs of vorticity and divergence
        """
        return self.ivsht(self.invlap * vrtdivspec / self.radius)

    def gethuv(self, uspec):
        """
        compute wind vector from spectral coeffs of vorticity and divergence
        """
        hgrid = self.spec2grid(uspec[:1])
        uvgrid = self.getuv(uspec[1:])
        return torch.cat((hgrid, uvgrid), dim=-3)

    def potential_vorticity(self, uspec):
        """
        Compute potential vorticity
        """
        ugrid = self.spec2grid(uspec)
        pvrt = (
            (0.5 * self.havg * self.gravity / self.omega)
            * (ugrid[1] + self.coriolis)
            / ugrid[0]
        )
        return pvrt

    def dimensionless(self, uspec):
        """
        Remove dimensions from variables
        """
        uspec[0] = (uspec[0] - self.havg * self.gravity) / self.hamp / self.gravity
        # vorticity is measured in 1/s so we normalize using sqrt(g h) / r
        uspec[1:] = uspec[1:] * self.radius / torch.sqrt(self.gravity * self.havg)
        return uspec

    def dudtspec(self, uspec):
        """
        Compute time derivatives from solution represented in spectral coefficients
        """

        dudtspec = torch.zeros_like(uspec)

        # compute the derivatives - this should be incorporated into the solver:
        ugrid = self.spec2grid(uspec)
        uvgrid = self.getuv(uspec[1:])

        # phi = ugrid[0]
        # vrtdiv = ugrid[1:]

        tmp = uvgrid * (ugrid[1] + self.coriolis)
        tmpspec = self.vrtdivspec(tmp)
        dudtspec[2] = tmpspec[0]
        dudtspec[1] = -1 * tmpspec[1]

        tmp = uvgrid * ugrid[0]
        tmp = self.vrtdivspec(tmp)
        dudtspec[0] = -1 * tmp[1]

        tmpspec = self.grid2spec(ugrid[0] + 0.5 * (uvgrid[0] ** 2 + uvgrid[1] ** 2))
        dudtspec[2] = dudtspec[2] - self.lap * tmpspec

        return dudtspec

    def galewsky_initial_condition(self):
        """
        Initializes non-linear barotropically unstable shallow water test case of Galewsky et al. (2004, Tellus, 56A, 429-440).

        [1] Galewsky; An initial-value problem for testing numerical models of the global shallow-water equations;
            DOI: 10.1111/j.1600-0870.2004.00071.x; http://www-vortex.mcs.st-and.ac.uk/~rks/reprints/galewsky_etal_tellus_2004.pdf
        """
        device = self.lap.device

        umax = 80.0
        phi0 = torch.asarray(torch.pi / 7.0, device=device)
        phi1 = torch.asarray(0.5 * torch.pi - phi0, device=device)
        phi2 = 0.25 * torch.pi
        en = torch.exp(torch.asarray(-4.0 / (phi1 - phi0) ** 2, device=device))
        alpha = 1.0 / 3.0
        beta = 1.0 / 15.0

        lats, lons = torch.meshgrid(self.lats, self.lons)

        u1 = (umax / en) * torch.exp(1.0 / ((lats - phi0) * (lats - phi1)))
        ugrid = torch.where(
            torch.logical_and(lats < phi1, lats > phi0),
            u1,
            torch.zeros(self.nlat, self.nlon, device=device),
        )
        vgrid = torch.zeros((self.nlat, self.nlon), device=device)
        hbump = (
            self.hamp
            * torch.cos(lats)
            * torch.exp(-(((lons - torch.pi) / alpha) ** 2))
            * torch.exp(-((phi2 - lats) ** 2) / beta)
        )

        # intial velocity field
        ugrid = torch.stack((ugrid, vgrid))
        # intial vorticity/divergence field
        vrtdivspec = self.vrtdivspec(ugrid)
        vrtdivgrid = self.spec2grid(vrtdivspec)

        # solve balance eqn to get initial zonal geopotential with a localized bump (not balanced).
        tmp = ugrid * (vrtdivgrid + self.coriolis)
        tmpspec = self.vrtdivspec(tmp)
        tmpspec[1] = self.grid2spec(0.5 * torch.sum(ugrid**2, dim=0))
        phispec = (
            self.invlap * tmpspec[0]
            - tmpspec[1]
            + self.grid2spec(self.gravity * (self.havg + hbump))
        )

        # assemble solution
        uspec = torch.zeros(
            3, self.lmax, self.mmax, dtype=vrtdivspec.dtype, device=device
        )
        uspec[0] = phispec
        uspec[1:] = vrtdivspec

        return torch.tril(uspec)

    def random_initial_condition(self, mach=0.1) -> torch.Tensor:
        """
        random initial condition on the sphere
        """
        device = self.lap.device
        ctype = torch.complex128 if self.lap.dtype == torch.float64 else torch.complex64

        # mach number relative to wave speed
        llimit = mlimit = 80

        # hgrid = self.havg + hamp * torch.randn(self.nlat, self.nlon, device=device, dtype=dtype)
        # ugrid = uamp * torch.randn(self.nlat, self.nlon, device=device, dtype=dtype)
        # vgrid = vamp * torch.randn(self.nlat, self.nlon, device=device, dtype=dtype)
        # ugrid = torch.stack((ugrid, vgrid))

        # initial geopotential
        uspec = torch.zeros(
            3, self.lmax, self.mmax, dtype=ctype, device=self.lap.device
        )
        uspec[:, :llimit, :mlimit] = torch.sqrt(
            torch.tensor(
                4 * torch.pi / llimit / (llimit + 1), device=device, dtype=ctype
            )
        ) * torch.randn_like(uspec[:, :llimit, :mlimit])

        uspec[0] = self.gravity * self.hamp * uspec[0]
        uspec[0, 0, 0] += (
            torch.sqrt(torch.tensor(4 * torch.pi, device=device, dtype=ctype))
            * self.havg
            * self.gravity
        )
        uspec[1:] = (
            mach * uspec[1:] * torch.sqrt(self.gravity * self.havg) / self.radius
        )
        # uspec[1:] = self.vrtdivspec(self.spec2grid(uspec[1:]) * torch.cos(self.lats.reshape(-1, 1)))

        # # intial velocity field
        # ugrid = uamp * self.spec2grid(uspec[1])
        # vgrid = vamp * self.spec2grid(uspec[2])
        # ugrid = torch.stack((ugrid, vgrid))

        # # intial vorticity/divergence field
        # vrtdivspec = self.vrtdivspec(ugrid)
        # vrtdivgrid = self.spec2grid(vrtdivspec)

        # # solve balance eqn to get initial zonal geopotential with a localized bump (not balanced).
        # tmp = ugrid * (vrtdivgrid + self.coriolis)
        # tmpspec = self.vrtdivspec(tmp)
        # tmpspec[1] = self.grid2spec(0.5 * torch.sum(ugrid**2, dim=0))
        # phispec = self.invlap*tmpspec[0] - tmpspec[1] + self.grid2spec(self.gravity * hgrid)

        # # assemble solution
        # uspec = torch.zeros(3, self.lmax, self.mmax, dtype=phispec.dtype, device=device)
        # uspec[0] = phispec
        # uspec[1:] = vrtdivspec

        return torch.tril(uspec)

    def williamson_case6_initial_condition(
        self,
        r_min: int = 1,
        r_max: int = 5,
        omega_min: float = 5e-6,
        omega_max: float = 1e-5,
        h0_min: float = 6000.0,
        h0_max: float = 10000.0,
        eps_cos: float = 1e-6,
    ) -> torch.Tensor:
        """Williamson test case 6: Rossby-Haurwitz wave.

        Randomizes wave number R in [r_min, r_max], angular frequency omega in
        [omega_min, omega_max], and reference height h0 in [h0_min, h0_max].
        K is set equal to omega (standard coupling).

        Args:
            r_min, r_max: Range for wave number R (inclusive).
            omega_min, omega_max: Range for angular frequency (rad/s).
            h0_min, h0_max: Range for reference geopotential height (m^2/s^2).
            eps_cos: Small value for numerical stability near the poles.

        Returns:
            uspec: Spectral state tensor (3, lmax, mmax).
        """
        device = self.lap.device
        dtype  = self.lap.dtype

        lat = self.lats.to(device=device, dtype=dtype).reshape(-1, 1)
        lon = self.lons.to(device=device, dtype=dtype).reshape(1, -1)

        a     = self.radius.to(dtype=dtype)
        Omega = self.omega.to(dtype=dtype)
        g     = self.gravity.to(dtype=dtype)

        R_int   = int(torch.randint(r_min, r_max + 1, (1,)).item())
        omega_t = torch.as_tensor(
            omega_min + (omega_max - omega_min) * torch.rand(1).item(),
            device=device, dtype=dtype,
        )
        K_t   = omega_t  # standard coupling
        h0_t  = torch.as_tensor(
            h0_min + (h0_max - h0_min) * torch.rand(1).item(),
            device=device, dtype=dtype,
        )

        sinlat   = torch.sin(lat)
        coslat   = torch.cos(lat)
        cos_safe = torch.clamp(torch.abs(coslat), min=eps_cos) * torch.sign(coslat + 0.0)

        A = (omega_t / 2.0) * (2.0 * Omega + omega_t) * coslat ** 2 + (K_t ** 2) / 4.0 * coslat ** (2 * R_int) * (
            (R_int + 1) * coslat ** 2
            + (2.0 * R_int ** 2 - R_int - 2.0)
            - 2.0 * R_int ** 2 * cos_safe ** (-2)
        )
        B = 2.0 * (Omega + omega_t) * K_t / ((R_int + 1) * (R_int + 2)) * coslat ** R_int * (
            (R_int ** 2 + 2 * R_int + 2) - (R_int + 1) ** 2 * coslat ** 2
        )
        C = (K_t ** 2) / 4.0 * coslat ** (2 * R_int) * (
            (R_int + 1) * coslat ** 2 - (R_int + 2.0)
        )

        h = h0_t + (a ** 2) * (A + B * torch.cos(R_int * lon) + C * torch.cos(2.0 * R_int * lon)) / g

        cos_pow = coslat ** (R_int - 1)
        u = (a * omega_t * coslat
             + a * K_t * cos_pow * (R_int * sinlat ** 2 - coslat ** 2) * torch.cos(R_int * lon))
        v = -a * K_t * R_int * cos_pow * sinlat * torch.sin(R_int * lon)

        uv_grid    = torch.stack([u, v], dim=0)
        vrtdiv_spec = self.vrtdivspec(uv_grid)
        phi_spec    = self.grid2spec(h.unsqueeze(0))[0]

        ctype = torch.complex128 if dtype == torch.float64 else torch.complex64
        uspec = torch.zeros(3, self.lmax, self.mmax, dtype=ctype, device=device)
        uspec[0]  = phi_spec
        uspec[1:] = vrtdiv_spec.to(dtype=ctype)

        return torch.tril(uspec)

    def gaussian_bells_height_initial_condition(
        self,
        ref_mean: torch.Tensor,
        ref_std: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Gaussian bells in the height channel only; vorticity and divergence are zero.

        Calls gaussian_bells_initial_condition then zeros out channels 1 and 2.
        """
        uspec = self.gaussian_bells_initial_condition(ref_mean, ref_std, **kwargs)
        uspec[1:] = 0.0
        return uspec

    def williamson_case2_initial_condition(
        self,
        alpha=None,
        gh0=None,
        u0=None,
        gh0_min: float = 20000.0,
        gh0_max: float = 35000.0,
        u0_min: float = 10.0,
        u0_max: float = 60.0,
    ) -> torch.Tensor:
        """Williamson test case 2: steady-state geostrophic flow on the sphere.

        The flow is a balanced solid-body rotation tilted at angle alpha from
        the equator.  Each call with alpha/gh0/u0=None draws from a random range
        so the dataset sees a variety of flow configurations.

        Args:
            alpha:   Tilt angle in radians.  If None, sampled from U(0, pi/2).
            gh0:     Reference geopotential height (m^2/s^2).  If None, sampled from U(gh0_min, gh0_max).
            u0:      Maximum wind speed (m/s).  If None, sampled from U(u0_min, u0_max).
            gh0_min: Lower bound for gh0 randomization.
            gh0_max: Upper bound for gh0 randomization.
            u0_min:  Lower bound for u0 randomization (m/s).
            u0_max:  Upper bound for u0 randomization (m/s).

        Returns:
            uspec: Spectral state tensor (3, lmax, mmax).
        """
        device = self.lap.device
        dtype  = self.lap.dtype

        lat = self.lats.to(device=device, dtype=dtype).reshape(-1, 1)
        lon = self.lons.to(device=device, dtype=dtype).reshape(1, -1)

        a     = self.radius.to(dtype=dtype)
        Omega = self.omega.to(dtype=dtype)

        if u0 is None:
            u0_t = torch.as_tensor(
                u0_min + (u0_max - u0_min) * torch.rand(1).item(), device=device, dtype=dtype
            )
        else:
            u0_t = torch.as_tensor(float(u0), device=device, dtype=dtype)

        if gh0 is None:
            gh0_val = gh0_min + (gh0_max - gh0_min) * torch.rand(1).item()
        else:
            gh0_val = float(gh0)

        if alpha is None:
            alpha_t = (0.5 * torch.pi * torch.rand(1, device=device, dtype=dtype)).item()
            alpha_t = torch.as_tensor(float(alpha_t), device=device, dtype=dtype)
        else:
            alpha_t = torch.as_tensor(float(alpha), device=device, dtype=dtype)

        sinlat = torch.sin(lat)
        coslat = torch.cos(lat)
        sinlon = torch.sin(lon)
        coslon = torch.cos(lon)
        sinalpha = torch.sin(alpha_t)
        cosalpha = torch.cos(alpha_t)

        u_grid = u0_t * (coslat * cosalpha + sinlat * coslon * sinalpha)
        v_grid = -u0_t * sinlon * sinalpha

        cterm = -coslon * coslat * sinalpha + sinlat * cosalpha
        phi_grid = torch.as_tensor(gh0_val, device=device, dtype=dtype) - (
            a * Omega * u0_t + 0.5 * u0_t * u0_t
        ) * cterm ** 2

        uv_grid    = torch.stack([u_grid, v_grid + torch.zeros_like(lat)], dim=0)
        vrtdiv_spec = self.vrtdivspec(uv_grid)
        phi_spec    = self.grid2spec(phi_grid.unsqueeze(0))[0]

        ctype = torch.complex128 if dtype == torch.float64 else torch.complex64
        uspec = torch.zeros(3, self.lmax, self.mmax, dtype=ctype, device=device)
        uspec[0]  = phi_spec
        uspec[1:] = vrtdiv_spec.to(dtype=ctype)

        return torch.tril(uspec)

    def gaussian_bells_initial_condition(
        self,
        ref_mean: torch.Tensor,
        ref_std: torch.Tensor,
        k_min: int = 1,
        k_max: int = 8,
        sigma_min_deg: float = 5.0,
        sigma_max_deg: float = 20.0,
        signed: bool = True,
        mean_scale: float = 1.0,
        std_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate a random Gaussian bells initial condition on the sphere.

        Places K random Gaussian bumps per channel, normalizes, then scales using
        ref_mean and ref_std so the IC has a similar magnitude to physical states.

        Args:
            ref_mean: Per-channel reference mean, shape (3,) or scalar.
            ref_std:  Per-channel reference std,  shape (3,) or scalar.
            k_min:    Minimum number of bells per channel.
            k_max:    Maximum number of bells per channel.
            sigma_min_deg: Minimum bell width in degrees.
            sigma_max_deg: Maximum bell width in degrees.
            signed:   If True, amplitudes are drawn from U(-1, 1); else U(0, 1).
            mean_scale: Multiplier applied to ref_mean.
            std_scale:  Multiplier applied to ref_std.

        Returns:
            uspec: Spectral state tensor (3, lmax, mmax) on the solver device.
        """
        device = self.lats.device
        dtype = self.lats.dtype

        sigma_min_rad = sigma_min_deg * (torch.pi / 180.0)
        sigma_max_rad = sigma_max_deg * (torch.pi / 180.0)

        # lat/lon grids: (nlat,), (nlon,)
        lats = self.lats  # shape (nlat,)
        lons = self.lons  # shape (nlon,)
        # broadcast to (nlat, nlon)
        lat_grid = lats.unsqueeze(1).expand(self.nlat, self.nlon)
        lon_grid = lons.unsqueeze(0).expand(self.nlat, self.nlon)

        channels = []
        for ch in range(3):
            # Use a channel-specific offset seed so channels are decorrelated
            K = torch.randint(k_min, k_max + 1, (1,)).item()

            # Sample bell centres: lat from asin(U(-1,1)), lon from U(0, 2pi)
            lat0 = torch.asin(2.0 * torch.rand(K, device=device, dtype=dtype) - 1.0)
            lon0 = 2.0 * torch.pi * torch.rand(K, device=device, dtype=dtype)

            # Sample widths and amplitudes
            sigma = sigma_min_rad + (sigma_max_rad - sigma_min_rad) * torch.rand(K, device=device, dtype=dtype)
            if signed:
                amp = 2.0 * torch.rand(K, device=device, dtype=dtype) - 1.0
            else:
                amp = torch.rand(K, device=device, dtype=dtype)

            # Accumulate bells: shape (nlat, nlon)
            field = torch.zeros(self.nlat, self.nlon, device=device, dtype=dtype)
            for j in range(K):
                dist = _great_circle_distance(lat_grid, lon_grid, lat0[j], lon0[j])
                field = field + amp[j] * torch.exp(-0.5 * (dist / sigma[j]) ** 2)

            # Normalize to zero mean, unit std
            field = (field - field.mean()) / (field.std() + 1e-6)

            # Scale to match physical distribution
            ch_mean = ref_mean[ch] if ref_mean.ndim > 0 else ref_mean
            ch_std  = ref_std[ch]  if ref_std.ndim  > 0 else ref_std
            field = mean_scale * ch_mean + std_scale * ch_std * field

            channels.append(field)

        # Stack to (3, nlat, nlon) and transform to spectral space
        ugrid = torch.stack(channels, dim=0)  # (3, nlat, nlon)
        uspec = torch.zeros(3, self.lmax, self.mmax, dtype=torch.complex64, device=device)
        for ch in range(3):
            uspec[ch] = self.grid2spec(ugrid[ch])

        return torch.tril(uspec)

    def precomputed_initial_condition(self, folder: str, index: int, step: int = 0) -> torch.Tensor:
        """Load a precomputed spectral state from disk.

        Expects a file named {folder}/{index}_{step}.pt containing a spectral state tensor.
        """
        import os
        path = os.path.join(folder, f"{index}_{step}.pt")
        uspec = torch.load(path, map_location=self.lap.device)
        return uspec

    def timestep(self, uspec: torch.Tensor, nsteps: int) -> torch.Tensor:
        """
        Integrate the solution using Adams-Bashforth / forward Euler for nsteps steps.
        """

        dudtspec = torch.zeros(
            3, 3, self.lmax, self.mmax, dtype=uspec.dtype, device=uspec.device
        )

        # pointers to indicate the most current result
        inew = 0
        inow = 1
        iold = 2

        for iter in range(nsteps):
            dudtspec[inew] = self.dudtspec(uspec)

            # update vort,div,phiv with third-order adams-bashforth.
            # forward euler, then 2nd-order adams-bashforth time steps to start.
            if iter == 0:
                dudtspec[inow] = dudtspec[inew]
                dudtspec[iold] = dudtspec[inew]
            elif iter == 1:
                dudtspec[iold] = dudtspec[inew]

            uspec = uspec + self.dt * (
                (23.0 / 12.0) * dudtspec[inew]
                - (16.0 / 12.0) * dudtspec[inow]
                + (5.0 / 12.0) * dudtspec[iold]
            )

            # implicit hyperdiffusion for vort and div.
            uspec[1:] = self.hyperdiff * uspec[1:]

            # cycle through the indices
            inew = (inew - 1) % 3
            inow = (inow - 1) % 3
            iold = (iold - 1) % 3

        return uspec

    def integrate_grid(self, ugrid, dimensionless=False, polar_opt=0):
        dlon = 2 * torch.pi / self.nlon
        radius = 1 if dimensionless else self.radius
        if polar_opt > 0:
            out = torch.sum(
                ugrid[..., polar_opt:-polar_opt, :]
                * self.quad_weights[polar_opt:-polar_opt]
                * dlon
                * radius**2,
                dim=(-2, -1),
            )
        else:
            out = torch.sum(ugrid * self.quad_weights * dlon * radius**2, dim=(-2, -1))
        return out

    def plot_griddata(
        self,
        data,
        fig,
        cmap="twilight_shifted",
        vmax=None,
        vmin=None,
        projection="3d",
        title=None,
        antialiased=False,
    ):
        """
        plotting routine for data on the grid. Requires cartopy for 3d plots.
        """
        import matplotlib.pyplot as plt

        lons = self.lons.squeeze() - torch.pi
        lats = self.lats.squeeze()

        if data.is_cuda:
            data = data.cpu()
            lons = lons.cpu()
            lats = lats.cpu()

        Lons, Lats = np.meshgrid(lons, lats)

        if projection == "mollweide":
            # ax = plt.gca(projection=projection)
            ax = fig.add_subplot(projection=projection)
            im = ax.pcolormesh(Lons, Lats, data, cmap=cmap, vmax=vmax, vmin=vmin)
            # ax.set_title("Elevation map of mars")
            ax.grid(True)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.colorbar(im, orientation="horizontal")
            plt.title(title)

        elif projection == "3d":
            import cartopy.crs as ccrs

            proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=25.0)

            # ax = plt.gca(projection=proj, frameon=True)
            ax = fig.add_subplot(projection=proj)
            Lons = Lons * 180 / np.pi
            Lats = Lats * 180 / np.pi

            # contour data over the map.
            im = ax.pcolormesh(
                Lons,
                Lats,
                data,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                antialiased=antialiased,
                vmax=vmax,
                vmin=vmin,
            )
            plt.title(title, y=1.05)

        else:
            raise NotImplementedError

        return im

    def plot_specdata(self, data, fig, **kwargs):
        return self.plot_griddata(self.isht(data), fig, **kwargs)
