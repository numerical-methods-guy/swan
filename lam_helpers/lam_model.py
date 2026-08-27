"""
lam_model.py — LAMParadis: Local Area Model with HR interior dynamics
and LR halo boundary forcing.

Architecture (HR center + fixed LR boundary collar)
---------------------------------------------------

Inputs for one forecast interval
--------------------------------

    lr_halo [B, 5, H_lr_win, W_lr_win]
        |
        |  LR window contains the surrounding coarse-scale context.
        |  The central LR footprint is excluded/masked upstream when
        |  strict LR-halo-only forcing is required.
        v
    LR Encoder
    (GMBlock at LR window mesh_size = pL + 2R, pN + 2R;
     spherical/global-aware padding is allowed here)
        |
        v
    LR latent
    [B, D, H_lr_win, W_lr_win]
        |
        |  bilinear upsample to extended HR-window resolution
        v
    LR boundary latent / forcing field
    [B, D, H_hr_win, W_hr_win]
        |
        |  Used only in the exterior collar surrounding the HR patch.
        |
        |-----------------------------------------------------+
                                                              |
    hr_patch_t0 [B, 5, H_hr_patch, W_hr_patch]                |
        |                                                     |
        v                                                     |
    HR Encoder                                                |
    (GMBlock at HR patch mesh_size)                           |
        |                                                     |
        v                                                     |
    HR interior latent                                        |
    [B, D, H_hr_patch, W_hr_patch]                            |
        |                                                     |
        +---- spatially insert into central region ----------+
                                                              |
    Extended latent state                                     |
    [B, D, H_hr_win, W_hr_win]                               |
                                                              |
    ┌────────────────────────────────────────────────────┐  |
    │ LR boundary collar | HR interior | LR boundary collar│  |
    └────────────────────────────────────────────────────┘  |
                                                              v
    num_layers × ADR step over the extended HR window
    --------------------------------------------------
    velocity network
        → semi-Lagrangian advection
        → diffusion
        → reaction

    After each ADR step, the exterior collar is restored from
    the same dataset-derived LR boundary latent. The HR interior
    remains free to evolve dynamically.

    All ADR operations use local-patch behavior:
    - LocalSepConv rather than spherical SepConv
    - replicate/zero-style local padding, not GeoCyclicPadding
    - no pole-continuity enforcement
    - lat/lon grid spans the full extended HR window:
      central HR patch + exterior LR collar
        |
        v
    HR Output Projection
    (GMBlock at extended HR-window mesh_size)
        |
        v
    Extended physical output
    [B, 5, H_hr_win, W_hr_win]
        |
        |  crop central HR region only
        v
    Central patch prediction increment
    [B, 5, H_hr_patch, W_hr_patch]
        |
        |  residual skip
        +-------------------------------+
        |                               |
        v                               |
    hr_patch_t0 ------------------------+
        |
        v
    Predicted HR patch at t + dt
    [B, 5, H_hr_patch, W_hr_patch]


Rollout behavior
----------------

For one single-step forecast:

    dataset LR halo at t_k + HR patch state at t_k
        -> model
        -> predicted HR patch at t_(k+1)

The LR halo is fixed throughout all internal ADR steps of that
single model call. During an autoregressive rollout, the HR patch
is fed back from the previous prediction, while the LR halo is
refreshed from the dataset for each new forecast interval:

    step k:
        LR halo from dataset at t_k
        + HR state at t_k
        -> predicted HR state at t_(k+1)

    step k+1:
        new LR halo from dataset at t_(k+1)
        + predicted HR state at t_(k+1)
        -> predicted HR state at t_(k+2)


Key design rules
----------------

1. LR data acts as exterior context and boundary forcing; it is not
   channel-wise fused with HR features in the central patch.

2. HR data provides the initial latent state within the central patch.

3. The LR forcing field is derived once per forecast interval and
   reimposed during ADR evolution; it is not autoregressively predicted.

4. GeoCyclicPadding is restricted to the LR encoder, which processes
   a window extracted from the global LR field.

5. HR-local ADR layers use local padding because the extended HR window
   is a local domain, not a global sphere and not a polar domain.

6. GMBlock mesh_size must always match the spatial dimensions of the
   tensor processed by that specific block.

7. The learnable PARADIS alpha_adv sigmoid gate is retained.
"""

# lam_model.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.advection import NeuralSemiLagrangian
from model.blocks import GMBlock, SepConv, LocalSepConv


_ACTIVATIONS = {
    "SiLU": nn.SiLU,
    "GELU": nn.GELU,
}


def _get_activation_cls(name: str):
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]


def _get_scaled_timestep(dt_seconds: float) -> float:
    """Mirror of paradis.get_scaled_timestep."""
    return dt_seconds * 7.29212e-5


def _localize_layers(layers):
    """
    Replace spherical SepConv layers with LocalSepConv in HR-local branches.
    Supports both string-based and class-based layer specifications.
    """
    localized = []
    for l in layers:
        if l == "SepConv" or l is SepConv:
            localized.append("LocalSepConv")
        else:
            localized.append(l)
    return localized


class PatchNeuralSemiLagrangian(NeuralSemiLagrangian):
    """NeuralSemiLagrangian adapted for a local HR window.

    Changes from the global version:
    1. Interpolation support padding is local replicate padding, not GeoCyclicPadding.
    2. grid_sample(..., padding_mode="border") clamps departure coordinates to the
       outer border of the full HR window, not the patch-halo interface.
    3. enforce_pole_continuity is a no-op because patch rows are interior rows.
    """

    def enforce_pole_continuity(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _local_interp_pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding == 0:
            return x
        return F.pad(
            x,
            (self.padding, self.padding, self.padding, self.padding),
            mode="replicate",
        )

    def forward(
        self,
        hidden_features: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        batch_size = hidden_features.shape[0]
        H, W = self.mesh_size

        projected_inputs = self.down_projection(hidden_features)

        lon_prime = -u * dt
        lat_prime = -v * dt

        lat_dep, lon_dep = self._transform_to_latlon(
            lat_prime, lon_prime, self.lat_grid, self.lon_grid
        )

        pix_x = (lon_dep - self.min_lon) / self.d_lon * (self.Wf - 1.0)
        pix_y = (lat_dep - self.min_lat) / self.d_lat * (self.Hf - 1.0)

        # Local interpolation support: edge values are replicated outward.
        projected_padded = self._local_interp_pad(projected_inputs)

        pix_x_pad = pix_x + self.padding
        pix_y_pad = pix_y + self.padding

        H_pad = H + 2 * self.padding
        W_pad = W + 2 * self.padding

        grid_x = 2.0 * (pix_x_pad / float(W_pad - 1)) - 1.0
        grid_y = 2.0 * (pix_y_pad / float(H_pad - 1)) - 1.0

        grid_x = grid_x.reshape(batch_size * self.num_vels, H, W)
        grid_y = grid_y.reshape(batch_size * self.num_vels, H, W)
        grid = torch.stack([grid_x, grid_y], dim=-1)

        projected_padded = projected_padded.reshape(
            batch_size * self.num_vels, 1, H_pad, W_pad
        )

        interpolated = F.grid_sample(
            projected_padded,
            grid,
            align_corners=True,
            mode=self.interpolation,
            padding_mode="border",
        )

        interpolated = self.up_projection(
            interpolated.reshape(batch_size, self.num_vels, H, W)
        )

        return interpolated


class LAMParadis(nn.Module):
    """Local HR forecasting model with HR interior dynamics and LR halo forcing."""

    def __init__(
        self,
        lam_cfg: dict,
        model_cfg: dict,
        patch_nlat_hr: int,
        patch_nlon_hr: int,
        halo_radius: int,
        upscale_factor: int,
        lr_nlat: int,
        lr_nlon: int,
    ):
        super().__init__()

        def _get(key, fallback_key=None):
            v = lam_cfg.get(key)
            if v is None and fallback_key is not None:
                v = model_cfg.get(fallback_key)
            return v

        hidden_dim = _get("hidden_dim", "hidden_dim")
        num_layers = max(1, int(_get("num_layers", "num_layers")))
        num_vels = int(_get("num_vels", "num_vels"))
        base_dt = float(_get("base_dt", "base_dt"))
        interpolation = _get("interpolation", "interpolation") or "bicubic"
        activation_name = _get("activation", "activation") or "SiLU"
        bias_channels = int(model_cfg.get("bias_channels", 2))

        activation_fn = _get_activation_cls(activation_name)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_vels = num_vels
        self.patch_nlat_hr = patch_nlat_hr
        self.patch_nlon_hr = patch_nlon_hr
        self.halo_radius = halo_radius
        self.upscale_factor = upscale_factor
        self.dt = _get_scaled_timestep(base_dt) / num_layers

        patch_nlat_lr = patch_nlat_hr // upscale_factor
        patch_nlon_lr = patch_nlon_hr // upscale_factor
        self.lr_win_nlat = patch_nlat_lr + 2 * halo_radius
        self.lr_win_nlon = patch_nlon_lr + 2 * halo_radius
        self.hr_halo_border = halo_radius * upscale_factor

        lr_win_mesh = (self.lr_win_nlat, self.lr_win_nlon)
        hr_patch_mesh = (patch_nlat_hr, patch_nlon_hr)

        hr_win_nlat = (patch_nlat_lr + 2 * halo_radius) * upscale_factor
        hr_win_nlon = (patch_nlon_lr + 2 * halo_radius) * upscale_factor
        hr_win_mesh = (hr_win_nlat, hr_win_nlon)

        self.hr_win_nlat = hr_win_nlat
        self.hr_win_nlon = hr_win_nlon

        halo_mask = torch.ones(
            1,
            1,
            hr_win_nlat,
            hr_win_nlon,
            dtype=torch.bool,
        )

        row_start = self.hr_halo_border
        row_end = row_start + patch_nlat_hr
        col_start = self.hr_halo_border
        col_end = col_start + patch_nlon_hr

        halo_mask[:, :, row_start:row_end, col_start:col_end] = False

        self.register_buffer("hr_halo_mask", halo_mask)

        physblock = model_cfg.get("physblock", {})

        def _layers(key, default):
            return physblock.get(key, {}).get("layers", default)

        def _ldim(key, default):
            return physblock.get(key, {}).get("hidden_dim", default)

        # LR branch keeps spherical SepConv behavior.
        lr_input_layers = _layers("input_proj", ["SepConv", "CLinear"])

        # HR-local branches convert SepConv -> LocalSepConv.
        hr_input_layers = _localize_layers(_layers("input_proj", ["SepConv", "CLinear"]))
        vnet_layers = _localize_layers(_layers("velocity_net", ["SepConv"]))
        diffusion_layers = _localize_layers(_layers("diffusion", ["SepConv"]))
        reaction_layers = _localize_layers(_layers("reaction", ["CLinear", "CLinear"]))
        output_layers = _localize_layers(_layers("output_proj", ["SepConv", "CLinear"]))

        input_ldim = _ldim("input_proj", hidden_dim)
        vnet_ldim = _ldim("velocity_net", hidden_dim)
        diff_ldim = _ldim("diffusion", hidden_dim)
        reac_ldim = _ldim("reaction", hidden_dim)
        output_ldim = _ldim("output_proj", hidden_dim)

        adv_block = physblock.get("advection", {})
        down_proj_layers = _localize_layers(
            adv_block.get("down_projection", {}).get("layers", ["CLinear"])
        )
        up_proj_layers = _localize_layers(
            adv_block.get("up_projection", {}).get("layers", ["SepConv"])
        )
        down_proj_ldim = adv_block.get("down_projection", {}).get("hidden_dim", 0)
        up_proj_ldim = adv_block.get("up_projection", {}).get("hidden_dim", 0)

        self.lr_encoder = GMBlock(
            layers=lr_input_layers,
            input_dim=5,
            output_dim=hidden_dim,
            hidden_dim=input_ldim,
            mesh_size=lr_win_mesh,
            activation=True,
            activation_fn=activation_fn,
            pre_normalize=False,
            bias_channels=0,
        )

        self.hr_encoder = GMBlock(
            layers=hr_input_layers,
            input_dim=5,
            output_dim=hidden_dim,
            hidden_dim=input_ldim,
            mesh_size=hr_patch_mesh,
            activation=True,
            activation_fn=activation_fn,
            pre_normalize=False,
            bias_channels=0,
        )

        hr_dlat = 180.0 / (lr_nlat * upscale_factor)
        hr_dlon = 360.0 / (lr_nlon * upscale_factor)

        lat_start = -(
            (patch_nlat_hr - 1) / 2.0 + self.hr_halo_border
        ) * hr_dlat

        lat_end = (
            (patch_nlat_hr - 1) / 2.0 + self.hr_halo_border
        ) * hr_dlat

        lon_start = -self.hr_halo_border * hr_dlon
        lon_end = (
            patch_nlon_hr - 1 + self.hr_halo_border
        ) * hr_dlon

        lat_vec = torch.linspace(lat_start, lat_end, hr_win_nlat, dtype=torch.float32)
        lon_vec = torch.linspace(lon_start, lon_end, hr_win_nlon, dtype=torch.float32)

        lat_grid, lon_grid = torch.meshgrid(lat_vec, lon_vec, indexing="ij")
        lat_grid_rad = torch.deg2rad(lat_grid)
        lon_grid_rad = torch.deg2rad(lon_grid)

        self.velocity_nets = nn.ModuleList([
            GMBlock(
                layers=vnet_layers,
                input_dim=hidden_dim,
                output_dim=2 * num_vels,
                hidden_dim=vnet_ldim,
                mesh_size=hr_win_mesh,
                bias_channels=bias_channels,
                activation_fn=activation_fn,
                pre_normalize=True,
            )
            for _ in range(num_layers)
        ])

        self.advection = nn.ModuleList([
            PatchNeuralSemiLagrangian(
                hidden_dim=hidden_dim,
                mesh_size=hr_win_mesh,
                num_vels=num_vels,
                lat_grid=lat_grid_rad,
                lon_grid=lon_grid_rad,
                interpolation=interpolation,
                down_proj_layers=down_proj_layers,
                up_proj_layers=up_proj_layers,
                down_proj_ldim=down_proj_ldim,
                up_proj_ldim=up_proj_ldim,
            )
            for _ in range(num_layers)
        ])

        self.diffusion = nn.ModuleList([
            GMBlock(
                layers=diffusion_layers,
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                hidden_dim=diff_ldim,
                mesh_size=hr_win_mesh,
                pre_normalize=True,
                activation_fn=activation_fn,
                bias_channels=bias_channels,
            )
            for _ in range(num_layers)
        ])

        self.reaction = nn.ModuleList([
            GMBlock(
                layers=reaction_layers,
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                hidden_dim=reac_ldim,
                mesh_size=hr_win_mesh,
                pre_normalize=True,
                activation_fn=activation_fn,
                bias_channels=bias_channels,
            )
            for _ in range(num_layers)
        ])

        self.alpha_adv = nn.Parameter(torch.full((num_layers, hidden_dim), -1.0))

        self.output_proj = GMBlock(
            layers=output_layers,
            input_dim=hidden_dim,
            output_dim=5,
            hidden_dim=output_ldim,
            mesh_size=hr_win_mesh,
            activation=False,
            activation_fn=activation_fn,
            bias_channels=bias_channels,
        )

    def _adr_step(self, i: int, hidden: torch.Tensor) -> torch.Tensor:
        B = hidden.shape[0]

        vel_raw = self.velocity_nets[i](hidden)
        vel = vel_raw.view(B, 2, self.num_vels, self.hr_win_nlat, self.hr_win_nlon)
        u, v = vel[:, 0], vel[:, 1]

        g_adv = torch.sigmoid(self.alpha_adv[i]).to(hidden.dtype).view(1, -1, 1, 1)
        advected = self.advection[i](hidden, u, v, self.dt)
        hidden = hidden + g_adv * (advected - hidden)

        hidden = hidden + self.diffusion[i](hidden)
        hidden = hidden + self.reaction[i](hidden)
        return hidden

    def forward(self, lr_halo, hr_patch_t0):
        border = self.hr_halo_border

        # LR branch: produce an HR-resolution forcing field.
        lr_latent = self.lr_encoder(lr_halo)
        lr_latent_up = F.interpolate(
            lr_latent,
            size=(self.hr_win_nlat, self.hr_win_nlon),
            mode="bilinear",
            align_corners=False,
        )

        # HR branch: encode only the central HR patch.
        hr_latent = self.hr_encoder(hr_patch_t0)

        # Assemble the extended state:
        # LR latent in the exterior collar, HR latent in the center.
        hidden = lr_latent_up.clone()

        row_start = border
        row_end = border + self.patch_nlat_hr
        col_start = border
        col_end = border + self.patch_nlon_hr

        hidden[:, :, row_start:row_end, col_start:col_end] = hr_latent

        # Evolve the full extended window. Reimpose the same
        # dataset-derived LR collar after every ADR substep.
        for i in range(self.num_layers):
            hidden = self._adr_step(i, hidden)

            hidden = torch.where(
                self.hr_halo_mask,
                lr_latent_up,
                hidden,
            )

        # Project the extended latent field, then return only its HR center.
        output_extended = self.output_proj(hidden)

        output_interior = output_extended[
            :,
            :,
            row_start:row_end,
            col_start:col_end,
        ]

        return hr_patch_t0 + output_interior



# ---------------------------------------------------------------------------
# Build LAMParadis from a full config dict
# ---------------------------------------------------------------------------

def build_lam_model(config: dict) -> LAMParadis:
    """Construct LAMParadis from the full config_paradis_lam.yaml dict.

    Usage:
        import yaml
        with open("config_paradis_lam.yaml") as f:
            cfg = yaml.safe_load(f)
        model = build_lam_model(cfg)
    """
    lam_cfg   = config["lam"]
    model_cfg = config["model"]["paradis"]

    s         = int(lam_cfg["refinement_factor_lat"])
    assert s == int(lam_cfg["refinement_factor_lon"]), "Non-isotropic upscale not supported"

    patch_nlat_hr = int(lam_cfg["patch_nlat_lr"]) * s
    patch_nlon_hr = int(lam_cfg["patch_nlon_lr"]) * s
    halo_radius   = int(lam_cfg["halo_radius"])

    return LAMParadis(
        lam_cfg       = lam_cfg,
        model_cfg     = model_cfg,
        patch_nlat_hr = patch_nlat_hr,
        patch_nlon_hr = patch_nlon_hr,
        halo_radius   = halo_radius,
        upscale_factor= s,
        lr_nlat       = int(config["data"]["nlat"]),
        lr_nlon       = int(config["data"]["nlon"]),
    )
