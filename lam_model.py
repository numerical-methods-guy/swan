"""
lam_model.py — LAMParadis: Local Area Model using dual-encoder PARADIS architecture.

Architecture (Option B — dual encoder, latent-space fusion)
------------------------------------------------------------

                lr_halo  [B, 5, H_lr_win, W_lr_win]
                    |
                    v
          LR Encoder (GMBlock at LR mesh_size_lr_win)
                    |
          [B, D, H_lr_win, W_lr_win]
                    |
          bilinear upsample to HR window size
                    |
          [B, D, H_hr_win, W_hr_win]      crop halo border (R*s each side)
                    |
          [B, D, H_hr_patch, W_hr_patch] ─────────────────┐
                                                           │  concat → [B, 2D, H_hr, W_hr]
          HR Encoder (GMBlock at HR patch mesh_size_hr)    │    → 1×1 proj → [B, D, H_hr, W_hr]
                    |                                      │
          [B, D, H_hr_patch, W_hr_patch] ─────────────────┘
                    |
          fused latent [B, D, H_hr_patch, W_hr_patch]
                    |
          num_layers × ADR step (velocity net + advection + diffusion + reaction)
          (all ops at HR patch resolution — patch uses zero/replicate boundary padding,
           NOT GeoCyclicPadding, since it is a local patch, not a global sphere)
                    |
          HR Output projection (GMBlock)
                    |
          + residual skip (hr_patch_t0)
                    |
          [B, 3, H_hr_patch, W_hr_patch]   ← prediction of hr_patch at t+dt

Key design notes
----------------
1. GeoCyclicPadding is ONLY used in the LR encoder (it processes a window extracted
   from a global spherical field).  The HR ADR layers use replicate padding because
   the HR patch is a local tile — its edges are NOT periodic and are NOT poles.

2. NeuralSemiLagrangian.enforce_pole_continuity() is suppressed in the HR ADR layers
   for the same reason: patch row 0 and row -1 are interior rows, not poles.

3. The LR encoder mesh_size is the LR window size (pL+2R, pN+2R), NOT the full global
   LR grid — because we feed it a cropped window tensor, not the global LR field.

4. GlobalBias in GMBlock requires a fixed mesh_size that matches the actual tensor
   spatial dimensions.  Each GMBlock therefore receives its own mesh_size.

5. alpha_adv gate (learnable sigmoid gate from Paradis) is retained.

6. lat/lon grids for NeuralSemiLagrangian are built for the HR patch using the patch's
   actual geographic extent, derived from its position.  Since patch position varies per
   sample, we build a canonical patch-centred grid (0,0 centred) and let the semi-
   Lagrangian work in local coordinates.  For a local patch this is a good approximation.
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
    """Dual-encoder LAM model for HR patch super-resolution."""

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

        self.fusion_proj = nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.fusion_proj.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.fusion_proj.bias)

        hr_dlat = 180.0 / (lr_nlat * upscale_factor)
        hr_dlon = 360.0 / (lr_nlon * upscale_factor)

        lat_start = -(patch_nlat_hr - 1) / 2.0 * hr_dlat
        lat_end = (patch_nlat_hr - 1) / 2.0 * hr_dlat
        lon_start = 0.0
        lon_end = (patch_nlon_hr - 1) * hr_dlon

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
            output_dim=3,
            hidden_dim=output_ldim,
            mesh_size=hr_patch_mesh,
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

        lr_latent = self.lr_encoder(lr_halo)
        lr_latent_up = F.interpolate(
            lr_latent,
            size=(self.hr_win_nlat, self.hr_win_nlon),
            mode="bilinear",
            align_corners=False,
        )

        hr_latent = self.hr_encoder(hr_patch_t0)
        hr_latent_padded = lr_latent_up.clone()
        hr_latent_padded[:, :, border:-border, border:-border] = hr_latent

        fused = self.fusion_proj(torch.cat([lr_latent_up, hr_latent_padded], dim=1))

        hidden = fused
        for i in range(self.num_layers):
            hidden = self._adr_step(i, hidden)

        hidden_interior = hidden[
            :,
            :,
            border : border + self.patch_nlat_hr,
            border : border + self.patch_nlon_hr,
        ]

        return hr_patch_t0[:, :3, :, :] + self.output_proj(hidden_interior)
    
    # def forward(
    #     self,
    #     lr_halo: torch.Tensor,
    #     hr_patch_t0: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     Parameters
    #     ----------
    #     lr_halo     : [B, 5, lr_win_nlat, lr_win_nlon]
    #                   LR fields (3) + winds (2) over the full window (patch + halo)
    #     hr_patch_t0 : [B, 3, patch_nlat_hr, patch_nlon_hr]
    #                   HR fields over the interior patch at time t

    #     Returns
    #     -------
    #     hr_patch_t1 : [B, 3, patch_nlat_hr, patch_nlon_hr]
    #                   Predicted HR fields at time t+dt (residual added to input)
    #     """
    #     B = lr_halo.shape[0]
    #     Hhr, Whr = self.patch_nlat_hr, self.patch_nlon_hr
    #     border   = self.hr_halo_border

    #     # --- LR encoder branch -------------------------------------------------
    #     lr_latent = self.lr_encoder(lr_halo)
    #     # [B, D, lr_win_nlat, lr_win_nlon]

    #     # Upsample LR latent to HR window size
    #     lr_win_hr_nlat = self.lr_win_nlat * self.upscale_factor
    #     lr_win_hr_nlon = self.lr_win_nlon * self.upscale_factor
    #     lr_latent_up = F.interpolate(
    #         lr_latent,
    #         size=(lr_win_hr_nlat, lr_win_hr_nlon),
    #         mode="bilinear",
    #         align_corners=False,
    #     )
    #     # [B, D, lr_win_nlat*s, lr_win_nlon*s]

    #     # Crop halo border to get patch-interior-sized latent
    #     lr_latent_crop = lr_latent_up[
    #         :, :,
    #         border : border + Hhr,
    #         border : border + Whr,
    #     ]
    #     # [B, D, patch_nlat_hr, patch_nlon_hr]

    #     # --- HR encoder branch -------------------------------------------------
    #     hr_latent = self.hr_encoder(hr_patch_t0)
    #     # [B, D, patch_nlat_hr, patch_nlon_hr]

    #     # --- Fusion: concat + 1x1 projection -----------------------------------
    #     fused = self.fusion_proj(torch.cat([lr_latent_crop, hr_latent], dim=1))
    #     # [B, D, patch_nlat_hr, patch_nlon_hr]

    #     # --- HR ADR integration ------------------------------------------------
    #     hidden = fused
    #     for i in range(self.num_layers):
    #         hidden = self._adr_step(i, hidden)

    #     # --- Output projection + residual skip ---------------------------------
    #     return hr_patch_t0 + self.output_proj(hidden)


# ---------------------------------------------------------------------------
# Factory helper: build LAMParadis from a full config dict
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
