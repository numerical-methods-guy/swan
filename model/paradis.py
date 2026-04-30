"""Paradis neural architecture adapted for shallow water equations."""

from torch.utils.checkpoint import checkpoint

import torch
from torch import nn
import torch.nn.functional as F


from model.advection import NeuralSemiLagrangian
from model.blocks import GMBlock, PhysicalDownsample, SepConv
from model.padding import GeoCyclicPadding


def get_scaled_timestep(original_timestep_seconds: float) -> float:
    return original_timestep_seconds * 7.29212e-5


_ACTIVATIONS = {
    "SiLU": nn.SiLU,
    "GELU": nn.GELU,
}


def _get_activation_cls(name: str) -> type[nn.Module]:
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation_fn '{name}'. Allowed: {list(_ACTIVATIONS.keys())}"
        )
    return _ACTIVATIONS[name]


class Paradis(nn.Module):
    """Paradis model adapted for shallow water equations."""

    def __init__(self, config):
        super().__init__()

        self.nlat = config["data"]["nlat"]
        self.nlon = config["data"]["nlon"]

        self.grid = "equiangular"

        if self.grid != "equiangular":
            raise ValueError(
                f"Paradis model only supports 'equiangular' grid, got '{self.grid}'. "
                "Please set data.grid='equiangular' in your config."
            )

        mesh_size = (self.nlat, self.nlon)

        model_config = config["model"]["paradis"]  # SWAN

        hidden_dim = model_config["hidden_dim"]

        self.num_vels = model_config["num_vels"]

        adv_interpolation = model_config["interpolation"]
        bias_channels = model_config.get("bias_channels", 4)

        self.num_layers = max(1, model_config["num_layers"])
        self.dt = get_scaled_timestep(model_config["base_dt"]) / self.num_layers

        # Input projection
        self.activation_function = _get_activation_cls(
            model_config.get("activation", "SiLU")
        )

        input_dim = 5

        # Wrapper for gradient checkpointing
        self.step_fn = self._layer_step
        self.gradient_checkpoint = config.get("training", {}).get(
            "gradient_checkpointing", False
        )

        self.downsample_diffusion = False

        if self.gradient_checkpoint:
            self.step_fn = lambda i, h: checkpoint(
                self._layer_step, i, h, use_reentrant=False
            )

        physblock = model_config.get("physblock", {})

        input_layers = physblock.get("input_proj", {}).get(
            "layers", ["SepConv", "CLinear"]
        )
        vnet_layers = physblock.get("velocity_net", {}).get("layers", ["SepConv"])
        diffusion_layers = physblock.get("diffusion", {}).get("layers", ["SepConv"])
        reaction_layers = physblock.get("reaction", {}).get(
            "layers", ["CLinear", "CLinear"]
        )
        output_layers = physblock.get("output_proj", {}).get(
            "layers", ["SepConv", "CLinear"]
        )

        input_ldim = physblock.get("input_proj", {}).get("hidden_dim", hidden_dim)
        vnet_ldim = physblock.get("velocity_net", {}).get("hidden_dim", hidden_dim)
        diff_ldim = physblock.get("diffusion", {}).get(
            "hidden_dim", model_config.get("diffusion_size", hidden_dim)
        )
        reac_ldim = physblock.get("reaction", {}).get(
            "hidden_dim", model_config.get("reaction_size", hidden_dim)
        )
        output_ldim = physblock.get("output_proj", {}).get("hidden_dim", hidden_dim)

        adv_block = physblock.get("advection", {})
        down_proj_layers = adv_block.get("down_projection", {}).get(
            "layers", ["CLinear"]
        )
        up_proj_layers = adv_block.get("up_projection", {}).get("layers", ["SepConv"])
        down_proj_ldim = adv_block.get("down_projection", {}).get("hidden_dim", 0)
        up_proj_ldim = adv_block.get("up_projection", {}).get("hidden_dim", 0)

        mesh_size_coarse = mesh_size

        dlon = 360.0 / self.nlon
        lat = torch.linspace(-90, 90, self.nlat, dtype=torch.float32)
        lon = torch.linspace(0, 360 - dlon, self.nlon, dtype=torch.float32)
        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
        lat_grid = torch.deg2rad(lat_grid)
        lon_grid = torch.deg2rad(lon_grid)

        self.input_proj = GMBlock(
            layers=input_layers,
            input_dim=input_dim,
            output_dim=hidden_dim,
            hidden_dim=input_ldim,
            mesh_size=mesh_size,
            activation=True,
            activation_fn=self.activation_function,
            pre_normalize=False,
            bias_channels=0,
        )

        self.velocity_nets = nn.ModuleList(
            [
                GMBlock(
                    layers=vnet_layers,
                    input_dim=hidden_dim,
                    output_dim=2 * self.num_vels,
                    hidden_dim=vnet_ldim,
                    mesh_size=mesh_size,
                    bias_channels=bias_channels,
                    activation_fn=self.activation_function,
                    pre_normalize=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.advection = nn.ModuleList(
            [
                NeuralSemiLagrangian(
                    hidden_dim,
                    mesh_size,
                    num_vels=self.num_vels,
                    lat_grid=lat_grid,
                    lon_grid=lon_grid,
                    interpolation=adv_interpolation,
                    down_proj_layers=down_proj_layers,
                    up_proj_layers=up_proj_layers,
                    down_proj_ldim=down_proj_ldim,
                    up_proj_ldim=up_proj_ldim,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.diffusion = nn.ModuleList(
            [
                GMBlock(
                    layers=diffusion_layers,
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    hidden_dim=diff_ldim,
                    mesh_size=mesh_size_coarse,
                    pre_normalize=True,
                    activation_fn=self.activation_function,
                    bias_channels=bias_channels,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.reaction = nn.ModuleList(
            [
                GMBlock(
                    layers=reaction_layers,
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    hidden_dim=reac_ldim,
                    mesh_size=mesh_size,
                    pre_normalize=True,
                    activation_fn=self.activation_function,
                    bias_channels=bias_channels,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.output_proj = GMBlock(
            layers=output_layers,
            input_dim=hidden_dim,
            output_dim=3,
            hidden_dim=output_ldim,
            mesh_size=mesh_size,
            activation=False,
            activation_fn=self.activation_function,
            bias_channels=bias_channels,
        )

        self.alpha_adv = nn.Parameter(torch.full((self.num_layers, hidden_dim), -1.0))

        self.downsample = lambda x: x
        self.upsample = lambda x: x

    def _apply_checkpoint(self, func, *args):
        if self.gradient_checkpoint:
            return checkpoint(func, *args, use_reentrant=False)
        else:
            return func(*args)

    def _diffusion(self, i: int, z: torch.Tensor) -> torch.Tensor:
        return self.upsample(self.diffusion[i](self.downsample(z)))

    def _layer_step(self, i: int, hidden: torch.Tensor) -> torch.Tensor:
        """Single physics-informed latent update."""
        B = hidden.shape[0]

        # Predict latent velocities (u, v) for advection
        velocities_raw = self.velocity_nets[i](hidden)
        velocities = velocities_raw.view(B, 2, self.num_vels, self.nlat, self.nlon)
        u, v = velocities[:, 0], velocities[:, 1]

        g_adv = torch.sigmoid(self.alpha_adv[i]).to(hidden.dtype).view(1, -1, 1, 1)

        # Transport: Semi-Lagrangian advection
        advected = self.advection[i](hidden, u, v, self.dt)
        hidden = hidden + g_adv * (advected - hidden)

        # Mixing: Learned diffusion
        hidden = hidden + self._diffusion(i, hidden)

        # Forcing: Pointwise reaction (primary nonlinearity)
        hidden = hidden + self.reaction[i](hidden)

        return hidden

    def forward(self, fields: torch.Tensor, winds: torch.Tensor) -> torch.Tensor:

        x = torch.cat([fields, winds], dim=1)

        # Encode physical variables to latent space
        hidden = self._apply_checkpoint(self.input_proj, x)

        # Recurrent integration through physics layers
        for i in range(self.num_layers):
            hidden = self.step_fn(i, hidden)

        return fields + self._apply_checkpoint(self.output_proj, hidden)
