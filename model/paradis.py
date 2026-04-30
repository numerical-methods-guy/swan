"""Paradis neural architecture."""

import math

import torch
from torch import nn

from model.advection import NeuralSemiLagrangian
from model.blocks import GMBlock


def get_scaled_timestep(original_timestep_seconds: float) -> float:
    return original_timestep_seconds * 7.29212e-5


class ParadisModel(nn.Module):
    """Paradis model adapted for shallow water equations."""

    def __init__(self, config):
        super().__init__()

        self.nlat = config["data"]["nlat"]
        self.nlon = config["data"]["nlon"]
        self.grid = config["data"]["grid"]

        if self.grid != "equiangular":
            raise ValueError(
                f"Paradis model only supports 'equiangular' grid, got '{self.grid}'. "
                "Please set data.grid='equiangular' in your config."
            )

        mesh_size = (self.nlat, self.nlon)

        model_config = config["model"]["paradis"]

        hidden_dim = model_config["hidden_dim"]
        self.num_vels = model_config["num_vels"]
        diffusion_size = model_config["diffusion_size"]
        reaction_size = model_config["reaction_size"]

        adv_interpolation = model_config["interpolation"]
        bias_channels = model_config.get("bias_channels", 4)
        num_encoder_layers = model_config.get("num_encoder_layers", 1)

        self.num_layers = model_config["num_layers"]
        self.dt = get_scaled_timestep(model_config["base_dt"]) / self.num_layers

        # Create lat/lon grid
        dlat = 180.0 / (self.nlat - 1)
        dlon = 360.0 / self.nlon

        lat = torch.linspace(-90, 90, self.nlat, dtype=torch.float32)
        lon = torch.linspace(0, 360 - dlon, self.nlon, dtype=torch.float32)

        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
        lat_grid = torch.deg2rad(lat_grid)
        lon_grid = torch.deg2rad(lon_grid)

        # Input projection
        self.activation_function = nn.SiLU

        # Input: 5 channels (3 fields + 2 winds)
        input_dim = 5
        current_dim = input_dim
        encoder_layers = []
        bias = False
        
        for l in range(num_encoder_layers - 1):
            fc = nn.Conv2d(current_dim, hidden_dim, 1, bias=True)

            # Initialize the weights correctly
            scale = math.sqrt(2.0 / current_dim)
            nn.init.normal_(fc.weight, mean=0.0, std=scale)

            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0.0)

            encoder_layers.append(fc)
            encoder_layers.append(self.activation_function())

            current_dim = hidden_dim

        fc = nn.Conv2d(current_dim, hidden_dim, 1, bias=bias)
        scale = math.sqrt(1.0 / current_dim)
        nn.init.normal_(fc.weight, mean=0.0, std=scale)
        if fc.bias is not None:
            nn.init.constant_(fc.bias, 0.0)
        encoder_layers.append(fc)

        self.input_proj = nn.Sequential(*encoder_layers)

        self.velocity_nets = nn.ModuleList(
            [
                GMBlock(
                    layers=["SepConv"],
                    input_dim=hidden_dim,
                    output_dim=2 * self.num_vels,
                    hidden_dim=hidden_dim,
                    kernel_size=3,
                    mesh_size=mesh_size,
                    bias_channels=bias_channels,
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
                    project_advection=model_config.get("projected_advection", True),
                )
                for _ in range(self.num_layers)
            ]
        )

        self.diffusion = nn.ModuleList(
            [
                GMBlock(
                    layers=["SepConv"],
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    hidden_dim=diffusion_size,
                    mesh_size=mesh_size,
                    pre_normalize=True,
                    bias_channels=bias_channels,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.reaction = nn.ModuleList(
            [
                GMBlock(
                    layers=["CLinear"] * 2,
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    hidden_dim=reaction_size,
                    kernel_size=1,
                    mesh_size=mesh_size,
                    pre_normalize=True,
                    bias_channels=bias_channels,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.output_proj = GMBlock(
            layers=["SepConv", "CLinear"],
            input_dim=hidden_dim,
            output_dim=3,
            hidden_dim=hidden_dim,
            mesh_size=mesh_size,
            kernel_size=3,
            activation=False,
            bias_channels=bias_channels,
        )

    def forward(self, fields: torch.Tensor, winds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fields and winds.

        Parameters
        ----------
        fields : torch.Tensor
            Input fields of shape (batch, 3, nlat, nlon)
        winds : torch.Tensor
            Input winds of shape (batch, 2, nlat, nlon)

        Returns
        -------
        torch.Tensor
            Output fields of shape (batch, 3, nlat, nlon)
        """
        # Concatenate fields and winds
        x = torch.cat([fields, winds], dim=1)
        batch_size = x.shape[0]

        # Project features to latent space
        hidden = self.input_proj(x)

        for i in range(self.num_layers):
            velocities_raw = self.velocity_nets[i](hidden)

            # Obtain velocities in latent space
            velocities = velocities_raw.reshape(
                batch_size, 2, self.num_vels, self.nlat, self.nlon
            )
            u = velocities[:, 0]
            v = velocities[:, 1]

            # Apply SL advection, reaction and diffusion blocks
            advected = self.advection[i](hidden, u, v, self.dt)
            hidden = hidden + advected

            diffused = self.diffusion[i](hidden)
            hidden = hidden + diffused

            reacted = self.reaction[i](hidden)
            hidden = hidden + reacted

        # Project back to physical space
        return self.output_proj(hidden)
