"""Paradis neural architecture adapted for shallow water equations."""

import math
import torch
from torch import nn
from collections import OrderedDict
from collections.abc import Sequence
from typing import Union, Type, Tuple


def get_scaled_timestep(original_timestep_seconds: float) -> float:
    return original_timestep_seconds * 7.29212e-5


class GeoCyclicPadding(torch.nn.Module):
    """Cyclic padding layer for equiangular grids with poles."""

    def __init__(self, pad_width):
        super().__init__()
        self.pad_width = pad_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cyclic padding to the input tensor."""
        if self.pad_width == 0:
            return x

        assert (
            len(x.shape) == 4
        ), "Input must be 4-dimensional [batch, channels, lat, lon]"
        batch_size, channels, height, width = x.shape
        assert width % 2 == 0, "Number of longitude points must be even"

        middle_index = width // 2

        top_source = x[:, :, 1 : self.pad_width + 1, :]
        bottom_source = x[:, :, -(self.pad_width + 1) : -1, :]

        top_padding = torch.roll(top_source, shifts=middle_index, dims=3)
        bottom_padding = torch.roll(bottom_source, shifts=middle_index, dims=3)

        x = torch.cat([top_padding.flip(2), x, bottom_padding.flip(2)], dim=2)

        x_padded = torch.cat(
            [x[:, :, :, -self.pad_width :], x, x[:, :, :, : self.pad_width]], dim=3
        )

        return x_padded


class CLinear(nn.Module):
    """Channel-wise linear transformation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mesh_size: tuple,
        kernel_size: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SepConv(nn.Module):
    """Separable convolution."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mesh_size: tuple,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) // 2
        self.geo_padding = GeoCyclicPadding(self.padding)

        self.depthwise = nn.Conv2d(
            input_dim, input_dim, kernel_size, groups=input_dim, bias=False
        )
        self.pointwise = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.geo_padding(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ChannelNorm(nn.Module):
    """Channel normalization layer."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        assert input_dim == output_dim
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(input_dim), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(input_dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cvar, cmean = torch.var_mean(x, dim=-3, keepdim=False)
        inv_std = (self.eps + cvar) ** -0.5
        shifted_x = x - cmean[..., None, :, :]
        x = torch.einsum("...cij,...ij,c->...cij", shifted_x, inv_std, self.weight)
        x = x + self.bias[..., :, None, None]
        return x


class GlobalBias(nn.Module):
    """Learned bias operator with geophysical features."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mesh_size: tuple,
        bias: bool = True,
        kernel_size: int = 0,
    ):
        super().__init__()
        self.bias = nn.Parameter(
            torch.zeros(((input_dim,) + mesh_size)), requires_grad=True
        )

        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.projection is None:
            y = self.bias
        else:
            y = torch.einsum("iab,ji->jab", self.bias, self.projection.weight)

        x = x + y[..., :, :, :]
        return x


BLOCK_REGISTRY = {
    "SepConv": SepConv,
    "CLinear": CLinear,
    "ChannelNorm": ChannelNorm,
    "GlobalBias": GlobalBias,
}


class GMBlock(nn.Sequential):
    """
    Generic Multilayer Block.
    Composes several simple blocks with activation functions.
    """

    def __init__(
        self,
        layers: Sequence[Union[str, Type[nn.Module]]],
        input_dim: int,
        output_dim: int,
        mesh_size: Tuple[int, int],
        kernel_size: int = 3,
        hidden_dim: Union[Sequence, int] = 0,
        activation_fn: Type[nn.Module] = nn.SiLU,
        bias_channels: int = 0,
        activation: Union[Sequence, bool] = False,
        pre_normalize: bool = False,
    ):
        num_layers = len(layers)
        if num_layers == 0:
            raise ValueError("GMBlock: must specify at least one layer")

        if isinstance(activation, Sequence):
            assert len(activation) == num_layers
        else:
            activation = (True,) * (num_layers - 1) + (activation,)

        if isinstance(hidden_dim, Sequence):
            assert len(hidden_dim) == num_layers - 1
        else:
            if hidden_dim <= 0:
                hidden_dim = max(input_dim, output_dim)
            hidden_dim = (hidden_dim,) * (num_layers - 1)

        blocks = []

        if pre_normalize:
            blocks.append(
                (
                    "0-ChannelNorm",
                    ChannelNorm(input_dim=input_dim, output_dim=input_dim),
                )
            )

        layer_in_size = input_dim

        for idx, l in enumerate(layers):
            if isinstance(l, str):
                if l not in BLOCK_REGISTRY:
                    raise ValueError(
                        f"Unknown layer type: {l}. Available: {list(BLOCK_REGISTRY.keys())}"
                    )
                ltype = BLOCK_REGISTRY[l]
            else:
                ltype = l

            if idx == num_layers - 1:
                layer_out_size = output_dim
            else:
                layer_out_size = hidden_dim[idx]

            layer_name = f"{idx}-{ltype.__name__}"
            layer_obj = ltype(
                input_dim=layer_in_size,
                output_dim=layer_out_size,
                mesh_size=mesh_size,
                kernel_size=kernel_size,
            )
            blocks.append((layer_name, layer_obj))

            if idx == 0 and bias_channels > 0:
                blocks.append(
                    (
                        f"0-GlobalBias",
                        GlobalBias(
                            input_dim=bias_channels,
                            output_dim=layer_out_size,
                            mesh_size=mesh_size,
                        ),
                    )
                )

            if activation[idx]:
                blocks.append((f"{idx}-{activation_fn.__name__}", activation_fn()))

            layer_in_size = layer_out_size

        super().__init__(OrderedDict(blocks))


class NeuralSemiLagrangian(nn.Module):
    """Neural semi-Lagrangian advection operator."""

    def __init__(
        self,
        hidden_dim: int,
        mesh_size: tuple,
        num_vels: int,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
        interpolation: str = "bicubic",
    ):
        super().__init__()

        self.padding = 1
        if interpolation == "bicubic":
            self.padding = 2

        self.padding_interp = GeoCyclicPadding(self.padding)
        self.hidden_dim = hidden_dim
        self.num_vels = num_vels
        self.mesh_size = mesh_size

        self.down_projection = GMBlock(
            layers=["CLinear"],
            input_dim=hidden_dim,
            output_dim=num_vels,
            mesh_size=mesh_size,
            kernel_size=1,
        )

        self.up_projection = GMBlock(
            layers=["SepConv"],
            input_dim=num_vels,
            output_dim=hidden_dim,
            mesh_size=mesh_size,
            kernel_size=1,
        )

        self.interpolation = interpolation

        H, W = mesh_size

        self.register_buffer(
            "lat_grid", lat_grid.unsqueeze(0).unsqueeze(0).contiguous().clone()
        )
        self.register_buffer(
            "lon_grid", lon_grid.unsqueeze(0).unsqueeze(0).contiguous().clone()
        )

        self.register_buffer("Hf", torch.tensor(float(H)))
        self.register_buffer("Wf", torch.tensor(float(W)))
        self.register_buffer("pad", torch.tensor(float(self.padding)))

        self.register_buffer("min_lat", torch.min(lat_grid))
        self.register_buffer("max_lat", torch.max(lat_grid))
        self.register_buffer("min_lon", torch.min(lon_grid))
        self.register_buffer("max_lon", torch.max(lon_grid))

        self.register_buffer("d_lon", self.max_lon - self.min_lon)
        self.register_buffer("d_lat", self.max_lat - self.min_lat)

    def _transform_to_latlon(
        self,
        lat_prime: torch.Tensor,
        lon_prime: torch.Tensor,
        lat_p: torch.Tensor,
        lon_p: torch.Tensor,
    ) -> tuple:
        """Transform from local rotated coordinates back to standard latlon coordinates."""
        sin_lat_prime = torch.sin(lat_prime)
        cos_lat_prime = torch.cos(lat_prime)
        sin_lon_prime = torch.sin(lon_prime)
        cos_lon_prime = torch.cos(lon_prime)
        sin_lat_p = torch.sin(lat_p)
        cos_lat_p = torch.cos(lat_p)

        sin_lat = sin_lat_prime * cos_lat_p + cos_lat_prime * cos_lon_prime * sin_lat_p
        lat = torch.arcsin(torch.clamp(sin_lat, -1 + 1e-7, 1 - 1e-7))

        num = cos_lat_prime * sin_lon_prime
        den = cos_lat_prime * cos_lon_prime * cos_lat_p - sin_lat_prime * sin_lat_p
        lon = lon_p + torch.atan2(num, den)

        lon = torch.remainder(lon + 2 * torch.pi, 2 * torch.pi)

        return lat, lon

    def forward(
        self,
        hidden_features: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute advection using rotated coordinate system."""
        batch_size = hidden_features.shape[0]
        H, W = self.mesh_size

        projected_inputs = self.down_projection(hidden_features)

        lon_prime = -u * dt
        lat_prime = -v * dt

        lat_grid = self.lat_grid.expand(batch_size, self.num_vels, -1, -1)
        lon_grid = self.lon_grid.expand(batch_size, self.num_vels, -1, -1)

        lat_dep, lon_dep = self._transform_to_latlon(
            lat_prime, lon_prime, lat_grid, lon_grid
        )

        pix_x = (lon_dep - self.min_lon) / self.d_lon * (self.Wf - 1.0)
        pix_y = (lat_dep - self.min_lat) / self.d_lat * (self.Hf - 1.0)

        dynamic_padded = self.padding_interp(projected_inputs)

        pix_x_pad = pix_x + self.pad
        pix_y_pad = pix_y + self.pad

        H_pad = H + 2 * self.padding
        W_pad = W + 2 * self.padding

        grid_x = 2.0 * (pix_x_pad / float(W_pad - 1)) - 1.0
        grid_y = 2.0 * (pix_y_pad / float(H_pad - 1)) - 1.0

        grid_x = grid_x.reshape(batch_size * self.num_vels, H, W)
        grid_y = grid_y.reshape(batch_size * self.num_vels, H, W)

        grid = torch.stack([grid_x, grid_y], dim=-1)

        dynamic_padded = dynamic_padded.reshape(
            batch_size * self.num_vels, 1, H_pad, W_pad
        )

        interpolated = torch.nn.functional.grid_sample(
            dynamic_padded,
            grid,
            align_corners=True,
            mode=self.interpolation,
            padding_mode="zeros",
        )

        interpolated = self.up_projection(
            interpolated.reshape(batch_size, self.num_vels, H, W)
        )

        return interpolated


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
        # Save num_vels to self so it can be used in forward() for reshaping
        self.num_vels = model_config["num_vels"]
        diffusion_size = model_config["diffusion_size"]
        reaction_size = model_config["reaction_size"]

        adv_interpolation = model_config["interpolation"]
        bias_channels = model_config.get("bias_channels", 4)
        num_encoder_layers = model_config.get("num_encoder_layers", 1)

        dlat = 180.0 / (self.nlat - 1)
        dlon = 360.0 / self.nlon

        lat = torch.linspace(-90, 90, self.nlat, dtype=torch.float32)
        lon = torch.linspace(0, 360 - dlon, self.nlon, dtype=torch.float32)

        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
        lat_grid = torch.deg2rad(lat_grid)
        lon_grid = torch.deg2rad(lon_grid)

        # Input projection
        self.activation_function = nn.SiLU

        input_dim = 5  # Number of channels in input
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

        self.num_layers = max(1, model_config["num_layers"])

        self.dt = get_scaled_timestep(config["data"]["dt"]) / self.num_layers

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
        x = torch.cat([fields, winds], dim=1)

        hidden = self.input_proj(x)
        batch_size = hidden.shape[0]

        for i in range(self.num_layers):
            velocities_raw = self.velocity_nets[i](hidden)

            velocities = velocities_raw.reshape(
                batch_size, 2, self.num_vels, self.nlat, self.nlon
            )
            u = velocities[:, 0]
            v = velocities[:, 1]

            advected = self.advection[i](hidden, u, v, self.dt)
            hidden = hidden + advected

            diffused = self.diffusion[i](hidden)
            hidden = hidden + diffused

            reacted = self.reaction[i](hidden)
            hidden = hidden + reacted

        return self.output_proj(hidden)
