"""Paradis neural architecture adapted for shallow water equations."""

import torch
from torch import nn
from collections import OrderedDict
from collections.abc import Sequence
from typing import Union, Type, Tuple


class GeoCyclicPadding(torch.nn.Module):
    """Cyclic padding layer for equiangular grids with poles."""

    def __init__(self, pad_width):
        super().__init__()
        self.pad_width = pad_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply cyclic padding to the input tensor."""
            # Validate input dimensions
            assert (
                len(x.shape) == 4
            ), "Input must be 4-dimensional [batch, channels, lat, lon]"
            batch_size, channels, height, width = x.shape
            assert width % 2 == 0, "Number of longitude points must be even"
    
            # For latitude padding, we need to rotate by 180° and account for longitude padding
            middle_index = width // 2
    
            # Take rows 1 to pad+1 (skipping row 0/South Pole)
            top_source = x[:, :, 1 : self.pad_width + 1, :] 
            
            # Take rows -(pad+1) to -1 (skipping row -1/North Pole)
            bottom_source = x[:, :, -(self.pad_width + 1) : -1, :] 
            
            # Apply 180° shift
            top_padding = torch.roll(top_source, shifts=middle_index, dims=3)
            bottom_padding = torch.roll(bottom_source, shifts=middle_index, dims=3)
            
            # Flip and Concatenate
            # The flip(2) correctly orders them: e.g., [89, 88] -> [88, 89] (ghosts above 90)
            x = torch.cat([top_padding.flip(2), x, bottom_padding.flip(2)], dim=2)
    
            # Longitude periodic padding
            x_padded = torch.cat(
                [x[:, :, :, -self.pad_width :], x, x[:, :, :, : self.pad_width]], dim=3
            )
    
            return x_padded







class CLinear(nn.Module):
    """Channel-wise linear layer (1x1 convolution)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mesh_size: tuple,
        kernel_size: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        # Kernel size is ignored or strictly 1 for CLinear
        self.layer = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class SepConv(nn.Module):
    """Separated convolution: 2D conv followed by channel-wise linear."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mesh_size: tuple,
        kernel_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size

        if kernel_size > 1:
            self.padding = GeoCyclicPadding(kernel_size // 2)

        self.conv = nn.Conv2d(
            input_dim, input_dim, kernel_size, groups=input_dim, bias=bias
        )
        self.linear = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size > 1:
            x = self.padding(x)
        x = self.conv(x)
        x = self.linear(x)
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


# Registry of available blocks for string lookup
BLOCK_REGISTRY = {
    "SepConv": SepConv,
    "CLinear": CLinear,
    "ChannelNorm": ChannelNorm,
    "GlobalBias": GlobalBias,
}


class GMBlock(nn.Sequential):
    """
    GMBlock: Generic Multilayer Block.
    Composes several simple blocks with activation functions.
    Matches original logic: Hidden layers have activation by default.
    """

    def __init__(
        self,
        layers: Sequence[Union[str, Type[nn.Module]]],  # List of layer types or names
        input_dim: int,
        output_dim: int,
        mesh_size: Tuple[int, int],
        kernel_size: int = 3,
        hidden_dim: Union[Sequence, int] = 0,
        activation_fn: Type[nn.Module] = nn.SiLU,
        bias_channels: int = 0,
        activation: Union[
            Sequence, bool
        ] = False,  # False means 'disable ONLY on last layer'
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

        # Optional Pre-normalization
        if pre_normalize:
            blocks.append(
                (
                    "0-ChannelNorm",
                    ChannelNorm(input_dim=input_dim, output_dim=input_dim),
                )
            )

        # Build Layers
        layer_in_size = input_dim

        for idx, l in enumerate(layers):
            # Resolve layer type from string or class
            if isinstance(l, str):
                if l not in BLOCK_REGISTRY:
                    raise ValueError(
                        f"Unknown layer type: {l}. Available: {list(BLOCK_REGISTRY.keys())}"
                    )
                ltype = BLOCK_REGISTRY[l]
            else:
                ltype = l

            # Determine output size for this specific layer
            if idx == num_layers - 1:
                layer_out_size = output_dim
            else:
                layer_out_size = hidden_dim[idx]

            # Construct the layer
            layer_name = f"{idx}-{ltype.__name__}"
            layer_obj = ltype(
                input_dim=layer_in_size,
                output_dim=layer_out_size,
                mesh_size=mesh_size,
                kernel_size=kernel_size,
            )
            blocks.append((layer_name, layer_obj))

            # Optional Global Bias (only after first layer if requested)
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

            # Activation
            if activation[idx]:
                blocks.append((f"{idx}-{activation_fn.__name__}", activation_fn()))

            # Update input size for next iteration
            layer_in_size = layer_out_size

        super().__init__(OrderedDict(blocks))


class NeuralSemiLagrangian(nn.Module):
    """Implements the semi-Lagrangian advection for shallow water equations."""

    def __init__(
        self,
        hidden_dim: int,
        mesh_size: tuple,
        num_vels: int,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
        interpolation: str = "bicubic",
        bias_channels: int = 4,
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

        self.velocity_net = GMBlock(
            layers=["SepConv"],
            input_dim=hidden_dim,
            output_dim=2 * num_vels,
            hidden_dim=hidden_dim,
            kernel_size=3,
            mesh_size=mesh_size,
            bias_channels=bias_channels,
            pre_normalize=True,
        )

        H, W = mesh_size

        # Store for later use
        self.register_buffer(
            "lat_grid", lat_grid.unsqueeze(0).unsqueeze(0).contiguous().clone()
        )
        self.register_buffer(
            "lon_grid", lon_grid.unsqueeze(0).unsqueeze(0).contiguous().clone()
        )

        # Buffers: normalization constants
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

        # Normalize longitude to [0, 2π]
        lon = torch.remainder(lon + 2 * torch.pi, 2 * torch.pi)

        return lat, lon

    def forward(
        self,
        hidden_features: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute advection using rotated coordinate system."""
        batch_size = hidden_features.shape[0]
        H, W = self.mesh_size

        velocities = self.velocity_net(hidden_features)
        velocities = velocities.reshape(batch_size, 2, self.num_vels, H, W)

        u = velocities[:, 0]
        v = velocities[:, 1]

        projected_inputs = self.down_projection(hidden_features)

        lon_prime = -u * dt
        lat_prime = -v * dt

        # Expand grids to match batch and num_vels dimensions
        lat_grid = self.lat_grid.expand(batch_size, self.num_vels, -1, -1)
        lon_grid = self.lon_grid.expand(batch_size, self.num_vels, -1, -1)

        lat_dep, lon_dep = self._transform_to_latlon(
            lat_prime, lon_prime, lat_grid, lon_grid
        )

        # Convert departure points to pixel locations
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

    SYNOPTIC_TIME_SCALE = 7.29212e5

    def __init__(self, config):
        super().__init__()

        self.nlat = config["data"]["nlat"]
        self.nlon = config["data"]["nlon"]
        self.grid = config["data"]["grid"]

        # Validate grid type
        if self.grid != "equiangular":
            raise ValueError(
                f"Paradis model only supports 'equiangular' grid, got '{self.grid}'. "
                "Please set data.grid='equiangular' in your config."
            )

        mesh_size = (self.nlat, self.nlon)

        model_config = config["model"]["paradis"]

        # Get channel sizes directly from config
        hidden_dim = model_config["hidden_dim"]
        num_vels = model_config["num_vels"]
        diffusion_size = model_config["diffusion_size"]
        reaction_size = model_config["reaction_size"]

        adv_interpolation = model_config["interpolation"]
        bias_channels = model_config.get("bias_channels", 4)

        # Create latitude and longitude grids for equiangular grid
        dlat = 180.0 / (self.nlat - 1)  # Spacing changes because endpoints are included
        dlon = 360.0 / self.nlon

        # Generate latitude including -90 and 90 exactly
        lat = torch.linspace(-90, 90, self.nlat, dtype=torch.float32)

        lon = torch.linspace(0, 360 - dlon, self.nlon, dtype=torch.float32)

        # Create 2D meshgrids and convert to radians
        # Use indexing='ij' to get (nlat, nlon) directly
        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
        lat_grid = torch.deg2rad(lat_grid)
        lon_grid = torch.deg2rad(lon_grid)

        # Input projection (3 channels -> hidden_dim)
        self.input_proj = GMBlock(
            layers=["SepConv"] * 2,
            input_dim=3,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            mesh_size=mesh_size,
            bias_channels=bias_channels,
            pre_normalize=True,
        )

        # Rescale the time step
        self.num_layers = max(1, model_config["num_layers"])
        dt = config["data"]["dt"]
        self.dt = dt / self.SYNOPTIC_TIME_SCALE / self.num_layers

        # Advection layer
        self.advection = nn.ModuleList(
            [
                NeuralSemiLagrangian(
                    hidden_dim,
                    mesh_size,
                    num_vels=num_vels,
                    lat_grid=lat_grid,
                    lon_grid=lon_grid,
                    interpolation=adv_interpolation,
                    bias_channels=bias_channels,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Diffusion-reaction layers
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

        # Output projection (hidden_dim -> 3 channels)
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

    def _DR(self, z: torch.Tensor, i: int) -> torch.Tensor:
        return self.diffusion[i](z) + self.reaction[i](z)

    def _step(self, z: torch.Tensor, i: int) -> torch.Tensor:
        # Lie-Trotter splitting with RK2 on diffusion and reaction layers
        zadv = self.advection[i](z, self.dt)
        k1 = self._DR(zadv, i)
        zmid = zadv + 0.5 * self.dt * k1
        k2 = self._DR(zmid, i)
        return zadv + self.dt * k2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project features to latent space
        z = self.input_proj(x)
        z0 = z.clone()

        # Compute advection and diffusion-reaction
        for i in range(self.num_layers):
            z = self._step(z, i)

        # Residual connection: input + change
        return x + self.output_proj(z - z0)
