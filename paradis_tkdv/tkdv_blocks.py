from collections import OrderedDict
from collections.abc import Sequence
from typing import Union, Type

import torch
from torch import nn

from paradis_tkdv.tkdv_padding import PeriodicPadding1D

"""
    Building blocks for the 1D periodic tKdV Paradis model.
    Tensors have shape (batch, channels, x) where x is the spatial grid of length J.
    mesh_size is an int (domain length J) throughout.
"""


def init_conv1d_default(conv: nn.Conv1d, *, scale: float = 1.0) -> None:
    nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
    if scale != 1.0:
        with torch.no_grad():
            conv.weight.mul_(scale)
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0.0)


def init_module_convs(m: nn.Module, *, last_conv_scale: float = 1.0) -> None:
    convs = []
    for module in m.modules():
        if isinstance(module, GlobalBias):
            continue
        if isinstance(module, nn.Conv1d):
            convs.append(module)

    for i, conv in enumerate(convs):
        scale = last_conv_scale if (i == len(convs) - 1) else 1.0
        init_conv1d_default(conv, scale=scale)


class CLinear(nn.Module):
    """Channel-wise linear transformation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mesh_size: int,
        kernel_size: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SepConv(nn.Module):
    """Separable convolution."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mesh_size: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) // 2
        self.periodic_padding = PeriodicPadding1D(self.padding)

        self.depthwise = nn.Conv1d(
            input_dim, input_dim, kernel_size, groups=input_dim, bias=False
        )
        self.pointwise = nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.periodic_padding(x)
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
        cvar, cmean = torch.var_mean(x, dim=-2, keepdim=False)
        inv_std = (self.eps + cvar) ** -0.5
        shifted_x = x - cmean[..., None, :]
        x = torch.einsum("...ci,...i,c->...ci", shifted_x, inv_std, self.weight)
        x = x + self.bias[..., :, None]
        return x


# LowRankBias -- low-rank factorized bias operator
class GlobalBias(nn.Module):
    """
    LowRankBias -- construct a low-rank factorized bias operator that reduces
    the number of parameters while maintaining expressiveness through separable
    rank-K decomposition.

    Uses factors:
    - A ∈ R^{C_in×K} (per-channel coefficients)
    - U ∈ R^{K×J} (spatial factors)

    The bias map is computed as: y_c = ∑_{k=1}^K A_{ck} u_k
    With optional projection to output channels.

    Parameters: K*(C_in + J) vs C_in*J for a full bias
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        bias: bool = True,  # Not used (would be redundant)
        kernel_size: int = 0,  # Not used
        mesh_size: int,  # domain length J
        rank: int = 128,  # K - rank of the factorization
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.width = mesh_size

        # Factor matrices
        self.A = nn.Parameter(torch.zeros(input_dim, rank), requires_grad=True)
        self.U = nn.Parameter(torch.zeros(rank, self.width), requires_grad=True)

        with torch.no_grad():
            nn.init.normal_(self.A, mean=0.0, std=1e-3)
            nn.init.normal_(self.U, mean=0.0, std=1e-3)

        # Optional projection to output channels
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # bias_maps: [C_in, J]
        bias_maps = torch.einsum("ck,kw->cw", self.A, self.U)

        if self.projection is not None:
            # [C_out, J]
            bias_maps = torch.einsum("oc,cw->ow", self.projection.weight, bias_maps)

        # x: [B, C_out, J] (or [B, C_in, J] if no projection)
        return x + bias_maps.unsqueeze(0)



ActivationType = Type[nn.Module]

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
        mesh_size: int,
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
        init_module_convs(self, last_conv_scale=0.1)
