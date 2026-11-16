import torch
import torch.nn.functional as F

from paradis_tkdv.tkdv_blocks import GMBlock
from paradis_tkdv.tkdv_padding import PeriodicPadding1D


class NeuralSemiLagrangian1D(torch.nn.Module):
    """Neural semi-Lagrangian advection operator."""

    def __init__(
        self,
        hidden_dim: int,
        mesh_size: int,
        num_vels: int,
        down_proj_layers: list = None,
        up_proj_layers: list = None,
        down_proj_ldim: int = 0,
        up_proj_ldim: int = 0,
    ):
        super().__init__()

        self.padding = 1

        self.padding_interp = PeriodicPadding1D(self.padding)
        self.hidden_dim = hidden_dim
        self.num_vels = num_vels
        self.mesh_size = mesh_size

        if down_proj_layers is None:
            down_proj_layers = ["CLinear"]
        if up_proj_layers is None:
            up_proj_layers = ["SepConv"]

        self.down_projection = GMBlock(
            layers=down_proj_layers,
            input_dim=hidden_dim,
            output_dim=num_vels,
            mesh_size=mesh_size,
            hidden_dim=down_proj_ldim,
        )

        self.up_projection = GMBlock(
            layers=up_proj_layers,
            input_dim=num_vels,
            output_dim=hidden_dim,
            mesh_size=mesh_size,
            hidden_dim=up_proj_ldim,
        )

        W = mesh_size

        self.register_buffer("Wf", torch.tensor(float(W)), persistent=False)


    def forward(
        self,
        hidden_features: torch.Tensor,
        u: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute 1D periodic semi-Lagrangian advection."""
        batch_size = hidden_features.shape[0]
        W = int(self.Wf.item())

        projected_inputs = self.down_projection(hidden_features)

        dep = torch.remainder(-u * dt, self.Wf)

        projected_padded = self.padding_interp(projected_inputs)

        pix_x_pad = dep + self.padding
        W_pad = W + 2 * self.padding

        grid_x = 2.0 * (pix_x_pad / float(W_pad - 1)) - 1.0
        grid_x = grid_x.reshape(batch_size * self.num_vels, 1, W)
        grid = torch.stack([grid_x, torch.zeros_like(grid_x)], dim=-1)

        projected_padded = projected_padded.reshape(batch_size * self.num_vels, 1, 1, W_pad)

        interpolated = F.grid_sample(
            projected_padded,
            grid,
            align_corners=True,
            mode="bilinear",
            padding_mode="zeros",
        )

        interpolated = interpolated.reshape(batch_size, self.num_vels, W)

        return self.up_projection(interpolated)
