import torch


class PeriodicPadding1D(torch.nn.Module):
    """Circular padding for 1D periodic domains."""

    def __init__(self, pad_width):
        super().__init__()
        self.pad_width = pad_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cyclic padding to the input tensor."""
        if self.pad_width == 0:
            return x

        assert len(x.shape) == 3, "Input must be 3-dimensional [batch, channels, x]"
        batch_size, channels, width = x.shape


        x_padded = torch.cat(
            [x[:, :, -self.pad_width:], x, x[:, :, :self.pad_width]], dim=2
        )

        return x_padded
