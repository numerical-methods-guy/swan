import torch
import torch.nn as nn


class ReversedHuberLoss(nn.Module):
    """Smooth reversed Huber (Berhu) loss.

    Blends L1 (small errors) and quadratic (large errors) via a sigmoid weight:
        weight = sigmoid(2 * (|e| - delta))
        loss   = (1 - weight) * delta * |e|  +  weight * (e^2 + delta^2) / (2 * delta)
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = torch.as_tensor(self.delta, device=pred.device, dtype=pred.dtype)
        error = pred - target
        abs_error = torch.abs(error)
        small = delta * abs_error
        large = (error ** 2 + delta ** 2) / (2 * delta)
        weight = torch.sigmoid(2 * (abs_error - delta))
        return ((1 - weight) * small + weight * large).mean()
