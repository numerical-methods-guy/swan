from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def build_lr_blend_weight_hr(
    patch_nlat_hr: int,
    patch_nlon_hr: int,
    blend_width_hr: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build the LR/coarse blend weight on the HR patch.

    Returns
    -------
    w_lr : torch.Tensor
        Shape [1, 1, H, W], values in [0, 1].
        - 1 at the outer patch edge
        - 0 in the free zone
        - cosine-squared taper across the blend ring
    """
    if blend_width_hr < 0:
        raise ValueError(f"blend_width_hr must be >= 0, got {blend_width_hr}")

    H = int(patch_nlat_hr)
    W = int(patch_nlon_hr)

    if blend_width_hr == 0:
        return torch.zeros((1, 1, H, W), device=device, dtype=dtype)

    yy = torch.arange(H, device=device, dtype=dtype).view(H, 1).expand(H, W)
    xx = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W)

    dist_top = yy
    dist_bottom = (H - 1) - yy
    dist_left = xx
    dist_right = (W - 1) - xx

    dist_to_edge = torch.minimum(
        torch.minimum(dist_top, dist_bottom),
        torch.minimum(dist_left, dist_right),
    )

    w_lr = torch.zeros((H, W), device=device, dtype=dtype)
    b = int(blend_width_hr)

    if b == 1:
        w_lr[dist_to_edge < 1] = 1.0
    else:
        mask = dist_to_edge < b
        xi = dist_to_edge[mask] / float(b - 1)
        w_lr[mask] = torch.cos(0.5 * math.pi * xi) ** 2

    return w_lr.unsqueeze(0).unsqueeze(0)


def build_hr_retention_weight(
    w_lr: torch.Tensor,
    *,
    power: float = 1.0,
    min_weight: float = 0.0,
) -> torch.Tensor:
    """
    Convert LR blend weight to HR-retention / training-loss weight.

    Parameters
    ----------
    w_lr : [1, 1, H, W]
        LR/coarse blend weight.
    power : float
        Optional sharpening of the HR weight.
    min_weight : float
        Floor so the blend zone still contributes to loss.
        0.0 means pure complement, 0.1 means at least 10% weight everywhere.
    """
    if power <= 0.0:
        raise ValueError(f"power must be > 0, got {power}")
    if not (0.0 <= min_weight <= 1.0):
        raise ValueError(f"min_weight must be in [0, 1], got {min_weight}")

    w_hr = (1.0 - w_lr).clamp(0.0, 1.0)
    if power != 1.0:
        w_hr = w_hr.pow(power)

    if min_weight > 0.0:
        w_hr = min_weight + (1.0 - min_weight) * w_hr

    return w_hr


def weighted_patch_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted MSE over [B, C, H, W].

    weight may be [1,1,H,W], [1,C,H,W], or [B,C,H,W].
    """
    err2 = (pred - target) ** 2
    w = weight.to(device=pred.device, dtype=pred.dtype)
    return (err2 * w).sum() / w.expand_as(err2).sum().clamp_min(1.0)


def crop_lr_patch_from_halo(
    lr_halo: torch.Tensor,
    patch_nlat_lr: int,
    patch_nlon_lr: int,
    halo_radius: int,
) -> torch.Tensor:
    """
    Extract the interior LR state patch from an LR halo window.

    lr_halo shape: [B, C, Hwin, Wwin], where C=5 for
    [height, divergence, vorticity, u, v].

    Returns: [B, C, patch_nlat_lr, patch_nlon_lr].
    """
    r = int(halo_radius)
    return lr_halo[:, :, r:r + patch_nlat_lr, r:r + patch_nlon_lr]


def upsample_lr_patch_to_hr(
    lr_patch: torch.Tensor,
    hr_size: tuple[int, int],
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Upsample LR prognostic patch to HR patch size.
    """
    if mode not in ("nearest", "bilinear", "bicubic"):
        raise ValueError(f"Unsupported interpolation mode '{mode}'")

    if mode == "nearest":
        return F.interpolate(lr_patch, size=hr_size, mode=mode)

    return F.interpolate(lr_patch, size=hr_size, mode=mode, align_corners=False)


def blend_hr_prediction_with_lr(
    hr_pred: torch.Tensor,
    lr_patch: torch.Tensor,
    w_lr_hr: torch.Tensor,
    *,
    interpolation: str = "bilinear",
) -> torch.Tensor:
    """
    Blend HR prediction with an LR prognostic patch.

    Parameters
    ----------
    hr_pred : [B, C, Hhr, Whr]
    HR prediction, typically in physical units during rollout.
    For this model, C=5: [height, divergence, vorticity, u, v].

    lr_patch : [B, C, Hlr, Wlr] or [B, C, Hhr, Whr]
    LR state patch in physical units, with the same channel order as hr_pred.
    w_lr_hr : [1, 1, Hhr, Whr] or [B, 1, Hhr, Whr]
        LR blend weight on HR grid.
    """
    if hr_pred.ndim != 4 or lr_patch.ndim != 4:
        raise ValueError(
            "hr_pred and lr_patch must both have shape [B, C, H, W]."
        )

    if hr_pred.shape[0] != lr_patch.shape[0]:
        raise ValueError(
            f"Batch mismatch: hr_pred has B={hr_pred.shape[0]}, "
            f"lr_patch has B={lr_patch.shape[0]}."
        )

    if hr_pred.shape[1] != lr_patch.shape[1]:
        raise ValueError(
            f"Channel mismatch: hr_pred has C={hr_pred.shape[1]}, "
            f"lr_patch has C={lr_patch.shape[1]}."
        )
    
    if lr_patch.shape[-2:] != hr_pred.shape[-2:]:
        lr_on_hr = upsample_lr_patch_to_hr(
            lr_patch,
            hr_size=hr_pred.shape[-2:],
            mode=interpolation,
        )
    else:
        lr_on_hr = lr_patch

    w_lr = w_lr_hr.to(device=hr_pred.device, dtype=hr_pred.dtype)
    return (1.0 - w_lr) * hr_pred + w_lr * lr_on_hr