"""Loss construction for SWAN training."""

import torch

from utils.loss import ParadisLoss


def build_paradis_loss(config: dict) -> ParadisLoss:
    """Construct a ParadisLoss for the shallow water equation setting.

    The SWE model has three output channels (geopotential, vorticity, divergence)
    with no pressure-level structure, so all variables are treated as surface
    variables and pressure weighting is effectively bypassed.
    """
    nlat = config["data"]["nlat"]
    loss_cfg = config.get("loss", {})

    loss_function = loss_cfg.get("loss_function", "reversed_huber")
    delta_loss = loss_cfg.get("delta_loss", 1.0)

    lat_grid = torch.linspace(-90.0, 90.0, nlat, dtype=torch.float32)

    num_features = 3
    num_surface_vars = 3
    output_name_order = ["h", "vorticity", "divergence"]

    pressure_levels = torch.tensor([1000.0], dtype=torch.float32)
    var_loss_weights = torch.ones(num_features, dtype=torch.float32)

    return ParadisLoss(
        loss_function=loss_function,
        lat_grid=lat_grid,
        pressure_levels=pressure_levels,
        num_features=num_features,
        num_surface_vars=num_surface_vars,
        var_loss_weights=var_loss_weights,
        output_name_order=output_name_order,
        delta_loss=delta_loss,
    )
