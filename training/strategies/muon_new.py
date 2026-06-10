"""Muon-new strategy.

Flattened the ([C_out, C_in, 1, 1]) weights in the input projection, velocity networks, diffusion blocks, reaction blocks, and the advection down/up projections into [C_out, C_in] proxy matrices before applying the Muon update. These parameters are flattened since they are equivalent to a hidden linear layer that is not spatially convolved, and fits the Muon intuition of orthogonalizing a matrix transformation.
Depthwise/spatial convolution kernels, biases, normalization
parameters, and output heads stay on AdamW. 
"""

from __future__ import annotations

from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from torch import nn

from training.strategies.base import TrainingStrategy


EXCLUDED_MUON_NEW_1X1_MODULES = {"output_proj", "velocity_nets", "advection", "reaction"}


def _resolve_flat_lr(train_cfg: dict, module: pl.LightningModule) -> float:
    lr = train_cfg["learning_rate"]
    if module.nfuture > 0:
        lr = train_cfg.get("finetune_learning_rate", lr)
    return lr


@dataclass
class Conv1x1Proxy:
    name: str
    original: nn.Parameter
    proxy: nn.Parameter
    matrix_shape: tuple[int, int]


@dataclass
class MuonParamSplit:
    native_muon_params: list[nn.Parameter]
    conv1x1_proxies: list[Conv1x1Proxy]
    excluded_conv1x1_params: list[nn.Parameter]
    other_params: list[nn.Parameter]

    @property
    def muon_params(self) -> list[nn.Parameter]:
        return self.native_muon_params + [spec.proxy for spec in self.conv1x1_proxies]


def _is_matrix_like_conv1x1(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.Conv2d)
        and module.kernel_size == (1, 1)
        and module.groups == 1
    )


def _is_excluded_muon_new_module(module_name: str) -> bool:
    top_level_name = module_name.split(".", 1)[0]
    return top_level_name in EXCLUDED_MUON_NEW_1X1_MODULES


def _make_conv1x1_proxy(name: str, param: nn.Parameter) -> Conv1x1Proxy:
    matrix_shape = (param.shape[0], param.shape[1])
    proxy = nn.Parameter(
        param.detach().reshape(matrix_shape).clone(),
        requires_grad=True,
    )
    return Conv1x1Proxy(
        name=name,
        original=param,
        proxy=proxy,
        matrix_shape=matrix_shape,
    )


def split_params_for_muon_new(model: nn.Module) -> MuonParamSplit:
    conv1x1_proxies: list[Conv1x1Proxy] = []
    excluded_conv1x1_params: list[nn.Parameter] = []
    proxied_param_ids: set[int] = set()

    for module_name, module in model.named_modules():
        if not _is_matrix_like_conv1x1(module):
            continue
        if module.weight.requires_grad:
            if _is_excluded_muon_new_module(module_name):
                excluded_conv1x1_params.append(module.weight)
                continue
            param_name = f"{module_name}.weight" if module_name else "weight"
            conv1x1_proxies.append(_make_conv1x1_proxy(param_name, module.weight))
            proxied_param_ids.add(id(module.weight))

    native_muon_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []

    for _name, param in model.named_parameters():
        if not param.requires_grad or id(param) in proxied_param_ids:
            continue
        if param.ndim == 2:
            native_muon_params.append(param)
        else:
            other_params.append(param)

    return MuonParamSplit(
        native_muon_params=native_muon_params,
        conv1x1_proxies=conv1x1_proxies,
        excluded_conv1x1_params=excluded_conv1x1_params,
        other_params=other_params,
    )


class MuonNewStrategy(TrainingStrategy):
    automatic_optimization = False

    def __init__(self, config: dict):
        super().__init__(config)
        if not hasattr(torch.optim, "Muon"):
            raise ImportError(
                "torch.optim.Muon is not available in this PyTorch build. "
                "Use a PyTorch build that provides torch.optim.Muon."
            )
        self._conv1x1_proxies: list[Conv1x1Proxy] = []

    def configure_optimizers(self, module: pl.LightningModule):
        train_cfg = self.config["training"]
        train_muon_cfg = {
            **self._optim_cfg("muon"),
            **self._optim_cfg("muon_new"),
        }
        train_adamw_cfg = self._optim_cfg("adamw")
        lr = train_muon_cfg.get("learning_rate")
        if lr is None:
            lr = train_cfg["learning_rate"]
        if module.nfuture > 0:
            lr = train_muon_cfg.get(
                "finetune_learning_rate",
                train_cfg.get("finetune_learning_rate", lr),
            )

        muon_lr = train_cfg.get(
            "muon_lr",
            train_muon_cfg.get("muon_lr", train_muon_cfg.get("learning_rate", lr)),
        )
        adamw_lr = train_cfg.get(
            "adamw_lr",
            train_muon_cfg.get(
                "adamw_lr",
                train_adamw_cfg.get("learning_rate", lr),
            ),
        )
        muon_momentum = train_cfg.get(
            "muon_momentum",
            train_cfg.get(
                "momentum",
                train_muon_cfg.get(
                    "momentum",
                    train_muon_cfg.get("muon_momentum", 0.95),
                ),
            ),
        )
        muon_wd = train_cfg.get(
            "muon_weight_decay",
            train_muon_cfg.get(
                "muon_weight_decay",
                train_muon_cfg.get("weight_decay", 0.0),
            ),
        )
        adamw_wd = train_cfg.get(
            "adamw_weight_decay",
            train_muon_cfg.get(
                "adamw_weight_decay",
                train_adamw_cfg.get("weight_decay", 1e-4),
            ),
        )

        split = split_params_for_muon_new(module.model)
        self._conv1x1_proxies = split.conv1x1_proxies

        print(
            "Muon-new parameter split: "
            f"{len(split.native_muon_params)} native matrix tensors -> Muon, "
            f"{len(split.conv1x1_proxies)} 1x1 conv matrices -> Muon proxies, "
            f"{len(split.excluded_conv1x1_params)} output/velocity 1x1 conv matrices -> AdamW, "
            f"{len(split.other_params)} other tensors -> AdamW"
        )

        muon_optimizer = torch.optim.Muon(
            split.muon_params,
            lr=muon_lr,
            momentum=muon_momentum,
            nesterov=True,
            ns_steps=5,
            weight_decay=muon_wd,
        )
        adamw_optimizer = torch.optim.AdamW(
            split.other_params,
            lr=adamw_lr,
            weight_decay=adamw_wd,
            foreach=True,
        )

        milestones = train_cfg.get("lr_milestones", None)
        gamma = train_cfg.get("lr_gamma", 0.5)
        cosine_eta_min = train_cfg.get("cosine_eta_min", None)
        num_epochs = train_cfg.get("pretrain_epochs")

        if cosine_eta_min is not None:
            schedulers = [
                {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                        muon_optimizer,
                        T_max=num_epochs,
                        eta_min=cosine_eta_min,
                    ),
                    "interval": "epoch",
                },
                {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                        adamw_optimizer,
                        T_max=num_epochs,
                        eta_min=cosine_eta_min,
                    ),
                    "interval": "epoch",
                },
            ]
        elif milestones is not None:
            schedulers = [
                {
                    "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                        muon_optimizer,
                        milestones=milestones,
                        gamma=gamma,
                    ),
                    "interval": "epoch",
                },
                {
                    "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                        adamw_optimizer,
                        milestones=milestones,
                        gamma=gamma,
                    ),
                    "interval": "epoch",
                },
            ]
        else:
            schedulers = [
                {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        adamw_optimizer,
                        mode="min",
                        factor=0.5,
                        patience=5,
                    ),
                    "monitor": "val_loss",
                },
            ]

        return [muon_optimizer, adamw_optimizer], schedulers

    def _zero_original_proxy_grads(self) -> None:
        for spec in self._conv1x1_proxies:
            spec.original.grad = None

    def _copy_original_grads_to_proxies(self) -> None:
        for spec in self._conv1x1_proxies:
            if spec.original.grad is None:
                spec.proxy.grad = None
            else:
                spec.proxy.grad = (
                    spec.original.grad.detach().reshape(spec.matrix_shape).clone()
                )

    @torch.no_grad()
    def _copy_proxy_weights_to_originals(self) -> None:
        for spec in self._conv1x1_proxies:
            spec.original.copy_(spec.proxy.reshape_as(spec.original))
            spec.original.grad = None

    def training_step(self, module: pl.LightningModule, loss: torch.Tensor) -> torch.Tensor:
        opt_muon, opt_adamw = module.optimizers()

        opt_muon.zero_grad(set_to_none=True)
        opt_adamw.zero_grad(set_to_none=True)
        self._zero_original_proxy_grads()

        module.manual_backward(loss)
        self._copy_original_grads_to_proxies()

        opt_muon.step()
        self._copy_proxy_weights_to_originals()
        opt_adamw.step()

        return loss

    def _get_schedulers(self, module: pl.LightningModule) -> list:
        schedulers = module.lr_schedulers()
        if not schedulers:
            return []
        return schedulers if isinstance(schedulers, list) else [schedulers]

    def on_train_epoch_end(self, module: pl.LightningModule) -> None:
        for scheduler in self._get_schedulers(module):
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

    def on_validation_epoch_end(self, module: pl.LightningModule) -> None:
        for scheduler in self._get_schedulers(module):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                val_loss = module.trainer.callback_metrics.get("val_loss")
                if val_loss is not None:
                    scheduler.step(val_loss)
