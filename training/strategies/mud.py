"""MUD + AdamW hybrid optimizer strategy.
Adapts the PyTorch MUD implementation in https://arxiv.org/html/2603.17970 into our training strategy. """
from __future__ import annotations

import ast
import math
from typing import Iterable, Tuple

import pytorch_lightning as pl
import torch

from training.strategies.base import TrainingStrategy


@torch.no_grad()
def mud_whiten(M: torch.Tensor, passes: int = 1, eps: float = 1e-8) -> torch.Tensor:
    # Multi-pass MUD transform on a 2D matrix M (n, m)
    assert M.ndim == 2, "mud_whiten expects a 2D tensor"
    if passes < 1:
        raise ValueError(f"passes must be >= 1, got {passes}")

    n, m = M.shape
    transposed = n > m
    Q = M.t().contiguous() if transposed else M.contiguous()
    Q = Q.float() 

    for _ in range(int(passes)):
        Q = Q / Q.norm(dim=1, keepdim=True).clamp_min(eps)  # Row normalization
        G = Q @ Q.t()  # Row Gram (k,k)
        T = torch.tril(G)  # Lower-triangular of Gram
        Q = torch.linalg.solve_triangular(T, Q, upper=False)  # Forward solve: T X = Q
        Q = Q / Q.norm(dim=1, keepdim=True).clamp_min(eps)  # Renormalize rows

    if transposed:
        Q = Q.t().contiguous()

    return Q.to(dtype=M.dtype)


class MUDOptimizer(torch.optim.Optimizer):
    # MUD + AdamW hybrid optimizer

    def __init__(
        self,
        mud_params: Iterable[torch.nn.Parameter],
        adamw_params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        beta_mud: float = 0.95,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        mud_passes: int = 1,
        eps: float = 1e-8,
    ):
        self._eps = float(eps)
        self._mud_passes = int(mud_passes)

        if self._mud_passes < 1:
            raise ValueError(f"mud_passes must be >= 1, got {mud_passes}")

        mud_params = list(mud_params)
        adamw_params = list(adamw_params)

        param_groups = [
            dict(
                params=mud_params,
                lr=lr,
                weight_decay=weight_decay,
                beta_mud=beta_mud,
                use_mud=True,
            ),
            dict(
                params=adamw_params,
                lr=lr,
                weight_decay=weight_decay,
                adamw_betas=adamw_betas,
                use_mud=False,
            ),
        ]
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_mud", False):
                self._mud_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _mud_step(self, group: dict):
        lr = float(group["lr"])
        wd = float(group["weight_decay"])
        beta = float(group["beta_mud"])

        for p in group["params"]:
            if p.grad is None:
                continue

            g = p.grad
            state = self.state[p]

            if len(state) == 0:
                state["momentum"] = torch.zeros_like(p)

            p.mul_(1.0 - lr * wd)  

            m = state["momentum"]
            m.mul_(beta).add_(g)  # momentum
            u = g + beta * m  # Nesterov lookahead

            if p.ndim == 2:
                q = mud_whiten(u, passes=self._mud_passes, eps=self._eps)
                scale = 0.2 * math.sqrt(max(int(p.size(0)), int(p.size(1))))
                p.add_(q, alpha=-lr * scale)
            else:
                p.add_(u, alpha=-lr)  # fallback for non-matrix params

    def _adamw_step(self, group: dict):
        lr = float(group["lr"])
        wd = float(group["weight_decay"])
        b1, b2 = group.get("adamw_betas", (0.9, 0.95))
        b1, b2 = float(b1), float(b2)
        eps = self._eps

        for p in group["params"]:
            if p.grad is None:
                continue

            g = p.grad
            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0
                state["m"] = torch.zeros_like(p)
                state["v"] = torch.zeros_like(p)

            state["step"] += 1
            t = state["step"]

            p.mul_(1.0 - lr * wd)  # decoupled weight decay

            m = state["m"]
            v = state["v"]
            m.mul_(b1).add_(g, alpha=1.0 - b1)
            v.mul_(b2).addcmul_(g, g, value=1.0 - b2)

            m_hat = m / (1.0 - b1**t)
            v_hat = v / (1.0 - b2**t)
            p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)


def split_params_for_mud(model: torch.nn.Module) -> tuple[list, list]:
    # Split trainable parameters into MUD and AdamW parameter lists 
    mud_params = []
    adamw_params = []

    for param in model.parameters():
        if not param.requires_grad:
            continue

        if param.ndim == 2:
            mud_params.append(param)
        else:
            adamw_params.append(param)

    return mud_params, adamw_params


def _resolve_adamw_betas(train_cfg: dict) -> Tuple[float, float]:
    raw_betas = train_cfg.get("adamw_betas", None)

    if isinstance(raw_betas, str):
        try:
            raw_betas = ast.literal_eval(raw_betas)
        except (ValueError, SyntaxError):
            raw_betas = None

    if isinstance(raw_betas, (tuple, list)) and len(raw_betas) == 2:
        return float(raw_betas[0]), float(raw_betas[1])

    beta1 = train_cfg.get("beta1", 0.9)
    beta2 = train_cfg.get("beta2", 0.95)
    return float(beta1), float(beta2)


class MudStrategy(TrainingStrategy):

    automatic_optimization = True

    def configure_optimizers(self, module: pl.LightningModule):
        train_cfg = self.config["training"]
        lr = self._resolve_lr(module)

        # The paper implementation uses one lr and one weight_decay for both
        # MUD and AdamW parameter groups
        weight_decay = train_cfg.get("weight_decay", 1e-2)
        beta_mud = train_cfg.get("beta_mud", train_cfg.get("mud_beta", 0.95))
        adamw_betas = _resolve_adamw_betas(train_cfg)
        mud_passes = train_cfg.get("mud_passes", 1)
        mud_eps = train_cfg.get("mud_eps", 1e-8)

        mud_params, adamw_params = split_params_for_mud(module.model)
        print(
            "MUD optimizer parameter split: "
            f"{len(mud_params)} matrix tensors -> MUD, "
            f"{len(adamw_params)} other tensors -> AdamW"
        )

        optimizer = MUDOptimizer(
            mud_params=mud_params,
            adamw_params=adamw_params,
            lr=lr,
            weight_decay=weight_decay,
            beta_mud=beta_mud,
            adamw_betas=adamw_betas,
            mud_passes=mud_passes,
            eps=mud_eps,
        )

        milestones = train_cfg.get("lr_milestones", None)
        gamma = train_cfg.get("lr_gamma", 0.5)

        if milestones is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=gamma,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def optimizer_zero_grad(
        self,
        module: pl.LightningModule,
        epoch: int,
        batch_idx: int,
        optimizer,
    ) -> None:
        optimizer.zero_grad(set_to_none=True)