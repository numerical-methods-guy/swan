"""Gauss-Newton update helpers for SWAN.

This file contains the numerical update logic only. It is kept separate from
the Lightning strategy so the math is not mixed with logging, checkpointing,
or trainer-specific code.

Two variants are available:

* ``MatrixFreeGaussNewton`` avoids forming the full Jacobian. It uses JVP/VJP
  products and conjugate gradient, so it is the practical default for PARADIS.
* ``ExplicitGaussNewton`` builds the Jacobian directly. This is easier to
  reason about, but is only meant for tiny debugging runs because memory grows
  quickly with model size.

Both variants solve a damped least-squares system based on the residual
``prediction - target`` and return the same diagnostics through
``GaussNewtonStats``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.func import functional_call


@dataclass
class GaussNewtonStats:
    """Diagnostics for one Gauss-Newton step."""

    loss: torch.Tensor
    residual_norm: torch.Tensor
    gradient_norm: torch.Tensor
    step_norm: torch.Tensor
    cg_iterations: int
    cg_residual_norm: torch.Tensor


class BaseGaussNewton:
    """Shared utilities for Gauss-Newton variants."""

    def __init__(
        self,
        damping: float = 1.0e-3,
        step_size: float = 1.0,
        max_step_norm: float | None = None,
    ):
        if damping < 0:
            raise ValueError("damping must be non-negative")
        if step_size <= 0:
            raise ValueError("step_size must be positive")

        self.damping = damping
        self.step_size = step_size
        self.max_step_norm = max_step_norm

    def step(
        self,
        model: nn.Module,
        fields: torch.Tensor,
        winds: torch.Tensor,
        target: torch.Tensor,
        nfuture: int = 0,
    ) -> GaussNewtonStats:
        raise NotImplementedError

    def _prepare_functional_state(
        self,
        model: nn.Module,
        fields: torch.Tensor,
    ) -> tuple[
        list[torch.Tensor],
        torch.Tensor,
        list[str],
        list[torch.Size],
        list[int],
        dict[str, torch.Tensor],
    ]:
        trainable = {
            name: param
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        if not trainable:
            raise ValueError("model has no trainable parameters")

        names = list(trainable.keys())
        params = [trainable[name] for name in names]
        shapes = [param.shape for param in params]
        numels = [param.numel() for param in params]
        buffers = dict(model.named_buffers())

        flat_params = torch.cat([param.detach().reshape(-1) for param in params])
        flat_params = flat_params.to(device=fields.device).requires_grad_(True)

        return params, flat_params, names, shapes, numels, buffers

    @staticmethod
    def _unflatten(
        vector: torch.Tensor,
        names: list[str],
        shapes: list[torch.Size],
        numels: list[int],
    ) -> dict[str, torch.Tensor]:
        pieces = torch.split(vector, numels)
        return {
            name: piece.reshape(shape)
            for name, piece, shape in zip(names, pieces, shapes)
        }

    def _make_residual_fn(
        self,
        model: nn.Module,
        fields: torch.Tensor,
        winds: torch.Tensor,
        target: torch.Tensor,
        nfuture: int,
        names: list[str],
        shapes: list[torch.Size],
        numels: list[int],
        buffers: dict[str, torch.Tensor],
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def residual_fn(vector: torch.Tensor) -> torch.Tensor:
            state = {**buffers, **self._unflatten(vector, names, shapes, numels)}
            prediction = functional_call(model, state, (fields, winds))
            for _ in range(nfuture):
                prediction = functional_call(model, state, (prediction, winds))
            residual = prediction - target
            return residual.reshape(-1) / residual.numel() ** 0.5

        return residual_fn

    def _finalize_step(
        self,
        params: list[torch.Tensor],
        flat_params: torch.Tensor,
        step: torch.Tensor,
    ) -> torch.Tensor:
        if self.max_step_norm is not None:
            step_norm = torch.linalg.vector_norm(step)
            if step_norm > self.max_step_norm:
                step = step * (self.max_step_norm / (step_norm + 1.0e-12))

        update = self.step_size * step
        self._copy_flat_params(params, flat_params.detach() + update)
        return update

    @staticmethod
    def _copy_flat_params(
        params: list[torch.Tensor],
        flat_params: torch.Tensor,
    ) -> None:
        offset = 0
        with torch.no_grad():
            for param in params:
                numel = param.numel()
                param.copy_(flat_params[offset : offset + numel].view_as(param))
                offset += numel


class MatrixFreeGaussNewton(BaseGaussNewton):
    """Gauss-Newton using CG and Jacobian-vector products."""

    def __init__(
        self,
        damping: float = 1.0e-3,
        cg_iters: int = 10,
        cg_tol: float = 1.0e-6,
        step_size: float = 1.0,
        max_step_norm: float | None = None,
    ):
        super().__init__(
            damping=damping,
            step_size=step_size,
            max_step_norm=max_step_norm,
        )
        if cg_iters < 1:
            raise ValueError("cg_iters must be >= 1")

        self.cg_iters = cg_iters
        self.cg_tol = cg_tol

    def step(
        self,
        model: nn.Module,
        fields: torch.Tensor,
        winds: torch.Tensor,
        target: torch.Tensor,
        nfuture: int = 0,
    ) -> GaussNewtonStats:
        params, flat_params, names, shapes, numels, buffers = (
            self._prepare_functional_state(model, fields)
        )
        residual_fn = self._make_residual_fn(
            model, fields, winds, target, nfuture, names, shapes, numels, buffers
        )

        residual = residual_fn(flat_params)
        loss = 0.5 * residual.dot(residual)
        gradient = torch.autograd.grad(loss, flat_params)[0].detach()
        rhs = -gradient

        def normal_matvec(vector: torch.Tensor) -> torch.Tensor:
            vector = vector.detach()
            _, jv = torch.autograd.functional.jvp(
                residual_fn,
                flat_params.detach(),
                vector,
                create_graph=False,
                strict=False,
            )
            _, jt_jv = torch.autograd.functional.vjp(
                residual_fn,
                flat_params.detach(),
                v=jv.detach(),
                create_graph=False,
                strict=False,
            )
            return jt_jv.detach() + self.damping * vector

        step, cg_iterations, cg_residual = self._conjugate_gradient(
            normal_matvec,
            rhs,
        )
        update = self._finalize_step(params, flat_params, step)

        return GaussNewtonStats(
            loss=(2.0 * loss).detach(),
            residual_norm=torch.linalg.vector_norm(residual.detach()),
            gradient_norm=torch.linalg.vector_norm(gradient),
            step_norm=torch.linalg.vector_norm(update.detach()),
            cg_iterations=cg_iterations,
            cg_residual_norm=torch.linalg.vector_norm(cg_residual.detach()),
        )

    def _conjugate_gradient(
        self,
        matvec: Callable[[torch.Tensor], torch.Tensor],
        rhs: torch.Tensor,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        x = torch.zeros_like(rhs)
        residual = rhs.clone()
        direction = residual.clone()
        residual_dot = residual.dot(residual)
        tolerance = self.cg_tol * max(float(torch.linalg.vector_norm(rhs)), 1.0)

        iterations = 0
        for iterations in range(1, self.cg_iters + 1):
            matvec_direction = matvec(direction)
            denom = direction.dot(matvec_direction).clamp_min(1.0e-30)
            alpha = residual_dot / denom
            x = x + alpha * direction
            residual = residual - alpha * matvec_direction

            if torch.linalg.vector_norm(residual) <= tolerance:
                break

            next_residual_dot = residual.dot(residual)
            beta = next_residual_dot / residual_dot.clamp_min(1.0e-30)
            direction = residual + beta * direction
            residual_dot = next_residual_dot

        return x, iterations, residual


class ExplicitGaussNewton(BaseGaussNewton):
    """Gauss-Newton using an explicitly materialized Jacobian.

    This method is intended for tiny debugging runs because the Jacobian is
    large for neural networks.
    """

    def step(
        self,
        model: nn.Module,
        fields: torch.Tensor,
        winds: torch.Tensor,
        target: torch.Tensor,
        nfuture: int = 0,
    ) -> GaussNewtonStats:
        params, flat_params, names, shapes, numels, buffers = (
            self._prepare_functional_state(model, fields)
        )
        residual_fn = self._make_residual_fn(
            model, fields, winds, target, nfuture, names, shapes, numels, buffers
        )

        residual = residual_fn(flat_params)
        loss = 0.5 * residual.dot(residual)
        jacobian = torch.autograd.functional.jacobian(
            residual_fn,
            flat_params,
            create_graph=False,
            strict=False,
        )

        gradient = jacobian.T @ residual.detach()
        normal_matrix = jacobian.T @ jacobian
        if self.damping > 0:
            eye = torch.eye(
                normal_matrix.shape[0],
                device=normal_matrix.device,
                dtype=normal_matrix.dtype,
            )
            normal_matrix = normal_matrix + self.damping * eye

        rhs = -gradient
        try:
            step = torch.linalg.solve(normal_matrix, rhs)
        except RuntimeError:
            step = torch.linalg.lstsq(normal_matrix, rhs).solution

        update = self._finalize_step(params, flat_params, step)
        zero = torch.zeros((), device=fields.device, dtype=fields.dtype)

        return GaussNewtonStats(
            loss=(2.0 * loss).detach(),
            residual_norm=torch.linalg.vector_norm(residual.detach()),
            gradient_norm=torch.linalg.vector_norm(gradient.detach()),
            step_norm=torch.linalg.vector_norm(update.detach()),
            cg_iterations=0,
            cg_residual_norm=zero,
        )


def build_gauss_newton(config: dict) -> BaseGaussNewton:
    """Build a Gauss-Newton helper from config.

    Supported methods are ``matrix_free`` and ``explicit``. ``original`` is
    accepted as an alias for ``explicit``.
    """
    gn_cfg = config.get("training", {}).get("gauss_newton", config)
    method = gn_cfg.get("method", gn_cfg.get("variant", "matrix_free")).lower()

    common = {
        "damping": gn_cfg.get("damping", 1.0e-3),
        "step_size": gn_cfg.get("step_size", 1.0),
        "max_step_norm": gn_cfg.get("max_step_norm", None),
    }

    if method == "matrix_free":
        return MatrixFreeGaussNewton(
            **common,
            cg_iters=gn_cfg.get("cg_iters", 10),
            cg_tol=gn_cfg.get("cg_tol", 1.0e-6),
        )
    if method in {"explicit", "original"}:
        return ExplicitGaussNewton(**common)

    raise ValueError(
        f"Unknown Gauss-Newton method '{method}'. Choose 'matrix_free' or 'explicit'."
    )


MatrixFreeDampedGaussNewton = MatrixFreeGaussNewton
OriginalGaussNewton = ExplicitGaussNewton
