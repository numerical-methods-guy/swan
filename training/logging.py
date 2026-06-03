"""Logger construction for SWAN training."""

from __future__ import annotations

from typing import Any

from pytorch_lightning.loggers import TensorBoardLogger


def build_loggers(config: dict, run_name: str | None = None) -> list[Any]:
    """Build Lightning loggers from config.

    TensorBoard stays enabled by default. MLflow is optional so cluster runs do
    not depend on it unless explicitly requested in the config.
    """
    experiment_name = config["experiment"]["name"]
    train_cfg = config["training"]
    logging_cfg = config.get("logging", {})

    loggers: list[Any] = []

    if logging_cfg.get("tensorboard", True):
        loggers.append(
            TensorBoardLogger(
                train_cfg["save_dir"],
                name=run_name or experiment_name,
            )
        )

    mlflow_cfg = logging_cfg.get("mlflow", {})
    if mlflow_cfg.get("enabled", False):
        try:
            from pytorch_lightning.loggers import MLFlowLogger
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MLflow logging is enabled, but mlflow is not installed. "
                "Install it with `pip install mlflow` or disable logging.mlflow.enabled."
            ) from exc

        loggers.append(
            MLFlowLogger(
                experiment_name=mlflow_cfg.get("experiment_name", experiment_name),
                run_name=run_name or mlflow_cfg.get("run_name"),
                tracking_uri=mlflow_cfg.get("tracking_uri", "./mlruns"),
            )
        )

    if not loggers:
        return []

    return loggers
