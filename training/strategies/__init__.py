"""Lightning training strategies (optimizer + step behavior)."""

from training.strategies.adam import AdamStrategy
from training.strategies.adamW import AdamWStrategy
from training.strategies.base import TrainingStrategy
from training.strategies.gauss_newton import GaussNewtonStrategy
from training.strategies.mud import MudStrategy
from training.strategies.mud_new import MudNewStrategy
from training.strategies.muon import MuonStrategy
from training.strategies.muon_new import MuonNewStrategy
from training.strategies.sgd import SGDStrategy

REGISTRY: dict[str, type[TrainingStrategy]] = {
    "adam": AdamStrategy,
    "adamw": AdamWStrategy,
    "gauss_newton": GaussNewtonStrategy,
    "mud": MudStrategy,
    "mud_new": MudNewStrategy,
    "muon": MuonStrategy,
    "muon_new": MuonNewStrategy,
    "sgd": SGDStrategy,
}


def available_optimizer_names() -> list[str]:
    """Return optimizer names supported by the strategy registry."""
    return sorted(REGISTRY)


def build_strategy(name: str, config: dict) -> TrainingStrategy:
    """Instantiate a training strategy by optimizer name."""
    key = name.lower()
    if key not in REGISTRY:
        allowed = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown optimizer '{name}'. Choose one of: {allowed}")
    return REGISTRY[key](config)


def resolve_optimizer_name(cli_value: str | None, config: dict) -> str:
    """Resolve optimizer: CLI flag > config > default 'adam'."""
    if cli_value is not None:
        return cli_value.lower()
    return config.get("training", {}).get("optimizer", "adam").lower()
