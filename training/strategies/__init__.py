"""Lightning training strategies (optimizer + step behavior)."""

from training.strategies.adam import AdamStrategy
from training.strategies.adamW import AdamWStrategy
from training.strategies.base import TrainingStrategy
from training.strategies.muon import MuonStrategy
from training.strategies.sgd import SGDStrategy


REGISTRY: dict[str, type[TrainingStrategy]] = {
    "adam": AdamStrategy,
    "adamw": AdamWStrategy,
    "muon": MuonStrategy,
    "sgd": SGDStrategy,
}


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
