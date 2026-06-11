"""
Configuration loader for SecureMed-LLM.
Loads and merges YAML config with optional CLI overrides.
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Hierarchical configuration object with dot-access."""

    def __init__(self, d: Dict[str, Any]):
        for k, v in d.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __repr__(self):
        return str(self.__dict__)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """
    Load YAML config file and apply optional overrides.

    Args:
        config_path: Path to config.yaml
        overrides: Dict of dot-separated key→value overrides,
                   e.g. {"training.epochs": 10, "differential_privacy.epsilon": 2.0}

    Returns:
        Config object with attribute access.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    if overrides:
        for dotted_key, value in overrides.items():
            _set_nested(raw, dotted_key.split("."), value)

    return Config(raw)


def _set_nested(d: Dict, keys: list, value: Any) -> None:
    """Recursively set a nested dict key from a list of key parts."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    # attempt type coercion from string CLI args
    d[keys[-1]] = _coerce(value)


def _coerce(value: Any) -> Any:
    """Try to coerce string CLI values to bool/int/float."""
    if not isinstance(value, str):
        return value
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
