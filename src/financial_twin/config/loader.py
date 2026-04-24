"""YAML loader -> validated Scenario."""

from __future__ import annotations

from pathlib import Path

import yaml

from .schema import Scenario


def load_yaml(path: str | Path) -> Scenario:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Scenario.model_validate(raw)


def load_dict(data: dict) -> Scenario:
    return Scenario.model_validate(data)
