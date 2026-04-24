"""Deterministic inflation indexer.

A single configurable rate (default 2.5%) is compounded annually and
applied to brackets, the SHS threshold, IRS contribution limits, and
nominal expense lines.
"""

from __future__ import annotations

import numpy as np


def index_factor(years_from_base: int, inflation_rate: float) -> float:
    return float((1.0 + inflation_rate) ** years_from_base)


def index_series(start_year: int, end_year: int, base_year: int, inflation_rate: float) -> np.ndarray:
    """Return cumulative inflation factors for years [start_year, end_year)."""
    n = end_year - start_year
    offsets = np.arange(start_year - base_year, start_year - base_year + n, dtype=np.float64)
    return (1.0 + inflation_rate) ** offsets
