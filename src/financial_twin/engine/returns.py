"""Cholesky-correlated stocks/bonds return draws.

For per-account asset allocation we draw a single (stocks, bonds) pair
per (run, year) and let each account combine them with its own weights.
That keeps draw count small and ensures all stochastic accounts share a
coherent view of the year's market.
"""

from __future__ import annotations

import numpy as np


def draw_returns(
    rng: np.random.Generator,
    n_runs: int,
    n_years: int,
    stocks_mean: float,
    stocks_stdev: float,
    bonds_mean: float,
    bonds_stdev: float,
    correlation: float,
) -> np.ndarray:
    """Return shape [n_runs, n_years, 2] where [..., 0] = stocks, [..., 1] = bonds."""
    if stocks_stdev <= 0.0 and bonds_stdev <= 0.0:
        # Degenerate (deterministic) case: no random component at all.
        out = np.zeros((n_runs, n_years, 2), dtype=np.float64)
        out[..., 0] = stocks_mean
        out[..., 1] = bonds_mean
        return out
    cov = np.array(
        [
            [stocks_stdev**2, correlation * stocks_stdev * bonds_stdev],
            [correlation * stocks_stdev * bonds_stdev, bonds_stdev**2],
        ],
        dtype=np.float64,
    )
    # Cholesky requires positive-definite; nudge with tiny diagonal jitter
    # if either stdev is zero (so one axis has no variance).
    if stocks_stdev <= 0.0:
        cov[0, 0] = 1e-24
    if bonds_stdev <= 0.0:
        cov[1, 1] = 1e-24
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal(size=(n_runs, n_years, 2))
    correlated = z @ L.T
    correlated[..., 0] += stocks_mean
    correlated[..., 1] += bonds_mean
    return correlated


def blend_account_return(
    market_draw: np.ndarray,  # [n_runs] (single year, 2 assets)
    stocks_weight: float,
) -> np.ndarray:
    """Blend the market draw using account-specific stocks weight."""
    return stocks_weight * market_draw[..., 0] + (1.0 - stocks_weight) * market_draw[..., 1]
