"""Oregon state income tax (2026 MFJ progressive brackets, up to 9.9%).

LTCG is taxed as ordinary income at the state level (per spec).
The 529 contribution credit ($360 max for joint filers) reduces the
*following* year's state liability — the runner manages that deferral.
"""

from __future__ import annotations

import numpy as np


# 2026 Oregon MFJ ordinary brackets.
OR_MFJ_2026_BRACKETS = np.array([0.0, 7_700.0, 19_300.0, 250_000.0], dtype=np.float64)
OR_MFJ_2026_RATES = np.array([0.0475, 0.0675, 0.0875, 0.099], dtype=np.float64)


def _piecewise(income: np.ndarray, brackets: np.ndarray, rates: np.ndarray) -> np.ndarray:
    income = np.maximum(income, 0.0)
    n = len(brackets)
    upper = np.concatenate([brackets[1:], np.array([np.inf])])
    tax = np.zeros_like(income)
    for i in range(n):
        lo = brackets[i]
        hi = upper[i]
        portion = np.clip(income - lo, 0.0, hi - lo)
        tax += portion * rates[i]
    return tax


def compute_oregon_tax(
    ordinary_plus_ltcg: np.ndarray,
    *,
    inflation_factor: float,
    deferred_credit: np.ndarray,
) -> np.ndarray:
    """Return Oregon income tax per run after applying any 529 credit carried in."""
    brackets = OR_MFJ_2026_BRACKETS * inflation_factor
    gross = _piecewise(ordinary_plus_ltcg, brackets, OR_MFJ_2026_RATES)
    return np.maximum(0.0, gross - deferred_credit)
