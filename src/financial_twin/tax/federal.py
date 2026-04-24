"""Federal income tax (post-TCJA-sunset 2026 brackets, MFJ).

Brackets are inflation-indexed each year. The function is fully
vectorized across runs via ``np.digitize``.
"""

from __future__ import annotations

import numpy as np


# 2026 MFJ ordinary brackets after TCJA sunset (10/15/25/28/33/35/39.6).
# Bracket starts and rates are best-known projected values; precise CPI
# indexing is handled at call time.
MFJ_2026_BRACKETS = np.array(
    [0.0, 24_000.0, 97_400.0, 196_550.0, 299_550.0, 535_550.0, 604_950.0],
    dtype=np.float64,
)
MFJ_2026_RATES = np.array(
    [0.10, 0.15, 0.25, 0.28, 0.33, 0.35, 0.396], dtype=np.float64
)
MFJ_2026_STANDARD_DEDUCTION = 16_600.0  # ~half of TCJA std deduction post-sunset

# LTCG: 0/15/20 with MFJ 2026 thresholds (post-sunset adjusted).
MFJ_2026_LTCG_BRACKETS = np.array([0.0, 96_700.0, 600_050.0], dtype=np.float64)
MFJ_2026_LTCG_RATES = np.array([0.00, 0.15, 0.20], dtype=np.float64)


def _piecewise_tax(
    income: np.ndarray, brackets: np.ndarray, rates: np.ndarray
) -> np.ndarray:
    """Vectorized piecewise tax: sum of rate_i * width_i within bracket."""
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


def compute_federal_tax(
    ordinary_income: np.ndarray,
    ltcg_income: np.ndarray,
    *,
    inflation_factor: float,
) -> np.ndarray:
    """Return federal tax liability per run, shape [n_runs]."""
    brackets = MFJ_2026_BRACKETS * inflation_factor
    ltcg_brackets = MFJ_2026_LTCG_BRACKETS * inflation_factor
    std_ded = MFJ_2026_STANDARD_DEDUCTION * inflation_factor

    taxable_ordinary = np.maximum(0.0, ordinary_income - std_ded)
    ord_tax = _piecewise_tax(taxable_ordinary, brackets, MFJ_2026_RATES)

    # LTCG stacks on top of ordinary in the LTCG-bracket lookup.
    stacked_low = taxable_ordinary
    stacked_high = taxable_ordinary + np.maximum(0.0, ltcg_income)
    ltcg_tax = _ltcg_tax(stacked_low, stacked_high, ltcg_brackets, MFJ_2026_LTCG_RATES)

    return ord_tax + ltcg_tax


def _ltcg_tax(
    stacked_low: np.ndarray,
    stacked_high: np.ndarray,
    brackets: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    n = len(brackets)
    upper = np.concatenate([brackets[1:], np.array([np.inf])])
    tax = np.zeros_like(stacked_low)
    for i in range(n):
        lo = brackets[i]
        hi = upper[i]
        portion = np.clip(stacked_high, lo, hi) - np.clip(stacked_low, lo, hi)
        tax += np.maximum(portion, 0.0) * rates[i]
    return tax
