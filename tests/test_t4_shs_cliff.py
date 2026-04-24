"""T4 — Metro SHS Cliff.

At joint taxable income $200k (below threshold), SHS = $0. At $210k,
SHS = $50 (= 1% × $5k over the 2026 threshold of $205k). The delta must
be exactly $50 to the cent.
"""

from __future__ import annotations

import math

import numpy as np

from financial_twin.tax.metro_shs import compute_metro_shs


def test_shs_cliff_exact_50():
    income_low = np.array([200_000.0])
    income_high = np.array([210_000.0])
    low = compute_metro_shs(
        income_low, threshold_2026=205_000.0, rate=0.01, inflation_factor=1.0
    )
    high = compute_metro_shs(
        income_high, threshold_2026=205_000.0, rate=0.01, inflation_factor=1.0
    )
    assert float(low[0]) == 0.0
    assert math.isclose(float(high[0]), 50.0, abs_tol=1e-9)
    assert math.isclose(float(high[0] - low[0]), 50.0, abs_tol=1e-9)
