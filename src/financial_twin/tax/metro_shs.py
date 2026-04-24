"""Metro Supportive Housing Services (SHS) tax.

Flat 1% on joint taxable income above the inflation-indexed $205,000
threshold (2026 base). Used by T4: at $200k → $0; at $210k → $50.
"""

from __future__ import annotations

import numpy as np


def compute_metro_shs(
    taxable_income: np.ndarray, *, threshold_2026: float, rate: float, inflation_factor: float
) -> np.ndarray:
    threshold = threshold_2026 * inflation_factor
    return rate * np.maximum(0.0, taxable_income - threshold)
