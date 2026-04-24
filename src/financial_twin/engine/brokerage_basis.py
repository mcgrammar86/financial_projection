"""Average-basis tracking for the taxable Brokerage account.

A withdrawal from Brokerage is split into return-of-basis (untaxed) and
LTCG (taxed at federal LTCG + Oregon ordinary). The basis fraction is
basis / pre-withdrawal balance.
"""

from __future__ import annotations

import numpy as np


def split_withdrawal(
    balance_before: np.ndarray, basis: np.ndarray, withdrawal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (basis_consumed, ltcg_realized) — both shape [n_runs]."""
    safe_balance = np.where(balance_before > 0, balance_before, 1.0)
    basis_fraction = np.clip(basis / safe_balance, 0.0, 1.0)
    basis_consumed = np.minimum(basis, withdrawal * basis_fraction)
    ltcg_realized = np.maximum(0.0, withdrawal - basis_consumed)
    return basis_consumed, ltcg_realized
