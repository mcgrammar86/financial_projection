"""Mortgage amortization schedule and the T3 lifestyle-split rule.

When the mortgage hits ``payoff_year``:
- ``standard_expenses`` permanently increases by 50% of the former
  annual P&I (locked in nominal dollars, never re-inflated).
- The other 50% is automatically contributed to the named Brokerage
  account.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config.schema import MortgageSpec


@dataclass
class MortgageSchedule:
    annual_p_and_i: np.ndarray  # [n_years] nominal $ paid in P&I that year
    payoff_year: int
    annual_p_and_i_at_payoff: float  # full-year baseline used for split


def amortize(
    mortgage: MortgageSpec | None, *, start_year: int, n_years: int
) -> MortgageSchedule | None:
    if mortgage is None:
        return None
    balance = mortgage.principal_remaining_2026
    monthly_rate = mortgage.annual_rate / 12.0
    monthly_payment = mortgage.monthly_payment
    annual = np.zeros(n_years, dtype=np.float64)
    full_year_p_and_i = monthly_payment * 12.0
    for t in range(n_years):
        sim_year = start_year + t
        if sim_year >= mortgage.payoff_year or balance <= 0.0:
            balance = 0.0
            annual[t] = 0.0
            continue
        year_paid = 0.0
        for _ in range(12):
            interest = balance * monthly_rate
            principal = monthly_payment - interest
            if principal >= balance:
                year_paid += balance + interest
                balance = 0.0
                break
            balance -= principal
            year_paid += monthly_payment
        annual[t] = year_paid
    return MortgageSchedule(
        annual_p_and_i=annual,
        payoff_year=mortgage.payoff_year,
        annual_p_and_i_at_payoff=full_year_p_and_i,
    )
