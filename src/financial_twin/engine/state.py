"""SimState: the dense NumPy arrays that carry the simulation forward.

One Monte Carlo run = one slice along axis 0 of every array. The entire
hot loop mutates these arrays in place; Polars only enters at the end
when we shape results for analytics and the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimState:
    n_runs: int
    n_years: int
    n_accounts: int
    start_year: int

    balances: np.ndarray = field(init=False)  # [n_runs, n_years+1, n_accounts]
    # ``balances`` is mutable scratch space: runner mutates ``balances[:, t, :]``
    # in-place during apply_rmd / withdraw_for_need at year t, then step_accounts
    # writes ``balances[:, t+1, :]`` as end-of-year-t. That value is then mutated
    # again by year t+1's withdrawals. Use ``initial_balances`` and
    # ``eoy_balances`` for any post-run analytics that need clean, never-mutated
    # snapshots of starting and end-of-year balances.
    initial_balances: np.ndarray = field(init=False)  # [n_runs, n_accounts]
    eoy_balances: np.ndarray = field(init=False)  # [n_runs, n_years, n_accounts]
    contributions: np.ndarray = field(init=False)
    employer_match: np.ndarray = field(init=False)
    withdrawals: np.ndarray = field(init=False)
    growth: np.ndarray = field(init=False)

    gross_income: np.ndarray = field(init=False)  # [n_runs, n_years]
    taxable_income: np.ndarray = field(init=False)
    ltcg_income: np.ndarray = field(init=False)
    tax_paid: np.ndarray = field(init=False)
    tax_breakdown: dict[str, np.ndarray] = field(init=False)
    expenses: np.ndarray = field(init=False)
    pension_income: np.ndarray = field(init=False)
    ss_income: np.ndarray = field(init=False)

    returns_draw: np.ndarray = field(init=False)  # [n_runs, n_years, 2]
    cost_basis_brokerage: np.ndarray = field(init=False)  # [n_runs, n_years+1]

    deferred_state_credit: np.ndarray = field(init=False)  # [n_runs, n_years+1]

    # Property tax tracking
    assessed_value: np.ndarray = field(init=False)  # [n_years+1]

    def __post_init__(self) -> None:
        nr, ny, na = self.n_runs, self.n_years, self.n_accounts
        self.balances = np.zeros((nr, ny + 1, na), dtype=np.float64)
        self.initial_balances = np.zeros((nr, na), dtype=np.float64)
        self.eoy_balances = np.zeros((nr, ny, na), dtype=np.float64)
        self.contributions = np.zeros((nr, ny, na), dtype=np.float64)
        self.employer_match = np.zeros((nr, ny, na), dtype=np.float64)
        self.withdrawals = np.zeros((nr, ny, na), dtype=np.float64)
        self.growth = np.zeros((nr, ny, na), dtype=np.float64)
        self.gross_income = np.zeros((nr, ny), dtype=np.float64)
        self.taxable_income = np.zeros((nr, ny), dtype=np.float64)
        self.ltcg_income = np.zeros((nr, ny), dtype=np.float64)
        self.tax_paid = np.zeros((nr, ny), dtype=np.float64)
        self.tax_breakdown = {
            "federal": np.zeros((nr, ny), dtype=np.float64),
            "oregon": np.zeros((nr, ny), dtype=np.float64),
            "metro_shs": np.zeros((nr, ny), dtype=np.float64),
            "property": np.zeros((nr, ny), dtype=np.float64),
            "payroll": np.zeros((nr, ny), dtype=np.float64),
        }
        self.expenses = np.zeros((nr, ny), dtype=np.float64)
        self.pension_income = np.zeros((nr, ny), dtype=np.float64)
        self.ss_income = np.zeros((nr, ny), dtype=np.float64)
        self.returns_draw = np.zeros((nr, ny, 2), dtype=np.float64)
        self.cost_basis_brokerage = np.zeros((nr, ny + 1), dtype=np.float64)
        self.deferred_state_credit = np.zeros((nr, ny + 1), dtype=np.float64)
        self.assessed_value = np.zeros(ny + 1, dtype=np.float64)
