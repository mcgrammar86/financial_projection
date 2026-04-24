"""FICA payroll tax during accumulation (employee portion)."""

from __future__ import annotations


SS_WAGE_BASE_2026 = 176_100.0  # placeholder; index annually
SS_RATE = 0.062
MEDICARE_RATE = 0.0145


def compute_payroll(salary: float, inflation_factor: float) -> float:
    wage_base = SS_WAGE_BASE_2026 * inflation_factor
    ss = min(salary, wage_base) * SS_RATE
    medicare = salary * MEDICARE_RATE
    return ss + medicare
