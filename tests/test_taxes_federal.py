"""Sanity checks for the federal tax module."""

from __future__ import annotations

import numpy as np

from financial_twin.tax.federal import (
    MFJ_2026_BRACKETS,
    MFJ_2026_RATES,
    MFJ_2026_STANDARD_DEDUCTION,
    compute_federal_tax,
)


def test_zero_income_zero_tax():
    out = compute_federal_tax(np.array([0.0]), np.array([0.0]), inflation_factor=1.0)
    assert float(out[0]) == 0.0


def test_below_standard_deduction_zero_tax():
    out = compute_federal_tax(
        np.array([MFJ_2026_STANDARD_DEDUCTION - 1.0]),
        np.array([0.0]),
        inflation_factor=1.0,
    )
    assert float(out[0]) == 0.0


def test_first_bracket_only():
    # Income exactly at the std deduction + $1k -> taxed only at 10%
    income = MFJ_2026_STANDARD_DEDUCTION + 1_000.0
    out = compute_federal_tax(
        np.array([income]), np.array([0.0]), inflation_factor=1.0
    )
    assert abs(float(out[0]) - 100.0) < 1e-6


def test_bracket_progression_monotonic():
    incomes = np.linspace(0, 1_000_000, 25)
    taxes = compute_federal_tax(incomes, np.zeros_like(incomes), inflation_factor=1.0)
    deltas = np.diff(taxes)
    assert (deltas >= -1e-6).all()
