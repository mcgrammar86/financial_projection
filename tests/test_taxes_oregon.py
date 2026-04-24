"""Sanity checks for the Oregon tax module."""

from __future__ import annotations

import numpy as np

from financial_twin.tax.oregon import compute_oregon_tax


def test_zero_income_zero_tax():
    out = compute_oregon_tax(
        np.array([0.0]), inflation_factor=1.0, deferred_credit=np.array([0.0])
    )
    assert float(out[0]) == 0.0


def test_529_credit_reduces_liability():
    income = np.array([100_000.0])
    no_credit = compute_oregon_tax(
        income, inflation_factor=1.0, deferred_credit=np.array([0.0])
    )
    with_credit = compute_oregon_tax(
        income, inflation_factor=1.0, deferred_credit=np.array([360.0])
    )
    assert float(no_credit[0]) - float(with_credit[0]) == 360.0


def test_credit_clamped_at_zero():
    out = compute_oregon_tax(
        np.array([0.0]), inflation_factor=1.0, deferred_credit=np.array([1000.0])
    )
    assert float(out[0]) == 0.0
