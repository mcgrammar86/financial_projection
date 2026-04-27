"""HealthcareSpec out-of-pocket line item applies every year, working or retired."""

from __future__ import annotations

import pytest

from financial_twin.config.loader import load_dict
from financial_twin.engine.cashflow import healthcare_for_year


def _scenario(oop_2026: float, oop_growth: float = 0.05) -> dict:
    return {
        "people": [{
            "name": "alex",
            "birth_year": 1985,
            "retirement_year": 2050,
            "salary_2026": 100_000.0,
        }],
        "accounts": [{
            "name": "k",
            "behavior": "stochastic_market",
            "balance_2026": 1000.0,
            "tax_type": "pre_tax",
            "contribution_limit_2026": 0.0,
            "stocks_weight": 0.0,
            "owner": "alex",
        }],
        "expenses": {"standard_annual_2026": 0.0},
        "healthcare": {
            "out_of_pocket_annual_2026": oop_2026,
            "out_of_pocket_growth": oop_growth,
        },
        "simulation": {"n_runs": 1, "start_year": 2026, "end_year": 2030},
        "tax": {"inflation_rate": 0.025},
    }


def test_oop_in_2026_matches_config():
    s = load_dict(_scenario(4_000.0))
    assert healthcare_for_year(s, 2026) == pytest.approx(4_000.0)


def test_oop_inflates_at_healthcare_rate_not_cpi():
    s = load_dict(_scenario(4_000.0, oop_growth=0.06))
    # 2028 = 2 years out at 6% -> 4000 * 1.06^2 = 4494.40
    assert healthcare_for_year(s, 2028) == pytest.approx(4_000.0 * 1.06**2)


def test_oop_applies_during_working_years():
    """Bridge/Medicare are 0 pre-retirement, but OOP still applies."""
    s = load_dict(_scenario(3_500.0, oop_growth=0.0))
    # 2026 is working: no bridge, no medicare, only OOP.
    assert healthcare_for_year(s, 2026) == pytest.approx(3_500.0)


def test_oop_zero_when_household_empty():
    """If everyone has died, no OOP."""
    s = load_dict({
        "people": [{
            "name": "alex", "birth_year": 1950, "retirement_year": 2026,
            "death_year": 2027,
        }],
        "accounts": [{
            "name": "k", "behavior": "stochastic_market", "balance_2026": 1000.0,
            "tax_type": "pre_tax", "contribution_limit_2026": 0.0,
            "stocks_weight": 0.0, "owner": "alex",
        }],
        "expenses": {"standard_annual_2026": 0.0},
        "healthcare": {"out_of_pocket_annual_2026": 5_000.0, "out_of_pocket_growth": 0.0},
        "simulation": {"n_runs": 1, "start_year": 2026, "end_year": 2030},
    })
    assert healthcare_for_year(s, 2026) > 0  # alive
    assert healthcare_for_year(s, 2030) == 0  # dead, no OOP
