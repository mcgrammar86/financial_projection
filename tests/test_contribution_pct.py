"""Percent-of-salary contributions: employee `contribution_pct` and
non-elective `employer_contribution_pct`.

These let the user write "6% of my pay goes to my 401k" and "my employer
puts an additional 6% in regardless" without computing dollar amounts by
hand.
"""

from __future__ import annotations

import pytest

from financial_twin.config.loader import load_dict
from financial_twin.engine.cashflow import planned_contributions


def _scenario(
    *,
    contribution_2026: float = 0.0,
    contribution_pct: float = 0.0,
    employer_contribution_pct: float = 0.0,
    employer_match_pct: float = 0.0,
    employer_match_max_pct: float = 0.0,
    salary_2026: float = 100_000.0,
    limit: float = 24_000.0,
) -> dict:
    return {
        "people": [{
            "name": "alex",
            "birth_year": 1985,
            "retirement_year": 2050,
            "salary_2026": salary_2026,
            "salary_growth": 0.0,
        }],
        "accounts": [{
            "name": "k401",
            "behavior": "stochastic_market",
            "balance_2026": 0.0,
            "tax_type": "pre_tax",
            "contribution_limit_2026": limit,
            "contribution_2026": contribution_2026,
            "contribution_pct": contribution_pct,
            "employer_contribution_pct": employer_contribution_pct,
            "employer_match_pct": employer_match_pct,
            "employer_match_max_pct": employer_match_max_pct,
            "stocks_weight": 0.0,
            "owner": "alex",
        }],
        "expenses": {"standard_annual_2026": 0.0},
        "simulation": {"n_runs": 1, "start_year": 2026, "end_year": 2030},
        "tax": {"inflation_rate": 0.0},
    }


def test_employee_contribution_pct_uses_salary():
    """6% of $100k = $6k employee contribution in 2026."""
    s = load_dict(_scenario(contribution_pct=0.06))
    employee, employer = planned_contributions(s, 2026)["k401"]
    assert employee == pytest.approx(6_000.0)
    assert employer == 0.0


def test_employee_contribution_pct_tracks_salary_growth():
    """5% of salary, with salary inflating, scales with the salary."""
    data = _scenario(contribution_pct=0.05)
    data["people"][0]["salary_growth"] = 0.04
    s = load_dict(data)
    # 2028: salary = 100k * 1.04^2 = 108_160; 5% = 5_408
    employee, _ = planned_contributions(s, 2028)["k401"]
    assert employee == pytest.approx(100_000.0 * 1.04**2 * 0.05)


def test_employee_pct_and_dollar_are_additive():
    """Setting both contribution_2026 and contribution_pct sums them."""
    s = load_dict(_scenario(contribution_2026=2_000.0, contribution_pct=0.05))
    employee, _ = planned_contributions(s, 2026)["k401"]
    # 2_000 (flat) + 5_000 (5% of 100k) = 7_000
    assert employee == pytest.approx(7_000.0)


def test_employee_combined_capped_at_irs_limit():
    """If pct + flat dollars exceed the cap, planned is clipped to the cap."""
    # 30% of 100k + 5k flat = 35k requested, cap = 24k -> planned = 24k.
    s = load_dict(_scenario(
        contribution_2026=5_000.0, contribution_pct=0.30, limit=24_000.0,
    ))
    employee, _ = planned_contributions(s, 2026)["k401"]
    assert employee == pytest.approx(24_000.0)


def test_employer_contribution_pct_is_non_elective():
    """Employer puts in 6% of salary even with $0 employee contribution."""
    s = load_dict(_scenario(employer_contribution_pct=0.06))
    employee, employer = planned_contributions(s, 2026)["k401"]
    assert employee == 0.0
    assert employer == pytest.approx(6_000.0)


def test_employer_contribution_pct_adds_to_match():
    """Non-elective % stacks on top of regular matching dollars."""
    # Employee puts in 6% (-> 6k); match is dollar-for-dollar up to 5%
    # of salary (-> 5k); plus a non-elective 3% (-> 3k). Total employer = 8k.
    s = load_dict(_scenario(
        contribution_pct=0.06,
        employer_match_pct=0.05,
        employer_match_max_pct=0.05,
        employer_contribution_pct=0.03,
    ))
    employee, employer = planned_contributions(s, 2026)["k401"]
    assert employee == pytest.approx(6_000.0)
    assert employer == pytest.approx(5_000.0 + 3_000.0)


def test_employer_contribution_pct_zero_after_retirement():
    """Both employee and employer contributions stop at retirement."""
    data = _scenario(
        contribution_pct=0.06, employer_contribution_pct=0.05,
    )
    data["people"][0]["retirement_year"] = 2027
    s = load_dict(data)
    # Working in 2026: both fire.
    e26, m26 = planned_contributions(s, 2026)["k401"]
    assert e26 > 0 and m26 > 0
    # Retired by 2027: both zero.
    e27, m27 = planned_contributions(s, 2027)["k401"]
    assert e27 == 0.0 and m27 == 0.0


def test_vehicle_group_validator_includes_pct():
    """Combined pct-based contributions in a vehicle_group still hit the cap."""
    data = {
        "people": [{
            "name": "alex", "birth_year": 1985, "retirement_year": 2050,
            "salary_2026": 100_000.0,
        }],
        "accounts": [
            {
                "name": "k_traditional", "behavior": "stochastic_market",
                "balance_2026": 0.0, "tax_type": "pre_tax",
                "contribution_limit_2026": 24_000.0,
                "contribution_pct": 0.20,  # 20% of 100k = $20k
                "vehicle_group": "k", "owner": "alex",
            },
            {
                "name": "k_roth", "behavior": "stochastic_market",
                "balance_2026": 0.0, "tax_type": "roth",
                "contribution_limit_2026": 24_000.0,
                "contribution_pct": 0.10,  # 10% of 100k = $10k. Combined $30k > $24k.
                "vehicle_group": "k", "owner": "alex",
            },
        ],
        "expenses": {"standard_annual_2026": 0.0},
        "simulation": {"n_runs": 1, "start_year": 2026, "end_year": 2030},
    }
    with pytest.raises(ValueError, match="exceeds shared IRS limit"):
        load_dict(data)
