"""Vehicle-group shared-limit enforcement."""

from __future__ import annotations

import pytest

from financial_twin.config.loader import load_dict


def _scenario_dict(traditional_contrib: float, roth_contrib: float, *, limit: float = 24_000.0) -> dict:
    return {
        "people": [
            {
                "name": "alex",
                "birth_year": 1985,
                "retirement_year": 2050,
                "salary_2026": 100_000.0,
            }
        ],
        "accounts": [
            {
                "name": "k_traditional",
                "behavior": "stochastic_market",
                "balance_2026": 0.0,
                "tax_type": "pre_tax",
                "contribution_limit_2026": limit,
                "contribution_2026": traditional_contrib,
                "vehicle_group": "k",
                "owner": "alex",
            },
            {
                "name": "k_roth",
                "behavior": "stochastic_market",
                "balance_2026": 0.0,
                "tax_type": "roth",
                "contribution_limit_2026": limit,
                "contribution_2026": roth_contrib,
                "vehicle_group": "k",
                "owner": "alex",
            },
        ],
        "expenses": {"standard_annual_2026": 0.0},
        "simulation": {"n_runs": 1, "start_year": 2026, "end_year": 2030},
        "tax": {"inflation_rate": 0.0},
    }


def test_within_combined_limit_is_valid():
    scenario = load_dict(_scenario_dict(18_000.0, 6_000.0))
    assert scenario is not None  # validates


def test_combined_over_limit_raises():
    with pytest.raises(ValueError, match="exceeds shared IRS limit"):
        load_dict(_scenario_dict(20_000.0, 6_000.0))


def test_conflicting_limits_in_group_raises():
    data = _scenario_dict(10_000.0, 5_000.0)
    data["accounts"][1]["contribution_limit_2026"] = 30_000.0  # disagrees
    with pytest.raises(ValueError, match="conflicting"):
        load_dict(data)
