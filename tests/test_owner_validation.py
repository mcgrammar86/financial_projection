"""Account-owner validation: a typo in `owner:` should be a hard error,
not a silently zeroed contribution."""

from __future__ import annotations

import pytest

from financial_twin.config.loader import load_dict


def _scenario_dict(account_owner: str) -> dict:
    return {
        "people": [
            {
                "name": "Mike",
                "birth_year": 1986,
                "retirement_year": 2050,
                "salary_2026": 100_000.0,
            }
        ],
        "accounts": [
            {
                "name": "k",
                "behavior": "stochastic_market",
                "balance_2026": 100_000.0,
                "tax_type": "pre_tax",
                "contribution_limit_2026": 24_000.0,
                "contribution_2026": 18_000.0,
                "owner": account_owner,
            }
        ],
        "expenses": {"standard_annual_2026": 0.0},
        "simulation": {"n_runs": 1, "start_year": 2026, "end_year": 2030},
        "tax": {"inflation_rate": 0.0},
    }


def test_matching_owner_is_valid():
    scenario = load_dict(_scenario_dict("Mike"))
    assert scenario is not None


def test_unknown_owner_raises():
    with pytest.raises(ValueError, match="unknown owner"):
        load_dict(_scenario_dict("husband"))  # typo: person is "Mike"


def test_unknown_owner_on_pension_raises():
    data = {
        "people": [{"name": "Mike", "birth_year": 1986, "retirement_year": 2050}],
        "accounts": [
            {
                "name": "pension",
                "behavior": "defined_benefit",
                "owner": "wife",  # not in people
                "service_start_year": 2020,
                "service_end_year": 2050,
            }
        ],
        "expenses": {"standard_annual_2026": 0.0},
        "simulation": {"n_runs": 1, "start_year": 2026, "end_year": 2030},
    }
    with pytest.raises(ValueError, match="unknown owner"):
        load_dict(data)
