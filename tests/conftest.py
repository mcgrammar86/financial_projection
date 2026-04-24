"""Shared pytest fixtures.

The four required tests use ``n_runs=1`` and a single deterministic seed
so the stochastic engine collapses to a deterministic check, and the
asserts can be exact-to-the-cent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from financial_twin.config.loader import load_dict
from financial_twin.config.schema import Scenario


@pytest.fixture
def base_scenario_dict() -> dict:
    """Minimal scenario covering one earner with one TCDRS account.

    Tests override fields as needed.
    """
    return {
        "people": [
            {
                "name": "primary",
                "birth_year": 1985,
                "retirement_year": 2050,
                "salary_2026": 100_000.0,
                "fte_salary_2026": 100_000.0,
                "salary_growth": 0.0,
                "pia_monthly": 0.0,
                "ss_claim_age": 67,
            },
        ],
        "accounts": [],
        "expenses": {"standard_annual_2026": 0.0, "discretionary_annual_2026": 0.0},
        "simulation": {
            "n_runs": 1,
            "start_year": 2026,
            "end_year": 2031,
            "horizon_to_age": 95,
            "seed": 42,
        },
        "tax": {"inflation_rate": 0.0},
    }


@pytest.fixture
def make_scenario(base_scenario_dict):
    def _make(**overrides) -> Scenario:
        data = {**base_scenario_dict, **overrides}
        return load_dict(data)

    return _make
