"""Target-date fund glide path on stocks_weight."""

from __future__ import annotations

import math

import numpy as np

from financial_twin.config.loader import load_dict
from financial_twin.simulate.runner import _glide_path, run_simulation


def _scenario(start_w: float, target_w: float, target_year: int | None = None):
    accounts = [
        {
            "name": "tdf_401k",
            "behavior": "stochastic_market",
            "balance_2026": 100_000.0,
            "tax_type": "pre_tax",
            "contribution_limit_2026": 24_000.0,
            "contribution_2026": 0.0,
            "stocks_weight": start_w,
            "stocks_weight_glide_target": target_w,
            "stocks_mean": 0.0,
            "stocks_stdev": 0.0,
            "bonds_mean": 0.0,
            "bonds_stdev": 0.0,
            "owner": "alex",
        }
    ]
    if target_year is not None:
        accounts[0]["stocks_weight_glide_target_year"] = target_year
    return load_dict({
        "people": [{
            "name": "alex",
            "birth_year": 1985,
            "retirement_year": 2046,
            "salary_2026": 100_000.0,
        }],
        "accounts": accounts,
        "expenses": {"standard_annual_2026": 0.0},
        "simulation": {"n_runs": 1, "start_year": 2026, "end_year": 2050},
        "tax": {"inflation_rate": 0.0},
    })


def test_glide_endpoints():
    scenario = _scenario(0.9, 0.4, target_year=2046)
    spec = scenario.accounts[0]
    weights = _glide_path(spec, scenario, n_years=24)
    # year 0 = start_year (2026)
    assert math.isclose(weights[0], 0.9, abs_tol=1e-9)
    # midpoint (10 years in) is halfway
    assert math.isclose(weights[10], 0.9 + 10 / 20 * (0.4 - 0.9), abs_tol=1e-9)
    # at and past target_year, hold at target
    assert math.isclose(weights[20], 0.4, abs_tol=1e-9)
    assert math.isclose(weights[23], 0.4, abs_tol=1e-9)


def test_glide_defaults_to_owner_retirement_year():
    scenario = _scenario(0.9, 0.4)  # no explicit target_year
    spec = scenario.accounts[0]
    weights = _glide_path(spec, scenario, n_years=24)
    # owner.retirement_year=2046; at year offset 20 (=2046) we're at target
    assert math.isclose(weights[20], 0.4, abs_tol=1e-9)


def test_no_glide_returns_none():
    scenario = load_dict({
        "people": [{
            "name": "alex",
            "birth_year": 1985,
            "retirement_year": 2046,
            "salary_2026": 100_000.0,
        }],
        "accounts": [{
            "name": "static_acct",
            "behavior": "stochastic_market",
            "balance_2026": 100_000.0,
            "tax_type": "pre_tax",
            "contribution_limit_2026": 24_000.0,
            "stocks_weight": 0.7,
            "owner": "alex",
        }],
        "expenses": {"standard_annual_2026": 0.0},
        "simulation": {"n_runs": 1, "start_year": 2026, "end_year": 2030},
    })
    assert _glide_path(scenario.accounts[0], scenario, n_years=4) is None


def test_runner_uses_glide_for_growth():
    """Deterministic check: set stocks_mean=0.10, bonds_mean=0.0, glide
    from 1.0 -> 0.0 over 4 years; final balance = 100k * prod(1 + r_t)
    where r_t = (1 - t/4) * 0.10."""
    scenario = load_dict({
        "people": [{
            "name": "alex",
            "birth_year": 1985,
            "retirement_year": 2030,
            "salary_2026": 100_000.0,
        }],
        "accounts": [{
            "name": "tdf",
            "behavior": "stochastic_market",
            "balance_2026": 100_000.0,
            "tax_type": "pre_tax",
            "contribution_limit_2026": 24_000.0,
            "contribution_2026": 0.0,
            "stocks_weight": 1.0,
            "stocks_weight_glide_target": 0.0,
            "stocks_weight_glide_target_year": 2030,
            "stocks_mean": 0.10,
            "stocks_stdev": 0.0,
            "bonds_mean": 0.0,
            "bonds_stdev": 0.0,
            "owner": "alex",
        }],
        "expenses": {"standard_annual_2026": 0.0},
        "simulation": {"n_runs": 1, "start_year": 2026, "end_year": 2030, "seed": 1},
        "tax": {"inflation_rate": 0.0},
    })
    results = run_simulation(scenario)
    idx = results.account_idx_map["tdf"]
    final = float(results.state.balances[0, 4, idx])
    expected = 100_000.0
    for t in range(4):
        weight = 1.0 - t / 4.0
        expected *= 1.0 + weight * 0.10
    assert math.isclose(final, expected, rel_tol=0, abs_tol=1e-6)
