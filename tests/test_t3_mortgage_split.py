"""T3 — Mortgage Lifestyle Split.

After ``payoff_year``, expenses must rise by exactly 50% of the former
annual P&I, and the other 50% must be auto-routed to the named
Brokerage account as an additional contribution.
"""

from __future__ import annotations

import math

from financial_twin.simulate.runner import run_simulation


def test_lifestyle_split_to_the_cent(make_scenario):
    monthly_payment = 2_000.0
    annual_pi = monthly_payment * 12.0
    payoff_year = 2030

    scenario = make_scenario(
        accounts=[
            {
                "name": "brokerage",
                "behavior": "stochastic_market",
                "balance_2026": 0.0,
                "tax_type": "taxable",
                "contribution_limit_2026": 1e9,
                "contribution_2026": 0.0,
                "stocks_weight": 0.0,
                "stocks_mean": 0.0,
                "stocks_stdev": 0.0,
                "bonds_mean": 0.0,
                "bonds_stdev": 0.0,
                "owner": "primary",
            }
        ],
        mortgage={
            "principal_remaining_2026": 200_000.0,
            "annual_rate": 0.0,
            "monthly_payment": monthly_payment,
            "payoff_year": payoff_year,
            "lifestyle_split_brokerage_account": "brokerage",
        },
        expenses={"standard_annual_2026": 50_000.0, "discretionary_annual_2026": 0.0},
        simulation={
            "n_runs": 1,
            "start_year": 2026,
            "end_year": 2032,
            "horizon_to_age": 95,
            "seed": 42,
        },
    )
    results = run_simulation(scenario)
    state = results.state
    payoff_idx = payoff_year - scenario.simulation.start_year

    # Expense bump = 50% of former annual P&I (held nominal, no inflation in test)
    pre = float(state.expenses[0, payoff_idx - 1])
    post = float(state.expenses[0, payoff_idx])
    # Pre-payoff expenses include the mortgage P&I; post-payoff expenses
    # drop the P&I but add half of it back as lifestyle inflation, so the
    # net delta from pre to post should be -annual_pi + 0.5 * annual_pi.
    expected_delta = -annual_pi + 0.5 * annual_pi
    assert math.isclose(post - pre, expected_delta, abs_tol=1e-6), (
        f"pre={pre} post={post} delta={post - pre} expected={expected_delta}"
    )

    # Brokerage receives exactly 50% of former annual P&I as forced contribution
    brokerage_idx = results.account_idx_map["brokerage"]
    contrib = float(state.contributions[0, payoff_idx, brokerage_idx])
    assert math.isclose(contrib, 0.5 * annual_pi, abs_tol=1e-6), (
        f"contrib={contrib} expected={0.5 * annual_pi}"
    )
