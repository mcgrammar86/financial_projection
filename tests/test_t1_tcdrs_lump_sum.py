"""T1 — TCDRS Lump-Sum Audit.

5-year horizon, $100k principal, 7% annual rate, **no contributions**.
Balance at end of year 5 must equal 100_000 * 1.07**5 to 6 decimal places.
"""

from __future__ import annotations

import math

from financial_twin.simulate.runner import run_simulation


def test_tcdrs_lump_sum_compound(make_scenario):
    scenario = make_scenario(
        accounts=[
            {
                "name": "tcdrs",
                "behavior": "fixed_deferred",
                "balance_2026": 100_000.0,
                "annual_rate": 0.07,
                "is_taxable": False,
                "contribution_2026": 0.0,
                "contribution_growth": 0.0,
                "interest_on_jan1_only": True,
            }
        ],
    )
    results = run_simulation(scenario)
    idx = results.account_idx_map["tcdrs"]
    final = float(results.state.balances[0, 5, idx])
    expected = 100_000.0 * (1.07 ** 5)
    assert math.isclose(final, expected, rel_tol=0, abs_tol=1e-6), (
        f"final={final:.6f} expected={expected:.6f}"
    )
