"""Tax-cashflow convergence + pretax benefits.

The runner iterates cash_need / withdraw / tax until tax stabilizes,
so the cashflow loop must reconcile (income IN = expenses + tax + contribs).
"""

from __future__ import annotations

import numpy as np

from financial_twin.config.loader import load_dict
from financial_twin.engine.cashflow import pretax_benefits_for_year
from financial_twin.simulate.runner import run_simulation


def _retiree_scenario(pretax_benefits: list[dict] | None = None) -> dict:
    """Single-person, single-account retiree drawing from pretax 401k.

    No salary, no SS yet. All cash need must come from pretax withdrawals,
    which are taxed as ordinary income -- the chicken-and-egg loop the
    convergence iteration is supposed to solve.
    """
    person: dict = {
        "name": "alex",
        "birth_year": 1960,
        "retirement_year": 2026,  # already retired in start_year
        "salary_2026": 0.0,
        "pia_monthly": 0.0,
    }
    if pretax_benefits is not None:
        person["pretax_benefits"] = pretax_benefits
    return {
        "people": [person],
        "accounts": [
            {
                "name": "k401",
                "behavior": "stochastic_market",
                "balance_2026": 1_000_000.0,
                "tax_type": "pre_tax",
                "contribution_limit_2026": 0.0,
                "stocks_mean": 0.0,
                "stocks_stdev": 0.0,
                "bonds_mean": 0.0,
                "bonds_stdev": 0.0,
                "stocks_weight": 0.0,
                "owner": "alex",
            }
        ],
        "expenses": {"standard_annual_2026": 60_000.0},
        "simulation": {"n_runs": 1, "start_year": 2026, "end_year": 2027, "seed": 1},
        "tax": {"inflation_rate": 0.0},
    }


def test_cashflow_loop_reconciles_at_convergence():
    """After convergence, IN (= withdrawals) ≈ OUT (= expenses + tax)."""
    scenario = load_dict(_retiree_scenario())
    results = run_simulation(scenario)
    state = results.state

    # Year 0 reconciliation
    drawn = float(state.withdrawals[0, 0, :].sum())
    expenses = float(state.expenses[0, 0])
    income_tax = float(
        state.tax_breakdown["federal"][0, 0]
        + state.tax_breakdown["oregon"][0, 0]
        + state.tax_breakdown["metro_shs"][0, 0]
    )
    # IN = drawn (no salary/pension/SS); OUT = expenses (incl property=0 here) + income tax
    # Property tax is 0 because there's no `house` block.
    assert abs(drawn - (expenses + income_tax)) < 1.0, (
        f"drawn={drawn:.2f} != expenses+tax={expenses + income_tax:.2f}"
    )
    assert state.tax_iterations[0] >= 1
    assert state.tax_iterations[0] < 25  # well under the safety cap


def _worker_scenario(pretax_benefits: list[dict] | None = None) -> dict:
    """Working person with W2 income. Pretax benefits should reduce tax."""
    person: dict = {
        "name": "alex",
        "birth_year": 1985,
        "retirement_year": 2050,
        "salary_2026": 120_000.0,
    }
    if pretax_benefits is not None:
        person["pretax_benefits"] = pretax_benefits
    return {
        "people": [person],
        "accounts": [
            {
                "name": "savings",
                "behavior": "stochastic_market",
                "balance_2026": 100_000.0,
                "tax_type": "taxable",
                "contribution_limit_2026": 1e9,
                "stocks_mean": 0.0,
                "stocks_stdev": 0.0,
                "stocks_weight": 0.0,
                "owner": "alex",
            }
        ],
        "expenses": {"standard_annual_2026": 30_000.0},
        "simulation": {"n_runs": 1, "start_year": 2026, "end_year": 2027, "seed": 1},
        "tax": {"inflation_rate": 0.0},
    }


def test_pretax_benefits_reduce_taxable_income_for_worker():
    """A $5k pretax health insurance line drops a worker's taxable income by $5k."""
    base = run_simulation(load_dict(_worker_scenario()))
    with_benefit = run_simulation(load_dict(_worker_scenario(
        pretax_benefits=[{"name": "health", "amount_2026": 5_000.0, "growth": 0.0}]
    )))
    base_taxable = float(base.state.taxable_income[0, 0])
    with_taxable = float(with_benefit.state.taxable_income[0, 0])
    assert abs((base_taxable - with_taxable) - 5_000.0) < 1.0, (
        f"taxable income should drop by exactly $5k; "
        f"base={base_taxable:.2f}, with={with_taxable:.2f}"
    )
    base_tax = float(base.state.tax_paid[0, 0])
    with_tax = float(with_benefit.state.tax_paid[0, 0])
    assert with_tax < base_tax, "pretax benefit should reduce total tax for a worker"


def test_pretax_benefit_helper():
    """Helper returns inflated amount and stops at retirement."""
    from financial_twin.config.schema import Person, PretaxBenefit
    p = Person(
        name="alex", birth_year=1960, retirement_year=2030, salary_2026=100_000.0,
        pretax_benefits=[
            PretaxBenefit(name="health", amount_2026=6_000.0, growth=0.05),
            PretaxBenefit(name="fsa", amount_2026=2_000.0, growth=0.0,
                          stops_at_retirement=False),
        ],
    )
    assert pretax_benefits_for_year(p, 2026) == 8_000.0
    # 2027: health = 6000*1.05 = 6300, fsa = 2000 → 8300
    assert abs(pretax_benefits_for_year(p, 2027) - 8_300.0) < 1e-6
    # 2030 (retirement): health stops, fsa keeps going
    assert pretax_benefits_for_year(p, 2030) == 2_000.0
