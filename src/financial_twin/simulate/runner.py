"""Monte Carlo runner.

Executes the full annual loop:

  for year in range(n_years):
      1. Pension + Social Security + salary income
      2. Mandatory expenses (incl. mortgage P&I + post-payoff lifestyle)
      3. Planned contributions + 529 withdrawals
      4. RMDs (forced from pre-tax accounts at age 73+)
      5. Tax-efficient drawdown for any spending shortfall
      6. Apply per-account growth via behavior registry
      7. Aggregate gross/taxable income, compute taxes
      8. Net taxes against cash; defer 529 credit to year+1

Polars only enters at the end via ``Results.to_polars()``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ..config.loader import load_yaml
from ..config.schema import (
    DefinedBenefitAccount,
    Education529Account,
    FixedDeferredAccount,
    Scenario,
    StochasticAccount,
)
from ..engine.accounts import AccountContext, step_accounts
from ..engine.cashflow import (
    compute_cashflow_year,
    planned_529_withdrawals,
    planned_contributions,
    pretax_benefits_for_year,
    salary_for_year,
)
from ..engine.mortgage import amortize
from ..engine.pension import compute_pension_stream
from ..engine.returns import draw_returns
from ..engine.rng import make_rng
from ..engine.social_security import compute_ss_stream
from ..engine.state import SimState
from ..engine.withdrawals import (
    AccountClassification,
    apply_rmd,
    classify_accounts,
    withdraw_for_need,
)
from ..tax import engine as tax_engine
from ..tax.property import project_assessed_values


@dataclass
class Results:
    scenario: Scenario
    state: SimState
    pension_streams: dict[str, np.ndarray]
    ss_streams: dict[str, np.ndarray]
    account_idx_map: dict[str, int]

    def to_polars(self) -> pl.DataFrame:
        nr, ny_plus_1, na = self.state.balances.shape
        ny = ny_plus_1 - 1
        names = [None] * na
        for name, idx in self.account_idx_map.items():
            names[idx] = name
        # End-of-year balances (clean, never mutated by next-year withdrawals)
        end_bal = self.state.eoy_balances  # [n_runs, ny, na]
        run_idx = np.repeat(np.arange(nr), ny * na)
        year_idx = np.tile(np.repeat(np.arange(ny), na), nr)
        acc_idx = np.tile(np.arange(na), nr * ny)
        sim_year = self.scenario.simulation.start_year + year_idx
        return pl.DataFrame(
            {
                "run": run_idx,
                "sim_year": sim_year,
                "account": [names[i] for i in acc_idx],
                "balance": end_bal.reshape(-1),
                "contribution": self.state.contributions.reshape(-1),
                "employer_match": self.state.employer_match.reshape(-1),
                "withdrawal": self.state.withdrawals.reshape(-1),
                "growth": self.state.growth.reshape(-1),
            }
        )

    @property
    def terminal_net_worth(self) -> np.ndarray:
        return self.state.eoy_balances[:, -1, :].sum(axis=1)

    def balance_history(self) -> np.ndarray:
        """Clean per-year balance snapshots, shape ``[n_runs, n_years+1, n_accounts]``.

        Index 0 is start-of-simulation (= ``balance_2026``); index t for t>=1
        is the true end-of-year-(t-1). Unlike ``state.balances``, this array
        is never mutated by the year loop's in-place withdrawals.
        """
        nr, na = self.state.initial_balances.shape
        history = np.empty((nr, self.state.eoy_balances.shape[1] + 1, na), dtype=np.float64)
        history[:, 0, :] = self.state.initial_balances
        history[:, 1:, :] = self.state.eoy_balances
        return history


def _glide_path(
    spec, scenario: Scenario, n_years: int
) -> np.ndarray | None:
    """Per-year stocks weight for a stochastic account, or None if static.

    Linearly interpolates from ``spec.stocks_weight`` (at start_year) to
    ``spec.stocks_weight_glide_target`` at ``stocks_weight_glide_target_year``
    (defaults to the owner's retirement_year). Held flat after the target.
    """
    if spec.behavior != "stochastic_market":
        return None
    target = getattr(spec, "stocks_weight_glide_target", None)
    if target is None:
        return None
    target_year = spec.stocks_weight_glide_target_year
    if target_year is None:
        owner = next((p for p in scenario.people if p.name == spec.owner), None)
        if owner is None:
            return None
        target_year = owner.retirement_year
    start_year = scenario.simulation.start_year
    glide_years = max(1, target_year - start_year)
    weights = np.empty(n_years, dtype=np.float64)
    for t in range(n_years):
        sim_year = start_year + t
        if sim_year >= target_year:
            weights[t] = target
        else:
            frac = (sim_year - start_year) / glide_years
            weights[t] = spec.stocks_weight + frac * (target - spec.stocks_weight)
    return weights


def run_simulation(scenario: Scenario) -> Results:
    nr = scenario.simulation.n_runs
    ny = scenario.n_years
    start_year = scenario.simulation.start_year

    # Index accounts in a stable order, defined-benefit handled separately.
    balance_specs = [a for a in scenario.accounts if a.behavior != "defined_benefit"]
    pension_specs = [a for a in scenario.accounts if a.behavior == "defined_benefit"]
    account_idx_map = {spec.name: i for i, spec in enumerate(balance_specs)}
    contexts = [
        AccountContext(
            spec=spec,
            idx=i,
            base_year=start_year,
            stocks_weight_by_year=_glide_path(spec, scenario, ny),
        )
        for i, spec in enumerate(balance_specs)
    ]
    pretax_specs_by_idx = {
        i: spec
        for i, spec in enumerate(balance_specs)
        if (spec.behavior == "stochastic_market" and getattr(spec, "tax_type", None) == "pre_tax")
        or (spec.behavior == "fixed_deferred" and not getattr(spec, "is_taxable", True))
    }
    person_by_name = {p.name: p for p in scenario.people}

    state = SimState(
        n_runs=nr, n_years=ny, n_accounts=len(balance_specs), start_year=start_year
    )

    # Initial balances
    for spec in balance_specs:
        idx = account_idx_map[spec.name]
        state.balances[:, 0, idx] = getattr(spec, "balance_2026", 0.0)
        if spec.behavior == "stochastic_market" and spec.tax_type == "taxable":
            # Brokerage cost basis tracked for whole household; sum across taxable accounts.
            state.cost_basis_brokerage[:, 0] += spec.balance_2026
    # Snapshot initial balances before the year loop mutates state.balances[:, 0].
    state.initial_balances[:, :] = state.balances[:, 0, :]

    # Stochastic returns (we use the first stochastic account's mean/stdev as
    # the *market* draw; per-account asset weights blend stocks vs bonds).
    stoch = next((a for a in balance_specs if a.behavior == "stochastic_market"), None)
    if stoch is None:
        # Fallback so the engine still runs in stochastic-free scenarios.
        stocks_mean, stocks_stdev = 0.07, 0.18
        bonds_mean, bonds_stdev = 0.03, 0.06
    else:
        stocks_mean = stoch.stocks_mean
        stocks_stdev = stoch.stocks_stdev
        bonds_mean = stoch.bonds_mean
        bonds_stdev = stoch.bonds_stdev
    rng = make_rng(scenario.simulation.seed)
    state.returns_draw = draw_returns(
        rng,
        n_runs=nr,
        n_years=ny,
        stocks_mean=stocks_mean,
        stocks_stdev=stocks_stdev,
        bonds_mean=bonds_mean,
        bonds_stdev=bonds_stdev,
        correlation=scenario.simulation.stocks_bonds_correlation,
    )

    # Mortgage
    mortgage = amortize(scenario.mortgage, start_year=start_year, n_years=ny)

    # Pre-compute deterministic pension + SS streams
    pension_streams: dict[str, np.ndarray] = {}
    for db in pension_specs:
        assert isinstance(db, DefinedBenefitAccount)
        person = person_by_name[db.owner]
        salary_history = np.array(
            [salary_for_year(person, start_year + t) for t in range(ny)],
            dtype=np.float64,
        )
        pension_streams[db.name] = compute_pension_stream(
            scenario, db, person, salary_history,
            n_years=ny, start_year=start_year,
        )

    ss_streams: dict[str, np.ndarray] = {
        p.name: compute_ss_stream(
            p, n_years=ny, start_year=start_year, inflation_rate=scenario.tax.inflation_rate
        )
        for p in scenario.people
    }

    classification = classify_accounts(scenario, account_idx_map)

    # Property tax projection (deterministic, household-level)
    avs, prop_taxes = project_assessed_values(scenario.house, ny)
    state.assessed_value[:] = np.array(avs, dtype=np.float64)

    # Year loop
    for t in range(ny):
        sim_year = start_year + t

        # 1. Income (deterministic)
        salary = sum(salary_for_year(p, sim_year) for p in scenario.people)
        pension = sum(stream[t] for stream in pension_streams.values())
        ss = sum(stream[t] for stream in ss_streams.values())

        # 2. Expenses (mortgage + lifestyle split)
        cf = compute_cashflow_year(scenario, sim_year, mortgage, year_index=t)
        property_tax = prop_taxes[t]
        total_expenses = (
            cf.standard_expenses
            + cf.discretionary_expenses
            + cf.healthcare_expense
            + cf.mortgage_p_and_i
            + property_tax
        )
        state.expenses[:, t] = total_expenses

        # 3. Contributions: plan is {name: (employee, employer_match)}
        plan = planned_contributions(scenario, sim_year)
        # Add lifestyle-split brokerage contribution to designated account
        if mortgage is not None and sim_year >= mortgage.payoff_year:
            target = scenario.mortgage.lifestyle_split_brokerage_account if scenario.mortgage else None
            if target is not None and target in account_idx_map:
                emp, match = plan.get(target, (0.0, 0.0))
                plan[target] = (emp + cf.lifestyle_split_brokerage_contrib, match)
        for name, (employee, match) in plan.items():
            idx = account_idx_map.get(name)
            if idx is None:
                continue
            state.contributions[:, t, idx] = employee
            state.employer_match[:, t, idx] = match

        # 529 withdrawals (deterministic)
        for name, amount in planned_529_withdrawals(scenario, sim_year).items():
            idx = account_idx_map[name]
            state.withdrawals[:, t, idx] += amount

        # 529 contribution credit accumulates across all 529 accounts this year
        contributed_529 = sum(
            employee for k, (employee, _match) in plan.items()
            if any(a.name == k and a.behavior == "education_529_glidepath" for a in scenario.accounts)
        )
        contributed_529_arr = np.full(nr, contributed_529, dtype=np.float64)

        # 4. RMDs (forced ordinary income) -- run once before the iteration.
        rmd_per_run = apply_rmd(
            state, t, scenario, classification, pretax_specs_by_idx, person_by_name
        )

        # Pre-tax payroll deductions (health insurance, FSA, etc.): a cash
        # outflow that ALSO reduces taxable ordinary income (Section 125 /
        # cafeteria-plan style). Deterministic per year.
        pretax_benefits_total = sum(
            pretax_benefits_for_year(p, sim_year) for p in scenario.people
        )
        state.pretax_benefits[:, t] = pretax_benefits_total

        # Pre-tax 401k/457 contributions reduce taxable ordinary income.
        pretax_employee_contrib = 0.0
        for spec in balance_specs:
            if spec.behavior == "stochastic_market" and spec.tax_type == "pre_tax":
                emp, _match = plan.get(spec.name, (0.0, 0.0))
                pretax_employee_contrib += emp

        # Snapshot post-RMD scratch state so we can iterate cash_need <->
        # withdraw <-> tax until tax converges. RMD is independent of tax
        # (depends only on jan1 balance), so we don't redo it.
        balances_snapshot = state.balances[:, t, :].copy()
        withdrawals_snapshot = state.withdrawals[:, t, :].copy()
        basis_snapshot = state.cost_basis_brokerage[:, t].copy()

        # 5. Iterate cash_need / withdraw / tax to convergence.
        # Initial guess: zero income tax (cash_need only covers expenses).
        # Each iteration: (a) compute cash_need including current tax estimate,
        # (b) withdraw to cover, (c) recompute tax from realized withdrawals,
        # (d) compare to previous estimate. Converges geometrically because
        # each $1 of additional pretax withdrawal adds only ~marginal-rate
        # extra tax (< $1).
        fixed_income = salary + pension + ss
        employee_outflow = sum(emp for emp, _m in plan.values())
        nontax_cash_need = (
            total_expenses + employee_outflow + pretax_benefits_total - fixed_income
        )
        # nontax_cash_need is a scalar; tax_estimate is per-run.
        tax_estimate = np.zeros(nr, dtype=np.float64)
        ordinary_drawn = np.zeros(nr, dtype=np.float64)
        ltcg_drawn = np.zeros(nr, dtype=np.float64)
        result = None
        deferred_in = state.deferred_state_credit[:, t]
        max_iter = 25
        for it in range(max_iter):
            # Restore the post-RMD baseline before re-running withdrawals.
            state.balances[:, t, :] = balances_snapshot
            state.withdrawals[:, t, :] = withdrawals_snapshot
            state.cost_basis_brokerage[:, t] = basis_snapshot

            cash_need = np.maximum(0.0, nontax_cash_need + tax_estimate)
            net_need = np.maximum(0.0, cash_need - rmd_per_run)
            ordinary_drawn, ltcg_drawn, _unmet = withdraw_for_need(
                state, t, net_need, classification
            )

            ordinary_income = np.maximum(
                0.0,
                salary + pension + ss + ordinary_drawn + rmd_per_run
                - pretax_employee_contrib - pretax_benefits_total,
            )
            result = tax_engine.compute_taxes(
                scenario,
                sim_year=sim_year,
                ordinary_income=ordinary_income,
                ltcg_income=ltcg_drawn,
                deferred_state_credit_in=deferred_in,
                contributed_to_529_this_year=contributed_529_arr,
            )
            new_tax = result.total
            if np.max(np.abs(new_tax - tax_estimate)) < 1.0:
                tax_estimate = new_tax
                state.tax_iterations[t] = it + 1
                break
            tax_estimate = new_tax
        else:
            state.tax_iterations[t] = max_iter

        ltcg_income = ltcg_drawn
        ordinary_income = np.maximum(
            0.0,
            salary + pension + ss + ordinary_drawn + rmd_per_run
            - pretax_employee_contrib - pretax_benefits_total,
        )

        # Track Brokerage cost basis: total inflow (employee + employer match)
        # adds to basis; withdrawal handling already deducted basis_consumed
        # in withdraw_for_need.
        for spec in balance_specs:
            if spec.behavior == "stochastic_market" and spec.tax_type == "taxable":
                idx = account_idx_map[spec.name]
                state.cost_basis_brokerage[:, t] += (
                    state.contributions[:, t, idx] + state.employer_match[:, t, idx]
                )

        # 6. Apply growth (after withdrawals/RMD) — balances[:,t+1,...] = ...
        step_accounts(state, t, contexts)
        # Snapshot the true end-of-year balance now, before next year's
        # apply_rmd / withdraw_for_need mutates balances[:, t+1, :] in place.
        state.eoy_balances[:, t, :] = state.balances[:, t + 1, :]

        # Carry brokerage basis forward — clamp so basis never exceeds balance
        next_brokerage_balance = np.zeros(nr, dtype=np.float64)
        for spec in balance_specs:
            if spec.behavior == "stochastic_market" and spec.tax_type == "taxable":
                idx = account_idx_map[spec.name]
                next_brokerage_balance += state.balances[:, t + 1, idx]
        state.cost_basis_brokerage[:, t + 1] = np.minimum(
            state.cost_basis_brokerage[:, t], next_brokerage_balance
        )

        # 7. Record final per-year aggregates.
        state.gross_income[:, t] = salary + pension + ss + ordinary_drawn + rmd_per_run + ltcg_drawn
        state.taxable_income[:, t] = ordinary_income + ltcg_income
        state.ltcg_income[:, t] = ltcg_income
        state.pension_income[:, t] = pension
        state.ss_income[:, t] = ss

        state.tax_paid[:, t] = result.total + property_tax
        state.tax_breakdown["federal"][:, t] = result.federal
        state.tax_breakdown["oregon"][:, t] = result.oregon
        state.tax_breakdown["metro_shs"][:, t] = result.metro_shs
        state.tax_breakdown["property"][:, t] = property_tax
        state.deferred_state_credit[:, t + 1] = result.deferred_state_credit_next_year

    return Results(
        scenario=scenario,
        state=state,
        pension_streams=pension_streams,
        ss_streams=ss_streams,
        account_idx_map=account_idx_map,
    )


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m financial_twin.simulate.runner <config.yaml>")
        return 2
    scenario = load_yaml(Path(argv[0]))
    results = run_simulation(scenario)
    tw = results.terminal_net_worth
    p10, p50, p90 = np.percentile(tw, [10, 50, 90])
    print(f"Terminal net worth (n_runs={scenario.simulation.n_runs}):")
    print(f"  P10: ${p10:,.0f}")
    print(f"  P50: ${p50:,.0f}")
    print(f"  P90: ${p90:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
