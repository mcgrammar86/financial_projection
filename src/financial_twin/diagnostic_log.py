"""Diagnostic dump: per-year per-account activity for a deterministic 1-run scenario.

Forces ``n_runs=1`` and runs the engine end-to-end, then writes a
human-readable log showing the parsed config and, for each year, the
cashflow inputs, planned contributions, per-account
Start/Contribution/Withdrawal/Growth/End, pension/SS streams, and tax
breakdown. Includes a check column ``S+C-W+G-E`` that should be ~0 for
every well-behaved account.

Usage:
  python -m financial_twin.diagnostic_log configs/sample.yaml --years 5 --out diag.log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

from .config.loader import load_yaml
from .config.schema import Scenario
from .engine.cashflow import (
    compute_cashflow_year,
    planned_529_withdrawals,
    planned_contributions,
    pretax_benefits_for_year,
    salary_for_year,
)
from .engine.mortgage import amortize
from .simulate.runner import _glide_path, run_simulation


def _money(x: float) -> str:
    return f"${x:>14,.2f}"


def _hdr(out: TextIO, title: str, char: str = "=") -> None:
    bar = char * 80
    print(bar, file=out)
    print(title, file=out)
    print(bar, file=out)


def _dump_config(scenario: Scenario, out: TextIO) -> None:
    _hdr(out, "CONFIG (as parsed by pydantic)")
    print(json.dumps(scenario.model_dump(mode="json"), indent=2, default=str), file=out)
    print(f"\nconfig_hash: {scenario.config_hash}", file=out)
    print(f"n_years (derived): {scenario.n_years}", file=out)
    print(file=out)


def _dump_glide_paths(scenario: Scenario, out: TextIO) -> None:
    _hdr(out, "TDF glide paths (per stochastic account)", "-")
    for spec in scenario.accounts:
        if spec.behavior != "stochastic_market":
            continue
        weights = _glide_path(spec, scenario, scenario.n_years)
        if weights is None:
            print(f"  {spec.name:30s} STATIC stocks_weight={spec.stocks_weight}", file=out)
        else:
            head = ", ".join(f"{w:.4f}" for w in weights[:6])
            print(f"  {spec.name:30s} glide [first 6]: {head}, ... -> {weights[-1]:.4f}", file=out)
    print(file=out)


def _dump_year(scenario: Scenario, results, mortgage, t: int, out: TextIO) -> None:
    sim_year = scenario.simulation.start_year + t
    state = results.state
    _hdr(out, f"YEAR {sim_year}  (t={t})")

    # --- Income (deterministic) ---
    print("\n[Income]", file=out)
    salary_total = 0.0
    for p in scenario.people:
        s = salary_for_year(p, sim_year)
        salary_total += s
        print(f"  {p.name:30s} salary             = {_money(s)}", file=out)
    pension_total = 0.0
    for name, stream in results.pension_streams.items():
        v = float(stream[t])
        pension_total += v
        print(f"  {name:30s} pension            = {_money(v)}", file=out)
    ss_total = 0.0
    for name, stream in results.ss_streams.items():
        v = float(stream[t])
        ss_total += v
        print(f"  {name:30s} social_security    = {_money(v)}", file=out)
    print(f"  {'TOTAL salary':30s}                    = {_money(salary_total)}", file=out)
    print(f"  {'TOTAL pension':30s}                    = {_money(pension_total)}", file=out)
    print(f"  {'TOTAL social_security':30s}                    = {_money(ss_total)}", file=out)

    # --- Pre-tax payroll deductions (health, FSA, etc.) ---
    pretax_total = 0.0
    for p in scenario.people:
        v = pretax_benefits_for_year(p, sim_year)
        pretax_total += v
        if v > 0 or p.pretax_benefits:
            print(f"  {p.name:30s} pretax_benefits    = {_money(v)}", file=out)
    print(f"  {'TOTAL pretax benefits':30s}                    = {_money(pretax_total)}", file=out)

    # --- Cashflow / expenses ---
    cf = compute_cashflow_year(scenario, sim_year, mortgage, year_index=t)
    print("\n[Expenses]", file=out)
    print(f"  standard_expenses                     = {_money(cf.standard_expenses)}", file=out)
    print(f"  discretionary_expenses                = {_money(cf.discretionary_expenses)}", file=out)
    print(f"  healthcare_expense                    = {_money(cf.healthcare_expense)}", file=out)
    print(f"  mortgage_p_and_i                      = {_money(cf.mortgage_p_and_i)}", file=out)
    print(f"  property_tax                          = {_money(float(state.tax_breakdown['property'][0, t]))}", file=out)
    print(f"  pretax_payroll_benefits               = {_money(pretax_total)}", file=out)
    print(f"  lifestyle_split_brokerage_contrib     = {_money(cf.lifestyle_split_brokerage_contrib)}", file=out)
    print(f"  TOTAL expenses (state.expenses)       = {_money(float(state.expenses[0, t]))}", file=out)

    # --- Planned contributions (per account) ---
    plan = planned_contributions(scenario, sim_year)
    print("\n[Planned contributions (employee, employer match)]", file=out)
    print(f"  {'Account':30s} {'Employee':>15s} {'Match':>15s}", file=out)
    for name, (emp, match) in plan.items():
        print(f"  {name:30s} {emp:>15,.2f} {match:>15,.2f}", file=out)

    # --- 529 planned withdrawals ---
    w529 = {k: v for k, v in planned_529_withdrawals(scenario, sim_year).items() if v > 0}
    if w529:
        print("\n[Planned 529 withdrawals]", file=out)
        for name, amt in w529.items():
            print(f"  {name:30s} {_money(amt)}", file=out)

    # --- Per-account activity (single run, so this IS the realized path) ---
    # Use the clean balance snapshots: history[t] = jan1 of year t (= initial
    # for t=0, end-of-year-(t-1) otherwise), so Start/End read true.
    history = results.balance_history()
    print("\n[Per-account activity (single deterministic run)]", file=out)
    print(
        f"  {'Account':30s} {'Start':>14s} {'Employee':>12s} {'Match':>10s} "
        f"{'Withdraw':>12s} {'Growth':>12s} {'End':>14s}   [check]",
        file=out,
    )
    for spec in scenario.accounts:
        if spec.behavior == "defined_benefit":
            continue
        idx = results.account_idx_map[spec.name]
        start = float(history[0, t, idx])
        employee = float(state.contributions[0, t, idx])
        match = float(state.employer_match[0, t, idx])
        withdraw = float(state.withdrawals[0, t, idx])
        growth = float(state.growth[0, t, idx])
        end = float(history[0, t + 1, idx])
        check = start + employee + match - withdraw + growth - end
        print(
            f"  {spec.name:30s} {start:>14,.2f} {employee:>12,.2f} {match:>10,.2f} "
            f"{withdraw:>12,.2f} {growth:>12,.2f} {end:>14,.2f}   [{check:>+8,.2f}]",
            file=out,
        )

    # --- Income aggregation + tax breakdown ---
    print("\n[Income aggregation & tax]", file=out)
    print(f"  gross_income                          = {_money(float(state.gross_income[0, t]))}", file=out)
    print(f"  taxable_income (after pretax deduct)  = {_money(float(state.taxable_income[0, t]))}", file=out)
    print(f"  ltcg_income                           = {_money(float(state.ltcg_income[0, t]))}", file=out)
    print(f"  pension_income (in tax)               = {_money(float(state.pension_income[0, t]))}", file=out)
    print(f"  ss_income (in tax)                    = {_money(float(state.ss_income[0, t]))}", file=out)
    print(f"  federal_tax                           = {_money(float(state.tax_breakdown['federal'][0, t]))}", file=out)
    print(f"  oregon_tax                            = {_money(float(state.tax_breakdown['oregon'][0, t]))}", file=out)
    print(f"  metro_shs_tax                         = {_money(float(state.tax_breakdown['metro_shs'][0, t]))}", file=out)
    print(f"  property_tax                          = {_money(float(state.tax_breakdown['property'][0, t]))}", file=out)
    print(f"  total_tax_paid                        = {_money(float(state.tax_paid[0, t]))}", file=out)
    print(f"  tax_iterations_to_converge            = {int(state.tax_iterations[t])}", file=out)

    # --- Cashflow reconciliation ---
    salary_total = sum(salary_for_year(p, sim_year) for p in scenario.people)
    pension_total = sum(float(stream[t]) for stream in results.pension_streams.values())
    ss_total = sum(float(stream[t]) for stream in results.ss_streams.values())
    drawn_total = float(state.withdrawals[0, t, :].sum())
    employee_total = float(state.contributions[0, t, :].sum())
    income_in = salary_total + pension_total + ss_total + drawn_total
    cash_out = (
        float(state.expenses[0, t])
        + float(state.tax_breakdown["federal"][0, t])
        + float(state.tax_breakdown["oregon"][0, t])
        + float(state.tax_breakdown["metro_shs"][0, t])
        + employee_total
        + pretax_total
    )
    surplus = income_in - cash_out
    print("\n[Cashflow reconciliation]", file=out)
    print(f"  IN  (salary + pension + SS + withdrawals)  = {_money(income_in)}", file=out)
    print(f"  OUT (expenses + income tax + employee contribs + pretax benefits)", file=out)
    print(f"                                             = {_money(cash_out)}", file=out)
    print(f"  Surplus (IN - OUT)                          = {_money(surplus)}", file=out)
    print(file=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Scenario YAML path")
    parser.add_argument("--years", type=int, default=5, help="Years to log (default 5)")
    parser.add_argument("--out", type=Path, default=None, help="Output file (default stdout)")
    args = parser.parse_args(argv)

    scenario = load_yaml(args.config)
    scenario.simulation.n_runs = 1  # force deterministic single-run

    results = run_simulation(scenario)
    mortgage = amortize(
        scenario.mortgage,
        start_year=scenario.simulation.start_year,
        n_years=scenario.n_years,
    )

    sink = open(args.out, "w") if args.out else sys.stdout
    try:
        _dump_config(scenario, sink)
        _dump_glide_paths(scenario, sink)
        n = min(args.years, scenario.n_years)
        for t in range(n):
            _dump_year(scenario, results, mortgage, t, sink)
    finally:
        if args.out:
            sink.close()
            print(f"Wrote diagnostic log to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
