"""Cash-flow orchestration: salary, expenses, contributions, the surplus
waterfall, and the T3 mortgage lifestyle split.

The runner calls this once per year per scenario:
1. Compute gross salary income for the year.
2. Compute mandatory expenses (standard + discretionary + property tax
   + healthcare bridge + mortgage P&I).
3. Compute contributions (capped by IRS limits, employer match).
4. Apply the surplus waterfall HSA -> 457 -> 401k -> Brokerage.
5. After mortgage payoff, half of the former P&I shows up as a
   permanent expense bump and the other half is auto-routed to the
   designated Brokerage account.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config.schema import (
    DefinedBenefitAccount,
    Education529Account,
    FixedDeferredAccount,
    Person,
    Scenario,
    StochasticAccount,
)
from .mortgage import MortgageSchedule


@dataclass
class CashflowYear:
    salary_total: float
    standard_expenses: float
    discretionary_expenses: float
    healthcare_expense: float
    mortgage_p_and_i: float
    lifestyle_split_brokerage_contrib: float


def salary_for_year(person: Person, sim_year: int) -> float:
    if sim_year >= person.retirement_year:
        return 0.0
    years_from_2026 = sim_year - 2026
    return person.salary_2026 * (1.0 + person.salary_growth) ** years_from_2026


def pretax_benefits_for_year(person: Person, sim_year: int) -> float:
    """Per-person pre-tax payroll deductions (health insurance, FSA, etc.).

    These reduce taxable ordinary income AND reduce take-home cash. By
    default each benefit stops at the person's retirement_year.
    """
    if not person.pretax_benefits:
        return 0.0
    total = 0.0
    for benefit in person.pretax_benefits:
        if benefit.stops_at_retirement and sim_year >= person.retirement_year:
            continue
        total += benefit.amount_2026 * (1.0 + benefit.growth) ** (sim_year - 2026)
    return total


def healthcare_for_year(scenario: Scenario, sim_year: int) -> float:
    bridge = 0.0
    medicare = 0.0
    inflation = scenario.tax.inflation_rate
    base_factor = (1.0 + inflation) ** (sim_year - 2026)
    bridge_per = scenario.healthcare.bridge_annual_premium_2026 * base_factor
    medicare_per = scenario.healthcare.medicare_supplement_annual_2026 * base_factor

    # Per-person bridge/Medicare (only after retirement; switches at 65).
    living_count = 0
    for person in scenario.people:
        if person.death_year is not None and sim_year >= person.death_year:
            continue
        living_count += 1
        if sim_year < person.retirement_year:
            continue
        age = sim_year - person.birth_year
        if age < 65:
            bridge += bridge_per
        else:
            medicare += medicare_per

    # Household out-of-pocket: every year (working or retired) until the
    # household empties out. Uses healthcare-specific inflation, not CPI.
    oop_factor = (1.0 + scenario.healthcare.out_of_pocket_growth) ** (sim_year - 2026)
    oop = scenario.healthcare.out_of_pocket_annual_2026 * oop_factor
    if living_count == 0:
        oop = 0.0

    return bridge + medicare + oop


def compute_cashflow_year(
    scenario: Scenario,
    sim_year: int,
    mortgage: MortgageSchedule | None,
    *,
    year_index: int,
) -> CashflowYear:
    inflation = scenario.tax.inflation_rate
    inflation_factor = (1.0 + inflation) ** (sim_year - 2026)

    salary_total = sum(salary_for_year(p, sim_year) for p in scenario.people)
    standard = scenario.expenses.standard_annual_2026 * inflation_factor
    discretionary = scenario.expenses.discretionary_annual_2026 * inflation_factor
    health = healthcare_for_year(scenario, sim_year)
    p_and_i = float(mortgage.annual_p_and_i[year_index]) if mortgage is not None else 0.0

    lifestyle_extra = 0.0
    lifestyle_brokerage_contrib = 0.0
    if mortgage is not None and sim_year >= mortgage.payoff_year:
        half = 0.5 * mortgage.annual_p_and_i_at_payoff
        lifestyle_extra = half
        lifestyle_brokerage_contrib = half

    return CashflowYear(
        salary_total=salary_total,
        standard_expenses=standard + lifestyle_extra,
        discretionary_expenses=discretionary,
        healthcare_expense=health,
        mortgage_p_and_i=p_and_i,
        lifestyle_split_brokerage_contrib=lifestyle_brokerage_contrib,
    )


def planned_contributions(
    scenario: Scenario, sim_year: int
) -> dict[str, tuple[float, float]]:
    """Return ``{name: (employee, employer_match)}`` for the year.

    Employee portion is the participant's own contribution (subject to
    pre-tax-income deduction logic in the runner). Employer match is
    counted toward the account balance but never reduces taxable income.
    """
    inflation = scenario.tax.inflation_rate
    factor = (1.0 + inflation) ** (sim_year - 2026)
    contribs: dict[str, tuple[float, float]] = {}
    for spec in scenario.accounts:
        if spec.behavior == "stochastic_market":
            assert isinstance(spec, StochasticAccount)
            owner = next((p for p in scenario.people if p.name == spec.owner), None)
            if owner is None or sim_year >= owner.retirement_year:
                contribs[spec.name] = (0.0, 0.0)
                continue
            cap = spec.contribution_limit_2026 * factor
            owner_salary = salary_for_year(owner, sim_year)
            # Employee contribution = flat dollar (inflated) + % of current salary.
            planned_raw = spec.contribution_2026 * factor + owner_salary * spec.contribution_pct
            planned = min(planned_raw, cap)
            if spec.employer_match_pct > 0.0:
                cap_pct = (
                    spec.employer_match_max_pct
                    if spec.employer_match_max_pct > 0.0
                    else spec.employer_match_pct
                )
                contribution_pct_of_salary = (
                    planned / owner_salary if owner_salary > 0 else 0.0
                )
                effective_pct = min(spec.employer_match_pct, cap_pct, contribution_pct_of_salary)
                match = owner_salary * effective_pct
            else:
                match = spec.employer_match_2026 * factor
            # Non-elective employer contribution: paid every working year
            # regardless of what the employee contributes.
            match += owner_salary * spec.employer_contribution_pct
            contribs[spec.name] = (planned, match)
        elif spec.behavior == "fixed_deferred":
            assert isinstance(spec, FixedDeferredAccount)
            growth = (1.0 + spec.contribution_growth) ** (sim_year - 2026)
            contribs[spec.name] = (spec.contribution_2026 * growth, 0.0)
        elif spec.behavior == "education_529_glidepath":
            assert isinstance(spec, Education529Account)
            growth = (1.0 + spec.contribution_growth) ** (sim_year - 2026)
            contribs[spec.name] = (spec.contribution_2026 * growth, 0.0)
        elif spec.behavior == "defined_benefit":
            assert isinstance(spec, DefinedBenefitAccount)
            contribs[spec.name] = (0.0, 0.0)  # handled by pension stream
    return contribs


def planned_529_withdrawals(
    scenario: Scenario, sim_year: int
) -> dict[str, float]:
    """Return planned 529 withdrawal by account name for the year."""
    out: dict[str, float] = {}
    inflation = scenario.tax.inflation_rate
    factor = (1.0 + inflation) ** (sim_year - 2026)
    for spec in scenario.accounts:
        if spec.behavior != "education_529_glidepath":
            continue
        assert isinstance(spec, Education529Account)
        age = sim_year - spec.beneficiary_birth_year
        if spec.college_start_age <= age < spec.college_start_age + spec.college_years:
            out[spec.name] = spec.annual_tuition_2026 * factor
        else:
            out[spec.name] = 0.0
    return out
