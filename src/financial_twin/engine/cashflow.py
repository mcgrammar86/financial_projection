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


def healthcare_for_year(scenario: Scenario, sim_year: int) -> float:
    bridge = 0.0
    medicare = 0.0
    inflation = scenario.tax.inflation_rate
    base_factor = (1.0 + inflation) ** (sim_year - 2026)
    bridge_per = scenario.healthcare.bridge_annual_premium_2026 * base_factor
    medicare_per = scenario.healthcare.medicare_supplement_annual_2026 * base_factor
    for person in scenario.people:
        if sim_year < person.retirement_year:
            continue
        age = sim_year - person.birth_year
        if person.death_year is not None and sim_year >= person.death_year:
            continue
        if age < 65:
            bridge += bridge_per
        else:
            medicare += medicare_per
    return bridge + medicare


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
) -> dict[str, float]:
    """Return planned contribution by account name for the year (deterministic)."""
    inflation = scenario.tax.inflation_rate
    factor = (1.0 + inflation) ** (sim_year - 2026)
    contribs: dict[str, float] = {}
    for spec in scenario.accounts:
        if spec.behavior == "stochastic_market":
            assert isinstance(spec, StochasticAccount)
            owner = next((p for p in scenario.people if p.name == spec.owner), None)
            if owner is None or sim_year >= owner.retirement_year:
                contribs[spec.name] = 0.0
                continue
            cap = spec.contribution_limit_2026 * factor
            planned = min(spec.contribution_2026 * factor, cap)
            contribs[spec.name] = planned + spec.employer_match_2026 * factor
        elif spec.behavior == "fixed_deferred":
            assert isinstance(spec, FixedDeferredAccount)
            growth = (1.0 + spec.contribution_growth) ** (sim_year - 2026)
            contribs[spec.name] = spec.contribution_2026 * growth
        elif spec.behavior == "education_529_glidepath":
            assert isinstance(spec, Education529Account)
            growth = (1.0 + spec.contribution_growth) ** (sim_year - 2026)
            contribs[spec.name] = spec.contribution_2026 * growth
        elif spec.behavior == "defined_benefit":
            assert isinstance(spec, DefinedBenefitAccount)
            contribs[spec.name] = 0.0  # handled by pension stream
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
