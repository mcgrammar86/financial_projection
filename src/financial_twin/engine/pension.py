"""Oregon PERS OPSRP pension calculator.

Benefit = formula_multiplier * Years_of_Service * FAS, with a 2%-capped
COLA each retirement year. When the pensioner's ``wife_phase`` is
``sprint``, FAS is replaced by the FTE salary across the full service
duration (a "what if she'd been full-time the whole career" scenario).

Survivor benefit: when the pensioner dies (``death_year``), the surviving
spouse continues to receive ``survivor_fraction`` of the COLA-adjusted
benefit until simulation end.
"""

from __future__ import annotations

import numpy as np

from ..config.schema import DefinedBenefitAccount, Person, Scenario


def compute_pension_stream(
    scenario: Scenario,
    account: DefinedBenefitAccount,
    person: Person,
    salary_history: np.ndarray,
    *,
    n_years: int,
    start_year: int,
) -> np.ndarray:
    """Return annual pension income shape [n_years] (deterministic).

    ``salary_history`` is a 1-D nominal salary by year for the pensioner
    across the simulation horizon; we use the last `fas_years` years of
    employed service to compute FAS in steady mode.
    """
    fas = _compute_fas(account, person, salary_history, start_year=start_year)
    yos = max(0, account.service_end_year - account.service_start_year)
    base_benefit = account.formula_multiplier * yos * fas

    stream = np.zeros(n_years, dtype=np.float64)
    retirement_year = max(account.service_end_year, person.retirement_year)
    death_year = person.death_year

    for t in range(n_years):
        sim_year = start_year + t
        if sim_year < retirement_year:
            continue
        years_in_retirement = sim_year - retirement_year
        cola_factor = (1.0 + account.cola_rate) ** years_in_retirement
        benefit = base_benefit * cola_factor
        if death_year is not None and sim_year >= death_year:
            benefit *= account.survivor_fraction
        stream[t] = benefit
    return stream


def _compute_fas(
    account: DefinedBenefitAccount,
    person: Person,
    salary_history: np.ndarray,
    *,
    start_year: int,
) -> float:
    """Compute Final Average Salary for the pension formula.

    ``salary_history`` is indexed by year offset from ``start_year``.
    """
    if person.wife_phase == "sprint":
        # T2: substitute the FTE salary for FAS across the full service duration.
        return float(person.fte_salary_2026 or person.salary_2026)

    end_offset = account.service_end_year - start_year
    fas_years = account.fas_years
    if end_offset <= 0 or fas_years <= 0:
        return float(person.salary_2026)

    lo = max(0, end_offset - fas_years)
    hi = max(lo + 1, min(len(salary_history), end_offset))
    window = salary_history[lo:hi]
    if window.size == 0:
        return float(person.salary_2026)
    return float(window.mean())
