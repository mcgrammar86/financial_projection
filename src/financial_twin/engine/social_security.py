"""Social Security benefit stream.

Per-person PIA at FRA (assumed 67) is provided in 2026 dollars; we apply
early-claim reductions / delayed retirement credits, then index annually
by the configured inflation rate. No WEP per spec.
"""

from __future__ import annotations

import numpy as np

from ..config.schema import Person


# Standard SSA reduction/credit factors anchored at FRA = 67.
_FRA = 67


def _claim_factor(claim_age: int) -> float:
    if claim_age == _FRA:
        return 1.0
    if claim_age < _FRA:
        # 5/9 of 1% per month for first 36 months, 5/12 of 1% thereafter.
        months_early = (_FRA - claim_age) * 12
        first = min(36, months_early) * (5.0 / 9.0) / 100.0
        rest = max(0, months_early - 36) * (5.0 / 12.0) / 100.0
        return 1.0 - (first + rest)
    months_late = (claim_age - _FRA) * 12
    return 1.0 + months_late * (8.0 / 12.0) / 100.0


def compute_ss_stream(
    person: Person,
    *,
    n_years: int,
    start_year: int,
    inflation_rate: float,
) -> np.ndarray:
    """Annual Social Security income for one person, shape [n_years]."""
    if person.pia_monthly <= 0:
        return np.zeros(n_years, dtype=np.float64)
    claim_year = person.birth_year + person.ss_claim_age
    factor = _claim_factor(person.ss_claim_age)
    annual_pia_2026 = person.pia_monthly * 12.0 * factor

    stream = np.zeros(n_years, dtype=np.float64)
    death_year = person.death_year
    for t in range(n_years):
        sim_year = start_year + t
        if sim_year < claim_year:
            continue
        if death_year is not None and sim_year >= death_year:
            continue  # SS stops at death; no auto-survivor in v2.0
        years_from_2026 = sim_year - 2026
        stream[t] = annual_pia_2026 * (1.0 + inflation_rate) ** years_from_2026
    return stream
