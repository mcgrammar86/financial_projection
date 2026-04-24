"""Vectorized tax aggregator called once per simulated year."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config.schema import Scenario
from . import federal, metro_shs, oregon


@dataclass
class TaxYearResult:
    federal: np.ndarray
    oregon: np.ndarray
    metro_shs: np.ndarray
    total: np.ndarray
    deferred_state_credit_next_year: np.ndarray  # 529 credit


def compute_taxes(
    scenario: Scenario,
    *,
    sim_year: int,
    ordinary_income: np.ndarray,
    ltcg_income: np.ndarray,
    deferred_state_credit_in: np.ndarray,
    contributed_to_529_this_year: np.ndarray,
) -> TaxYearResult:
    inflation_factor = (1.0 + scenario.tax.inflation_rate) ** (sim_year - 2026)

    fed = federal.compute_federal_tax(
        ordinary_income, ltcg_income, inflation_factor=inflation_factor
    )
    or_tax = oregon.compute_oregon_tax(
        ordinary_income + ltcg_income,
        inflation_factor=inflation_factor,
        deferred_credit=deferred_state_credit_in,
    )
    shs = metro_shs.compute_metro_shs(
        ordinary_income + ltcg_income,
        threshold_2026=scenario.tax.metro_shs_threshold_2026,
        rate=scenario.tax.metro_shs_rate,
        inflation_factor=inflation_factor,
    )
    total = fed + or_tax + shs

    # 529 credit: 100% of contribution up to $360 (joint), carried forward.
    next_credit = np.minimum(contributed_to_529_this_year, scenario.tax.state_credit_529_max)

    return TaxYearResult(
        federal=fed,
        oregon=or_tax,
        metro_shs=shs,
        total=total,
        deferred_state_credit_next_year=next_credit,
    )
