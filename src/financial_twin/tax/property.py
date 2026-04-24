"""Oregon Measure 50 property tax.

Assessed Value (AV) grows at most ``measure50_growth_cap`` (3%) per
year. The solar premium is excluded from AV. Annual property tax is
``av * millage_rate``.
"""

from __future__ import annotations

from ..config.schema import HouseSpec


def project_assessed_values(
    house: HouseSpec | None, n_years: int
) -> tuple[list[float], list[float]]:
    """Return (assessed_value_by_year, property_tax_by_year)."""
    if house is None:
        return [0.0] * (n_years + 1), [0.0] * n_years
    av = max(0.0, house.assessed_value_2026 - house.solar_premium_excluded_av)
    avs = [av]
    taxes = []
    for _ in range(n_years):
        taxes.append(av * house.millage_rate)
        av = av * (1.0 + house.measure50_growth_cap)
        avs.append(av)
    return avs, taxes
