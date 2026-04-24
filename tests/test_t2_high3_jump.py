"""T2 — High-3 Jump.

When ``wife_phase == "sprint"``, the pension formula uses the FTE salary
as FAS across the entire service duration, regardless of the actual
salary history. The sprint pension should equal exactly
``1.5% * total_YOS * fte_salary``.
"""

from __future__ import annotations

import math

from financial_twin.simulate.runner import run_simulation


def _make_pension_scenario(make_scenario, *, phase: str):
    return make_scenario(
        people=[
            {
                "name": "wife",
                "birth_year": 1985,
                "retirement_year": 2046,
                "salary_2026": 50_000.0,  # 20-hr "steady" actual
                "fte_salary_2026": 100_000.0,  # 40-hr "sprint" rate
                "salary_growth": 0.0,
                "pia_monthly": 0.0,
                "ss_claim_age": 67,
                "wife_phase": phase,
            },
        ],
        accounts=[
            {
                "name": "pers",
                "behavior": "defined_benefit",
                "owner": "wife",
                "formula_multiplier": 0.015,
                "fas_years": 5,
                "service_start_year": 2026,
                "service_end_year": 2046,
                "cola_rate": 0.0,
                "survivor_fraction": 0.5,
            },
        ],
        simulation={
            "n_runs": 1,
            "start_year": 2026,
            "end_year": 2050,
            "horizon_to_age": 95,
            "seed": 42,
        },
        tax={"inflation_rate": 0.0},
    )


def test_sprint_uses_fte_fas(make_scenario):
    sprint = run_simulation(_make_pension_scenario(make_scenario, phase="sprint"))
    sprint_benefit_2046 = sprint.pension_streams["pers"][2046 - 2026]
    yos = 20  # 2026 -> 2046
    expected = 0.015 * yos * 100_000.0
    assert math.isclose(sprint_benefit_2046, expected, rel_tol=0, abs_tol=1e-6)


def test_sprint_exceeds_steady(make_scenario):
    steady = run_simulation(_make_pension_scenario(make_scenario, phase="steady"))
    sprint = run_simulation(_make_pension_scenario(make_scenario, phase="sprint"))
    t = 2046 - 2026
    assert sprint.pension_streams["pers"][t] > steady.pension_streams["pers"][t]
