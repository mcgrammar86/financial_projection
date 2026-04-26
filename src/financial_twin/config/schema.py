"""Pydantic schema for the household scenario YAML.

The schema is organized by *account behavior* (how money acts), not by
account location. One schema entry can describe a whole family of
TCDRS/HYSA accounts (Fixed-Rate Deferred), a 401k/457/HSA tier
(Stochastic Tax-Advantaged), 529 plans (Restricted Education), or PERS
(Defined Benefit).
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Behavior = Literal[
    "fixed_deferred",
    "stochastic_market",
    "education_529_glidepath",
    "defined_benefit",
]
TaxType = Literal["taxable", "pre_tax", "roth", "triple_exempt"]
ClaimAge = Literal[62, 63, 64, 65, 66, 67, 68, 69, 70]
WifePhase = Literal["steady", "sprint"]


class PretaxBenefit(BaseModel):
    """Pre-tax payroll deduction (health insurance, dental, FSA, etc.).

    Reduces both take-home cash AND taxable income for federal/Oregon
    purposes (Section 125 cafeteria-plan style). Stops at the owner's
    retirement_year by default since employer-sponsored benefits typically
    end with employment.
    """
    model_config = ConfigDict(extra="forbid")
    name: str
    amount_2026: float
    growth: float = 0.05  # healthcare inflation default
    stops_at_retirement: bool = True


class Person(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    birth_year: int
    retirement_year: int
    salary_2026: float = 0.0
    fte_salary_2026: float | None = None  # used for High-3 sprint
    salary_growth: float = 0.025
    pia_monthly: float = 0.0  # Social Security PIA at FRA (67), in 2026 dollars
    ss_claim_age: ClaimAge = 67
    death_year: int | None = None  # for survivor-benefit modeling
    wife_phase: WifePhase = "steady"  # only meaningful for the pensioner
    pretax_benefits: list[PretaxBenefit] = Field(default_factory=list)


class FixedDeferredAccount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    behavior: Literal["fixed_deferred"] = "fixed_deferred"
    balance_2026: float
    annual_rate: float
    is_taxable: bool = False
    contribution_2026: float = 0.0
    contribution_growth: float = 0.025
    # TCDRS-style: interest credited Dec 31 on Jan 1 balance only.
    interest_on_jan1_only: bool = True


class StochasticAccount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    behavior: Literal["stochastic_market"] = "stochastic_market"
    balance_2026: float
    tax_type: TaxType
    contribution_limit_2026: float
    contribution_2026: float = 0.0
    employer_match_2026: float = 0.0  # flat $ match (used when employer_match_pct == 0)
    employer_match_pct: float = 0.0  # match expressed as fraction of owner's salary
    employer_match_max_pct: float = 0.0  # optional cap on % of salary that gets matched
    vehicle_group: str | None = None  # accounts in the same group share an IRS limit
    stocks_weight: float = Field(0.7, ge=0.0, le=1.0)  # initial / static weight
    # Target-date glide path: when both glide fields are set, the engine
    # linearly interpolates `stocks_weight` -> `stocks_weight_glide_target`
    # from `start_year` to `stocks_weight_glide_target_year` (defaults to
    # the owner's retirement_year), and holds at the target afterward.
    stocks_weight_glide_target: float | None = Field(default=None, ge=0.0, le=1.0)
    stocks_weight_glide_target_year: int | None = None
    stocks_mean: float = 0.07
    stocks_stdev: float = 0.18
    bonds_mean: float = 0.03
    bonds_stdev: float = 0.06
    waterfall_priority: int = 100  # lower = filled first; HSA(1) -> 457(2) -> 401k(3)
    owner: str  # Person.name


class Education529Account(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    behavior: Literal["education_529_glidepath"] = "education_529_glidepath"
    balance_2026: float
    contribution_2026: float = 0.0
    contribution_growth: float = 0.025
    beneficiary_birth_year: int
    college_start_age: int = 18
    annual_tuition_2026: float = 25_000.0
    college_years: int = 4
    glide_aggressive_rate: float = 0.07  # young child
    glide_conservative_rate: float = 0.025  # at age 18
    state_credit_max: float = 360.0  # Oregon 2026 joint filer max


class DefinedBenefitAccount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    behavior: Literal["defined_benefit"] = "defined_benefit"
    owner: str
    formula_multiplier: float = 0.015  # OPSRP General
    fas_years: int = 5
    service_start_year: int
    service_end_year: int
    cola_rate: float = 0.02  # PERS COLA (2% capped)
    survivor_fraction: float = 0.5


AccountSpec = Annotated[
    Union[
        FixedDeferredAccount,
        StochasticAccount,
        Education529Account,
        DefinedBenefitAccount,
    ],
    Field(discriminator="behavior"),
]


class MortgageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    principal_remaining_2026: float
    annual_rate: float
    monthly_payment: float  # P&I only
    payoff_year: int
    # When the mortgage is paid off, half of the former annual P&I
    # increases lifestyle expenses; the other half is automatically
    # contributed to the named taxable Brokerage account.
    lifestyle_split_brokerage_account: str


class HouseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assessed_value_2026: float
    market_value_2026: float
    measure50_growth_cap: float = 0.03
    millage_rate: float = 0.015  # combined Clackamas-area rate, configurable
    solar_premium_excluded_av: float = 0.0


class TaxSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filing_status: Literal["mfj", "single"] = "mfj"
    inflation_rate: float = 0.025
    metro_shs_threshold_2026: float = 205_000.0
    metro_shs_rate: float = 0.01
    state_credit_529_max: float = 360.0
    federal_ltcg_rate: float = 0.15  # broad-bracket simplification
    state_ltcg_as_ordinary: bool = True


class HealthcareSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bridge_annual_premium_2026: float = 0.0  # ACA-bridge expense pre-65
    medicare_supplement_annual_2026: float = 0.0  # post-65


class ExpensesSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    standard_annual_2026: float
    discretionary_annual_2026: float = 0.0


class SimulationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_runs: int = 1000
    start_year: int = 2026
    end_year: int | None = None  # if None, derived from horizon_to_age
    horizon_to_age: int = 95
    seed: int = 20260424
    stocks_bonds_correlation: float = 0.05
    rmd_start_age: int = 73


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    people: list[Person]
    accounts: list[AccountSpec]
    mortgage: MortgageSpec | None = None
    house: HouseSpec | None = None
    expenses: ExpensesSpec
    healthcare: HealthcareSpec = Field(default_factory=HealthcareSpec)
    tax: TaxSpec = Field(default_factory=TaxSpec)
    simulation: SimulationSpec = Field(default_factory=SimulationSpec)

    @model_validator(mode="after")
    def _derive_end_year(self) -> "Scenario":
        if self.simulation.end_year is None:
            youngest_birth = min(p.birth_year for p in self.people)
            self.simulation.end_year = youngest_birth + self.simulation.horizon_to_age
        if self.simulation.end_year < self.simulation.start_year + 1:
            raise ValueError("end_year must be at least start_year+1")
        return self

    @model_validator(mode="after")
    def _validate_account_owners(self) -> "Scenario":
        """Every account that declares ``owner`` must name a real Person.

        Without this, ``planned_contributions`` and the pension stream
        builder silently treat orphaned accounts as having no owner --
        zeroing all employee contributions for that account. The most
        common cause is renaming a Person without updating the
        ``owner:`` lines on their accounts.
        """
        person_names = {p.name for p in self.people}
        for spec in self.accounts:
            owner = getattr(spec, "owner", None)
            if owner is None:
                continue
            if owner not in person_names:
                raise ValueError(
                    f"account '{spec.name}' references unknown owner "
                    f"'{owner}'; valid Person names are {sorted(person_names)}."
                )
        return self

    @model_validator(mode="after")
    def _enforce_vehicle_group_limits(self) -> "Scenario":
        """Stochastic accounts in the same `vehicle_group` share an IRS cap.

        All members must declare the same `contribution_limit_2026`, and the
        sum of `contribution_2026` across the group must not exceed it.
        Employer match is excluded from the participant cap.
        """
        groups: dict[str, list[StochasticAccount]] = {}
        for spec in self.accounts:
            if spec.behavior != "stochastic_market":
                continue
            assert isinstance(spec, StochasticAccount)
            if spec.vehicle_group is None:
                continue
            groups.setdefault(spec.vehicle_group, []).append(spec)
        for group_name, members in groups.items():
            limits = {m.contribution_limit_2026 for m in members}
            if len(limits) > 1:
                raise ValueError(
                    f"vehicle_group '{group_name}': members declare conflicting "
                    f"contribution_limit_2026 values {sorted(limits)}; they must match."
                )
            limit = next(iter(limits))
            total = sum(m.contribution_2026 for m in members)
            if total > limit + 1e-9:
                names = ", ".join(m.name for m in members)
                raise ValueError(
                    f"vehicle_group '{group_name}': combined contribution_2026 "
                    f"({total:.2f}) exceeds shared IRS limit ({limit:.2f}) "
                    f"across [{names}]."
                )
        return self

    @field_validator("people")
    @classmethod
    def _names_unique(cls, v: list[Person]) -> list[Person]:
        names = [p.name for p in v]
        if len(set(names)) != len(names):
            raise ValueError("Person names must be unique")
        return v

    @property
    def n_years(self) -> int:
        assert self.simulation.end_year is not None
        return self.simulation.end_year - self.simulation.start_year

    @property
    def config_hash(self) -> str:
        """Stable hash of the scenario for Streamlit caching."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
