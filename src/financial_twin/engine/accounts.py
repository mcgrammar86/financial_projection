"""Account behavior dispatch.

Each behavior is a vectorized growth function that takes the global
SimState and a per-account context (config + index in the balances
array). It mutates `balances[:, year+1, idx]` in place.

Defined-benefit accounts have no balance; they're handled separately by
the pension module and skip dispatch here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from ..config.schema import (
    AccountSpec,
    Education529Account,
    FixedDeferredAccount,
    StochasticAccount,
)
from .returns import blend_account_return
from .state import SimState


@dataclass
class AccountContext:
    spec: AccountSpec
    idx: int  # column in balances array
    base_year: int  # scenario.simulation.start_year
    # Per-year stocks weight for stochastic accounts (TDF glide path).
    # None = use spec.stocks_weight as a static value.
    stocks_weight_by_year: np.ndarray | None = None


class GrowthFn(Protocol):
    def __call__(self, state: SimState, year: int, ctx: AccountContext) -> None: ...


def _fixed_deferred(state: SimState, year: int, ctx: AccountContext) -> None:
    """TCDRS / HYSA: simple compound interest on Jan 1 balance.

    Contribution lands in the year but, when ``interest_on_jan1_only`` is
    true (TCDRS), it does not earn interest until the following year.
    """
    spec: FixedDeferredAccount = ctx.spec  # type: ignore[assignment]
    jan1 = state.balances[:, year, ctx.idx]
    # Total inflow = employee contribution + employer match.
    contrib = (
        state.contributions[:, year, ctx.idx]
        + state.employer_match[:, year, ctx.idx]
    )
    rate = spec.annual_rate
    if spec.interest_on_jan1_only:
        interest = jan1 * rate
        new_balance = jan1 + interest + contrib
    else:
        # Approximate mid-year compounding: contributions earn half-year of interest.
        interest = jan1 * rate + contrib * (rate / 2.0)
        new_balance = jan1 + interest + contrib
    state.growth[:, year, ctx.idx] = interest
    state.balances[:, year + 1, ctx.idx] = new_balance


def _stochastic_market(state: SimState, year: int, ctx: AccountContext) -> None:
    """401k/457/HSA/Brokerage growth: balance * (1 + blended return).

    When ``ctx.stocks_weight_by_year`` is provided (TDF glide path), the
    stocks/bonds blend uses the year-specific weight; otherwise the
    spec's static ``stocks_weight`` is used.
    """
    spec: StochasticAccount = ctx.spec  # type: ignore[assignment]
    market = state.returns_draw[:, year, :]  # [n_runs, 2]
    if ctx.stocks_weight_by_year is not None:
        weight = float(ctx.stocks_weight_by_year[year])
    else:
        weight = spec.stocks_weight
    r = blend_account_return(market, weight)
    jan1 = state.balances[:, year, ctx.idx]
    # Total inflow = employee contribution + employer match.
    contrib = (
        state.contributions[:, year, ctx.idx]
        + state.employer_match[:, year, ctx.idx]
    )
    # Mid-year contribution convention: contributions earn half a year of return.
    growth = jan1 * r + contrib * (r * 0.5)
    state.growth[:, year, ctx.idx] = growth
    state.balances[:, year + 1, ctx.idx] = jan1 + growth + contrib


def _education_529_glidepath(state: SimState, year: int, ctx: AccountContext) -> None:
    """Linear glide path from aggressive to conservative as the child approaches 18."""
    spec: Education529Account = ctx.spec  # type: ignore[assignment]
    sim_year = ctx.base_year + year
    child_age = sim_year - spec.beneficiary_birth_year
    years_to_college = max(0, spec.college_start_age - child_age)
    horizon = max(spec.college_start_age, 1)
    fraction_aggressive = min(1.0, years_to_college / horizon)
    rate = (
        fraction_aggressive * spec.glide_aggressive_rate
        + (1.0 - fraction_aggressive) * spec.glide_conservative_rate
    )
    jan1 = state.balances[:, year, ctx.idx]
    # Total inflow = employee contribution + employer match (529 plans rarely
    # have employer match but support it for symmetry).
    contrib = (
        state.contributions[:, year, ctx.idx]
        + state.employer_match[:, year, ctx.idx]
    )
    withdraw = state.withdrawals[:, year, ctx.idx]
    growth = jan1 * rate + contrib * (rate * 0.5)
    state.growth[:, year, ctx.idx] = growth
    state.balances[:, year + 1, ctx.idx] = np.maximum(0.0, jan1 + growth + contrib - withdraw)


BEHAVIOR_REGISTRY: dict[str, Callable[[SimState, int, AccountContext], None]] = {
    "fixed_deferred": _fixed_deferred,
    "stochastic_market": _stochastic_market,
    "education_529_glidepath": _education_529_glidepath,
}


def step_accounts(state: SimState, year: int, contexts: list[AccountContext]) -> None:
    for ctx in contexts:
        beh = ctx.spec.behavior
        if beh == "defined_benefit":
            continue
        BEHAVIOR_REGISTRY[beh](state, year, ctx)
