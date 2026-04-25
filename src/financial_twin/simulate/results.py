"""Post-simulation analytics: percentile fans, SWR, ruin year, drawdown."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .runner import Results


@dataclass
class FanChart:
    sim_years: np.ndarray
    p05: np.ndarray
    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray
    p95: np.ndarray


def net_worth_fan(results: Results) -> FanChart:
    bal = results.balance_history().sum(axis=2)  # [n_runs, n_years+1]
    sim_years = np.arange(
        results.scenario.simulation.start_year,
        results.scenario.simulation.start_year + bal.shape[1],
    )
    pcts = np.percentile(bal, [5, 10, 50, 90, 95], axis=0)
    return FanChart(sim_years, pcts[0], pcts[1], pcts[2], pcts[3], pcts[4])


def safe_withdrawal_rate(
    results: Results, *, success_threshold: float = 0.95
) -> float:
    """Estimate the constant inflation-adjusted SWR (fraction of initial
    portfolio) such that ``success_threshold`` of runs end with > $0."""
    history = results.balance_history()  # [n_runs, n_years+1, n_accounts]
    nr, nyp1, _ = history.shape
    initial = history[:, 0, :].sum(axis=1)

    # Coarse search over candidate withdrawal rates 1%..8%.
    candidates = np.linspace(0.01, 0.08, 36)
    successes = np.zeros_like(candidates)
    inflation = results.scenario.tax.inflation_rate
    n_years = nyp1 - 1
    bal_total = history.sum(axis=2)  # [n_runs, n_years+1]
    for k, w in enumerate(candidates):
        # Simulate a simple constant-real-withdrawal sweep against the
        # already-realized growth path of the aggregated portfolio.
        # This is an approximation — true SWR re-runs the engine.
        broke = np.zeros(nr, dtype=bool)
        cur = bal_total[:, 0].copy()
        for t in range(n_years):
            growth = bal_total[:, t + 1] / np.where(bal_total[:, t] > 0, bal_total[:, t], 1.0)
            cur = cur * np.where(bal_total[:, t] > 0, growth, 1.0)
            real_withdrawal = w * initial * (1.0 + inflation) ** t
            cur = cur - real_withdrawal
            broke |= cur <= 0
            cur = np.maximum(cur, 0.0)
        successes[k] = np.mean(~broke)
    successful = candidates[successes >= success_threshold]
    return float(successful.max()) if successful.size else 0.0


def ruin_year_distribution(results: Results) -> np.ndarray:
    bal = results.balance_history().sum(axis=2)
    n_years = bal.shape[1]
    ruin = np.full(bal.shape[0], n_years, dtype=int)
    for t in range(n_years):
        broke = (bal[:, t] <= 0) & (ruin == n_years)
        ruin[broke] = t
    return results.scenario.simulation.start_year + ruin


def max_drawdown(results: Results) -> np.ndarray:
    bal = results.balance_history().sum(axis=2)
    running_max = np.maximum.accumulate(bal, axis=1)
    dd = (bal - running_max) / np.where(running_max > 0, running_max, 1.0)
    return dd.min(axis=1)
