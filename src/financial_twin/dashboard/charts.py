"""The Scrutiny Suite charts: Sankey, waterfall pie, SWR heatmap, fan."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from financial_twin.simulate.results import FanChart
from financial_twin.simulate.runner import Results


def income_sankey(results: Results, year_index: int) -> go.Figure:
    state = results.state
    salary = float(state.gross_income[0, year_index] - state.pension_income[0, year_index]
                   - state.ss_income[0, year_index]
                   - state.ltcg_income[0, year_index])
    pension = float(state.pension_income[0, year_index])
    ss = float(state.ss_income[0, year_index])
    ltcg = float(state.ltcg_income[0, year_index])
    fed = float(state.tax_breakdown["federal"][0, year_index])
    ore = float(state.tax_breakdown["oregon"][0, year_index])
    shs = float(state.tax_breakdown["metro_shs"][0, year_index])
    prop = float(state.tax_breakdown["property"][0, year_index])
    expenses = float(state.expenses[0, year_index])
    saving = max(0.0, salary + pension + ss + ltcg - fed - ore - shs - prop - expenses)

    labels = ["Salary", "Pension", "Soc Sec", "LTCG", "Income",
              "Federal", "Oregon", "SHS", "Property", "Expenses", "Saving"]
    src = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4]
    tgt = [4, 4, 4, 4, 5, 6, 7, 8, 9, 10]
    val = [salary, pension, ss, ltcg, fed, ore, shs, prop, expenses, saving]
    return go.Figure(go.Sankey(
        node=dict(label=labels, pad=15, thickness=15),
        link=dict(source=src, target=tgt, value=val),
    ))


def next_dollar_pie(results: Results, year_index: int) -> go.Figure:
    state = results.state
    fed = float(state.tax_breakdown["federal"][0, year_index])
    ore = float(state.tax_breakdown["oregon"][0, year_index])
    shs = float(state.tax_breakdown["metro_shs"][0, year_index])
    expenses = float(state.expenses[0, year_index])
    saving = max(0.0, float(state.gross_income[0, year_index]) - fed - ore - shs - expenses)
    fig = go.Figure(go.Pie(
        labels=["Federal", "Oregon", "Metro SHS", "Expenses", "Saving"],
        values=[fed, ore, shs, expenses, saving],
        hole=0.4,
    ))
    return fig


def fan_chart(fan: FanChart) -> go.Figure:
    fig = go.Figure()
    fig.add_traces([
        go.Scatter(x=fan.sim_years, y=fan.p95, name="P95", line=dict(width=0), showlegend=False),
        go.Scatter(x=fan.sim_years, y=fan.p05, name="P05",
                   fill="tonexty", line=dict(width=0)),
        go.Scatter(x=fan.sim_years, y=fan.p90, name="P90", line=dict(width=0), showlegend=False),
        go.Scatter(x=fan.sim_years, y=fan.p10, name="P10",
                   fill="tonexty", line=dict(width=0)),
        go.Scatter(x=fan.sim_years, y=fan.p50, name="P50", line=dict(width=2)),
    ])
    fig.update_layout(yaxis_title="Net worth ($)", xaxis_title="Year")
    return fig


def account_balances_stacked(results: Results, percentile: int = 50) -> go.Figure:
    """Stacked area of per-account balances over time at the given percentile.

    Bottom-most (largest terminal-balance) accounts are stacked first so the
    legend reads top-to-bottom in the same order as the visual stack.
    """
    bal = results.state.balances  # [n_runs, n_years+1, n_accounts]
    pct = np.percentile(bal, percentile, axis=0)  # [n_years+1, n_accounts]
    sim_years = np.arange(
        results.scenario.simulation.start_year,
        results.scenario.simulation.start_year + pct.shape[0],
    )
    names_by_idx = [""] * pct.shape[1]
    for name, idx in results.account_idx_map.items():
        names_by_idx[idx] = name
    order = np.argsort(-pct[-1, :])  # largest terminal balance first
    fig = go.Figure()
    for idx in order:
        fig.add_trace(go.Scatter(
            x=sim_years,
            y=pct[:, idx],
            name=names_by_idx[idx],
            stackgroup="one",
            mode="lines",
            hovertemplate="%{y:$,.0f}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        yaxis_title="Balance ($)",
        xaxis_title="Year",
        hovermode="x unified",
    )
    return fig


def swr_heatmap(results: Results) -> go.Figure:
    """SWR vs Pension Gap heatmap.

    Y-axis: candidate withdrawal rate (1%..8%).
    X-axis: simulation horizon (years from start).
    Color: probability of portfolio survival up to year t at rate w.
    """
    bal = results.state.balances.sum(axis=2)  # [n_runs, n_years+1]
    inflation = results.scenario.tax.inflation_rate
    initial = bal[:, 0]
    rates = np.linspace(0.01, 0.08, 36)
    n_years = bal.shape[1] - 1
    grid = np.zeros((len(rates), n_years), dtype=np.float64)
    for k, w in enumerate(rates):
        cur = initial.copy()
        broke = np.zeros_like(cur, dtype=bool)
        for t in range(n_years):
            growth = bal[:, t + 1] / np.where(bal[:, t] > 0, bal[:, t], 1.0)
            cur = cur * np.where(bal[:, t] > 0, growth, 1.0)
            cur = cur - w * initial * (1.0 + inflation) ** t
            broke |= cur <= 0
            grid[k, t] = float(np.mean(~broke))
    return go.Figure(go.Heatmap(
        z=grid,
        x=results.scenario.simulation.start_year + np.arange(1, n_years + 1),
        y=(rates * 100).round(1),
        colorbar=dict(title="P(success)"),
    )).update_layout(xaxis_title="Year", yaxis_title="Withdrawal rate (%)")
