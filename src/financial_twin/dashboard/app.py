"""Streamlit entry point.

Usage:
  streamlit run src/financial_twin/dashboard/app.py -- --config configs/sample.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import streamlit as st

from financial_twin.config.loader import load_yaml
from financial_twin.simulate import results as result_mod
from financial_twin.dashboard import cache, charts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sample.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Financial Twin", layout="wide")
    st.title("The Financial Twin")

    config_path = Path(args.config)
    scenario = load_yaml(config_path)
    st.caption(f"Scenario: `{config_path.name}`  ·  config_hash: `{scenario.config_hash[:12]}`")

    results = cache.cached_run(scenario)

    overview_tab, balances_tab = st.tabs(["Overview", "Account balances"])

    with overview_tab:
        sim_year = st.slider(
            "Year to scrutinize",
            scenario.simulation.start_year,
            scenario.simulation.end_year - 1,
            scenario.simulation.start_year,
        )
        year_idx = sim_year - scenario.simulation.start_year

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Income flow")
            st.plotly_chart(charts.income_sankey(results, year_idx), use_container_width=True)
        with col2:
            st.subheader("Next dollar")
            st.plotly_chart(charts.next_dollar_pie(results, year_idx), use_container_width=True)

        st.subheader("Net worth fan (P05/P10/P50/P90/P95)")
        fan = result_mod.net_worth_fan(results)
        st.plotly_chart(charts.fan_chart(fan), use_container_width=True)

        st.subheader("Safe Withdrawal Rate heatmap")
        st.plotly_chart(charts.swr_heatmap(results), use_container_width=True)

        swr = result_mod.safe_withdrawal_rate(results, success_threshold=0.95)
        terminal = results.terminal_net_worth
        p10, p50, p90 = np.percentile(terminal, [10, 50, 90])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Success-95 SWR", f"{swr*100:.2f}%")
        c2.metric("P10 terminal", f"${p10:,.0f}")
        c3.metric("P50 terminal", f"${p50:,.0f}")
        c4.metric("P90 terminal", f"${p90:,.0f}")

    with balances_tab:
        percentile = st.select_slider(
            "Percentile across runs",
            options=[10, 25, 50, 75, 90],
            value=50,
            help="P50 = median across the Monte Carlo runs.",
        )
        st.subheader(f"Per-account balance over time (P{percentile})")
        st.plotly_chart(
            charts.account_balances_stacked(results, percentile),
            use_container_width=True,
        )

        # Year-end balances table. ``balance_history()`` returns the clean
        # snapshots (initial + step_accounts outputs) so values aren't
        # corrupted by next-year in-place withdrawal mutations.
        st.subheader(f"Year-end balances (P{percentile})")
        history = results.balance_history()  # [n_runs, n_years+1, n_accounts]
        # Drop t=0 (= start-of-sim initial balances) so each row labels the year that ended.
        end_pct = np.percentile(history[:, 1:, :], percentile, axis=0)  # [n_years, n_accounts]
        names_by_idx = [""] * end_pct.shape[1]
        for name, idx in results.account_idx_map.items():
            names_by_idx[idx] = name
        end_years = np.arange(
            scenario.simulation.start_year,
            scenario.simulation.start_year + end_pct.shape[0],
        )
        cols = {"Year": end_years}
        for idx, name in enumerate(names_by_idx):
            cols[name] = end_pct[:, idx]
        cols["Total"] = end_pct.sum(axis=1)
        df = pl.DataFrame(cols)
        money_cols = [c for c in df.columns if c != "Year"]
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                c: st.column_config.NumberColumn(format="$%.0f") for c in money_cols
            },
        )

        # Per-account drill-down: starting balance + contribution -
        # withdrawal + growth = ending balance, with transparent year-by-year math.
        st.subheader("Per-account activity (drill-down)")
        chosen = st.selectbox("Account", names_by_idx, index=0)
        cidx = names_by_idx.index(chosen)
        starts = np.percentile(history[:, :-1, cidx], percentile, axis=0)
        ends = np.percentile(history[:, 1:, cidx], percentile, axis=0)
        contribs = np.percentile(results.state.contributions[:, :, cidx], percentile, axis=0)
        withdraws = np.percentile(results.state.withdrawals[:, :, cidx], percentile, axis=0)
        growths = np.percentile(results.state.growth[:, :, cidx], percentile, axis=0)
        activity = pl.DataFrame({
            "Year": end_years,
            "Start balance": starts,
            "Contribution": contribs,
            "Withdrawal": withdraws,
            "Growth": growths,
            "End balance": ends,
        })
        st.dataframe(
            activity,
            use_container_width=True,
            column_config={
                c: st.column_config.NumberColumn(format="$%.0f")
                for c in activity.columns if c != "Year"
            },
        )
        st.caption(
            "Note: each column is the per-year P-percentile across runs computed "
            "independently, so Start + Contribution − Withdrawal + Growth may not "
            "exactly equal End for any single Monte Carlo path."
        )


if __name__ == "__main__":
    main()
