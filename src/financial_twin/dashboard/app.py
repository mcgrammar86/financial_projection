"""Streamlit entry point.

Usage:
  streamlit run src/financial_twin/dashboard/app.py -- --config configs/sample.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import streamlit as st

from ..config.loader import load_yaml
from ..simulate import results as result_mod
from . import cache, charts


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


if __name__ == "__main__":
    main()
