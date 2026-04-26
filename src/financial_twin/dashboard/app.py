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
from financial_twin.engine.cashflow import (
    compute_cashflow_year,
    planned_529_withdrawals,
    pretax_benefits_for_year,
    salary_for_year,
)
from financial_twin.engine.mortgage import amortize
from financial_twin.engine.withdrawals import classify_accounts
from financial_twin.simulate import results as result_mod
from financial_twin.dashboard import cache, charts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sample.yaml")
    return parser.parse_args()


def _build_cashflow_table(scenario, results, percentile: int) -> pl.DataFrame:
    """One row per simulated year, all cashflow categories side-by-side.

    Deterministic columns (salary, pension, SS, expenses, mortgage,
    property tax, 529 withdrawals) are the same across all runs.
    Stochastic columns (drawdowns, taxes) are taken at the requested
    percentile across runs.

    The Surplus/Gap column reconciles money in vs money out:
      In  = Salary + Pension + SS + Drawn (pretax) + Drawn (taxable) + Drawn (Roth/HSA)
      Out = Standard + Discretionary + Healthcare + Mortgage + PropertyTax
            + Federal + Oregon + MetroSHS + EmployeeContribs + PretaxBenefits
      Surplus = In - Out
    The runner iterates cash_need ↔ withdraw ↔ tax until tax converges,
    so at P50 the surplus should be ~0 every year. Non-zero residuals at
    other percentiles are from the percentile mix (each column percentile
    computed independently), not engine error.
    """
    state = results.state
    ny = scenario.n_years
    start_year = scenario.simulation.start_year
    sim_years = np.arange(start_year, start_year + ny)

    mortgage = amortize(scenario.mortgage, start_year=start_year, n_years=ny)
    classification = classify_accounts(scenario, results.account_idx_map)

    # --- Income (deterministic) ---
    salary = np.array(
        [sum(salary_for_year(p, start_year + t) for p in scenario.people) for t in range(ny)],
        dtype=np.float64,
    )
    pension = np.array(
        [sum(stream[t] for stream in results.pension_streams.values()) for t in range(ny)],
        dtype=np.float64,
    )
    ss = np.array(
        [sum(stream[t] for stream in results.ss_streams.values()) for t in range(ny)],
        dtype=np.float64,
    )

    # --- Withdrawals split by tax tier (stochastic across runs) ---
    def _pct_sum(idxs: list[int]) -> np.ndarray:
        if not idxs:
            return np.zeros(ny, dtype=np.float64)
        per_run_per_year = state.withdrawals[:, :, idxs].sum(axis=2)  # [n_runs, ny]
        return np.percentile(per_run_per_year, percentile, axis=0)

    pretax_drawn = _pct_sum(classification.pretax_idx)
    brokerage_drawn = _pct_sum(classification.brokerage_idx)
    roth_drawn = _pct_sum(classification.roth_or_hsa_idx)

    # --- Expenses + mortgage + pretax benefits (deterministic) ---
    standard = np.zeros(ny, dtype=np.float64)
    discretionary = np.zeros(ny, dtype=np.float64)
    healthcare = np.zeros(ny, dtype=np.float64)
    mortgage_pi = np.zeros(ny, dtype=np.float64)
    lifestyle_split = np.zeros(ny, dtype=np.float64)
    pretax_benefits = np.zeros(ny, dtype=np.float64)
    for t in range(ny):
        cf = compute_cashflow_year(scenario, start_year + t, mortgage, year_index=t)
        standard[t] = cf.standard_expenses
        discretionary[t] = cf.discretionary_expenses
        healthcare[t] = cf.healthcare_expense
        mortgage_pi[t] = cf.mortgage_p_and_i
        lifestyle_split[t] = cf.lifestyle_split_brokerage_contrib
        pretax_benefits[t] = sum(
            pretax_benefits_for_year(p, start_year + t) for p in scenario.people
        )

    # --- Taxes (stochastic except property which is deterministic) ---
    federal = np.percentile(state.tax_breakdown["federal"], percentile, axis=0)
    oregon = np.percentile(state.tax_breakdown["oregon"], percentile, axis=0)
    shs = np.percentile(state.tax_breakdown["metro_shs"], percentile, axis=0)
    property_tax = np.percentile(state.tax_breakdown["property"], percentile, axis=0)

    # --- Contributions (sum across all accounts) ---
    employee = np.percentile(state.contributions.sum(axis=2), percentile, axis=0)
    match = np.percentile(state.employer_match.sum(axis=2), percentile, axis=0)

    # --- 529 outflows (deterministic) ---
    w529 = np.zeros(ny, dtype=np.float64)
    for t in range(ny):
        for _, amt in planned_529_withdrawals(scenario, start_year + t).items():
            w529[t] += amt

    # --- Reconciliation ---
    income_in = salary + pension + ss + pretax_drawn + brokerage_drawn + roth_drawn
    cash_out = (
        standard + discretionary + healthcare + mortgage_pi + property_tax
        + federal + oregon + shs
        + employee
        + pretax_benefits
    )
    surplus = income_in - cash_out

    return pl.DataFrame({
        "Year": sim_years,
        "Salary": salary,
        "Pension": pension,
        "SocSec": ss,
        "Drawn pretax": pretax_drawn,
        "Drawn taxable": brokerage_drawn,
        "Drawn roth/HSA": roth_drawn,
        "Income IN": income_in,
        "Standard exp": standard,
        "Discretionary": discretionary,
        "Healthcare": healthcare,
        "Mortgage P&I": mortgage_pi,
        "Property tax": property_tax,
        "Pretax benefits": pretax_benefits,
        "Federal tax": federal,
        "Oregon tax": oregon,
        "Metro SHS": shs,
        "Employee contrib": employee,
        "Employer match": match,
        "529 outflow": w529,
        "Cash OUT": cash_out,
        "Surplus (IN-OUT)": surplus,
    })


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Financial Twin", layout="wide")
    st.title("The Financial Twin")

    config_path = Path(args.config)
    scenario = load_yaml(config_path)
    st.caption(f"Scenario: `{config_path.name}`  ·  config_hash: `{scenario.config_hash[:12]}`")

    results = cache.cached_run(scenario)

    overview_tab, balances_tab, cashflow_tab = st.tabs(
        ["Overview", "Account balances", "Yearly cashflow"]
    )

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
        employee = np.percentile(results.state.contributions[:, :, cidx], percentile, axis=0)
        match = np.percentile(results.state.employer_match[:, :, cidx], percentile, axis=0)
        withdraws = np.percentile(results.state.withdrawals[:, :, cidx], percentile, axis=0)
        growths = np.percentile(results.state.growth[:, :, cidx], percentile, axis=0)
        activity = pl.DataFrame({
            "Year": end_years,
            "Start balance": starts,
            "Employee contrib": employee,
            "Employer match": match,
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
            "Employee contrib = exactly what you set as `contribution_2026` "
            "(inflated each year); Employer match is computed from "
            "salary × match_pct. Each column is the per-year P-percentile "
            "across runs computed independently, so Start + Employee + Match "
            "− Withdrawal + Growth may not exactly equal End for any single "
            "Monte Carlo path."
        )

    with cashflow_tab:
        _render_cashflow_tab(scenario, results)


def _render_cashflow_tab(scenario, results) -> None:
    percentile = st.select_slider(
        "Percentile across runs",
        options=[10, 25, 50, 75, 90],
        value=50,
        key="cashflow_percentile",
        help="P50 = median across the Monte Carlo runs. Salary, pension, "
        "SS, expenses, mortgage, property tax, and 529 outflows are "
        "deterministic; withdrawals and income taxes vary by run.",
    )
    df = _build_cashflow_table(scenario, results, percentile)

    st.subheader(f"Year-by-year cashflow (P{percentile})")
    money_cols = [c for c in df.columns if c != "Year"]
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            c: st.column_config.NumberColumn(format="$%.0f") for c in money_cols
        },
    )

    # Single-year deep dive: vertical breakdown for the selected year.
    st.subheader("Single-year deep dive")
    pick = st.slider(
        "Year",
        scenario.simulation.start_year,
        scenario.simulation.end_year - 1,
        scenario.simulation.start_year,
        key="cashflow_year_pick",
    )
    row = df.filter(pl.col("Year") == pick).row(0, named=True)

    def _section(title: str, items: list[tuple[str, float]]) -> None:
        st.markdown(f"**{title}**")
        section_df = pl.DataFrame({
            "Item": [k for k, _ in items],
            "Amount": [v for _, v in items],
        })
        total = sum(v for _, v in items)
        section_df = section_df.vstack(pl.DataFrame({"Item": [f"{title} total"], "Amount": [total]}))
        st.dataframe(
            section_df,
            use_container_width=True,
            hide_index=True,
            column_config={"Amount": st.column_config.NumberColumn(format="$%.0f")},
        )

    col_left, col_right = st.columns(2)
    with col_left:
        _section("Income IN", [
            ("Salary", row["Salary"]),
            ("Pension", row["Pension"]),
            ("Social Security", row["SocSec"]),
            ("Drawn from pretax", row["Drawn pretax"]),
            ("Drawn from taxable (HYSA + brokerage)", row["Drawn taxable"]),
            ("Drawn from Roth/HSA", row["Drawn roth/HSA"]),
        ])
        _section("Living expenses", [
            ("Standard", row["Standard exp"]),
            ("Discretionary", row["Discretionary"]),
            ("Healthcare", row["Healthcare"]),
            ("Mortgage P&I", row["Mortgage P&I"]),
            ("Property tax", row["Property tax"]),
            ("Pretax payroll benefits", row["Pretax benefits"]),
        ])
    with col_right:
        _section("Income taxes", [
            ("Federal", row["Federal tax"]),
            ("Oregon", row["Oregon tax"]),
            ("Metro SHS", row["Metro SHS"]),
        ])
        _section("Investments / outflows", [
            ("Employee contributions (sum)", row["Employee contrib"]),
            ("Employer match (sum)", row["Employer match"]),
            ("529 tuition outflow", row["529 outflow"]),
        ])

    surplus = row["Surplus (IN-OUT)"]
    color = "green" if abs(surplus) < 1.0 else ("red" if surplus < 0 else "orange")
    st.markdown(
        f"**Reconciliation for {pick}:** "
        f"Income IN ${row['Income IN']:,.0f}  −  Cash OUT ${row['Cash OUT']:,.0f}  "
        f"= :{color}[**${surplus:,.0f} surplus**]"
    )
    if abs(surplus) > 100.0:
        # Cashflow loop iterates to convergence so any residual is from
        # the percentile mixing across columns, not from the engine.
        st.info(
            "Each column is the per-year P-percentile across runs computed "
            "independently, so the IN/OUT totals don't have to reconcile "
            "exactly at non-50 percentiles. Switch to P50 for the cleanest read."
        )


if __name__ == "__main__":
    main()
