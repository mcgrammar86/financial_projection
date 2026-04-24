# Financial Twin v2.0

Monte Carlo household financial simulator. Projects net worth and Safe
Withdrawal Rates for an Oregon/Clackamas household with Oregon PERS,
TCDRS, 401k/457/HSA, 529s, and a mortgage. Renders an interactive
Streamlit dashboard.

The engine is organized by **account behavior** (how money acts), not by
account location: one growth function handles all "Fixed-Rate Deferred"
accounts (TCDRS, HYSA), another handles "Stochastic Tax-Advantaged"
(401k/457/HSA/Brokerage), etc.

## Install

```sh
pip install -e ".[test]"
```

## Run the simulator

```sh
python -m financial_twin.simulate.runner configs/sample.yaml
```

## Launch the dashboard

```sh
streamlit run src/financial_twin/dashboard/app.py -- --config configs/sample.yaml
```

## Tests

```sh
python -m pytest -q
```

The four required gate tests (T1–T4) are in `tests/test_t1_*` through
`test_t4_*` and must pass before any deploy.

## Architecture summary

- **NumPy** carries the hot loop; arrays are shaped
  `[n_runs, n_years, n_accounts]`.
- **Polars** appears only in `Results.to_polars()` for analytics.
- **Streamlit** caches results by `Scenario.config_hash`; sliders
  operate on the cached `Results` object instead of re-running.
- **Behaviors**: `fixed_deferred`, `stochastic_market`,
  `education_529_glidepath`, `defined_benefit`. Defined Benefit has no
  balance — it's a deterministic income stream.

## Locked design decisions

- Returns: per-account asset allocation (stocks/bonds weights) drawing
  from a single Cholesky-correlated market.
- Inflation: deterministic 2.5%; indexes brackets, SHS threshold,
  contribution limits, expense lines.
- Withdrawal order: Brokerage → Pre-tax → Roth/HSA.
- PERS: OPSRP General `1.5% × YOS × FAS5` with 2%-capped COLA and
  configurable survivor fraction.
- Social Security: per-person PIA + claim age (62/67/70); standard
  early/delayed factors anchored at FRA = 67.
- RMDs: enforced at age 73 from pre-tax accounts.
- Healthcare bridge: configurable annual premium pre-65 + post-65
  Medicare supplement.
- Brokerage capital gains: average-basis approximation.

## Known simplifications

- No stochastic mortality; survivor benefit triggered by configured
  `death_year`.
- No ACA premium-tax-credit math; healthcare bridge is a flat expense.
- No Roth conversion modeling.
- Brokerage uses average-basis (not lot-level FIFO).
- No state-residency change mid-retirement.
- 529 surplus auto-rolls to Brokerage; shortfalls drawn from cashflow.
- HSA QME draw is a config'd annual figure, not health-cost-modeled.
- Pension COLA capped at 2%; no Tier 1/2 support.
