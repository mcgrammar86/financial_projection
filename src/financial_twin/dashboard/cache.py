"""Streamlit caching wrappers."""

from __future__ import annotations

import streamlit as st

from ..config.schema import Scenario
from ..simulate.runner import Results, run_simulation


@st.cache_data(hash_funcs={Scenario: lambda s: s.config_hash})
def cached_run(scenario: Scenario) -> Results:
    return run_simulation(scenario)
