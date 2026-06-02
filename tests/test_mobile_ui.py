"""Tests for mobile UI helpers (no Streamlit runtime required)."""

import pandas as pd

from mobile_ui import mobile_saved_transactions_display_columns


def test_mobile_saved_columns_hides_japanese_on_compact(monkeypatch):
    import streamlit as st

    monkeypatch.setitem(st.session_state, "compact_layout", True)
    df = pd.DataFrame(
        columns=["date", "description", "original_description", "amount_jpy", "category", "transaction_type"]
    )
    cols = mobile_saved_transactions_display_columns(df)
    assert "original_description" not in cols
    assert "description" in cols


def test_mobile_saved_columns_shows_japanese_on_desktop(monkeypatch):
    import streamlit as st

    monkeypatch.setitem(st.session_state, "compact_layout", False)
    df = pd.DataFrame(
        columns=["date", "description", "original_description", "amount_jpy", "category", "transaction_type"]
    )
    cols = mobile_saved_transactions_display_columns(df)
    assert "original_description" in cols
