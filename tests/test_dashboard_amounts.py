"""Tests for dashboard amount normalization."""

import pandas as pd

from dashboard import _expense_rows, _income_rows


def test_dashboard_uses_transaction_type_and_amount_jpy():
    df = pd.DataFrame(
        [
            {"amount": 12.0, "amount_jpy": 1800.0, "transaction_type": "Expense"},
            {"amount": 2000.0, "amount_jpy": 300000.0, "transaction_type": "Credit"},
        ]
    )

    expenses = _expense_rows(df)
    income = _income_rows(df)

    assert expenses["_spend_amount"].sum() == 1800.0
    assert income["_spend_amount"].sum() == 300000.0
