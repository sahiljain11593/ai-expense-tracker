"""Insights engine edge-case tests."""

import pandas as pd

from insights_engine import InsightsEngine


def test_insights_report_with_uncategorized_expenses():
    """Expenses without categories should not crash insights generation."""
    transactions = [
        {
            "date": "2025-08-01",
            "description": "Test Shop",
            "amount": -1000.0,
            "category": None,
            "transaction_type": "Expense",
        },
        {
            "date": "2025-08-02",
            "description": "Another Shop",
            "amount": -500.0,
            "category": "",
            "transaction_type": "Expense",
        },
    ]
    report = InsightsEngine().generate_comprehensive_report(transactions)
    assert "summary" in report
    assert "insights" in report
