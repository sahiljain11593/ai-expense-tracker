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


def test_insights_use_transaction_type_for_positive_expenses():
    """Japanese statements often store expenses as positive debit amounts."""
    transactions = [
        {
            "date": "2025-08-01",
            "description": "Lawson",
            "amount": 1200.0,
            "amount_jpy": 1200.0,
            "category": "Food",
            "transaction_type": "Expense",
        },
        {
            "date": "2025-08-02",
            "description": "Salary",
            "amount": 300000.0,
            "amount_jpy": 300000.0,
            "category": "Income",
            "transaction_type": "Credit",
        },
    ]

    report = InsightsEngine().generate_comprehensive_report(transactions)

    assert report["summary"]["total_expenses"] == 1200.0
    assert report["summary"]["total_income"] == 300000.0
    assert report["summary"]["net_cashflow"] == 298800.0
    assert report["category_breakdown"]["Food"]["total_spent"] == 1200.0
