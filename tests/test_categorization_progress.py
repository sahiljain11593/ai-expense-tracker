"""Tests for categorization session progress persistence."""

import pandas as pd

from data_store import (
    init_db,
    create_categorization_session,
    save_categorization_progress,
    load_categorization_progress,
    normalize_date_for_db,
)


def test_normalize_date_for_db_handles_pandas_timestamp():
    ts = pd.Timestamp("2025-08-16")
    assert normalize_date_for_db(ts) == "2025-08-16"


def test_save_categorization_progress_with_pandas_timestamp(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    session_id = create_categorization_session("sample.csv", 1, db_path=db_path)

    save_categorization_progress(
        session_id,
        {
            "date": pd.Timestamp("2025-08-16"),
            "description": "STARBUCKS COFFEE",
            "amount": -450.0,
            "category": "Food",
            "subcategory": "Restaurant",
            "transaction_type": "Expense",
            "confidence_score": 0.85,
        },
        db_path=db_path,
    )

    rows = load_categorization_progress(session_id, db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["date"] == "2025-08-16"
    assert rows[0]["category"] == "Food"


def test_contextual_learning_uses_real_merchant(tmp_path):
    """Contextual merchant suggestions should be stored under the merchant name."""
    from data_store import get_connection, get_learning_suggestions, learn_from_categorization

    db_path = str(tmp_path / "learning.db")
    init_db(db_path)

    learn_from_categorization(
        description="Lawson Shibuya",
        category="Food",
        subcategory="Groceries",
        amount=850,
        date="2025-01-06",
        db_path=db_path,
    )

    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT merchant, context_key FROM merchant_context_learning ORDER BY context_key"
        ).fetchall()
    finally:
        conn.close()

    assert rows
    assert {row["merchant"] for row in rows} == {"lawson shibuya"}
    assert "amount_range" in {row["context_key"] for row in rows}

    suggestions = get_learning_suggestions(
        "Lawson Shibuya",
        amount=820,
        date="2025-01-13",
        db_path=db_path,
    )
    assert any(s["category"] == "Food" for s in suggestions)
