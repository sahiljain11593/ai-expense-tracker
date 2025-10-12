import os
import tempfile
import pandas as pd
from datetime import date, timedelta

from data_store import init_db, insert_transactions, create_import_record, load_all_transactions


def test_strict_dedupe(tmp_path):
    db_dir = tmp_path / "data"
    db_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(tmp_path)
    init_db()

    batch_id = create_import_record("test.csv", 2)
    rows = [
        {
            "date": date(2024, 1, 1),
            "description": "Starbucks Shibuya",
            "original_description": "スターバックス 渋谷店",
            "amount": -500.0,
            "currency": "JPY",
            "fx_rate": 1.0,
            "amount_jpy": -500.0,
            "category": "Food",
            "subcategory": "Beverages",
            "transaction_type": "Expense",
        },
        {
            "date": date(2024, 1, 1),
            "description": "Starbucks Shibuya",
            "original_description": "スターバックス 渋谷店",
            "amount": -500.0,
            "currency": "JPY",
            "fx_rate": 1.0,
            "amount_jpy": -500.0,
            "category": "Food",
            "subcategory": "Beverages",
            "transaction_type": "Expense",
        },
    ]

    inserted, dupes, _ = insert_transactions(rows, batch_id)
    assert inserted == 1
    assert dupes == 1

    inserted2, dupes2, _ = insert_transactions(rows, batch_id)
    assert inserted2 == 0
    assert dupes2 == 2


def test_recurring_generation_does_not_break(tmp_path):
    # This is a smoke test that ensures list/load works and we can insert generated rows
    os.chdir(tmp_path)
    init_db()

    # Insert an existing transaction to collide later
    batch_id = create_import_record("seed.csv", 1)
    seed_row = [{
        "date": date.today(),
        "description": "Recurring: Netflix",
        "original_description": "Recurring: Netflix",
        "amount": -1000.0,
        "currency": "JPY",
        "fx_rate": 1.0,
        "amount_jpy": -1000.0,
        "category": "Entertainment",
        "subcategory": "Streaming",
        "transaction_type": "Expense",
    }]
    insert_transactions(seed_row, batch_id)

    # Ensure DB has at least 1 row
    rows = load_all_transactions()
    assert len(rows) >= 1
