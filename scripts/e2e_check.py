#!/usr/bin/env python3
"""
Minimal end-to-end smoke test without authentication:
 - Initialize DB
 - Load sample CSV
 - Normalize currency (JPY), detect type
 - Insert with dedupe, report counts
 - Export CSV and backup DB
"""

import os
import sys
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_store import init_db, create_import_record, insert_transactions, export_transactions_to_csv, backup_database
from transaction_web_app import detect_transaction_type, enrich_currency_columns


def main():
    init_db()
    sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bank", "statements", "enavi202509(5734) (2).csv")
    if not os.path.exists(sample_path):
        print("Sample CSV not found:", sample_path)
        return 1

    df = pd.read_csv(sample_path)
    # Handle Japanese CSV format - map known columns
    # Expected columns: "利用日","利用店名・商品名","利用者","支払方法","利用金額","支払手数料","支払総額","9月支払金額","10月繰越残高","新規サイン"
    # Map to: date, description, amount
    date_col = "利用日"  # Usage date
    desc_col = "利用店名・商品名"  # Store name/product name
    amount_col = "利用金額"  # Usage amount
    
    df2 = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce").dt.date,
        "description": df[desc_col].astype(str),
        "original_description": df[desc_col].astype(str),
        "amount": pd.to_numeric(df[amount_col], errors="coerce"),
    }).dropna(subset=["date", "description", "amount"])[:50]

    df2 = enrich_currency_columns(df2, "JPY")
    df2 = detect_transaction_type(df2)
    batch_id = create_import_record(os.path.basename(sample_path), len(df2))
    rows = []
    for _, r in df2.iterrows():
        rows.append({
            "date": r.get("date"),
            "description": r.get("description"),
            "original_description": r.get("original_description"),
            "amount": float(r.get("amount", 0.0)),
            "currency": r.get("currency", "JPY"),
            "fx_rate": float(r.get("fx_rate", 1.0)),
            "amount_jpy": float(r.get("amount_jpy", r.get("amount", 0.0))),
            "category": r.get("category"),
            "subcategory": r.get("subcategory"),
            "transaction_type": r.get("transaction_type"),
        })
    inserted, dupes, _ = insert_transactions(rows, batch_id)
    print(f"Inserted: {inserted}, Duplicates: {dupes}")
    csv_path = export_transactions_to_csv()
    print("Exported CSV:", csv_path)
    bkp_path = backup_database()
    print("Backup DB:", bkp_path)
    print("E2E check completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

