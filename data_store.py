import os
import hashlib
import sqlite3
from typing import Tuple, List, Optional

import pandas as pd


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def init_db(db_path: str) -> None:
    ensure_parent_dir(db_path)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id TEXT PRIMARY KEY,
                date TEXT,
                description TEXT,
                original_description TEXT,
                amount REAL,
                currency TEXT,
                fx_rate REAL,
                amount_jpy REAL,
                transaction_type TEXT,
                category TEXT,
                subcategory TEXT,
                timestamp TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date)
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS imports (
                import_id TEXT PRIMARY KEY,
                source_filename TEXT,
                imported_at TEXT
            )
            """
        )
        conn.commit()


def normalize_str(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def compute_txn_id(date_str: str, description: str, amount: float) -> str:
    key = f"{pd.to_datetime(date_str).date().isoformat()}|{normalize_str(description)}|{amount:.2f}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def df_with_ids(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2['id'] = df2.apply(
        lambda r: compute_txn_id(str(r['date']), str(r.get('description', '')), float(r['amount'])), axis=1
    )
    return df2


def existing_ids(db_path: str) -> set:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM transactions")
        return {row[0] for row in cur.fetchall()}


def find_duplicates_against_store(db_path: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ids_in_db = existing_ids(db_path)
    df_ids = df_with_ids(df)
    dup_mask = df_ids['id'].isin(ids_in_db)
    return df_ids[dup_mask].copy(), df_ids[~dup_mask].copy()


def insert_transactions(db_path: str, df_with_id: pd.DataFrame) -> int:
    if df_with_id.empty:
        return 0
    with sqlite3.connect(db_path) as conn:
        records = [
            (
                r['id'],
                pd.to_datetime(r['date']).date().isoformat() if pd.notna(r['date']) else None,
                r.get('description'),
                r.get('original_description'),
                float(r.get('amount', 0)),
                r.get('currency', 'JPY'),
                float(r.get('fx_rate', 1) or 1),
                float(r.get('amount_jpy', r.get('amount', 0))),
                r.get('transaction_type'),
                r.get('category'),
                r.get('subcategory'),
                str(r.get('timestamp') or '')
            )
            for _, r in df_with_id.iterrows()
        ]
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT OR IGNORE INTO transactions (
                id, date, description, original_description, amount, currency, fx_rate, amount_jpy,
                transaction_type, category, subcategory, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()
        return cur.rowcount


def load_all_transactions(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM transactions ORDER BY date", conn)


def backup_database(db_path: str, backup_dir: str) -> str:
    ensure_parent_dir(os.path.join(backup_dir, "placeholder"))
    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    backup_path = os.path.join(backup_dir, f"expenses_backup_{ts}.db")
    with sqlite3.connect(db_path) as src, sqlite3.connect(backup_path) as dst:
        src.backup(dst)
    return backup_path


def export_csv(db_path: str, export_dir: str) -> str:
    ensure_parent_dir(os.path.join(export_dir, "placeholder"))
    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    out_path = os.path.join(export_dir, f"transactions_{ts}.csv")
    df = load_all_transactions(db_path)
    df.to_csv(out_path, index=False)
    return out_path

