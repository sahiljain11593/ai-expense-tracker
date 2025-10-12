"""
SQLite data layer for the AI Expense Tracker.

Schema (initial):
  - transactions
      id INTEGER PRIMARY KEY AUTOINCREMENT
      date TEXT (YYYY-MM-DD)
      description TEXT
      original_description TEXT
      amount REAL
      currency TEXT DEFAULT 'JPY'
      fx_rate REAL DEFAULT 1.0  -- rate to convert currency -> JPY for this row/date
      amount_jpy REAL           -- amount converted to JPY using fx_rate
      category TEXT
      subcategory TEXT
      transaction_type TEXT     -- 'Expense' | 'Credit'
      import_batch_id INTEGER   -- FK to imports.id
      dedupe_hash TEXT          -- hash(date|description|amount)
      created_at TEXT           -- ISO timestamp

  - imports
      id INTEGER PRIMARY KEY AUTOINCREMENT
      file_name TEXT
      imported_at TEXT
      num_rows INTEGER

  - settings (key TEXT PRIMARY KEY, value TEXT)
  - recurring_rules
      id INTEGER PRIMARY KEY AUTOINCREMENT
      merchant_pattern TEXT NOT NULL
      frequency TEXT NOT NULL  -- 'weekly' | 'monthly'
      next_date TEXT NOT NULL  -- YYYY-MM-DD
      amount REAL              -- optional fixed amount
      category TEXT
      subcategory TEXT
      currency TEXT DEFAULT 'JPY'
      active INTEGER NOT NULL DEFAULT 1

Notes
 - This module owns dedupe logic (exact hash on date|description|amount). Advanced
   fuzzy dedupe is added later.
 - Do not log PII. Keep outputs minimal.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_DB_PATH = os.path.join("data", "expenses.db")


def _ensure_dirs() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("exports", exist_ok=True)
    os.makedirs("backups", exist_ok=True)


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    _ensure_dirs()
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        # transactions
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              date TEXT NOT NULL,
              description TEXT NOT NULL,
              original_description TEXT,
              amount REAL NOT NULL,
              currency TEXT NOT NULL DEFAULT 'JPY',
              fx_rate REAL NOT NULL DEFAULT 1.0,
              amount_jpy REAL NOT NULL,
              category TEXT,
              subcategory TEXT,
              transaction_type TEXT,
              import_batch_id INTEGER,
              dedupe_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE(dedupe_hash)
            )
            """
        )

        # imports
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS imports (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              file_name TEXT,
              imported_at TEXT NOT NULL,
              num_rows INTEGER NOT NULL
            )
            """
        )

        # settings
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
              key TEXT PRIMARY KEY,
              value TEXT
            )
            """
        )

        # recurring rules
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recurring_rules (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              merchant_pattern TEXT NOT NULL,
              frequency TEXT NOT NULL,
              next_date TEXT NOT NULL,
              amount REAL,
              category TEXT,
              subcategory TEXT,
              currency TEXT NOT NULL DEFAULT 'JPY',
              active INTEGER NOT NULL DEFAULT 1
            )
            """
        )

        conn.commit()
    finally:
        conn.close()


def compute_dedupe_hash(date_str: str, description: str, amount: float) -> str:
    # Normalize description by trimming spaces and lowering
    normalized_desc = (description or "").strip().lower()
    key = f"{date_str}|{normalized_desc}|{round(float(amount), 2)}"
    # Lightweight hash (stable and readable)
    import hashlib

    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def create_import_record(file_name: Optional[str], num_rows: int, db_path: str = DEFAULT_DB_PATH) -> int:
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO imports (file_name, imported_at, num_rows) VALUES (?, ?, ?)",
            (
                file_name or "manual",
                datetime.utcnow().isoformat(timespec="seconds"),
                int(num_rows),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def insert_transactions(
    rows: Iterable[Dict],
    import_batch_id: Optional[int] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> Tuple[int, int, List[str]]:
    """
    Insert transactions with exact-dedupe on (date|description|amount).

    Params
      - rows: iterable of dicts with keys: date (datetime/date or 'YYYY-MM-DD'),
              description, original_description, amount, currency, fx_rate,
              amount_jpy, category, subcategory, transaction_type
      - import_batch_id: FK to imports.id (optional)

    Returns
      (inserted_count, duplicate_count, duplicate_hashes)
    """
    conn = get_connection(db_path)
    inserted = 0
    dupes = 0
    dupe_hashes: List[str] = []
    try:
        cur = conn.cursor()
        for r in rows:
            date_val = r.get("date")
            if hasattr(date_val, "strftime"):
                date_str = date_val.strftime("%Y-%m-%d")
            else:
                date_str = str(date_val)

            dedupe_hash = compute_dedupe_hash(date_str, r.get("description", ""), float(r.get("amount", 0.0)))
            created_at = datetime.utcnow().isoformat(timespec="seconds")

            try:
                cur.execute(
                    """
                    INSERT INTO transactions (
                      date, description, original_description, amount,
                      currency, fx_rate, amount_jpy, category, subcategory,
                      transaction_type, import_batch_id, dedupe_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        date_str,
                        r.get("description"),
                        r.get("original_description"),
                        float(r.get("amount", 0.0)),
                        r.get("currency", "JPY"),
                        float(r.get("fx_rate", 1.0)),
                        float(r.get("amount_jpy", r.get("amount", 0.0))),
                        r.get("category"),
                        r.get("subcategory"),
                        r.get("transaction_type"),
                        import_batch_id,
                        dedupe_hash,
                        created_at,
                    ),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                # UNIQUE(dedupe_hash) violated => duplicate
                dupes += 1
                dupe_hashes.append(dedupe_hash)

        conn.commit()
        return inserted, dupes, dupe_hashes
    finally:
        conn.close()


def load_all_transactions(db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM transactions ORDER BY date ASC, id ASC")
        rows = [dict(r) for r in cur.fetchall()]
        return rows
    finally:
        conn.close()


def backup_database(db_path: str = DEFAULT_DB_PATH, backups_dir: str = "backups") -> str:
    _ensure_dirs()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backups_dir, f"expenses_{ts}.db")
    # Use SQLite online backup API for safety
    src = get_connection(db_path)
    try:
        dst = sqlite3.connect(backup_path)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    return backup_path


def export_transactions_to_csv(csv_path: str = None, db_path: str = DEFAULT_DB_PATH) -> str:
    """
    Export all transactions to a CSV file under exports/ by default.
    Returns the file path.
    """
    import csv

    _ensure_dirs()
    if csv_path is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join("exports", f"transactions_{ts}.csv")

    rows = load_all_transactions(db_path)
    if not rows:
        # Create an empty file with headers
        headers = [
            "id",
            "date",
            "description",
            "original_description",
            "amount",
            "currency",
            "fx_rate",
            "amount_jpy",
            "category",
            "subcategory",
            "transaction_type",
            "import_batch_id",
            "dedupe_hash",
            "created_at",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        return csv_path

    headers = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def get_setting(key: str, db_path: str = DEFAULT_DB_PATH) -> Optional[str]:
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT value FROM settings WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def set_setting(key: str, value: str, db_path: str = DEFAULT_DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        conn.commit()
    finally:
        conn.close()


# Dedupe configuration functions
def get_dedupe_settings(db_path: str = DEFAULT_DB_PATH) -> Dict[str, float]:
    """Get dedupe configuration settings with defaults."""
    return {
        "merchant_similarity_threshold": float(get_setting("dedupe_merchant_similarity_threshold", db_path) or "0.8"),
        "amount_tolerance_percent": float(get_setting("dedupe_amount_tolerance_percent", db_path) or "1.0"),
        "date_tolerance_days": int(get_setting("dedupe_date_tolerance_days", db_path) or "1"),
    }


def set_dedupe_settings(settings: Dict[str, float], db_path: str = DEFAULT_DB_PATH) -> None:
    """Set dedupe configuration settings."""
    for key, value in settings.items():
        set_setting(f"dedupe_{key}", str(value), db_path)


def find_potential_duplicates(
    new_transactions: List[Dict], 
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict]:
    """Find potential duplicates using fuzzy matching based on configured settings.
    
    Returns list of potential duplicate matches with similarity scores.
    """
    from difflib import SequenceMatcher
    from datetime import datetime, timedelta
    
    settings = get_dedupe_settings(db_path)
    merchant_threshold = settings["merchant_similarity_threshold"]
    amount_tolerance = settings["amount_tolerance_percent"] / 100.0
    date_tolerance = timedelta(days=settings["date_tolerance_days"])
    
    # Load existing transactions
    existing = load_all_transactions(db_path)
    if not existing:
        return []
    
    potential_dupes = []
    
    for new_tx in new_transactions:
        new_date = new_tx.get("date")
        new_desc = (new_tx.get("description") or "").strip().lower()
        new_amount = float(new_tx.get("amount", 0))
        
        # Convert new_date to datetime if it's a string
        if isinstance(new_date, str):
            new_date = datetime.strptime(new_date, "%Y-%m-%d").date()
        elif hasattr(new_date, 'date'):
            new_date = new_date.date()
        
        for existing_tx in existing:
            existing_date = existing_tx.get("date")
            existing_desc = (existing_tx.get("description") or "").strip().lower()
            existing_amount = float(existing_tx.get("amount", 0))
            
            # Convert existing_date to datetime if it's a string
            if isinstance(existing_date, str):
                existing_date = datetime.strptime(existing_date, "%Y-%m-%d").date()
            elif hasattr(existing_date, 'date'):
                existing_date = existing_date.date()
            
            # Check date tolerance
            date_diff = abs((new_date - existing_date).days)
            if date_diff > settings["date_tolerance_days"]:
                continue
            
            # Check amount tolerance
            amount_diff = abs(new_amount - existing_amount)
            if new_amount != 0 and amount_diff / abs(new_amount) > amount_tolerance:
                continue
            
            # Check merchant similarity
            similarity = SequenceMatcher(None, new_desc, existing_desc).ratio()
            if similarity >= merchant_threshold:
                potential_dupes.append({
                    "new_transaction": new_tx,
                    "existing_transaction": existing_tx,
                    "similarity_score": similarity,
                    "date_diff_days": date_diff,
                    "amount_diff": amount_diff,
                    "amount_diff_percent": (amount_diff / abs(new_amount) * 100) if new_amount != 0 else 0
                })
    
    # Sort by similarity score (highest first)
    potential_dupes.sort(key=lambda x: x["similarity_score"], reverse=True)
    return potential_dupes


def upsert_recurring_rule(rule: Dict, db_path: str = DEFAULT_DB_PATH) -> int:
    """Insert or update a recurring rule. Returns rule id.
    Recognized keys: id (optional), merchant_pattern, frequency, next_date,
    amount, category, subcategory, currency, active
    """
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        if rule.get("id"):
            cur.execute(
                """
                UPDATE recurring_rules SET merchant_pattern=?, frequency=?, next_date=?, amount=?,
                  category=?, subcategory=?, currency=?, active=? WHERE id=?
                """,
                (
                    rule.get("merchant_pattern"),
                    rule.get("frequency"),
                    rule.get("next_date"),
                    rule.get("amount"),
                    rule.get("category"),
                    rule.get("subcategory"),
                    rule.get("currency", "JPY"),
                    int(bool(rule.get("active", 1))),
                    int(rule.get("id")),
                ),
            )
            conn.commit()
            return int(rule.get("id"))
        else:
            cur.execute(
                """
                INSERT INTO recurring_rules (merchant_pattern, frequency, next_date, amount, category, subcategory, currency, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rule.get("merchant_pattern"),
                    rule.get("frequency"),
                    rule.get("next_date"),
                    rule.get("amount"),
                    rule.get("category"),
                    rule.get("subcategory"),
                    rule.get("currency", "JPY"),
                    int(bool(rule.get("active", 1))),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
    finally:
        conn.close()


def list_recurring_rules(db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM recurring_rules WHERE active=1 ORDER BY next_date ASC, id ASC")
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()



