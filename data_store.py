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

  - categorization_sessions
      id INTEGER PRIMARY KEY AUTOINCREMENT
      file_name TEXT NOT NULL
      started_at TEXT NOT NULL
      last_updated TEXT NOT NULL
      total_transactions INTEGER NOT NULL
      reviewed_transactions INTEGER NOT NULL DEFAULT 0
      status TEXT NOT NULL DEFAULT 'in_progress'  -- 'in_progress' | 'completed' | 'abandoned'
      session_data TEXT  -- JSON data for session state

  - categorization_progress
      id INTEGER PRIMARY KEY AUTOINCREMENT
      session_id INTEGER NOT NULL
      transaction_hash TEXT NOT NULL  -- hash of transaction data
      date TEXT NOT NULL
      description TEXT NOT NULL
      amount REAL NOT NULL
      category TEXT
      subcategory TEXT
      transaction_type TEXT
      reviewed_at TEXT
      confidence_score REAL DEFAULT 0.0
      FOREIGN KEY (session_id) REFERENCES categorization_sessions(id)
      UNIQUE(session_id, transaction_hash)

Notes
 - This module owns dedupe logic (exact hash on date|description|amount). Advanced
   fuzzy dedupe is added later.
 - Do not log PII. Keep outputs minimal.
"""

from __future__ import annotations

import json
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

        # categorization sessions
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS categorization_sessions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              file_name TEXT NOT NULL,
              started_at TEXT NOT NULL,
              last_updated TEXT NOT NULL,
              total_transactions INTEGER NOT NULL,
              reviewed_transactions INTEGER NOT NULL DEFAULT 0,
              status TEXT NOT NULL DEFAULT 'in_progress',
              session_data TEXT
            )
            """
        )

        # categorization progress
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS categorization_progress (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id INTEGER NOT NULL,
              transaction_hash TEXT NOT NULL,
              date TEXT NOT NULL,
              description TEXT NOT NULL,
              amount REAL NOT NULL,
              category TEXT,
              subcategory TEXT,
              transaction_type TEXT,
              reviewed_at TEXT,
              confidence_score REAL DEFAULT 0.0,
              FOREIGN KEY (session_id) REFERENCES categorization_sessions(id),
              UNIQUE(session_id, transaction_hash)
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


def get_dedupe_settings(db_path: str = DEFAULT_DB_PATH) -> Dict:
    """Get dedupe configuration settings with defaults."""
    defaults = {
        'similarity_threshold': 0.85,  # 85% similarity for fuzzy matching
        'check_date_range_days': 0,    # 0 = exact date only
        'check_amount_tolerance': 0.0  # 0.0 = exact amount only
    }
    
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        settings = defaults.copy()
        for key in defaults:
            cur.execute('SELECT value FROM settings WHERE key = ?', (f'dedupe_{key}',))
            row = cur.fetchone()
            if row:
                val = row[0]
                # Convert to appropriate type
                if key == 'check_date_range_days':
                    settings[key] = int(val)
                else:
                    settings[key] = float(val)
        return settings
    finally:
        conn.close()


def save_dedupe_settings(similarity_threshold: float, date_range_days: int, amount_tolerance: float, db_path: str = DEFAULT_DB_PATH) -> None:
    """Save dedupe configuration to settings."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ('dedupe_similarity_threshold', str(similarity_threshold))
        )
        cur.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ('dedupe_check_date_range_days', str(date_range_days))
        )
        cur.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ('dedupe_check_amount_tolerance', str(amount_tolerance))
        )
        conn.commit()
    finally:
        conn.close()


def find_potential_duplicates_fuzzy(date: str, description: str, amount: float, transaction_id: Optional[int] = None, db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """
    Find potential duplicates using configurable fuzzy matching.
    
    Args:
        date: Transaction date (YYYY-MM-DD)
        description: Transaction description
        amount: Transaction amount
        transaction_id: If provided, exclude this ID from results
        db_path: Path to database
    
    Returns:
        List of potential duplicate transaction dicts with similarity scores
    """
    from difflib import SequenceMatcher
    from datetime import datetime as dt, timedelta
    
    settings = get_dedupe_settings(db_path)
    threshold = settings['similarity_threshold']
    date_range = settings['check_date_range_days']
    amount_tol = settings['check_amount_tolerance']
    
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        
        # Build date range query
        if date_range == 0:
            date_query = "date = ?"
            date_params = [date]
        else:
            # Calculate date range
            try:
                base_date = dt.strptime(date, '%Y-%m-%d')
                start_date = (base_date - timedelta(days=date_range)).strftime('%Y-%m-%d')
                end_date = (base_date + timedelta(days=date_range)).strftime('%Y-%m-%d')
                date_query = "date BETWEEN ? AND ?"
                date_params = [start_date, end_date]
            except:
                date_query = "date = ?"
                date_params = [date]
        
        # Build amount range query
        if amount_tol == 0.0:
            amount_query = "amount = ?"
            amount_params = [amount]
        else:
            lower = amount * (1 - amount_tol)
            upper = amount * (1 + amount_tol)
            amount_query = "amount BETWEEN ? AND ?"
            amount_params = [lower, upper]
        
        # Build full query
        query = f'''
            SELECT id, date, description, amount, category, subcategory, transaction_type
            FROM transactions
            WHERE {date_query} AND {amount_query}
        '''
        
        params = date_params + amount_params
        if transaction_id:
            query += " AND id != ?"
            params.append(transaction_id)
        
        cur.execute(query, params)
        rows = cur.fetchall()
        
        # Filter by description similarity
        duplicates = []
        desc_lower = (description or '').lower().strip()
        
        for row in rows:
            candidate_desc = (row['description'] or '').lower().strip()
            similarity = SequenceMatcher(None, desc_lower, candidate_desc).ratio()
            
            if similarity >= threshold:
                duplicates.append({
                    'id': row['id'],
                    'date': row['date'],
                    'description': row['description'],
                    'amount': row['amount'],
                    'category': row['category'],
                    'subcategory': row['subcategory'],
                    'transaction_type': row['transaction_type'],
                    'similarity': round(similarity * 100, 1)  # As percentage
                })
        
        # Sort by similarity descending
        duplicates.sort(key=lambda x: x['similarity'], reverse=True)
        return duplicates
    finally:
        conn.close()


# ============================================================================
# CATEGORIZATION PROGRESS MANAGEMENT
# ============================================================================

def create_categorization_session(file_name: str, total_transactions: int, db_path: str = DEFAULT_DB_PATH) -> int:
    """Create a new categorization session and return the session ID."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        now = datetime.now().isoformat()
        
        cur.execute(
            """
            INSERT INTO categorization_sessions 
            (file_name, started_at, last_updated, total_transactions, status)
            VALUES (?, ?, ?, ?, 'in_progress')
            """,
            (file_name, now, now, total_transactions)
        )
        
        session_id = cur.lastrowid
        conn.commit()
        return session_id
    finally:
        conn.close()


def save_categorization_progress(
    session_id: int, 
    transaction_data: Dict, 
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """Save categorization progress for a single transaction."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        
        # Create transaction hash for uniqueness
        transaction_hash = compute_dedupe_hash(
            str(transaction_data['date']),
            transaction_data['description'],
            float(transaction_data['amount'])
        )
        
        now = datetime.now().isoformat()
        
        # Upsert the progress record
        cur.execute(
            """
            INSERT OR REPLACE INTO categorization_progress
            (session_id, transaction_hash, date, description, amount, 
             category, subcategory, transaction_type, reviewed_at, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                transaction_hash,
                transaction_data['date'],
                transaction_data['description'],
                transaction_data['amount'],
                transaction_data.get('category'),
                transaction_data.get('subcategory'),
                transaction_data.get('transaction_type'),
                now,
                transaction_data.get('confidence_score', 0.0)
            )
        )
        
        # Update session's reviewed count and last_updated
        cur.execute(
            """
            UPDATE categorization_sessions 
            SET reviewed_transactions = (
                SELECT COUNT(*) FROM categorization_progress 
                WHERE session_id = ?
            ),
            last_updated = ?
            WHERE id = ?
            """,
            (session_id, now, session_id)
        )
        
        conn.commit()
    finally:
        conn.close()


def load_categorization_progress(session_id: int, db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """Load all categorization progress for a session."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.row_factory = sqlite3.Row
        
        cur.execute(
            """
            SELECT * FROM categorization_progress 
            WHERE session_id = ?
            ORDER BY reviewed_at DESC
            """,
            (session_id,)
        )
        
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_active_categorization_session(file_name: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict]:
    """Get the active categorization session for a file."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.row_factory = sqlite3.Row
        
        cur.execute(
            """
            SELECT * FROM categorization_sessions 
            WHERE file_name = ? AND status = 'in_progress'
            ORDER BY last_updated DESC
            LIMIT 1
            """,
            (file_name,)
        )
        
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def complete_categorization_session(session_id: int, db_path: str = DEFAULT_DB_PATH) -> None:
    """Mark a categorization session as completed."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        now = datetime.now().isoformat()
        
        cur.execute(
            """
            UPDATE categorization_sessions 
            SET status = 'completed', last_updated = ?
            WHERE id = ?
            """,
            (now, session_id)
        )
        
        conn.commit()
    finally:
        conn.close()


def get_categorization_session_stats(session_id: int, db_path: str = DEFAULT_DB_PATH) -> Dict:
    """Get statistics for a categorization session."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.row_factory = sqlite3.Row
        
        # Get session info
        cur.execute("SELECT * FROM categorization_sessions WHERE id = ?", (session_id,))
        session = cur.fetchone()
        if not session:
            return {}
        
        # Get progress stats
        cur.execute(
            """
            SELECT 
                COUNT(*) as total_reviewed,
                COUNT(CASE WHEN category IS NOT NULL THEN 1 END) as categorized,
                COUNT(CASE WHEN category IS NULL THEN 1 END) as uncategorized,
                AVG(confidence_score) as avg_confidence
            FROM categorization_progress 
            WHERE session_id = ?
            """,
            (session_id,)
        )
        
        stats = cur.fetchone()
        
        return {
            'session': dict(session),
            'total_transactions': session['total_transactions'],
            'reviewed_transactions': stats['total_reviewed'],
            'categorized_transactions': stats['categorized'],
            'uncategorized_transactions': stats['uncategorized'],
            'completion_percentage': round((stats['total_reviewed'] / session['total_transactions']) * 100, 1) if session['total_transactions'] > 0 else 0,
            'average_confidence': round(stats['avg_confidence'] or 0, 2)
        }
    finally:
        conn.close()


def get_merchant_categorization_suggestions(limit: int = 20, db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """Get merchant categorization suggestions based on historical data."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.row_factory = sqlite3.Row
        
        cur.execute(
            """
            SELECT 
                description,
                category,
                subcategory,
                COUNT(*) as frequency,
                AVG(amount_jpy) as avg_amount
            FROM transactions 
            WHERE category IS NOT NULL
            GROUP BY description, category, subcategory
            ORDER BY frequency DESC, avg_amount DESC
            LIMIT ?
            """,
            (limit,)
        )
        
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def apply_bulk_categorization_rules(
    session_id: int, 
    rules: List[Dict], 
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """Apply bulk categorization rules to uncategorized transactions in a session.
    
    Args:
        session_id: The categorization session ID
        rules: List of rules with pattern, category, subcategory
        db_path: Database path
        
    Returns:
        Number of transactions updated
    """
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        updated_count = 0
        
        for rule in rules:
            pattern = rule.get('pattern', '').lower()
            category = rule.get('category')
            subcategory = rule.get('subcategory')
            transaction_type = rule.get('transaction_type')
            
            if not pattern or not category:
                continue
            
            # Update transactions matching the pattern
            cur.execute(
                """
                UPDATE categorization_progress 
                SET category = ?, subcategory = ?, transaction_type = ?, reviewed_at = ?
                WHERE session_id = ? 
                AND category IS NULL 
                AND LOWER(description) LIKE ?
                """,
                (category, subcategory, transaction_type, datetime.now().isoformat(), 
                 session_id, f'%{pattern}%')
            )
            
            updated_count += cur.rowcount
        
        # Update session stats
        cur.execute(
            """
            UPDATE categorization_sessions 
            SET reviewed_transactions = (
                SELECT COUNT(*) FROM categorization_progress 
                WHERE session_id = ?
            ),
            last_updated = ?
            WHERE id = ?
            """,
            (session_id, datetime.now().isoformat(), session_id)
        )
        
        conn.commit()
        return updated_count
    finally:
        conn.close()



