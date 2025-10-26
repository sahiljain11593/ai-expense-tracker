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

  - merchant_learning
      id INTEGER PRIMARY KEY AUTOINCREMENT
      merchant TEXT NOT NULL
      category TEXT NOT NULL
      subcategory TEXT
      frequency INTEGER NOT NULL DEFAULT 1
      confidence_score REAL DEFAULT 0.5
      last_updated TEXT NOT NULL
      UNIQUE(merchant, category, subcategory)

  - learning_patterns
      id INTEGER PRIMARY KEY AUTOINCREMENT
      pattern_type TEXT NOT NULL  -- 'merchant', 'amount_range', 'date_pattern', 'time_pattern', 'merchant_context'
      pattern_value TEXT NOT NULL
      category TEXT NOT NULL
      subcategory TEXT
      frequency INTEGER NOT NULL DEFAULT 1
      confidence_score REAL DEFAULT 0.5
      last_updated TEXT NOT NULL
      UNIQUE(pattern_type, pattern_value, category, subcategory)

  - merchant_context_learning
      id INTEGER PRIMARY KEY AUTOINCREMENT
      merchant TEXT NOT NULL
      context_key TEXT NOT NULL  -- 'amount_range', 'day_of_week', 'time_of_day', 'amount_pattern'
      context_value TEXT NOT NULL

  - discarded_duplicates
      id INTEGER PRIMARY KEY AUTOINCREMENT
      import_batch_id INTEGER NOT NULL
      date TEXT NOT NULL
      description TEXT NOT NULL
      amount REAL NOT NULL
      dedupe_hash TEXT NOT NULL
      reason TEXT NOT NULL  -- 'exact_duplicate', 'similar_duplicate', 'manual_skip'
      discarded_at TEXT NOT NULL
      original_data TEXT  -- JSON of original transaction data
      FOREIGN KEY (import_batch_id) REFERENCES imports(id)
      UNIQUE(merchant, context_key, context_value, category, subcategory)

Notes
 - This module owns dedupe logic (exact hash on date|description|amount). Advanced
   fuzzy dedupe is added later.
 - Do not log PII. Keep outputs minimal.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta
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


def _create_missing_tables(cur: sqlite3.Cursor) -> None:
    """Create missing tables safely (migration-friendly)"""
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

    # merchant learning
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS merchant_learning (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          merchant TEXT NOT NULL,
          category TEXT NOT NULL,
          subcategory TEXT,
          frequency INTEGER NOT NULL DEFAULT 1,
          confidence_score REAL DEFAULT 0.5,
          last_updated TEXT NOT NULL,
          UNIQUE(merchant, category, subcategory)
        )
        """
    )

    # learning patterns
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS learning_patterns (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          pattern_type TEXT NOT NULL,
          pattern_value TEXT NOT NULL,
          category TEXT NOT NULL,
          subcategory TEXT,
          frequency INTEGER NOT NULL DEFAULT 1,
          confidence_score REAL DEFAULT 0.5,
          last_updated TEXT NOT NULL,
          UNIQUE(pattern_type, pattern_value, category, subcategory)
        )
        """
    )

    # merchant context learning
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS merchant_context_learning (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          merchant TEXT NOT NULL,
          context_key TEXT NOT NULL,
          context_value TEXT NOT NULL,
          category TEXT NOT NULL,
          subcategory TEXT,
          frequency INTEGER NOT NULL DEFAULT 1,
          confidence_score REAL DEFAULT 0.5,
          last_updated TEXT NOT NULL,
          UNIQUE(merchant, context_key, context_value, category, subcategory)
        )
        """
    )

    # discarded duplicates
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS discarded_duplicates (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          import_batch_id INTEGER NOT NULL,
          date TEXT NOT NULL,
          description TEXT NOT NULL,
          amount REAL NOT NULL,
          dedupe_hash TEXT NOT NULL,
          reason TEXT NOT NULL,
          discarded_at TEXT NOT NULL,
          original_data TEXT,
          FOREIGN KEY (import_batch_id) REFERENCES imports(id)
        )
        """
    )


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        # Ensure all tables exist (migration-safe)
        _create_missing_tables(cur)
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

        # merchant learning
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS merchant_learning (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              merchant TEXT NOT NULL,
              category TEXT NOT NULL,
              subcategory TEXT,
              frequency INTEGER NOT NULL DEFAULT 1,
              confidence_score REAL DEFAULT 0.5,
              last_updated TEXT NOT NULL,
              UNIQUE(merchant, category, subcategory)
            )
            """
        )

        # learning patterns
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_patterns (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              pattern_type TEXT NOT NULL,
              pattern_value TEXT NOT NULL,
              category TEXT NOT NULL,
              subcategory TEXT,
              frequency INTEGER NOT NULL DEFAULT 1,
              confidence_score REAL DEFAULT 0.5,
              last_updated TEXT NOT NULL,
              UNIQUE(pattern_type, pattern_value, category, subcategory)
            )
            """
        )

        # merchant context learning
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS merchant_context_learning (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              merchant TEXT NOT NULL,
              context_key TEXT NOT NULL,
              context_value TEXT NOT NULL,
              category TEXT NOT NULL,
              subcategory TEXT,
              frequency INTEGER NOT NULL DEFAULT 1,
              confidence_score REAL DEFAULT 0.5,
              last_updated TEXT NOT NULL,
              UNIQUE(merchant, context_key, context_value, category, subcategory)
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
                
                # Track discarded duplicate
                import json
                cur.execute(
                    """
                    INSERT INTO discarded_duplicates (
                        import_batch_id, date, description, amount, dedupe_hash,
                        reason, discarded_at, original_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        import_batch_id,
                        date_str,
                        r.get("description"),
                        float(r.get("amount", 0.0)),
                        dedupe_hash,
                        "exact_duplicate",
                        datetime.utcnow().isoformat(timespec="seconds"),
                        json.dumps(r)
                    )
                )

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
    from datetime import datetime, timedelta as dt, timedelta
    
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


def get_all_active_categorization_sessions(db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """Get all active categorization sessions."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.row_factory = sqlite3.Row
        
        cur.execute(
            """
            SELECT 
                cs.*,
                COUNT(cp.id) as reviewed_count,
                COUNT(CASE WHEN cp.category IS NOT NULL THEN 1 END) as categorized_count
            FROM categorization_sessions cs
            LEFT JOIN categorization_progress cp ON cs.id = cp.session_id
            WHERE cs.status = 'in_progress'
            GROUP BY cs.id
            ORDER BY cs.last_updated DESC
            """,
        )
        
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def load_session_transactions(session_id: int, db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """Load all transactions for a categorization session."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.row_factory = sqlite3.Row
        
        cur.execute(
            """
            SELECT * FROM categorization_progress 
            WHERE session_id = ?
            ORDER BY date DESC
            """,
            (session_id,)
        )
        
        return [dict(row) for row in cur.fetchall()]
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


# ============================================================================
# PERSISTENT LEARNING SYSTEM
# ============================================================================

def learn_from_categorization(
    description: str,
    category: str,
    subcategory: str = None,
    amount: float = None,
    date: str = None,
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """Learn from a user's categorization decision and store it persistently with contextual patterns."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        now = datetime.now().isoformat()
        
        # Extract merchant name
        merchant = _extract_merchant_name(description)
        
        # Learn basic merchant patterns
        if merchant and category:
            cur.execute(
                """
                INSERT OR REPLACE INTO merchant_learning
                (merchant, category, subcategory, frequency, confidence_score, last_updated)
                VALUES (?, ?, ?, 
                    COALESCE((SELECT frequency FROM merchant_learning WHERE merchant = ? AND category = ? AND subcategory = ?), 0) + 1,
                    MIN(0.95, COALESCE((SELECT confidence_score FROM merchant_learning WHERE merchant = ? AND category = ? AND subcategory = ?), 0.5) + 0.05),
                    ?)
                """,
                (merchant, category, subcategory, merchant, category, subcategory, merchant, category, subcategory, now)
            )
        
        # Learn contextual patterns for merchants (amount-based context)
        if merchant and amount and category:
            amount_range = _get_amount_range(amount)
            cur.execute(
                """
                INSERT OR REPLACE INTO merchant_context_learning
                (merchant, context_key, context_value, category, subcategory, frequency, confidence_score, last_updated)
                VALUES ('amount_range', ?, ?, ?, ?, 
                    COALESCE((SELECT frequency FROM merchant_context_learning WHERE merchant = ? AND context_key = 'amount_range' AND context_value = ? AND category = ? AND subcategory = ?), 0) + 1,
                    MIN(0.95, COALESCE((SELECT confidence_score FROM merchant_context_learning WHERE merchant = ? AND context_key = 'amount_range' AND context_value = ? AND category = ? AND subcategory = ?), 0.5) + 0.05),
                    ?)
                """,
                (merchant, amount_range, category, subcategory, merchant, amount_range, category, subcategory, merchant, amount_range, category, subcategory, now)
            )
        
        # Learn day-of-week context for merchants
        if merchant and date and category:
            day_of_week = _get_day_of_week(date)
            cur.execute(
                """
                INSERT OR REPLACE INTO merchant_context_learning
                (merchant, context_key, context_value, category, subcategory, frequency, confidence_score, last_updated)
                VALUES ('day_of_week', ?, ?, ?, ?, 
                    COALESCE((SELECT frequency FROM merchant_context_learning WHERE merchant = ? AND context_key = 'day_of_week' AND context_value = ? AND category = ? AND subcategory = ?), 0) + 1,
                    MIN(0.95, COALESCE((SELECT confidence_score FROM merchant_context_learning WHERE merchant = ? AND context_key = 'day_of_week' AND context_value = ? AND category = ? AND subcategory = ?), 0.5) + 0.05),
                    ?)
                """,
                (merchant, day_of_week, category, subcategory, merchant, day_of_week, category, subcategory, merchant, day_of_week, category, subcategory, now)
            )
        
        # Learn amount patterns for merchants (specific amount ranges)
        if merchant and amount and category:
            amount_pattern = _get_amount_pattern(amount)
            cur.execute(
                """
                INSERT OR REPLACE INTO merchant_context_learning
                (merchant, context_key, context_value, category, subcategory, frequency, confidence_score, last_updated)
                VALUES ('amount_pattern', ?, ?, ?, ?, 
                    COALESCE((SELECT frequency FROM merchant_context_learning WHERE merchant = ? AND context_key = 'amount_pattern' AND context_value = ? AND category = ? AND subcategory = ?), 0) + 1,
                    MIN(0.95, COALESCE((SELECT confidence_score FROM merchant_context_learning WHERE merchant = ? AND context_key = 'amount_pattern' AND context_value = ? AND category = ? AND subcategory = ?), 0.5) + 0.05),
                    ?)
                """,
                (merchant, amount_pattern, category, subcategory, merchant, amount_pattern, category, subcategory, merchant, amount_pattern, category, subcategory, now)
            )
        
        # Learn general amount range patterns
        if amount and category:
            amount_range = _get_amount_range(amount)
            cur.execute(
                """
                INSERT OR REPLACE INTO learning_patterns
                (pattern_type, pattern_value, category, subcategory, frequency, confidence_score, last_updated)
                VALUES ('amount_range', ?, ?, ?, 
                    COALESCE((SELECT frequency FROM learning_patterns WHERE pattern_type = 'amount_range' AND pattern_value = ? AND category = ? AND subcategory = ?), 0) + 1,
                    MIN(0.95, COALESCE((SELECT confidence_score FROM learning_patterns WHERE pattern_type = 'amount_range' AND pattern_value = ? AND category = ? AND subcategory = ?), 0.5) + 0.05),
                    ?)
                """,
                (amount_range, category, subcategory, amount_range, category, subcategory, amount_range, category, subcategory, now)
            )
        
        # Learn date patterns (day of week, month patterns)
        if date and category:
            day_pattern = _get_day_pattern(date)
            cur.execute(
                """
                INSERT OR REPLACE INTO learning_patterns
                (pattern_type, pattern_value, category, subcategory, frequency, confidence_score, last_updated)
                VALUES ('day_pattern', ?, ?, ?, 
                    COALESCE((SELECT frequency FROM learning_patterns WHERE pattern_type = 'day_pattern' AND pattern_value = ? AND category = ? AND subcategory = ?), 0) + 1,
                    MIN(0.95, COALESCE((SELECT confidence_score FROM learning_patterns WHERE pattern_type = 'day_pattern' AND pattern_value = ? AND category = ? AND subcategory = ?), 0.5) + 0.05),
                    ?)
                """,
                (day_pattern, category, subcategory, day_pattern, category, subcategory, day_pattern, category, subcategory, now)
            )
        
        conn.commit()
    finally:
        conn.close()


def get_learning_suggestions(
    description: str,
    amount: float = None,
    date: str = None,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict]:
    """Get category suggestions based on learned patterns with contextual awareness."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.row_factory = sqlite3.Row
        
        suggestions = []
        merchant = _extract_merchant_name(description)
        
        # Get contextual merchant suggestions (highest priority)
        if merchant and amount and date:
            amount_range = _get_amount_range(amount)
            day_of_week = _get_day_of_week(date)
            amount_pattern = _get_amount_pattern(amount)
            
            # Try amount range context first
            cur.execute(
                """
                SELECT category, subcategory, frequency, confidence_score
                FROM merchant_context_learning
                WHERE merchant = ? AND context_key = 'amount_range' AND context_value = ?
                ORDER BY frequency * confidence_score DESC
                LIMIT 3
                """,
                (merchant, amount_range)
            )
            
            for row in cur.fetchall():
                suggestions.append({
                    'method': 'merchant_context_amount',
                    'category': row['category'],
                    'subcategory': row['subcategory'],
                    'confidence': row['confidence_score'] * 1.2,  # Boost for contextual matches
                    'frequency': row['frequency'],
                    'reason': f"'{merchant}' with similar amount (¥{amount_range}) → {row['category']} {row['frequency']} times"
                })
            
            # Try day of week context
            cur.execute(
                """
                SELECT category, subcategory, frequency, confidence_score
                FROM merchant_context_learning
                WHERE merchant = ? AND context_key = 'day_of_week' AND context_value = ?
                ORDER BY frequency * confidence_score DESC
                LIMIT 2
                """,
                (merchant, day_of_week)
            )
            
            for row in cur.fetchall():
                suggestions.append({
                    'method': 'merchant_context_day',
                    'category': row['category'],
                    'subcategory': row['subcategory'],
                    'confidence': row['confidence_score'] * 1.1,  # Slight boost for day context
                    'frequency': row['frequency'],
                    'reason': f"'{merchant}' on {day_of_week}s → {row['category']} {row['frequency']} times"
                })
            
            # Try amount pattern context
            cur.execute(
                """
                SELECT category, subcategory, frequency, confidence_score
                FROM merchant_context_learning
                WHERE merchant = ? AND context_key = 'amount_pattern' AND context_value = ?
                ORDER BY frequency * confidence_score DESC
                LIMIT 2
                """,
                (merchant, amount_pattern)
            )
            
            for row in cur.fetchall():
                suggestions.append({
                    'method': 'merchant_context_pattern',
                    'category': row['category'],
                    'subcategory': row['subcategory'],
                    'confidence': row['confidence_score'] * 1.15,  # Boost for pattern matches
                    'frequency': row['frequency'],
                    'reason': f"'{merchant}' with {amount_pattern} pattern → {row['category']} {row['frequency']} times"
                })
        
        # Get general merchant-based suggestions (medium priority)
        if merchant:
            cur.execute(
                """
                SELECT category, subcategory, frequency, confidence_score
                FROM merchant_learning
                WHERE merchant = ?
                ORDER BY frequency * confidence_score DESC
                LIMIT 3
                """,
                (merchant,)
            )
            
            for row in cur.fetchall():
                suggestions.append({
                    'method': 'merchant_general',
                    'category': row['category'],
                    'subcategory': row['subcategory'],
                    'confidence': row['confidence_score'] * 0.9,  # Lower weight for general matches
                    'frequency': row['frequency'],
                    'reason': f"'{merchant}' generally categorized as {row['category']} {row['frequency']} times"
                })
        
        # Get amount-based suggestions (lower priority)
        if amount:
            amount_range = _get_amount_range(amount)
            cur.execute(
                """
                SELECT category, subcategory, frequency, confidence_score
                FROM learning_patterns
                WHERE pattern_type = 'amount_range' AND pattern_value = ?
                ORDER BY frequency * confidence_score DESC
                LIMIT 2
                """,
                (amount_range,)
            )
            
            for row in cur.fetchall():
                suggestions.append({
                    'method': 'amount_range',
                    'category': row['category'],
                    'subcategory': row['subcategory'],
                    'confidence': row['confidence_score'] * 0.6,  # Lower weight for amount patterns
                    'frequency': row['frequency'],
                    'reason': f"Amount range ¥{amount_range} → {row['category']} {row['frequency']} times"
                })
        
        # Get date-based suggestions (lowest priority)
        if date:
            day_pattern = _get_day_pattern(date)
            cur.execute(
                """
                SELECT category, subcategory, frequency, confidence_score
                FROM learning_patterns
                WHERE pattern_type = 'day_pattern' AND pattern_value = ?
                ORDER BY frequency * confidence_score DESC
                LIMIT 2
                """,
                (day_pattern,)
            )
            
            for row in cur.fetchall():
                suggestions.append({
                    'method': 'day_pattern',
                    'category': row['category'],
                    'subcategory': row['subcategory'],
                    'confidence': row['confidence_score'] * 0.4,  # Lowest weight for date patterns
                    'frequency': row['frequency'],
                    'reason': f"{day_pattern} transactions → {row['category']} {row['frequency']} times"
                })
        
        # Sort by confidence and remove duplicates
        seen = set()
        unique_suggestions = []
        for suggestion in sorted(suggestions, key=lambda x: x['confidence'], reverse=True):
            key = (suggestion['category'], suggestion['subcategory'])
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:5]  # Return top 5 suggestions
        
    finally:
        conn.close()


def get_learning_statistics(db_path: str = DEFAULT_DB_PATH) -> Dict:
    """Get statistics about the learning system."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        
        # Merchant learning stats
        cur.execute("SELECT COUNT(*) FROM merchant_learning")
        total_merchant_patterns = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(DISTINCT merchant) FROM merchant_learning")
        unique_merchants = cur.fetchone()[0]
        
        # Pattern learning stats
        cur.execute("SELECT COUNT(*) FROM learning_patterns")
        total_patterns = cur.fetchone()[0]
        
        # Recent learning activity (last 30 days)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        cur.execute("SELECT COUNT(*) FROM merchant_learning WHERE last_updated > ?", (thirty_days_ago,))
        recent_learning = cur.fetchone()[0]
        
        return {
            'total_merchant_patterns': total_merchant_patterns,
            'unique_merchants': unique_merchants,
            'total_patterns': total_patterns,
            'recent_learning': recent_learning
        }
    finally:
        conn.close()


def _extract_merchant_name(description: str) -> str:
    """Extract merchant name from transaction description."""
    import re
    
    # Remove common prefixes and suffixes
    desc = description.lower().strip()
    
    # Common patterns to remove
    patterns_to_remove = [
        r'visa\s+domestic\s+use\s+vs\s+',
        r'credit\s+card\s+',
        r'debit\s+card\s+',
        r'atm\s+',
        r'pos\s+',
        r'\d+',  # Remove numbers
        r'[^\w\s]',  # Remove special characters
    ]
    
    merchant = desc
    for pattern in patterns_to_remove:
        merchant = re.sub(pattern, '', merchant)
    
    # Clean up whitespace
    merchant = ' '.join(merchant.split())
    
    return merchant if merchant else "unknown"


def _get_amount_range(amount: float) -> str:
    """Get amount range for pattern learning."""
    abs_amount = abs(amount)
    
    if abs_amount < 100:
        return "0-100"
    elif abs_amount < 500:
        return "100-500"
    elif abs_amount < 1000:
        return "500-1000"
    elif abs_amount < 5000:
        return "1000-5000"
    elif abs_amount < 10000:
        return "5000-10000"
    elif abs_amount < 50000:
        return "10000-50000"
    else:
        return "50000+"


def _get_day_pattern(date: str) -> str:
    """Get day pattern for learning (weekday/weekend)."""
    try:
        from datetime import datetime, timedelta
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        if date_obj.weekday() < 5:  # Monday = 0, Sunday = 6
            return "weekday"
        else:
            return "weekend"
    except:
        return "unknown"


def _get_day_of_week(date: str) -> str:
    """Get day of week for contextual learning."""
    try:
        from datetime import datetime, timedelta
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        return days[date_obj.weekday()]
    except:
        return "unknown"


def _get_amount_pattern(amount: float) -> str:
    """Get specific amount pattern for contextual learning."""
    abs_amount = abs(amount)
    
    # Common transaction patterns
    if abs_amount == int(abs_amount):  # Round numbers
        if abs_amount < 100:
            return "small_round"
        elif abs_amount < 1000:
            return "medium_round"
        else:
            return "large_round"
    elif abs_amount % 100 == 0:  # Hundreds
        return "hundreds"
    elif abs_amount % 50 == 0:  # Fifties
        return "fifties"
    elif abs_amount % 10 == 0:  # Tens
        return "tens"
    else:
        return "irregular"


def get_discarded_duplicates(import_batch_id: Optional[int] = None, db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """Get discarded duplicates, optionally filtered by import batch."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        
        if import_batch_id:
            cur.execute(
                """
                SELECT dd.*, i.file_name, i.imported_at
                FROM discarded_duplicates dd
                JOIN imports i ON dd.import_batch_id = i.id
                WHERE dd.import_batch_id = ?
                ORDER BY dd.discarded_at DESC
                """,
                (import_batch_id,)
            )
        else:
            cur.execute(
                """
                SELECT dd.*, i.file_name, i.imported_at
                FROM discarded_duplicates dd
                JOIN imports i ON dd.import_batch_id = i.id
                ORDER BY dd.discarded_at DESC
                """
            )
        
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()


def get_import_history(db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """Get import history with statistics."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT 
                i.*,
                COUNT(t.id) as inserted_count,
                COUNT(dd.id) as discarded_count
            FROM imports i
            LEFT JOIN transactions t ON i.id = t.import_batch_id
            LEFT JOIN discarded_duplicates dd ON i.id = dd.import_batch_id
            GROUP BY i.id
            ORDER BY i.imported_at DESC
            """
        )
        
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()


def restore_discarded_duplicate(discarded_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Restore a discarded duplicate as a legitimate transaction."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        
        # Get the discarded duplicate data
        cur.execute("SELECT * FROM discarded_duplicates WHERE id = ?", (discarded_id,))
        discarded = cur.fetchone()
        
        if not discarded:
            return False
        
        # Parse original data
        import json
        original_data = json.loads(discarded[8])  # original_data column
        
        # Insert as new transaction
        date_str = discarded[2]  # date
        dedupe_hash = discarded[6]  # dedupe_hash
        created_at = datetime.utcnow().isoformat(timespec="seconds")
        
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
                original_data.get("description"),
                original_data.get("original_description"),
                float(original_data.get("amount", 0.0)),
                original_data.get("currency", "JPY"),
                float(original_data.get("fx_rate", 1.0)),
                float(original_data.get("amount_jpy", original_data.get("amount", 0.0))),
                original_data.get("category"),
                original_data.get("subcategory"),
                original_data.get("transaction_type"),
                discarded[1],  # import_batch_id
                dedupe_hash,
                created_at,
            ),
        )
        
        # Remove from discarded duplicates
        cur.execute("DELETE FROM discarded_duplicates WHERE id = ?", (discarded_id,))
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        return False
    finally:
        conn.close()


