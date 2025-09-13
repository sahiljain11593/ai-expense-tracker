#!/usr/bin/env python3
"""
Reconcile uploaded bank statements (CSV and PDF) under bank/statements/.

Outputs a concise markdown report at bank/statements/reconciliation_report.md
that summarizes totals, duplicates, anomalies, and cross-source differences.
"""

import os
import io
import sys
import re
from datetime import datetime
from typing import List, Tuple, Dict

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

STATEMENTS_DIR = os.path.join(PROJECT_ROOT, "bank", "statements")
REPORT_PATH = os.path.join(STATEMENTS_DIR, "reconciliation_report.md")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Try to map arbitrary CSV columns to a common schema: date, description, amount.
    Handles English and common Japanese column names.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "description", "amount"])

    original_cols = list(df.columns)
    cols_lower = {str(c).strip().lower(): c for c in original_cols}

    date_candidates = [
        "date", "transaction_date", "posting_date", "date_posted", "transaction date",
        "利用日", "取引日", "決済日"
    ]
    desc_candidates = [
        "description", "merchant", "payee", "transaction_description", "details",
        "利用店名・商品名", "店舗名", "商品名", "取引内容"
    ]
    amount_candidates = [
        "amount", "debit", "credit", "transaction_amount", "amount_debited", "amount_credited",
        "利用金額", "支払金額", "取引金額"
    ]

    def pick(candidates: List[str]) -> str:
        for key_lower, orig in cols_lower.items():
            if key_lower in [c.lower() for c in candidates] or orig in candidates:
                return orig
        # Heuristic fallback
        for c in original_cols:
            if any(tok in str(c).lower() for tok in ["date", "日", "日時"]):
                return c
        return ""

    date_col = pick(date_candidates)
    desc_col = pick(desc_candidates)
    amount_col = pick(amount_candidates)

    missing = [x for x in [date_col, desc_col, amount_col] if not x]
    if missing:
        # Try very simple defaults for 3-column CSVs
        if len(df.columns) >= 3:
            date_col = date_col or df.columns[0]
            desc_col = desc_col or df.columns[1]
            amount_col = amount_col or df.columns[2]
        else:
            raise RuntimeError(f"Unable to identify columns: have {list(df.columns)}")

    out = pd.DataFrame()

    # Date parsing with multiple formats
    def parse_date(x: str):
        if pd.isna(x):
            return pd.NaT
        s = str(x).strip()
        fmts = ["%Y/%m/%d", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y.%m.%d"]
        for fmt in fmts:
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        # Try pandas fallback
        try:
            return pd.to_datetime(s, errors="coerce")
        except Exception:
            return pd.NaT

    out["date"] = df[date_col].apply(parse_date)

    # Description
    out["description"] = df[desc_col].astype(str).fillna("").str.strip()

    # Amount: strip currency symbols and spaces, keep sign and dot
    def parse_amount(val) -> float:
        if pd.isna(val):
            return float("nan")
        s = str(val)
        s = re.sub(r"[^0-9\-\.]+", "", s)
        if s in ("", ".", "-", "-."):
            return float("nan")
        try:
            return float(s)
        except Exception:
            return float("nan")

    out["amount"] = df[amount_col].apply(parse_amount)

    # Drop rows without essentials
    out = out.dropna(subset=["date", "description", "amount"], how="any").copy()

    # Ensure types
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["description"] = out["description"].astype(str)
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")

    return out


def load_csv_files(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            # Try UTF-8 then Latin-1
            try:
                df_raw = pd.read_csv(p, encoding="utf-8")
            except UnicodeDecodeError:
                df_raw = pd.read_csv(p, encoding="latin-1")
            df_norm = normalize_columns(df_raw)
            df_norm["source_file"] = os.path.basename(p)
            frames.append(df_norm)
        except Exception as e:
            print(f"CSV load failed for {p}: {e}")
    if not frames:
        return pd.DataFrame(columns=["date", "description", "amount", "source_file"])
    return pd.concat(frames, ignore_index=True)


def load_pdf_files(paths: List[str]) -> pd.DataFrame:
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        print("pdfplumber not installed; skipping PDF extraction")
        return pd.DataFrame(columns=["date", "description", "amount", "source_file"])

    def extract_transactions_from_pdf_bytes(pdf_bytes: bytes) -> pd.DataFrame:
        transactions = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages[:5]:
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []
                for table in tables:
                    if not table or len(table) <= 1:
                        continue
                    header = [str(h).strip().lower() for h in table[0]]
                    # Try to find columns by header keywords
                    def find_idx(names: List[str]) -> int:
                        for nm in names:
                            if nm in header:
                                return header.index(nm)
                        return -1
                    date_idx = find_idx(["date", "取引日", "利用日", "決済日"])
                    desc_idx = find_idx(["description", "details", "内容", "摘要", "利用店名・商品名"])
                    amt_idx = find_idx(["amount", "金額", "利用金額", "支払金額"]) 
                    if min(date_idx, desc_idx, amt_idx) < 0:
                        continue
                    for row in table[1:]:
                        try:
                            date_raw = str(row[date_idx]).strip()
                            desc_raw = str(row[desc_idx]).strip()
                            amt_raw = str(row[amt_idx]).strip()
                            # Parse date
                            parsed_date = None
                            for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
                                try:
                                    parsed_date = datetime.strptime(date_raw, fmt)
                                    break
                                except Exception:
                                    pass
                            if parsed_date is None:
                                try:
                                    parsed_date = pd.to_datetime(date_raw, errors="coerce")
                                except Exception:
                                    parsed_date = pd.NaT
                            # Parse amount
                            amt_clean = re.sub(r"[^0-9\-\.]+", "", amt_raw)
                            amount_val = float(amt_clean) if amt_clean not in ("", ".", "-", "-.") else float("nan")
                            if pd.isna(parsed_date) or pd.isna(amount_val) or not desc_raw:
                                continue
                            transactions.append({"date": parsed_date, "description": desc_raw, "amount": amount_val})
                        except Exception:
                            continue
                if transactions:
                    break
        return pd.DataFrame(transactions)

    frames = []
    for p in paths:
        try:
            with open(p, "rb") as f:
                data = f.read()
            df = extract_transactions_from_pdf_bytes(data)
            if not df.empty:
                df = df[["date", "description", "amount"]]
                df["source_file"] = os.path.basename(p)
                frames.append(df)
        except Exception as e:
            print(f"PDF load failed for {p}: {e}")
    if not frames:
        return pd.DataFrame(columns=["date", "description", "amount", "source_file"])
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {
            "count": 0,
            "sum": 0.0,
            "abs_sum": 0.0,
            "min_date": None,
            "max_date": None,
            "duplicates": 0,
        }
    dup_mask = df.duplicated(subset=["date", "description", "amount"], keep=False)
    return {
        "count": int(len(df)),
        "sum": float(df["amount"].sum()),
        "abs_sum": float(df["amount"].abs().sum()),
        "min_date": pd.to_datetime(df["date"]).min(),
        "max_date": pd.to_datetime(df["date"]).max(),
        "duplicates": int(dup_mask.sum()),
    }


def normalize_key_row(row: pd.Series) -> Tuple[str, str, float]:
    d = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
    desc = re.sub(r"\s+", " ", str(row["description"]).strip()).lower()
    amount = round(float(row["amount"]), 2)
    return d, desc, amount


def compare_sources(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (only_in_a, only_in_b) by key (date, normalized description, rounded amount)."""
    if a.empty and b.empty:
        return a, b
    a_key = a.apply(normalize_key_row, axis=1)
    b_key = b.apply(normalize_key_row, axis=1)
    a = a.copy(); b = b.copy()
    a["_key"] = a_key
    b["_key"] = b_key
    only_a = a[~a["_key"].isin(set(b["_key"]))].drop(columns=["_key"])
    only_b = b[~b["_key"].isin(set(a["_key"]))].drop(columns=["_key"])
    return only_a, only_b


def ensure_deps():
    # Best effort import to detect missing deps early
    missing = []
    try:
        import pdfplumber  # noqa: F401
    except Exception:
        missing.append("pdfplumber")
    try:
        import pandas  # noqa: F401
    except Exception:
        missing.append("pandas")
    if missing:
        print(f"Warning: missing dependencies: {', '.join(missing)}")


def main() -> int:
    ensure_deps()

    csv_paths = []
    pdf_paths = []

    if not os.path.isdir(STATEMENTS_DIR):
        print(f"No directory: {STATEMENTS_DIR}")
        return 1

    for name in os.listdir(STATEMENTS_DIR):
        p = os.path.join(STATEMENTS_DIR, name)
        if os.path.isfile(p):
            if name.lower().endswith(".csv"):
                csv_paths.append(p)
            elif name.lower().endswith(".pdf"):
                pdf_paths.append(p)

    csv_df = load_csv_files(csv_paths)
    pdf_df = load_pdf_files(pdf_paths)

    # Summaries
    csv_sum = summarize(csv_df)
    pdf_sum = summarize(pdf_df)

    only_csv = pd.DataFrame()
    only_pdf = pd.DataFrame()
    if not csv_df.empty and not pdf_df.empty:
        only_csv, only_pdf = compare_sources(csv_df, pdf_df)

    # Write markdown report
    lines: List[str] = []
    lines.append("# Reconciliation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- CSV files: {', '.join([os.path.basename(p) for p in csv_paths]) or 'None'}")
    lines.append(f"- PDF files: {', '.join([os.path.basename(p) for p in pdf_paths]) or 'None'}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- CSV: {csv_sum['count']} tx | Sum: {csv_sum['sum']:.2f} | Abs Sum: {csv_sum['abs_sum']:.2f} | Duplicates: {csv_sum['duplicates']}")
    lines.append(f"- PDF: {pdf_sum['count']} tx | Sum: {pdf_sum['sum']:.2f} | Abs Sum: {pdf_sum['abs_sum']:.2f} | Duplicates: {pdf_sum['duplicates']}")
    if csv_sum["min_date"] is not None:
        lines.append(f"- CSV Range: {csv_sum['min_date'].strftime('%Y-%m-%d')} → {csv_sum['max_date'].strftime('%Y-%m-%d')}")
    if pdf_sum["min_date"] is not None:
        lines.append(f"- PDF Range: {pdf_sum['min_date'].strftime('%Y-%m-%d')} → {pdf_sum['max_date'].strftime('%Y-%m-%d')}")
    lines.append("")

    if not csv_df.empty and not pdf_df.empty:
        lines.append("## Cross-check (CSV vs PDF)")
        lines.append(f"- Only in CSV: {len(only_csv)}")
        lines.append(f"- Only in PDF: {len(only_pdf)}")
        lines.append("")
        if len(only_csv) > 0:
            lines.append("### Sample: Only in CSV (first 10)")
            head = only_csv.sort_values("date").head(10)
            for _, r in head.iterrows():
                lines.append(f"- {pd.to_datetime(r['date']).strftime('%Y-%m-%d')} | {r['description'][:60]} | {r['amount']:.2f} ({r.get('source_file','')})")
            lines.append("")
        if len(only_pdf) > 0:
            lines.append("### Sample: Only in PDF (first 10)")
            head = only_pdf.sort_values("date").head(10)
            for _, r in head.iterrows():
                lines.append(f"- {pd.to_datetime(r['date']).strftime('%Y-%m-%d')} | {r['description'][:60]} | {r['amount']:.2f} ({r.get('source_file','')})")
            lines.append("")
    else:
        lines.append("## Cross-check")
        if csv_df.empty:
            lines.append("- CSV data: not available or failed to parse")
        if pdf_df.empty:
            lines.append("- PDF data: not available or failed to parse")
        lines.append("")

    # Save report
    os.makedirs(STATEMENTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report written: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

