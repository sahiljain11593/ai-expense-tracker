"""
Bank statement parsers for PDF/text formats not covered by generic table extraction.
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from typing import List, Optional, Union

import pandas as pd

try:
    import pdfplumber  # type: ignore
except ImportError:
    pdfplumber = None

# Rakuten card statement line (利用明細 section)
_RAKUTEN_LINE_RE = re.compile(
    r"^(\d{4}/\d{2}/\d{2})\s+(.+?)\s+本人\*?\s+"
    r"(?:\d+回払い|分割\d+回払い[^ ]*)\s+"
    r"([\d,]+)\s+\d+\s+([\d,]+)\s+([\d,]+)\s+\d+\s*$"
)

_RAKUTEN_SKIP_PREFIXES = (
    "ご利用",
    "利用日",
    "（単位",
    "現地利用額",
    "換レート",
    "※",
    "楽天カード",
    "会員様",
    "お支払",
    "ポイント",
    "ジャン",
)


def is_rakuten_card_statement_text(text: str) -> bool:
    """Detect Rakuten card PDF text layout."""
    if not text:
        return False
    return (
        "楽天カード" in text or "楽天ゴールドカード" in text
    ) and "利用日" in text and "利用店名" in text


def parse_rakuten_card_text(text: str) -> List[dict]:
    """
    Parse Rakuten credit card ご利用明細 lines from extracted PDF text.

    Uses 当月請求額 (billed this cycle) as amount for totals alignment.
    """
    transactions: List[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(_RAKUTEN_SKIP_PREFIXES):
            continue
        m = _RAKUTEN_LINE_RE.match(line)
        if not m:
            continue
        date = datetime.strptime(m.group(1), "%Y/%m/%d").date()
        merchant = m.group(2).strip()
        billed = float(m.group(4).replace(",", ""))
        transactions.append(
            {
                "date": date,
                "description": merchant,
                "original_description": merchant,
                "amount": billed,
            }
        )
    return transactions


def extract_rakuten_card_pdf(file_stream: Union[str, io.BytesIO]) -> Optional[pd.DataFrame]:
    """Extract transactions from a Rakuten card PDF statement."""
    if pdfplumber is None:
        return None

    all_text_parts: List[str] = []
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            all_text_parts.append(page.extract_text() or "")

    full_text = "\n".join(all_text_parts)
    if not is_rakuten_card_statement_text(full_text):
        return None

    rows = parse_rakuten_card_text(full_text)
    if not rows:
        return None

    return pd.DataFrame(rows)
