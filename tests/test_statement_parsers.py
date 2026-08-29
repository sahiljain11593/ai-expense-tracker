"""Tests for Rakuten PDF statement parsing."""

from datetime import date

from statement_parsers import is_rakuten_card_statement_text, parse_rakuten_card_text

SAMPLE_TEXT = """
ご利用代金請求明細書
楽天カード株式会社
利用日 利用店名 利用者 支払方法 利用金額 手数料/利息 支払総額 当月請求額 翌月繰越残高
2026/04/30 焼肉ホルモンアポロン溝の口 本人* 1回払い 14,190 0 14,190 14,190 0
2026/04/26 セブン−イレブン 本人* 1回払い 192 0 192 192 0
2026/04/23 HOTEL AT BOOKING.COM利用国NL 本人* 1回払い 79,211 0 79,211 79,211 0
2026/04/11 楽天モバイル通信料 本人* 1回払い 3,297 0 3,297 3,297 0
"""


def test_detect_rakuten_statement():
    assert is_rakuten_card_statement_text(SAMPLE_TEXT)


def test_parse_rakuten_lines():
    rows = parse_rakuten_card_text(SAMPLE_TEXT)
    assert len(rows) == 4
    assert rows[0]["date"] == date(2026, 4, 30)
    assert "焼肉" in rows[0]["original_description"]
    assert rows[0]["amount"] == 14190.0
    booking = [r for r in rows if "BOOKING" in r["description"]][0]
    assert booking["amount"] == 79211.0
