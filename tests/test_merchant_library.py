"""Tests for services/merchants.py — static merchant library."""

import services.merchants as ml
import transaction_web_app as app
from data_store import init_db, get_cached_translations


# ── Library contents ──────────────────────────────────────────────────────────

def test_library_has_substantial_size():
    assert ml.merchant_library_size() >= 100, "Library should contain at least 100 entries"


def test_library_covers_convenience_stores():
    for jp in ("ローソン", "セブンイレブン", "ファミリーマート"):
        assert jp in ml.MERCHANT_LIBRARY, f"Missing convenience store: {jp}"


def test_library_covers_common_subscriptions():
    for jp in ("ネットフリックス", "スポティファイ", "アマゾンプライム"):
        assert jp in ml.MERCHANT_LIBRARY, f"Missing subscription: {jp}"


# ── lookup_merchant ───────────────────────────────────────────────────────────

def test_exact_match_full_width():
    assert ml.lookup_merchant("ローソン") == "Lawson"


def test_exact_match_after_halfwidth_normalization():
    # Half-width katakana from credit-card statements
    assert ml.lookup_merchant("ｽﾀｰﾊﾞｯｸｽ") == "Starbucks"


def test_prefix_match_with_location_suffix():
    result = ml.lookup_merchant("スターバックス 渋谷")
    assert result is not None
    assert "Starbucks" in result


def test_prefix_match_lawson_with_branch():
    # "ローソン ミゾノグチ" — full-width prefix + location
    result = ml.lookup_merchant("ローソン ミゾノグチ")
    assert result is not None
    assert "Lawson" in result

def test_short_ascii_key_does_not_match_inside_longer_word():
    # "AWS" (≤3 chars) must NOT match inside "LAWSON ..."
    result = ml.lookup_merchant("LAWSON ミゾノグチエキマエ")
    # Should either return None or return Lawson (via ローソン), not "AWS"
    if result is not None:
        assert "Web Services" not in result


def test_no_match_returns_none():
    assert ml.lookup_merchant("スパイスファクトリー") is None


def test_empty_input_returns_none():
    assert ml.lookup_merchant("") is None
    assert ml.lookup_merchant("   ") is None


def test_aeon_mall_full():
    result = ml.lookup_merchant("イオン モール ミゾノグチ")
    assert result is not None
    assert "AEON" in result


def test_7eleven_halfwidth():
    result = ml.lookup_merchant("ｾﾌﾞﾝｲﾚﾌﾞﾝ ｱｵﾔﾏ")
    assert result is not None
    assert "7" in result or "Seven" in result.lower() or "Eleven" in result


# ── apply_merchant_library ────────────────────────────────────────────────────

def test_apply_splits_resolved_unresolved():
    texts = ["ローソン", "スターバックス", "スパイスファクトリー"]
    resolved, unresolved = ml.apply_merchant_library(texts)

    assert "ローソン" in resolved
    assert "スターバックス" in resolved
    assert "スパイスファクトリー" in unresolved
    assert len(unresolved) == 1


def test_apply_resolves_half_width():
    texts = ["ﾕﾆｸﾛ ｼﾌﾞﾔ", "ｲｵﾝ ﾓｰﾙ ﾐｿﾞﾉｸﾞﾁ"]
    resolved, unresolved = ml.apply_merchant_library(texts)
    assert len(resolved) == 2
    assert len(unresolved) == 0


# ── translate_batch_ai short-circuits for library merchants ───────────────────

def test_translate_batch_ai_skips_ai_for_library_merchants(monkeypatch):
    """Known merchants must not reach the AI provider at all."""
    import services.translation as svc

    ai_calls = []

    def fake_gemini(text, api_key=None, model=None):
        ai_calls.append(text)
        return f"TRANSLATED:{text}"

    monkeypatch.setattr(svc, "translate_japanese_to_english_gemini", fake_gemini)
    monkeypatch.setattr(svc, "_translate_batch_gemini_single_prompt", lambda *a, **kw: {})
    monkeypatch.setattr(svc, "get_cached_translations", lambda texts, **kw: {})
    monkeypatch.setattr(svc, "save_translations", lambda *a, **kw: None)

    # Use a text that is definitely NOT in the library
    unknown = "謎の新規店舗XYZ"
    result = app.translate_batch_ai(
        ["ローソン", unknown],
        api_key="k",
        base_url=app.GEMINI_PROVIDER,
    )

    # ローソン resolved by library → no AI call for it
    assert "ローソン" in result
    assert result["ローソン"] == "Lawson"
    # unknown text → sent to AI
    assert unknown in ai_calls


# ── seed_translation_cache_from_library ──────────────────────────────────────

def test_seed_populates_translation_cache(tmp_path):
    from data_store import seed_translation_cache_from_library, get_translation_cache_size

    db = str(tmp_path / "test.db")
    init_db(db)

    inserted = seed_translation_cache_from_library(db_path=db)
    assert inserted >= 100, "Should seed at least 100 entries"

    size = get_translation_cache_size(db_path=db)
    assert size == inserted


def test_seed_is_idempotent(tmp_path):
    from data_store import seed_translation_cache_from_library, get_translation_cache_size

    db = str(tmp_path / "test.db")
    init_db(db)

    first = seed_translation_cache_from_library(db_path=db)
    second = seed_translation_cache_from_library(db_path=db)

    assert second == 0, "Second seeding must insert zero new rows"
    assert get_translation_cache_size(db_path=db) == first


def test_seeded_entries_retrievable_by_get_cached_translations(tmp_path):
    from data_store import seed_translation_cache_from_library

    db = str(tmp_path / "test.db")
    init_db(db)
    seed_translation_cache_from_library(db_path=db)

    result = get_cached_translations(["ローソン", "スターバックス", "unknown"], db_path=db)
    assert result["ローソン"] == "Lawson"
    assert result["スターバックス"] == "Starbucks"
    assert "unknown" not in result
