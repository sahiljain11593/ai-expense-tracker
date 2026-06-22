"""Tests for the persistent SQLite translation cache (data_store + translate_batch_ai)."""

import transaction_web_app as app
from data_store import (
    init_db,
    get_cached_translations,
    save_translations,
    get_translation_cache_size,
    clear_translation_cache,
)


def test_save_and_retrieve_translations(tmp_path):
    db = str(tmp_path / "test.db")
    init_db(db)

    save_translations({"ローソン": "Lawson", "スタバ": "Starbucks"}, model="gemini-3.1-flash-lite", provider="gemini", db_path=db)

    result = get_cached_translations(["ローソン", "スタバ", "unknown"], db_path=db)
    assert result["ローソン"] == "Lawson"
    assert result["スタバ"] == "Starbucks"
    assert "unknown" not in result


def test_cache_size_and_clear(tmp_path):
    db = str(tmp_path / "test.db")
    init_db(db)

    save_translations({"A": "a", "B": "b"}, db_path=db)
    assert get_translation_cache_size(db_path=db) == 2

    clear_translation_cache(db_path=db)
    assert get_translation_cache_size(db_path=db) == 0


def test_upsert_overwrites_old_entry(tmp_path):
    db = str(tmp_path / "test.db")
    init_db(db)

    save_translations({"ローソン": "Lawson OLD"}, db_path=db)
    save_translations({"ローソン": "Lawson NEW"}, db_path=db)

    result = get_cached_translations(["ローソン"], db_path=db)
    assert result["ローソン"] == "Lawson NEW"


_CACHE_UNKNOWN_A = "謎のキャッシュ店舗A"
_CACHE_UNKNOWN_B = "謎のキャッシュ店舗B"


def test_translate_batch_ai_skips_provider_for_cached(monkeypatch, tmp_path):
    """Provider must NOT be called for texts already in the DB cache."""
    import services.translation as svc
    db = str(tmp_path / "test.db")
    init_db(db)
    save_translations({_CACHE_UNKNOWN_A: "Cached Translation A"}, db_path=db)

    provider_calls = []

    def fake_gemini(text, api_key=None, model=None):
        provider_calls.append(text)
        return f"FRESH:{text}"

    monkeypatch.setattr(svc, "translate_japanese_to_english_gemini", fake_gemini)
    monkeypatch.setattr(svc, "_translate_batch_gemini_single_prompt", lambda texts, api_key, model: {})
    monkeypatch.setattr(svc, "get_cached_translations", lambda texts, **kw: get_cached_translations(texts, db_path=db))
    monkeypatch.setattr(svc, "save_translations", lambda m, **kw: save_translations(m, db_path=db))

    result = app.translate_batch_ai([_CACHE_UNKNOWN_A, _CACHE_UNKNOWN_B], api_key="k", base_url=app.GEMINI_PROVIDER)

    assert provider_calls == [_CACHE_UNKNOWN_B]
    assert result[_CACHE_UNKNOWN_A] == "Cached Translation A"
    assert result[_CACHE_UNKNOWN_B] == f"FRESH:{_CACHE_UNKNOWN_B}"


def test_translate_batch_ai_writes_new_translations_to_cache(monkeypatch, tmp_path):
    """Fresh translations must be written back to the DB cache."""
    import services.translation as svc
    db = str(tmp_path / "test.db")
    init_db(db)
    saved = {}

    def fake_gemini(text, api_key=None, model=None):
        return f"EN:{text}"

    monkeypatch.setattr(svc, "translate_japanese_to_english_gemini", fake_gemini)
    monkeypatch.setattr(svc, "_translate_batch_gemini_single_prompt", lambda texts, api_key, model: {})
    monkeypatch.setattr(svc, "get_cached_translations", lambda texts, **kw: {})
    monkeypatch.setattr(svc, "save_translations", lambda mapping, **kw: saved.update(mapping))

    app.translate_batch_ai([_CACHE_UNKNOWN_A], api_key="k", base_url=app.GEMINI_PROVIDER)

    assert saved == {_CACHE_UNKNOWN_A: f"EN:{_CACHE_UNKNOWN_A}"}
