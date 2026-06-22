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


def test_translate_batch_ai_skips_provider_for_cached(monkeypatch, tmp_path):
    """Provider must NOT be called for texts already in the DB cache."""
    db = str(tmp_path / "test.db")
    init_db(db)
    # Pre-populate the cache
    save_translations({"ローソン": "Lawson (cached)"}, db_path=db)

    provider_calls = []

    def fake_gemini(text, api_key=None, model=None):
        provider_calls.append(text)
        return f"FRESH:{text}"

    monkeypatch.setattr(app, "translate_japanese_to_english_gemini", fake_gemini)
    # Point translate_batch_ai at our tmp DB
    monkeypatch.setattr(app, "get_cached_translations", lambda texts, **kw: get_cached_translations(texts, db_path=db))
    monkeypatch.setattr(app, "save_translations", lambda m, **kw: save_translations(m, db_path=db))

    result = app.translate_batch_ai(
        ["ローソン", "スタバ"],
        api_key="k",
        base_url=app.GEMINI_PROVIDER,
    )

    # ローソン was cached → no provider call for it
    assert provider_calls == ["スタバ"]
    assert result["ローソン"] == "Lawson (cached)"
    assert result["スタバ"] == "FRESH:スタバ"


def test_translate_batch_ai_writes_new_translations_to_cache(monkeypatch, tmp_path):
    """Fresh translations must be written back to the DB cache."""
    db = str(tmp_path / "test.db")
    init_db(db)

    saved = {}

    def fake_gemini(text, api_key=None, model=None):
        return f"EN:{text}"

    def fake_save(mapping, model=None, provider=None, **kw):
        saved.update(mapping)

    monkeypatch.setattr(app, "translate_japanese_to_english_gemini", fake_gemini)
    monkeypatch.setattr(app, "get_cached_translations", lambda texts, **kw: {})
    monkeypatch.setattr(app, "save_translations", fake_save)

    app.translate_batch_ai(["ローソン"], api_key="k", base_url=app.GEMINI_PROVIDER)

    assert saved == {"ローソン": "EN:ローソン"}
