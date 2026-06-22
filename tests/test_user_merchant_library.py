"""Tests for user_merchant_library: DB CRUD, auto-learn flow, and lookup integration."""

import services.merchants as ml
import transaction_web_app as app
from data_store import (
    init_db,
    upsert_user_merchant,
    bulk_upsert_user_merchants,
    load_user_merchant_library,
    get_user_merchant_translations,
    delete_user_merchant,
    get_user_merchant_library_size,
)


# ── CRUD ─────────────────────────────────────────────────────────────────────

def test_upsert_and_retrieve(tmp_path):
    db = str(tmp_path / "t.db")
    init_db(db)
    upsert_user_merchant("スパイスファクトリー", "Spice Factory", source="user", db_path=db)
    rows = load_user_merchant_library(db_path=db)
    assert len(rows) == 1
    assert rows[0]["jp_text"] == "スパイスファクトリー"
    assert rows[0]["en_text"] == "Spice Factory"
    assert rows[0]["source"] == "user"


def test_upsert_updates_en_text(tmp_path):
    db = str(tmp_path / "t.db")
    init_db(db)
    upsert_user_merchant("コインランドリー", "Coin Laundry", source="auto", db_path=db)
    upsert_user_merchant("コインランドリー", "Coin Laundry Mizonoguchi", source="user", db_path=db)
    rows = load_user_merchant_library(db_path=db)
    assert rows[0]["en_text"] == "Coin Laundry Mizonoguchi"


def test_user_source_preserved_on_update(tmp_path):
    """Once marked 'user', source must not be overwritten by 'auto' upsert."""
    db = str(tmp_path / "t.db")
    init_db(db)
    upsert_user_merchant("トリキゾク", "Torikizoku", source="user", db_path=db)
    upsert_user_merchant("トリキゾク", "Torikizoku Izakaya", source="auto", db_path=db)
    rows = load_user_merchant_library(db_path=db)
    assert rows[0]["source"] == "user"  # user source wins


def test_delete_removes_entry(tmp_path):
    db = str(tmp_path / "t.db")
    init_db(db)
    upsert_user_merchant("ゴールドジム", "Gold's Gym", db_path=db)
    assert get_user_merchant_library_size(db_path=db) == 1
    delete_user_merchant("ゴールドジム", db_path=db)
    assert get_user_merchant_library_size(db_path=db) == 0


def test_bulk_upsert(tmp_path):
    db = str(tmp_path / "t.db")
    init_db(db)
    mapping = {"A店": "Shop A", "B店": "Shop B", "C店": "Shop C"}
    inserted = bulk_upsert_user_merchants(mapping, db_path=db)
    assert inserted == 3
    assert get_user_merchant_library_size(db_path=db) == 3


def test_get_user_merchant_translations_lookup(tmp_path):
    db = str(tmp_path / "t.db")
    init_db(db)
    bulk_upsert_user_merchants({"謎の店": "Mystery Shop", "別の店": "Another Shop"}, db_path=db)
    result = get_user_merchant_translations(["謎の店", "unknown"], db_path=db)
    assert result["謎の店"] == "Mystery Shop"
    assert "unknown" not in result


# ── apply_merchant_library now consults user DB ───────────────────────────────

def test_apply_merchant_library_uses_user_db(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    init_db(db)
    upsert_user_merchant("スパイスファクトリー", "Spice Factory", db_path=db)

    # Patch get_user_merchant_translations to use our tmp DB
    monkeypatch.setattr(
        "data_store.get_user_merchant_translations",
        lambda texts, **kw: get_user_merchant_translations(texts, db_path=db),
        raising=False,
    )
    # Also patch inside merchants module
    import services.merchants as _ml
    monkeypatch.setattr(
        _ml,
        "get_user_merchant_translations" if hasattr(_ml, "get_user_merchant_translations") else "_dummy",
        lambda texts, **kw: get_user_merchant_translations(texts, db_path=db),
        raising=False,
    )

    resolved, unresolved = ml.apply_merchant_library(["スパイスファクトリー", "謎の未知店"])
    assert "スパイスファクトリー" in resolved
    assert resolved["スパイスファクトリー"] == "Spice Factory"
    assert "謎の未知店" in unresolved


# ── Auto-learn via translate_batch_ai ────────────────────────────────────────

def test_translate_batch_ai_auto_saves_to_user_library(monkeypatch, tmp_path):
    """New AI translations must be auto-saved to user_merchant_library."""
    import services.translation as svc

    saved_to_lib = {}

    def fake_gemini(text, api_key=None, model=None):
        return f"EN:{text}"

    def fake_bulk_upsert(mapping, source="auto", **kw):
        saved_to_lib.update(mapping)

    monkeypatch.setattr(svc, "translate_japanese_to_english_gemini", fake_gemini)
    monkeypatch.setattr(svc, "_translate_batch_gemini_single_prompt", lambda *a, **kw: {})
    monkeypatch.setattr(svc, "get_cached_translations", lambda texts, **kw: {})
    monkeypatch.setattr(svc, "save_translations", lambda *a, **kw: None)
    monkeypatch.setattr(svc, "bulk_upsert_user_merchants", fake_bulk_upsert)

    unknown = "謎の新規店舗XYZ"
    app.translate_batch_ai([unknown], api_key="k", base_url=app.GEMINI_PROVIDER)

    assert unknown in saved_to_lib
    assert saved_to_lib[unknown] == f"EN:{unknown}"


def test_known_merchants_not_saved_to_user_library(monkeypatch):
    """Static library hits must NOT trigger auto-save (they are already known)."""
    import services.translation as svc

    saved_to_lib = {}

    def fake_bulk_upsert(mapping, source="auto", **kw):
        saved_to_lib.update(mapping)

    monkeypatch.setattr(svc, "translate_japanese_to_english_gemini", lambda t, **kw: f"EN:{t}")
    monkeypatch.setattr(svc, "_translate_batch_gemini_single_prompt", lambda *a, **kw: {})
    monkeypatch.setattr(svc, "get_cached_translations", lambda texts, **kw: {})
    monkeypatch.setattr(svc, "save_translations", lambda *a, **kw: None)
    monkeypatch.setattr(svc, "bulk_upsert_user_merchants", fake_bulk_upsert)

    # ローソン is in static library → resolved without AI → not in saved_to_lib
    app.translate_batch_ai(["ローソン"], api_key="k", base_url=app.GEMINI_PROVIDER)

    assert "ローソン" not in saved_to_lib
