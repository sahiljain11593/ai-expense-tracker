"""Contract tests for AI translation/categorization helper entry points."""

import io
import pandas as pd
import pytest

import transaction_web_app as app


_UNKNOWN_A = "謎の食堂アルファ"
_UNKNOWN_B = "不思議な店ベータ"


def test_translate_batch_ai_routes_to_gemini(monkeypatch):
    """Texts not in the merchant library must reach the Gemini provider."""
    import services.translation as svc
    calls = []

    def fake_gemini(text, api_key=None, model=None):
        calls.append((text, api_key, model))
        return f"translated:{text}"

    monkeypatch.setattr(svc, "translate_japanese_to_english_gemini", fake_gemini)
    monkeypatch.setattr(svc, "_translate_batch_gemini_single_prompt", lambda texts, api_key, model: {})
    monkeypatch.setattr(svc, "get_cached_translations", lambda texts, **kw: {}, raising=False)
    monkeypatch.setattr(svc, "save_translations", lambda *a, **kw: None, raising=False)

    result = app.translate_batch_ai(
        [_UNKNOWN_A, _UNKNOWN_B],
        api_key="test-key",
        base_url=app.GEMINI_PROVIDER,
    )

    assert result == {
        _UNKNOWN_A: f"translated:{_UNKNOWN_A}",
        _UNKNOWN_B: f"translated:{_UNKNOWN_B}",
    }
    assert calls == [
        (_UNKNOWN_A, "test-key", app.DEFAULT_GEMINI_MODEL),
        (_UNKNOWN_B, "test-key", app.DEFAULT_GEMINI_MODEL),
    ]


def test_translate_batch_ai_deduplicates_inputs(monkeypatch):
    """Repeated unknown texts should produce only one provider call each."""
    import services.translation as svc
    calls = []

    def fake_gemini(text, api_key=None, model=None):
        calls.append(text)
        return f"EN:{text}"

    monkeypatch.setattr(svc, "translate_japanese_to_english_gemini", fake_gemini)
    monkeypatch.setattr(svc, "_translate_batch_gemini_single_prompt", lambda texts, api_key, model: {})
    monkeypatch.setattr(svc, "get_cached_translations", lambda texts, **kw: {}, raising=False)
    monkeypatch.setattr(svc, "save_translations", lambda *a, **kw: None, raising=False)

    result = app.translate_batch_ai(
        [_UNKNOWN_A, _UNKNOWN_A, _UNKNOWN_A, _UNKNOWN_B],
        api_key="k",
        base_url=app.GEMINI_PROVIDER,
    )

    assert calls == [_UNKNOWN_A, _UNKNOWN_B]
    assert result[_UNKNOWN_A] == f"EN:{_UNKNOWN_A}"
    assert result[_UNKNOWN_B] == f"EN:{_UNKNOWN_B}"


def test_extract_csv_uses_batch_translation(monkeypatch, tmp_path):
    """extract_transactions_from_csv must call translate_batch_ai once, not per-row."""
    batch_calls = []

    def fake_batch(texts, api_key=None, model=None, base_url=None):
        batch_calls.append(list(texts))
        return {t: f"EN:{t}" for t in texts}

    monkeypatch.setattr(app, "translate_batch_ai", fake_batch)
    # Also stub the Streamlit calls so the function runs outside Streamlit
    import streamlit as st
    monkeypatch.setattr(st, "write", lambda *a, **kw: None)
    monkeypatch.setattr(st, "warning", lambda *a, **kw: None)
    monkeypatch.setattr(st, "progress", lambda *a, **kw: type("P", (), {"progress": lambda *a, **kw: None, "empty": lambda *a, **kw: None})())
    monkeypatch.setattr(st, "empty", lambda *a, **kw: type("E", (), {"text": lambda *a, **kw: None, "empty": lambda *a, **kw: None})())
    monkeypatch.setattr(st, "columns", lambda n, **kw: [type("C", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None})() for _ in range(n)])
    monkeypatch.setattr(st, "selectbox", lambda *a, **kw: a[1][0] if len(a) > 1 else None)

    csv_content = (
        "Date,Description,Amount\n"
        "2025-01-01,ローソン,500\n"
        "2025-01-02,ローソン,300\n"
        "2025-01-03,スタバ,600\n"
    )
    stream = io.BytesIO(csv_content.encode())

    df = app.extract_transactions_from_csv(
        stream,
        translation_mode=app.GEMINI_TRANSLATION_MODE,
        api_key="test-key",
    )

    # translate_batch_ai called exactly once (not 3 times)
    assert len(batch_calls) == 1
    # Descriptions in result are the translated versions
    assert list(df["description"]) == ["EN:ローソン", "EN:ローソン", "EN:スタバ"]
    # Original Japanese preserved
    assert list(df["original_description"]) == ["ローソン", "ローソン", "スタバ"]


def test_categorise_transactions_ai_returns_categories():
    df = pd.DataFrame(
        [
            {"description": "Starbucks Shibuya", "amount": 540},
            {"description": "Netflix", "amount": 1490},
        ]
    )

    result = app.categorise_transactions_ai(
        df,
        categories=["Food", "Subscriptions", "Uncategorised"],
        subcategories={},
        base_url=app.GEMINI_PROVIDER,
    )

    assert list(result["category"]) == ["Food", "Subscriptions"]
    assert (result["confidence"] > 0).all()
