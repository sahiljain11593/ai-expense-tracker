"""Contract tests for AI translation/categorization helper entry points."""

import pandas as pd

import transaction_web_app as app


def test_translate_batch_ai_routes_to_gemini(monkeypatch):
    calls = []

    def fake_gemini(text, api_key=None, model=None):
        calls.append((text, api_key, model))
        return f"translated:{text}"

    monkeypatch.setattr(app, "translate_japanese_to_english_gemini", fake_gemini)

    result = app.translate_batch_ai(
        ["ローソン", "ローソン", "スターバックス"],
        api_key="test-key",
        base_url=app.GEMINI_PROVIDER,
    )

    assert result == {
        "ローソン": "translated:ローソン",
        "スターバックス": "translated:スターバックス",
    }
    assert calls == [
        ("ローソン", "test-key", app.DEFAULT_GEMINI_MODEL),
        ("スターバックス", "test-key", app.DEFAULT_GEMINI_MODEL),
    ]


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
