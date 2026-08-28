"""Tests for _recover_partial_json and the batch Gemini prompt path."""

import transaction_web_app as app


INDEX_MAP = {"0": "ローソン", "1": "スタバ", "2": "ドンキ"}


def test_valid_json_all_keys():
    raw = '{"0": "Lawson", "1": "Starbucks", "2": "Don Quijote"}'
    result = app._recover_partial_json(raw, INDEX_MAP)
    assert result == {"ローソン": "Lawson", "スタバ": "Starbucks", "ドンキ": "Don Quijote"}


def test_valid_json_partial_keys():
    raw = '{"0": "Lawson", "2": "Don Quijote"}'
    result = app._recover_partial_json(raw, INDEX_MAP)
    assert result == {"ローソン": "Lawson", "ドンキ": "Don Quijote"}
    assert "スタバ" not in result


def test_truncated_json_recovers_available_pairs():
    # Simulates Gemini cutting off mid-response
    raw = '{"0": "Lawson", "1": "Starbucks", "2": "Don'
    result = app._recover_partial_json(raw, INDEX_MAP)
    # Must recover at least indices 0 and 1
    assert result.get("ローソン") == "Lawson"
    assert result.get("スタバ") == "Starbucks"


def test_recovery_rate_above_80_pct():
    """80%+ of rows must survive a truncated JSON response."""
    n = 10
    index_map = {str(i): f"text_{i}" for i in range(n)}
    # Truncate after 8 complete entries
    pairs = [f'"{i}": "EN_{i}"' for i in range(8)]
    raw = "{" + ", ".join(pairs) + ", \"8\": \"EN_8"  # cut off
    result = app._recover_partial_json(raw, index_map)
    assert len(result) >= int(n * 0.8)


def test_empty_response_returns_empty_dict():
    result = app._recover_partial_json("", INDEX_MAP)
    assert result == {}


def test_markdown_fenced_json_is_handled():
    raw = '```json\n{"0": "Lawson", "1": "Starbucks"}\n```'
    # _translate_batch_gemini_single_prompt strips fences before calling
    # _recover_partial_json; test the stripping logic via the function directly
    import re
    stripped = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    stripped = re.sub(r"\s*```$", "", stripped, flags=re.MULTILINE)
    result = app._recover_partial_json(stripped, INDEX_MAP)
    assert result["ローソン"] == "Lawson"
    assert result["スタバ"] == "Starbucks"


_BATCH_TEXT_A = "謎のバッチ店舗アルファ"
_BATCH_TEXT_B = "謎のバッチ店舗ベータ"
_BATCH_TEXT_C = "謎のバッチ店舗ガンマ"


def test_translate_batch_ai_uses_single_prompt_for_multiple_gemini_texts(monkeypatch):
    """For Gemini + multiple uncached/unlibrary texts, one batch call must be used."""
    import services.translation as svc
    single_prompt_calls = []

    def fake_batch(texts, api_key, model):
        single_prompt_calls.append(list(texts))
        return {t: f"BATCH:{t}" for t in texts}

    per_item_calls = []

    def fake_single(text, api_key=None, model=None):
        per_item_calls.append(text)
        return f"SINGLE:{text}"

    monkeypatch.setattr(svc, "_translate_batch_gemini_single_prompt", fake_batch)
    monkeypatch.setattr(svc, "translate_japanese_to_english_gemini", fake_single)
    monkeypatch.setattr(svc, "get_cached_translations", lambda texts, **kw: {})
    monkeypatch.setattr(svc, "save_translations", lambda *a, **kw: None)

    result = app.translate_batch_ai(
        [_BATCH_TEXT_A, _BATCH_TEXT_B, _BATCH_TEXT_C],
        api_key="k",
        base_url=app.GEMINI_PROVIDER,
    )

    assert len(single_prompt_calls) == 1
    assert set(single_prompt_calls[0]) == {_BATCH_TEXT_A, _BATCH_TEXT_B, _BATCH_TEXT_C}
    assert per_item_calls == []
    assert result == {
        _BATCH_TEXT_A: f"BATCH:{_BATCH_TEXT_A}",
        _BATCH_TEXT_B: f"BATCH:{_BATCH_TEXT_B}",
        _BATCH_TEXT_C: f"BATCH:{_BATCH_TEXT_C}",
    }


def test_translate_batch_ai_falls_back_per_item_when_batch_misses(monkeypatch):
    """If batch misses a text, per-item fallback must cover it."""
    import services.translation as svc

    def fake_batch(texts, api_key, model):
        return {texts[0]: f"BATCH:{texts[0]}"}

    per_item_calls = []

    def fake_single(text, api_key=None, model=None):
        per_item_calls.append(text)
        return f"SINGLE:{text}"

    monkeypatch.setattr(svc, "_translate_batch_gemini_single_prompt", fake_batch)
    monkeypatch.setattr(svc, "translate_japanese_to_english_gemini", fake_single)
    monkeypatch.setattr(svc, "get_cached_translations", lambda texts, **kw: {})
    monkeypatch.setattr(svc, "save_translations", lambda *a, **kw: None)

    result = app.translate_batch_ai([_BATCH_TEXT_A, _BATCH_TEXT_B], api_key="k", base_url=app.GEMINI_PROVIDER)

    assert result[_BATCH_TEXT_A].startswith("BATCH:")
    assert result[_BATCH_TEXT_B].startswith("SINGLE:")
    assert _BATCH_TEXT_B in per_item_calls
