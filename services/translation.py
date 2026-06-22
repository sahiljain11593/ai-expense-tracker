"""
services/translation.py — AI translation service layer.

All AI provider calls, retry logic, token usage tracking, and text-normalisation
helpers live here.  Streamlit UI code stays in transaction_web_app.py.

Usage:
    from services.translation import translate_batch_ai, translate_japanese_to_english
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import unicodedata
from typing import Optional

import streamlit as st

# ── Provider / model constants (single source of truth) ───────────────────────
GEMINI_PROVIDER = "gemini"
OPENAI_PROVIDER = "openai"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
GEMINI_TRANSLATION_MODE = "AI-Powered (Gemini 3.1 Flash-Lite)"
OPENAI_TRANSLATION_MODE = "AI-Powered (OpenAI)"
LEGACY_OPENAI_TRANSLATION_MODE = "AI-Powered (GPT-3.5)"

_COST_PER_1M: dict = {
    DEFAULT_GEMINI_MODEL: (0.25, 1.50),
    DEFAULT_OPENAI_MODEL: (0.15, 0.60),
}
_AI_USAGE_KEY = "ai_usage"


# ── Key / model resolution ─────────────────────────────────────────────────────

def _streamlit_secret(section: str, key: str) -> Optional[str]:
    try:
        value = st.secrets.get(section, {}).get(key)
        return value or None
    except Exception:
        return None


def _default_key_for(provider: str, explicit_key: str = None) -> Optional[str]:
    if explicit_key:
        return explicit_key
    if provider == GEMINI_PROVIDER:
        return (
            _streamlit_secret("gemini", "api_key")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
    return _streamlit_secret("openai", "api_key") or os.getenv("OPENAI_API_KEY")


def _default_model_for(provider: str, explicit_model: str = None) -> str:
    if explicit_model:
        return explicit_model
    return DEFAULT_GEMINI_MODEL if provider == GEMINI_PROVIDER else DEFAULT_OPENAI_MODEL


def _provider_from_mode(mode: str, model: str = None, base_url: str = None) -> str:
    if base_url == GEMINI_PROVIDER or (model and model.startswith("gemini")):
        return GEMINI_PROVIDER
    if mode == GEMINI_TRANSLATION_MODE:
        return GEMINI_PROVIDER
    return OPENAI_PROVIDER


# ── Retry helpers ──────────────────────────────────────────────────────────────

def _is_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        m in msg
        for m in ("429", "500", "502", "503", "504", "rate limit", "quota",
                  "too many requests", "timeout", "connection", "network")
    )


def _call_with_retry(fn, *args, max_attempts: int = 3, base_delay: float = 1.0, **kwargs):
    """Call fn up to max_attempts times with exponential backoff on transient errors."""
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _is_retryable_error(exc):
                raise
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 0.5))
    raise last_exc


# ── Usage tracking ─────────────────────────────────────────────────────────────

def _record_usage(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    was_fallback: bool = False,
) -> None:
    try:
        usage = st.session_state.setdefault(
            _AI_USAGE_KEY,
            {"calls": 0, "retries": 0, "fallbacks": 0, "input_tokens": 0,
             "output_tokens": 0, "cost_usd": 0.0, "provider": provider, "model": model},
        )
        usage["calls"] += 1
        usage["input_tokens"] += input_tokens
        usage["output_tokens"] += output_tokens
        if was_fallback:
            usage["fallbacks"] += 1
        rates = _COST_PER_1M.get(model, (0.0, 0.0))
        usage["cost_usd"] += (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000
        usage["provider"] = provider
        usage["model"] = model
    except Exception:
        pass


# ── Text normalisation ─────────────────────────────────────────────────────────

def normalize_japanese_text(text: str) -> str:
    try:
        s = unicodedata.normalize("NFKC", text)
        s = _replace_hyphen_between_katakana(s)
        return re.sub(r"\s+", " ", s).strip()
    except Exception:
        return text


def _replace_hyphen_between_katakana(s: str) -> str:
    try:
        hyphens = "-‐‑–—ｰ"
        pattern = re.compile(rf"([\u30A0-\u30FF])[{hyphens}]([\u30A0-\u30FF])")
        prev = None
        out = s
        while prev != out:
            prev = out
            out = pattern.sub(r"\1ー\2", out)
        return out
    except Exception:
        return s


def protect_known_merchants(text: str):
    """Replace known JP merchant names with placeholders to prevent mistranslation.

    Uses the full MERCHANT_LIBRARY from services.merchants so there is a single
    source of truth.  Falls back to a small inline dict if the import fails.
    """
    try:
        from services.merchants import MERCHANT_LIBRARY as _lib  # type: ignore
        merchant_map = _lib
    except Exception:
        merchant_map = {
            "ドンキホーテ": "Don Quijote", "ドン・キホーテ": "Don Quijote",
            "ローソン": "Lawson", "セブンイレブン": "7-Eleven",
            "ファミリーマート": "FamilyMart", "イオン": "AEON",
            "ニトリ": "Nitori", "マクドナルド": "McDonald's",
            "ケンタッキー": "KFC", "スターバックス": "Starbucks",
            "イトーヨーカドー": "Ito-Yokado", "西友": "Seiyu", "ライフ": "LIFE",
        }
    placeholders: dict = {}
    processed = text
    for i, (jp, en) in enumerate(merchant_map.items()):
        if jp in processed:
            token = f"[[BRAND_{i}]]"
            processed = processed.replace(jp, token)
            placeholders[token] = en
    return processed, placeholders


def restore_known_merchants(translated: str, placeholders: dict) -> str:
    try:
        restored = translated
        for token, en in placeholders.items():
            restored = restored.replace(token, en)
        if "Don't" in restored and "Don " in restored.replace("Don't", "Don "):
            restored = restored.replace("Don't", "Don")
        return restored
    except Exception:
        return translated


# ── Single-text translators ────────────────────────────────────────────────────

def translate_japanese_to_english_fallback(text: str) -> str:
    try:
        text_norm = normalize_japanese_text(text)
        protected_text, placeholders = protect_known_merchants(text_norm)
        from deep_translator import GoogleTranslator  # type: ignore
        if protected_text and any(ord(c) > 127 for c in protected_text):
            translated = GoogleTranslator(source="ja", target="en").translate(protected_text)
            return restore_known_merchants(translated, placeholders)
        return protected_text
    except Exception as e:
        st.warning(f"Fallback translation failed for '{text}': {e}")
        return text


def translate_japanese_to_english_ai(text: str, api_key: str = None) -> str:
    try:
        text_norm = normalize_japanese_text(text)
        protected_text, placeholders = protect_known_merchants(text_norm)
        import openai  # type: ignore
        if not protected_text or not any(ord(c) > 127 for c in protected_text):
            return protected_text
        api_key = _default_key_for(OPENAI_PROVIDER, api_key)
        if not api_key:
            st.warning("No OpenAI API key found. Using free translation fallback.")
            return translate_japanese_to_english_fallback(text)
        client = openai.OpenAI(api_key=api_key)
        prompt = (
            "Translate the following Japanese text to English. This is from a credit card "
            "statement, so maintain accuracy for financial terms and merchant names.\n\n"
            f"Japanese text: {protected_text}\n\nEnglish translation:"
        )
        def _do():
            return client.chat.completions.create(
                model=DEFAULT_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional translator specializing in financial documents."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
                temperature=0.1,
            )
        response = _call_with_retry(_do)
        try:
            u = response.usage
            _record_usage(OPENAI_PROVIDER, DEFAULT_OPENAI_MODEL, u.prompt_tokens, u.completion_tokens)
        except Exception:
            pass
        return restore_known_merchants(response.choices[0].message.content.strip(), placeholders)
    except Exception as e:
        st.warning(f"AI translation failed for '{text}': {e}")
        _record_usage(OPENAI_PROVIDER, DEFAULT_OPENAI_MODEL, 0, 0, was_fallback=True)
        return translate_japanese_to_english_fallback(text)


def translate_japanese_to_english_gemini(
    text: str,
    api_key: str = None,
    model: str = None,
) -> str:
    try:
        text_norm = normalize_japanese_text(text)
        protected_text, placeholders = protect_known_merchants(text_norm)
        if not protected_text or not any(ord(c) > 127 for c in protected_text):
            return protected_text
        api_key = _default_key_for(GEMINI_PROVIDER, api_key)
        if not api_key:
            st.warning("No Gemini API key found. Using free translation fallback.")
            return translate_japanese_to_english_fallback(text)
        model = _default_model_for(GEMINI_PROVIDER, model)
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
        cfg: dict = {"temperature": 0.1, "max_output_tokens": 120}
        try:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass
        client = genai.Client(api_key=api_key)
        def _do():
            return client.models.generate_content(
                model=model,
                contents=(
                    "Translate this Japanese credit-card statement merchant or memo to concise English. "
                    "Preserve merchant names and financial terms. Return only the English translation.\n\n"
                    f"Japanese text: {protected_text}"
                ),
                config=types.GenerateContentConfig(**cfg),
            )
        response = _call_with_retry(_do)
        translated = (getattr(response, "text", "") or "").strip()
        if not translated:
            raise RuntimeError("Gemini returned an empty translation")
        try:
            um = response.usage_metadata
            _record_usage(GEMINI_PROVIDER, model, um.prompt_token_count, um.candidates_token_count)
        except Exception:
            pass
        return restore_known_merchants(translated, placeholders)
    except Exception as e:
        st.warning(f"Gemini translation failed for '{text}': {e}")
        _record_usage(GEMINI_PROVIDER, model or DEFAULT_GEMINI_MODEL, 0, 0, was_fallback=True)
        return translate_japanese_to_english_fallback(text)


def translate_japanese_to_english(text: str, mode: str = "Free Fallback", api_key: str = None) -> str:
    if mode == GEMINI_TRANSLATION_MODE:
        return translate_japanese_to_english_gemini(text, api_key)
    if mode in (OPENAI_TRANSLATION_MODE, LEGACY_OPENAI_TRANSLATION_MODE, "AI-Powered (GPT-3.5)"):
        return translate_japanese_to_english_ai(text, api_key)
    if mode == "Free Fallback":
        return translate_japanese_to_english_fallback(text)
    return text  # No Translation


# ── Partial JSON recovery ──────────────────────────────────────────────────────

def _recover_partial_json(raw: str, index_to_text: dict) -> dict:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {index_to_text[str(k)]: str(v) for k, v in parsed.items() if str(k) in index_to_text}
    except (json.JSONDecodeError, ValueError):
        pass
    recovered = {}
    for m in re.finditer(r'"(\d+)"\s*:\s*"((?:[^"\\]|\\.)*)"', raw):
        idx, val = m.group(1), m.group(2)
        if idx in index_to_text:
            try:
                val = val.encode("raw_unicode_escape").decode("unicode_escape")
            except Exception:
                pass
            recovered[index_to_text[idx]] = val
    return recovered


def _translate_batch_gemini_single_prompt(texts: list, api_key: str, model: str) -> dict:
    if not texts:
        return {}
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
        index_to_text = {str(i): t for i, t in enumerate(texts)}
        numbered_lines = "\n".join(f'{i}: "{t}"' for i, t in enumerate(texts))
        prompt = (
            'Translate each numbered Japanese credit-card merchant / memo to English.\n'
            'Return ONLY a JSON object: {"0": "translation", "1": "translation", ...}\n'
            f"Preserve proper nouns and brand names. No explanations.\n\n{numbered_lines}"
        )
        cfg: dict = {"temperature": 0.1, "max_output_tokens": max(800, 80 * len(texts))}
        try:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass
        client = genai.Client(api_key=api_key)
        def _do():
            return client.models.generate_content(
                model=model, contents=prompt, config=types.GenerateContentConfig(**cfg)
            )
        response = _call_with_retry(_do)
        raw = (getattr(response, "text", "") or "").strip()
        try:
            um = response.usage_metadata
            _record_usage(GEMINI_PROVIDER, model, um.prompt_token_count, um.candidates_token_count)
        except Exception:
            pass
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
        return _recover_partial_json(raw, index_to_text)
    except Exception:
        return {}


# ── Batch entry point (used by the app and smoke tests) ───────────────────────

try:
    from data_store import get_cached_translations, save_translations  # type: ignore
except Exception:
    get_cached_translations = None  # type: ignore
    save_translations = None  # type: ignore


def translate_batch_ai(
    texts,
    api_key: str = None,
    model: str = None,
    base_url: str = None,
) -> dict:
    """Translate a batch of descriptions.  Resolution order (cheapest first):

    Step 0 │ Static MERCHANT_LIBRARY  (in-memory, ~0 μs, zero cost)
    Step 1 │ SQLite translation_cache (DB lookup, ~1 ms, zero cost)
    Step 2 │ AI provider call         (~1–5 s, may cost tokens)
    """
    from services.merchants import apply_merchant_library  # type: ignore

    import services.translation as _self
    _get_cached = _self.get_cached_translations
    _save = _self.save_translations

    provider = _provider_from_mode("", model=model, base_url=base_url)
    mode = GEMINI_TRANSLATION_MODE if provider == GEMINI_PROVIDER else OPENAI_TRANSLATION_MODE
    resolved_model = _default_model_for(provider, model)

    unique_texts = list(dict.fromkeys(str(t) for t in texts if str(t).strip()))

    # ── Step 0: static merchant library ──────────────────────────────────────
    library_hits, after_library = apply_merchant_library(unique_texts)

    # ── Step 1: SQLite translation_cache ─────────────────────────────────────
    db_cached: dict = {}
    if _get_cached is not None and after_library:
        try:
            db_cached = _get_cached(after_library)
        except Exception:
            pass

    uncached = [t for t in after_library if t not in db_cached]
    new_translations: dict = {}

    # Use module-level refs so tests can monkeypatch them
    import services.translation as _self
    _single_prompt = _self._translate_batch_gemini_single_prompt
    _gemini = _self.translate_japanese_to_english_gemini

    if provider == GEMINI_PROVIDER and len(uncached) > 1:
        batch_result = _single_prompt(uncached, api_key, resolved_model)
        new_translations.update(batch_result)
        for text in uncached:
            if text not in new_translations:
                new_translations[text] = _gemini(text, api_key, resolved_model)
    else:
        for text in uncached:
            if provider == GEMINI_PROVIDER:
                new_translations[text] = _gemini(text, api_key, resolved_model)
            else:
                new_translations[text] = translate_japanese_to_english(text, mode, api_key)

    if new_translations and _save is not None:
        try:
            _save(new_translations, model=resolved_model, provider=provider)
        except Exception:
            pass

    # ── Merge: library hits + DB cache + fresh AI translations ───────────────
    return {**library_hits, **db_cached, **new_translations}
