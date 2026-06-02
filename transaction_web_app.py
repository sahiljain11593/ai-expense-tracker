"""
transaction_web_app.py

This Streamlit application provides a simple user interface for
uploading credit card statements in PDF or image (screenshot) format,
extracting transaction data, automatically assigning categories based on
keyword rules, and allowing the user to review and adjust categories.

The app assumes that the user has installed the following Python
packages in their environment:

* streamlit — for building the web interface
* pandas — for data manipulation
* pdfplumber — for reading table data from PDF files
* pytesseract — for Optical Character Recognition on images
* Pillow (PIL) — for image handling

You also need to have the Tesseract OCR engine installed on your
system for pytesseract to work.

Usage: run this app with

    streamlit run transaction_web_app.py
"""

import io
import os
import re
from datetime import datetime
import unicodedata
from typing import Optional


import pandas as pd
import streamlit as st

# Data layer
try:
    from data_store import (
        init_db,
        create_import_record,
        insert_transactions,
        export_transactions_to_csv,
        backup_database,
        load_all_transactions,
        upsert_recurring_rule,
        list_recurring_rules,
        get_setting,
        set_setting,
        get_dedupe_settings,
        save_dedupe_settings,
        find_potential_duplicates_fuzzy,
        create_categorization_session,
        save_categorization_progress,
        load_categorization_progress,
        get_active_categorization_session,
        complete_categorization_session,
        get_categorization_session_stats,
        get_merchant_categorization_suggestions,
        apply_bulk_categorization_rules,
        get_all_active_categorization_sessions,
        load_session_transactions,
        learn_from_categorization,
        get_learning_suggestions,
        get_learning_statistics,
        load_merchant_learning,
    )
except Exception as _e:
    # Allow the app to still render other parts; show a soft warning
    init_db = None  # type: ignore
    create_import_record = None  # type: ignore
    insert_transactions = None  # type: ignore
    export_transactions_to_csv = None  # type: ignore
    backup_database = None  # type: ignore
    load_all_transactions = None  # type: ignore
    upsert_recurring_rule = None  # type: ignore
    list_recurring_rules = None  # type: ignore
    create_categorization_session = None  # type: ignore
    save_categorization_progress = None  # type: ignore
    load_categorization_progress = None  # type: ignore
    get_active_categorization_session = None  # type: ignore
    complete_categorization_session = None  # type: ignore
    get_categorization_session_stats = None  # type: ignore
    get_merchant_categorization_suggestions = None  # type: ignore
    apply_bulk_categorization_rules = None  # type: ignore
    get_all_active_categorization_sessions = None  # type: ignore
    load_session_transactions = None  # type: ignore
    learn_from_categorization = None  # type: ignore
    get_learning_suggestions = None  # type: ignore
    get_learning_statistics = None  # type: ignore
    load_merchant_learning = None  # type: ignore

# Auth UI (Firebase Google Sign-In)
try:
    from auth_ui import require_auth
except Exception:
    def require_auth() -> bool:  # fallback no-auth when module unavailable
        return True

# Advanced AI and Analytics modules
try:
    from ml_engine import EnsembleCategorizationEngine, LocalMLTrainer
    from insights_engine import InsightsEngine, SpendingAnalytics
    from dashboard import ModernDashboard, InteractiveFilters
    ADVANCED_FEATURES_AVAILABLE = True
except Exception as _e:
    # Graceful fallback if advanced features not available
    EnsembleCategorizationEngine = None  # type: ignore
    LocalMLTrainer = None  # type: ignore
    InsightsEngine = None  # type: ignore
    SpendingAnalytics = None  # type: ignore
    ModernDashboard = None  # type: ignore
    InteractiveFilters = None  # type: ignore
    ADVANCED_FEATURES_AVAILABLE = False

# Wide layout for more horizontal space
st.set_page_config(layout="wide", page_title="AI Expense Tracker", page_icon="💰")

# Smart Learning System (now using built-in MerchantLearningSystem class)

try:
    import pdfplumber  # type: ignore
except ImportError:
    pdfplumber = None

try:
    import pytesseract  # type: ignore
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None


class MerchantLearningSystem:
    """Learning system that improves categorization based on user feedback.

    Loads previously-learned merchant-category mappings from the SQLite database
    at startup so knowledge persists across sessions, and writes corrections back
    to the database on every user feedback event.
    """

    def __init__(self):
        self.merchant_categories: dict[str, dict[str, int]] = {}
        self.merchant_patterns: dict = {}
        self.user_corrections: list[dict] = []
        self.confidence_scores: dict[str, dict[str, float]] = {}
        # Load persisted learning from database
        self._load_from_db()

    # ── persistence helpers ──────────────────────────────────────────

    def _load_from_db(self):
        """Populate in-memory merchant mappings from the merchant_learning table."""
        if load_merchant_learning is None:
            return
        try:
            rows = load_merchant_learning()
            for r in rows:
                merchant = r["merchant"]
                category = r["category"]
                freq = r.get("frequency", 1)
                conf = r.get("confidence_score", 0.5)
                if merchant not in self.merchant_categories:
                    self.merchant_categories[merchant] = {}
                self.merchant_categories[merchant][category] = (
                    self.merchant_categories[merchant].get(category, 0) + freq
                )
                if merchant not in self.confidence_scores:
                    self.confidence_scores[merchant] = {}
                self.confidence_scores[merchant][category] = max(
                    self.confidence_scores[merchant].get(category, 0.0), conf
                )
        except Exception:
            pass  # Gracefully continue without DB data

    def _persist_correction(self, description: str, category: str, subcategory: str = "",
                            amount: float = None, date: str = None):
        """Write a single correction to the database."""
        if learn_from_categorization is None:
            return
        try:
            learn_from_categorization(
                description=description,
                category=category,
                subcategory=subcategory,
                amount=amount,
                date=date,
            )
        except Exception:
            pass  # Non-critical; in-memory learning still active

    # ── public API ───────────────────────────────────────────────────

    def learn_from_user_feedback(self, transaction_id: str, old_category: str, new_category: str, transaction_data: dict):
        """Learn from user corrections to improve future categorizations."""
        merchant = self._extract_merchant(transaction_data['description'])
        original_desc = transaction_data.get('original_description', '')

        correction = {
            'transaction_id': transaction_id,
            'merchant': merchant,
            'old_category': old_category,
            'new_category': new_category,
            'description': transaction_data['description'],
            'original_description': original_desc,
            'amount': transaction_data.get('amount'),
            'timestamp': pd.Timestamp.now()
        }
        self.user_corrections.append(correction)

        # Update in-memory merchant category mapping
        if merchant not in self.merchant_categories:
            self.merchant_categories[merchant] = {}
        if new_category not in self.merchant_categories[merchant]:
            self.merchant_categories[merchant][new_category] = 0
        self.merchant_categories[merchant][new_category] += 1
        self._update_confidence(merchant, new_category)

        # Persist to database
        self._persist_correction(
            description=transaction_data['description'],
            category=new_category,
            subcategory=transaction_data.get('subcategory', ''),
            amount=transaction_data.get('amount'),
            date=str(transaction_data.get('date', '')),
        )

    def suggest_category(self, description: str, original_description: str = "") -> tuple:
        """Suggest category based on learned patterns."""
        merchant = self._extract_merchant(description)

        if merchant in self.merchant_categories:
            categories = self.merchant_categories[merchant]
            if categories:
                best_category = max(categories, key=categories.get)
                confidence = self.confidence_scores.get(merchant, {}).get(best_category, 0.5)
                return best_category, confidence

        # Also try matching on original (Japanese) description
        if original_description:
            merchant_orig = self._extract_merchant(original_description)
            if merchant_orig in self.merchant_categories:
                categories = self.merchant_categories[merchant_orig]
                if categories:
                    best_category = max(categories, key=categories.get)
                    confidence = self.confidence_scores.get(merchant_orig, {}).get(best_category, 0.5)
                    return best_category, confidence

        return "Uncategorised", 0.0

    def predict_category(self, transaction_data: dict) -> tuple:
        """Predict category for a transaction with confidence and breakdown."""
        description = transaction_data.get('description', '')
        original_description = transaction_data.get('original_description', '')

        suggested_category, confidence = self.suggest_category(description, original_description)

        breakdown = {
            'method': 'merchant_learning',
            'confidence': confidence,
            'merchant': self._extract_merchant(description),
            'suggested_category': suggested_category
        }

        return suggested_category, confidence, breakdown

    # ── internal helpers ─────────────────────────────────────────────

    def _extract_merchant(self, description: str) -> str:
        """Extract merchant name from transaction description."""
        description = description.lower()
        patterns_to_remove = [
            r'visa\s+domestic\s+use\s+vs\s+',
            r'credit\s+card\s+',
            r'debit\s+card\s+',
            r'atm\s+',
            r'pos\s+',
            r'\d+',
            r'[^\w\s\u3040-\u30FF\u4E00-\u9FFF]',  # Keep Japanese characters
        ]
        merchant = description
        for pattern in patterns_to_remove:
            merchant = re.sub(pattern, '', merchant)
        merchant = ' '.join(merchant.split())
        return merchant if merchant else "unknown"

    def _update_confidence(self, merchant: str, category: str):
        """Update confidence score for merchant-category pair."""
        if merchant not in self.confidence_scores:
            self.confidence_scores[merchant] = {}
        if category not in self.confidence_scores[merchant]:
            self.confidence_scores[merchant][category] = 0.5
        current_confidence = self.confidence_scores[merchant][category]
        self.confidence_scores[merchant][category] = min(0.95, current_confidence + 0.1)

    def get_learning_stats(self) -> dict:
        """Get statistics about the learning system."""
        total_corrections = len(self.user_corrections)
        unique_merchants = len(self.merchant_categories)
        recent_corrections = [c for c in self.user_corrections
                              if (pd.Timestamp.now() - c['timestamp']).days < 30]
        return {
            'total_corrections': total_corrections,
            'unique_merchants': unique_merchants,
            'recent_corrections': len(recent_corrections),
            'merchant_categories': self.merchant_categories,
            'confidence_scores': self.confidence_scores
        }


def extract_transactions_from_pdf(file_stream: io.BytesIO) -> pd.DataFrame:
    """Extract transactions from a PDF statement using pdfplumber.

    Assumes the PDF contains a table with columns Date, Description and
    Amount.  This function looks for the largest table on the first few
    pages.  It may need adaptation for your specific statement layout.
    """
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is not installed; please install it to process PDFs.")

    transactions = []
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages[:5]:
            tables = page.extract_tables()
            for table in tables:
                if len(table) > 1 and len(table[0]) >= 3:
                    header = [(h or "").strip().lower() for h in table[0]]
                    try:
                        date_idx = header.index("date")
                        desc_idx = header.index("description")
                        amt_idx = header.index("amount")
                    except ValueError:
                        continue
                    for row in table[1:]:
                        try:
                            cell_date = row[date_idx] or ""
                            date = datetime.strptime(cell_date.strip(), "%d/%m/%Y")
                        except Exception:
                            try:
                                date = datetime.strptime(cell_date.strip(), "%Y-%m-%d")
                            except Exception:
                                continue
                        description = (row[desc_idx] or "").strip()
                        try:
                            amount = float(row[amt_idx].replace(",", ""))
                        except Exception:
                            continue
                        transactions.append({"date": date, "description": 
description, "amount": amount})
            if transactions:
                break
    if not transactions:
        raise RuntimeError("No transaction table detected in the uploaded PDF.")
    df = pd.DataFrame(transactions)
    return df


def extract_transactions_from_image(file_stream: io.BytesIO) -> pd.DataFrame:
    """Extract transactions from an image using OCR.

    This function reads the entire image as text and then attempts to
    parse lines that contain a date, description and amount.  It is
    simplistic and may need refinement for real statement layouts.  If
    pytesseract is not available, an error is raised.
    """
    if pytesseract is None or Image is None:
        raise RuntimeError(
            "pytesseract or PIL is not installed; please install them and ensure tesseract is available."
        )
    image = Image.open(file_stream)
    text = pytesseract.image_to_string(image)
    lines = text.splitlines()
    pattern = re.compile(r"(\\d{2}/\\d{2}/\\d{4})\\s+(.+?)\\s+(-?\\d+[.,]?\\d*)")
    records = []
    for line in lines:
        match = pattern.search(line)
        if match:
            date_str, desc, amt_str = match.groups()
            try:
                date = datetime.strptime(date_str, "%d/%m/%Y")
            except Exception:
                continue
            amount = float(amt_str.replace(",", ""))
            records.append({"date": date, "description": desc.strip(), 
"amount": amount})
    if not records:
        raise RuntimeError(
            "No transactions detected in the image.  Ensure the statement is clearly legible and try again."
        )
    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# Gemini / OpenAI unified AI helper
# ---------------------------------------------------------------------------
GEMINI_PROVIDER = "gemini"  # sentinel value passed as base_url to select Gemini

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"  # newest, free-tier, optimized for translation


def _is_gemini(base_url: str | None) -> bool:
    return base_url == GEMINI_PROVIDER


def _default_key_for(base_url: str | None) -> str | None:
    """Return the env-var-based default API key for the selected provider."""
    if _is_gemini(base_url):
        return os.getenv("GEMINI_API_KEY")
    return os.getenv("OPENAI_API_KEY")


def _default_model_for(base_url: str | None) -> str:
    return DEFAULT_GEMINI_MODEL if _is_gemini(base_url) else DEFAULT_OPENAI_MODEL


def _resolve_provider(api_key: str | None, model: str | None, base_url: str | None) -> tuple[str, str, str | None]:
    """Resolve api_key and model with provider-aware defaults. Returns (api_key, model, base_url)."""
    api_key = api_key or _default_key_for(base_url)
    if not model or model == DEFAULT_OPENAI_MODEL and _is_gemini(base_url):
        model = _default_model_for(base_url)
    return api_key, model, base_url


def _is_retryable_error(exc: Exception) -> bool:
    """Detect transient errors worth retrying (rate limits, transient network issues)."""
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "resource_exhausted" in msg or "quota" in msg:
        return True
    if "timeout" in msg or "connection" in msg or "503" in msg or "502" in msg or "unavailable" in msg:
        return True
    return False


def _ai_generate(
    messages: list[dict],
    model: str,
    api_key: str,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    base_url: str = None,
    json_mode: bool = False,
    max_retries: int = 3,
) -> str:
    """Call an AI model and return the response text.

    Works with both OpenAI and Google Gemini (native SDK).
    Pass base_url=GEMINI_PROVIDER to use Gemini, or None for OpenAI.
    Set json_mode=True to force valid JSON output.

    Includes exponential backoff retry for transient errors (429, 5xx, network).
    """
    import time
    import random

    api_key, model, base_url = _resolve_provider(api_key, model, base_url)
    if not api_key:
        provider = "Gemini" if _is_gemini(base_url) else "OpenAI"
        raise RuntimeError(f"No {provider} API key available")

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            if _is_gemini(base_url):
                return _gemini_call(messages, model, api_key, max_tokens, temperature, json_mode)
            return _openai_call(messages, model, api_key, max_tokens, temperature, json_mode)
        except Exception as exc:  # noqa: BLE001 - we re-raise non-retryable below
            last_exc = exc
            if attempt == max_retries - 1 or not _is_retryable_error(exc):
                raise
            # Exponential backoff with jitter: 1s, 2s, 4s + up to 1s jitter
            delay = (2 ** attempt) + random.random()
            time.sleep(delay)

    # Should be unreachable but keeps the type-checker happy
    raise last_exc if last_exc else RuntimeError("AI generate failed without exception")


def _gemini_call(
    messages: list[dict],
    model: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
) -> str:
    """Single Gemini call (no retry). Handles thinking-budget and empty responses."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # Split system instruction from conversation messages
    system_instruction = None
    user_parts = []
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        else:
            user_parts.append(msg["content"])
    prompt = user_parts[0] if len(user_parts) == 1 else "\n".join(user_parts)

    # Gemini 2.5+ Flash defaults to "thinking" mode where reasoning tokens consume
    # the same max_output_tokens budget. For deterministic structured tasks we
    # disable thinking entirely, which both saves tokens and avoids empty
    # responses where thinking ate the whole budget.
    config_kwargs = dict(
        system_instruction=system_instruction,
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    try:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:
        # Older SDKs may not have ThinkingConfig; safe to skip
        pass
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    config = types.GenerateContentConfig(**config_kwargs)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    text = response.text or ""
    if not text:
        finish = None
        try:
            if response.candidates:
                finish = getattr(response.candidates[0], "finish_reason", None)
        except Exception:
            pass
        raise RuntimeError(f"Gemini returned empty response (finish_reason={finish})")
    return text.strip()


def _openai_call(
    messages: list[dict],
    model: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
) -> str:
    """Single OpenAI chat completion call (no retry)."""
    import openai

    client = openai.OpenAI(api_key=api_key)
    kwargs = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


def translate_japanese_to_english_ai(text: str, api_key: str = None, model: str = None, base_url: str = None) -> str:
    """Translate a single Japanese text to English using OpenAI or Gemini."""
    try:
        text_norm = normalize_japanese_text(text)
        protected_text, placeholders = protect_known_merchants(text_norm)

        if not protected_text or not any(ord(char) > 127 for char in protected_text):
            return restore_known_merchants(protected_text, placeholders) if protected_text else text

        api_key = api_key or _default_key_for(base_url)
        model = model or _default_model_for(base_url)

        if not api_key:
            st.warning("No API key found. Using free translation fallback.")
            return translate_japanese_to_english_fallback(text)

        translated = _ai_generate(
            messages=[
                {"role": "system", "content": (
                    "You translate abbreviated Japanese bank and credit-card statement "
                    "descriptions into concise English. Input lines are NOT normal prose; "
                    "they are short, abbreviated merchant/payee names from financial "
                    "statements (e.g. 'ミゾノグチエキマエ' means 'Mizonokuchi Station'). "
                    "Use your knowledge of real Japanese businesses to identify the "
                    "actual merchant. Include the business type when helpful, "
                    "e.g. 'Shoukoktei (Chinese Restaurant)' or 'Torikizoku (Izakaya)'. "
                    "Return ONLY the English translation, nothing else. "
                    "Keep merchant/brand names in their well-known English form "
                    "(e.g. スタバ→Starbucks, マツキヨ→Matsumoto Kiyoshi)."
                )},
                {"role": "user", "content": protected_text}
            ],
            model=model,
            api_key=api_key,
            max_tokens=120,
            temperature=0.0,
            base_url=base_url,
        )

        translated = restore_known_merchants(translated, placeholders)
        return translated

    except Exception as e:
        st.warning(f"AI translation failed for '{text}': {e}")
        return translate_japanese_to_english_fallback(text)


def translate_batch_ai(texts: list[str], api_key: str = None, batch_size: int = 25, model: str = None, base_url: str = None) -> dict[str, str]:
    """Batch-translate a list of unique Japanese descriptions via AI (OpenAI or Gemini).

    Returns a dict mapping each original text to its English translation.
    Falls back to one-by-one free translation on failure.
    """
    import json as _json

    api_key = api_key or _default_key_for(base_url)
    model = model or _default_model_for(base_url)

    results: dict[str, str] = {}

    # Pre-process: normalise & protect merchants, skip non-Japanese.
    # NB: We use INLINE protection in batch mode (real English brand names instead
    # of [[BRAND_N]] placeholders) — placeholder tokens cause the model to copy
    # translations across rows because every row looks like "[[BRAND_X]] LOCATION".
    to_translate: list[tuple[str, str]] = []  # (original, protected_inline)
    for text in texts:
        text_norm = normalize_japanese_text(text)
        protected = protect_known_merchants_inline(text_norm)
        if not protected or not any(ord(c) > 127 for c in protected):
            results[text] = protected if protected else text
        else:
            to_translate.append((text, protected))

    if not to_translate or not api_key:
        # Fall back to free translation for anything remaining
        for original, _ in to_translate:
            results[original] = translate_japanese_to_english_fallback(original)
        return results

    # Process in batches
    for batch_start in range(0, len(to_translate), batch_size):
        batch = to_translate[batch_start:batch_start + batch_size]
        numbered_lines = "\n".join(
            f"{i+1}. {item[1]}" for i, item in enumerate(batch)
        )

        try:
            raw = _ai_generate(
                messages=[
                    {"role": "system", "content": (
                        "You translate abbreviated Japanese bank and credit-card statement "
                        "descriptions into concise English. Input lines are NOT normal prose; "
                        "they are short, abbreviated merchant/payee names from financial "
                        "statements (e.g. 'ミゾノグチエキマエ' means 'Mizonokuchi Station'). "
                        "Use your knowledge of real Japanese businesses to identify the "
                        "actual merchant. For example: ｼｮｳｺﾃｲ is a Chinese restaurant "
                        "(小籠亭→Shoukoktei Restaurant), ﾄﾘｷｿﾞｸ is an izakaya chain "
                        "(鳥貴族→Torikizoku Izakaya). "
                        "Include the business type in your translation when helpful, "
                        "e.g. 'Shoukoktei (Chinese Restaurant)' or 'Torikizoku (Izakaya)'. "
                        "Keep well-known brand names in English form "
                        "(e.g. スタバ→Starbucks, マツキヨ→Matsumoto Kiyoshi). "
                        "Reply with a JSON object mapping each input number (as a string) "
                        "to its English translation. Each translation MUST correspond to its "
                        "OWN input — do not copy translations across inputs. Output schema: "
                        "{\"1\": \"<english translation of input 1>\", \"2\": \"<english translation of input 2>\", ...}"
                    )},
                    {"role": "user", "content": numbered_lines}
                ],
                model=model,
                api_key=api_key,
                max_tokens=max(800, 80 * len(batch)),
                temperature=0.0,
                base_url=base_url,
                json_mode=True,
            )

            # Strip markdown fences if model adds them
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            mapping = _json.loads(raw)

            for i, (original, _protected) in enumerate(batch):
                translated = mapping.get(str(i + 1), "") or ""
                translated = translated.strip()
                if translated:
                    results[original] = translated
                else:
                    results[original] = translate_japanese_to_english_fallback(original)

        except Exception as e:
            st.warning(f"Batch translation failed (batch starting at {batch_start}): {e}. Falling back to free translation.")
            for original, _ in batch:
                results[original] = translate_japanese_to_english_fallback(original)

    return results

def translate_japanese_to_english_fallback(text: str) -> str:
    """Fallback translation using deep-translator when AI translation fails."""
    try:
        # Normalize first to convert half-width Katakana to standard form
        text_norm = normalize_japanese_text(text)
        # Protect known merchant names
        protected_text, placeholders = protect_known_merchants(text_norm)
        from deep_translator import GoogleTranslator
        if protected_text and any(ord(char) > 127 for char in protected_text):
            translated = GoogleTranslator(source='ja', target='en').translate(protected_text)
            if translated is None:
                return restore_known_merchants(text, placeholders)
            translated = restore_known_merchants(translated, placeholders)
            return translated
        # No Japanese characters left (all were known merchants) – restore placeholders
        return restore_known_merchants(protected_text, placeholders) if protected_text else text
    except Exception as e:
        st.warning(f"Fallback translation failed for '{text}': {e}")
        return text

def translate_japanese_to_english(text: str, mode: str = "Free Fallback", api_key: str = None, model: str = "gpt-4o-mini", base_url: str = None) -> str:
    """Main translation function - handles different translation modes."""
    if mode in ("AI-Powered (GPT-4o-mini)", "AI-Powered"):
        return translate_japanese_to_english_ai(text, api_key, model=model, base_url=base_url)
    elif mode == "Free Fallback":
        return translate_japanese_to_english_fallback(text)
    else:  # No Translation
        return text


def normalize_japanese_text(text: str) -> str:
    try:
        # Convert half-width kana to regular width, normalize compatibility characters
        s = unicodedata.normalize('NFKC', text)
        # Replace ASCII/Unicode hyphens between Katakana with prolonged sound mark 'ー'
        s = _replace_hyphen_between_katakana(s)
        # Collapse extra spaces
        s = re.sub(r"\s+", " ", s).strip()
        return s
    except Exception:
        return text


def protect_known_merchants(text: str):
    """Replace known JP merchant names with placeholders to avoid mistranslation.
    Returns (processed_text, placeholders_dict).
    """
    merchant_map = {
        # ── Convenience stores ──
        'ローソン': 'Lawson',
        'セブンイレブン': '7-Eleven',
        'ファミリーマート': 'FamilyMart',
        'ミニストップ': 'Ministop',
        'デイリーヤマザキ': 'Daily Yamazaki',
        'ポプラ': 'Poplar',
        # ── Supermarkets / general retail ──
        'イオン': 'AEON',
        'イトーヨーカドー': 'Ito-Yokado',
        '西友': 'Seiyu',
        'ライフ': 'LIFE Supermarket',
        'マルエツ': 'Maruetsu',
        'サミット': 'Summit Supermarket',
        'オーケー': 'OK Store',
        'コストコ': 'Costco',
        '業務スーパー': 'Gyomu Super',
        'ドンキホーテ': 'Don Quijote',
        'ドン・キホーテ': 'Don Quijote',
        'ダイソー': 'Daiso',
        'キャンドゥ': 'Can Do',
        'セリア': 'Seria',
        # ── Drugstores ──
        'マツモトキヨシ': 'Matsumoto Kiyoshi',
        'マツキヨ': 'Matsumoto Kiyoshi',
        'ウエルシア': 'Welcia',
        'ツルハ': 'Tsuruha',
        'サンドラッグ': 'Sundrug',
        'ココカラファイン': 'Cocokara Fine',
        # ── Fast food / cafes ──
        'マクドナルド': "McDonald's",
        'ケンタッキー': 'KFC',
        'スターバックス': 'Starbucks',
        'スタバ': 'Starbucks',
        'タリーズ': "Tully's Coffee",
        'ドトール': 'Doutor Coffee',
        'モスバーガー': 'Mos Burger',
        'すき家': 'Sukiya',
        '吉野家': 'Yoshinoya',
        '松屋': 'Matsuya',
        'サイゼリヤ': 'Saizeriya',
        'ガスト': 'Gusto',
        'ココイチ': 'CoCo Ichibanya',
        # ── Home / electronics ──
        'ニトリ': 'Nitori',
        'ヤマダ電機': 'Yamada Denki',
        'ビックカメラ': 'Bic Camera',
        'ヨドバシ': 'Yodobashi Camera',
        'ケーズデンキ': "K's Denki",
        # ── Fashion ──
        'ユニクロ': 'UNIQLO',
        'ジーユー': 'GU',
        'しまむら': 'Shimamura',
        'ZOZOTOWN': 'ZOZOTOWN',
        # ── Transport ──
        'スイカ': 'Suica',
        'パスモ': 'PASMO',
        'モバイルSuica': 'Mobile Suica',
        # ── Utilities / telecom ──
        '東京電力': 'TEPCO',
        '東京ガス': 'Tokyo Gas',
        '東京都水道': 'Tokyo Waterworks',
        'ソフトバンク': 'SoftBank',
        'ドコモ': 'NTT Docomo',
        'エーユー': 'au (KDDI)',
        '楽天モバイル': 'Rakuten Mobile',
        # ── Online / subscriptions ──
        'アマゾン': 'Amazon',
        '楽天': 'Rakuten',
        'ヤフー': 'Yahoo Japan',
        'メルカリ': 'Mercari',
        # ── Credit card / payment terms ──
        '年会費': 'Annual Fee',
        '楽天カード': 'Rakuten Card',
        '楽天ゴールドカード': 'Rakuten Gold Card',
    }
    placeholders = {}
    processed = text
    idx = 0
    for jp, en in merchant_map.items():
        if jp in processed:
            token = f"[[BRAND_{idx}]]"
            processed = processed.replace(jp, token)
            placeholders[token] = en
            idx += 1
    return processed, placeholders


def protect_known_merchants_inline(text: str) -> str:
    """Like protect_known_merchants, but inline-replaces brands with their English form.

    This is used in batch translation where placeholder tokens like [[BRAND_0]] can
    confuse the model — it sees the same structural pattern across many rows and
    starts copying translations from the first row. Using the English brand name
    directly (e.g. "UNIQLO Shibuya") gives the model a real, distinct token to
    work with for each row, eliminating that whole class of bug.

    Returns the protected text only — no placeholders dict, since restoration is
    a no-op when brands are already in English.
    """
    try:
        # Reuse the same merchant_map by calling the placeholder version, then
        # post-substituting placeholders with their English values.
        protected, placeholders = protect_known_merchants(text)
        for token, en in placeholders.items():
            protected = protected.replace(token, en)
        return protected
    except Exception:
        return text


def restore_known_merchants(translated: str, placeholders: dict) -> str:
    try:
        if translated is None:
            return ""
        restored = translated
        for token, en in placeholders.items():
            restored = restored.replace(token, en)
        # Guard against specific bad expansion like "Don't" from Don-*
        if "Don't" in restored and 'Don ' in restored.replace("Don't", 'Don '):
            restored = restored.replace("Don't", 'Don')
        return restored
    except Exception:
        return translated


def _replace_hyphen_between_katakana(s: str) -> str:
    """Replace hyphen-like chars between Katakana letters with the prolonged sound mark 'ー'.
    Handles '-', '‐', '‑', '–', '—', and halfwidth 'ｰ'.
    """
    try:
        # Katakana block \u30A0-\u30FF
        hyphens = "-‐‑–—ｰ"
        pattern = re.compile(rf"([\u30A0-\u30FF])[{hyphens}]([\u30A0-\u30FF])")
        prev = None
        out = s
        # Iteratively replace until stable (for multiple hyphens)
        while prev != out:
            prev = out
            out = pattern.sub(r"\1ー\2", out)
        return out
    except Exception:
        return s

def extract_transactions_from_csv(file_stream: io.BytesIO, translation_mode: str = "Free Fallback", api_key: str = None, ai_model: str = "gpt-4o-mini", ai_base_url: str = None) -> pd.DataFrame:
    """Extract transactions from a CSV file.
    
    This function reads CSV files and attempts to identify date, description, and amount columns.
    It handles common CSV formats from different banks and financial institutions.
    Now includes Japanese translation support.
    """
    try:
        # Try to read the CSV with different encodings
        df = pd.read_csv(file_stream, encoding='utf-8')
    except UnicodeDecodeError:
        file_stream.seek(0)  # Reset file pointer
        df = pd.read_csv(file_stream, encoding='latin-1')
    
    # Common column names for different banks (including Japanese)
    date_columns = ['date', 'transaction_date', 'posting_date', 'date_posted', 'transaction date',
                    '利用日', '取引日', '決済日']
    desc_columns = ['description', 'merchant', 'payee', 'transaction_description', 'details',
                    '利用店名・商品名', '店舗名', '商品名', '取引内容']
    amount_columns = ['amount', 'debit', 'credit', 'transaction_amount', 'amount_debited', 'amount_credited',
                      '支払総額', '利用金額', '支払金額', '取引金額']
    time_columns = ['time', 'transaction_time', 'time_posted', 'timestamp', '取引時刻', '利用時刻', '決済時刻']
    
    # Find the actual column names in the CSV
    date_col = None
    desc_col = None
    amount_col = None
    time_col = None
    
    for col in df.columns:
        col_lower = col.lower().strip()
        # Normalize quotes and BOM for exact-name checks in Japanese
        col_original = col.strip().lstrip('\ufeff').strip('"').strip("'")
        
        # Check English column names
        if col_lower in [name.lower() for name in date_columns]:
            date_col = col
        elif col_lower in [name.lower() for name in desc_columns]:
            desc_col = col
        elif col_lower in [name.lower() for name in amount_columns]:
            amount_col = col
        elif col_lower in [name.lower() for name in time_columns]:
            time_col = col
        
        # Check Japanese column names
        if col_original in date_columns:
            date_col = col
        elif col_original in desc_columns:
            desc_col = col
        elif col_original in amount_columns:
            amount_col = col
        elif col_original in time_columns:
            time_col = col

    # Prefer 支払総額 when available
    if amount_col:
        for col in df.columns:
            col_norm = col.strip().lstrip('\ufeff').strip('"').strip("'")
            if col_norm == '支払総額':
                amount_col = col
                break
    
    if not all([date_col, desc_col, amount_col]):
        # If we can't find the expected columns, show available columns and let user choose
        st.warning(f"Could not automatically identify columns. Available columns: {list(df.columns)}")
        st.write("Please select the correct columns:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            date_col = st.selectbox("Date column:", df.columns, index=0)
        with col2:
            desc_col = st.selectbox("Description column:", df.columns, index=1)
        with col3:
            amount_col = st.selectbox("Amount column:", df.columns, index=2)
    
    # Process the data with progress bar
    st.write("🔄 **Processing transactions...**")
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # ── Pass 1: parse rows (dates, amounts, descriptions) ──
    parsed_rows: list[dict] = []
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        try:
            progress = (idx + 1) / total_rows * 0.4  # 0-40% for parsing
            progress_bar.progress(progress)
            status_text.text(f"Parsing row {idx + 1} of {total_rows}…")
            
            # Handle different date formats
            date_str = str(row[date_col]).strip()
            if pd.isna(date_str) or date_str == '':
                continue
                
            date = None
            date_formats = ['%Y/%m/%d', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y']
            for fmt in date_formats:
                try:
                    date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            
            if date is None:
                continue
            
            # Extract timestamp if available
            timestamp = None
            if time_col:
                time_str = str(row[time_col]).strip()
                if not pd.isna(time_str) and time_str != '':
                    try:
                        time_formats = ['%H:%M:%S', '%H:%M', '%H.%M.%S', '%H.%M']
                        for fmt in time_formats:
                            try:
                                time_obj = datetime.strptime(time_str, fmt).time()
                                timestamp = time_obj
                                break
                            except ValueError:
                                continue
                    except Exception:
                        pass
                
            description = str(row[desc_col]).strip()
            if pd.isna(description) or description == '':
                continue
                
            amount_str = str(row[amount_col]).strip()
            if pd.isna(amount_str) or amount_str == '':
                continue
                
            amount_str = re.sub(r'[^\d.-]', '', amount_str)
            amount = float(amount_str)
            
            parsed_rows.append({
                "date": date,
                "timestamp": timestamp,
                "original_description": description,
                "amount": amount,
            })
            
        except Exception as e:
            st.warning(f"Error processing row {idx + 1}: {row}. Error: {e}")
            continue
    
    # ── Pass 2: batch translate all unique descriptions ──
    status_text.text("🌐 Translating descriptions…")
    progress_bar.progress(0.45)
    
    unique_descs = list({r["original_description"] for r in parsed_rows})
    
    # Use translation cache from session state
    if "translation_cache" not in st.session_state:
        st.session_state["translation_cache"] = {}
    cache: dict[str, str] = st.session_state["translation_cache"]
    
    # Split into cached and uncached
    uncached = [d for d in unique_descs if d not in cache]
    
    if uncached and translation_mode in ("AI-Powered (GPT-4o-mini)", "AI-Powered") and api_key:
        batch_results = translate_batch_ai(uncached, api_key, model=ai_model, base_url=ai_base_url)
        cache.update(batch_results)
    elif uncached and translation_mode == "Free Fallback":
        for i, desc in enumerate(uncached):
            cache[desc] = translate_japanese_to_english_fallback(desc)
            if len(uncached) > 1:
                progress_bar.progress(0.45 + 0.45 * (i + 1) / len(uncached))
                status_text.text(f"Translating {i + 1}/{len(uncached)}…")
    elif uncached and translation_mode == "No Translation":
        for desc in uncached:
            cache[desc] = desc
    
    progress_bar.progress(0.90)
    status_text.text("Assembling results…")
    
    # ── Pass 3: assemble final transactions list ──
    transactions = []
    for r in parsed_rows:
        original = r["original_description"]
        translated = cache.get(original, original)
        transactions.append({
            "date": r["date"],
            "timestamp": r["timestamp"],
            "description": translated,
            "original_description": original,
            "amount": r["amount"],
        })
    
    # Complete progress bar
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    
    if not transactions:
        raise RuntimeError("No valid transactions found in the CSV file.")
    
    # Clear progress elements
    progress_bar.empty()
    status_text.empty()
    
    df_result = pd.DataFrame(transactions)
    return df_result


@st.cache_data(show_spinner=False)
def get_fx_rate_to_jpy(currency: str, for_date: Optional[str]) -> float:
    """Fetch FX rate to JPY for a given date using exchangerate.host.

    Returns 1.0 for JPY or on error. for_date format: YYYY-MM-DD.
    """
    try:
        currency = (currency or "JPY").upper().strip()
        if currency == "JPY":
            return 1.0
        import requests
        # Historical if date provided, else latest
        if for_date:
            url = f"https://api.exchangerate.host/{for_date}"
        else:
            url = "https://api.exchangerate.host/latest"
        params = {"base": currency, "symbols": "JPY"}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        rate = float(data.get("rates", {}).get("JPY", 1.0))
        return rate if rate > 0 else 1.0
    except Exception:
        return 1.0


def enrich_currency_columns(df: pd.DataFrame, default_currency: str = "JPY") -> pd.DataFrame:
    df2 = df.copy()
    # If a currency column exists, use it; otherwise apply default
    currency_col = None
    for c in df2.columns:
        if str(c).lower().strip() in ["currency", "curr", "ccy"]:
            currency_col = c
            break
    if currency_col is None:
        df2["currency"] = default_currency
    else:
        df2.rename(columns={currency_col: "currency"}, inplace=True)

    # Compute fx_rate and amount_jpy per row
    fx_rates = []
    amts_jpy = []
    for _, r in df2.iterrows():
        date_val = r.get("date")
        if hasattr(date_val, "strftime"):
            date_str = date_val.strftime("%Y-%m-%d")
        else:
            date_str = str(date_val)
        ccy = str(r.get("currency", default_currency) or default_currency).upper()
        rate = get_fx_rate_to_jpy(ccy, date_str)
        fx_rates.append(rate)
        try:
            amt = float(r.get("amount", 0.0))
        except Exception:
            amt = 0.0
        amts_jpy.append(amt * rate)
    df2["fx_rate"] = fx_rates
    df2["amount_jpy"] = amts_jpy
    return df2

def validate_transaction_data(df: pd.DataFrame) -> dict:
    """Comprehensive data validation for transaction data."""
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'duplicates': [],
        'anomalies': []
    }
    
    try:
        # Check for duplicate transactions
        duplicate_mask = df.duplicated(subset=['date', 'description', 'amount'], keep=False)
        if duplicate_mask.any():
            duplicates = df[duplicate_mask]
            validation_results['duplicates'] = duplicates.to_dict('records')
            validation_results['warnings'].append(f"Found {len(duplicates)} duplicate transactions")
        
        # Check for amount anomalies
        if 'amount' in df.columns:
            amounts = df['amount']
            mean_amount = amounts.mean()
            std_amount = amounts.std()
            
            # Detect amounts that are 3 standard deviations from mean
            anomaly_mask = abs(amounts - mean_amount) > (3 * std_amount)
            if anomaly_mask.any():
                anomalies = df[anomaly_mask]
                validation_results['anomalies'] = anomalies.to_dict('records')
                validation_results['warnings'].append(f"Found {len(anomalies)} transactions with unusual amounts")
            
            # Check for zero amounts
            zero_amounts = df[amounts == 0]
            if len(zero_amounts) > 0:
                validation_results['warnings'].append(f"Found {len(zero_amounts)} transactions with zero amounts")
        
        # Check date range validity
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            min_date = dates.min()
            max_date = dates.max()
            current_date = pd.Timestamp.now()
            
            # Check for future dates
            future_dates = df[dates > current_date]
            if len(future_dates) > 0:
                validation_results['warnings'].append(f"Found {len(future_dates)} transactions with future dates")
            
            # Check for very old dates (more than 10 years ago)
            ten_years_ago = current_date - pd.DateOffset(years=10)
            old_dates = df[dates < ten_years_ago]
            if len(old_dates) > 0:
                validation_results['warnings'].append(f"Found {len(old_dates)} transactions older than 10 years")
        
        # Check for missing critical data
        missing_descriptions = df[df['description'].isna() | (df['description'] == '')]
        if len(missing_descriptions) > 0:
            validation_results['errors'].append(f"Found {len(missing_descriptions)} transactions with missing descriptions")
            validation_results['is_valid'] = False
        
        missing_amounts = df[df['amount'].isna()]
        if len(missing_amounts) > 0:
            validation_results['errors'].append(f"Found {len(missing_amounts)} transactions with missing amounts")
            validation_results['is_valid'] = False
        
        # Check for extreme amounts (suspicious transactions)
        if 'amount' in df.columns:
            amounts = df['amount']
            # Flag amounts over ¥1,000,000 as potentially suspicious
            large_amounts = df[abs(amounts) > 1000000]
            if len(large_amounts) > 0:
                validation_results['warnings'].append(f"Found {len(large_amounts)} transactions with amounts over ¥1,000,000")
        
    except Exception as e:
        validation_results['errors'].append(f"Validation error: {str(e)}")
        validation_results['is_valid'] = False
    
    return validation_results

def validate_financial_data(df: pd.DataFrame, expected_total: float = None) -> dict:
    """
    Comprehensive financial data validation following industry best practices.
    This is the type of validation used in professional financial applications.
    """
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'data_quality_score': 0.0,
        'reconciliation_status': 'unknown',
        'recommended_actions': []
    }
    
    try:
        # 1. Basic data integrity checks
        if df.empty:
            validation_results['errors'].append("No transaction data found")
            validation_results['is_valid'] = False
            return validation_results
        
        # 2. Amount data validation
        if 'amount' not in df.columns:
            validation_results['errors'].append("Missing 'amount' column")
            validation_results['is_valid'] = False
            return validation_results
        
        amounts = df['amount']
        
        # Check for invalid amounts
        invalid_amounts = amounts[~amounts.apply(lambda x: isinstance(x, (int, float)) and not pd.isna(x))]
        if len(invalid_amounts) > 0:
            validation_results['errors'].append(f"Found {len(invalid_amounts)} invalid amount values")
            validation_results['is_valid'] = False
        
        # Check for zero amounts (suspicious in financial data)
        zero_amounts = amounts[amounts == 0]
        if len(zero_amounts) > 0:
            validation_results['warnings'].append(f"Found {len(zero_amounts)} transactions with zero amounts")
        
        # 3. Duplicate detection (critical for financial accuracy)
        try:
            # Safe duplicate detection that handles mixed data types
            duplicate_mask = df.duplicated(subset=['date', 'description', 'amount'], keep=False)
            if duplicate_mask.any():
                duplicates = df[duplicate_mask]
                validation_results['warnings'].append(f"Found {len(duplicates)} duplicate transactions")
                validation_results['recommended_actions'].append("Review and remove duplicate transactions")
        except Exception as e:
            # Fallback to string-based duplicate detection
            try:
                df_string = df.astype(str)
                duplicate_mask = df_string.duplicated(subset=['date', 'description', 'amount'], keep=False)
                if duplicate_mask.any():
                    duplicates = df_string[duplicate_mask]
                    validation_results['warnings'].append(f"Found {len(duplicates)} potential duplicate transactions (string-based detection)")
                    validation_results['recommended_actions'].append("Review and remove duplicate transactions")
            except Exception as e2:
                validation_results['warnings'].append("Unable to detect duplicates due to data type complexity")
                validation_results['recommended_actions'].append("Manual review of transactions recommended")
        
        # 4. Amount range validation
        min_amount = amounts.min()
        max_amount = amounts.max()
        mean_amount = amounts.mean()
        
        # Flag suspicious amounts
        if abs(min_amount) > 1000000:  # Over ¥1M
            validation_results['warnings'].append(f"Found extremely large negative amount: ¥{min_amount:,.0f}")
        if max_amount > 1000000:  # Over ¥1M
            validation_results['warnings'].append(f"Found extremely large positive amount: ¥{max_amount:,.0f}")
        
        # 5. Financial reconciliation (if expected total provided)
        if expected_total is not None:
            actual_total = amounts.abs().sum()
            difference = abs(actual_total - expected_total)
            difference_percentage = (difference / expected_total) * 100
            
            validation_results['reconciliation_status'] = 'reconciled' if difference < 100 else 'unreconciled'
            
            if difference > 100:  # More than ¥100 difference
                validation_results['errors'].append(f"Financial reconciliation failed: Expected ¥{expected_total:,.0f}, Got ¥{actual_total:,.0f}")
                validation_results['errors'].append(f"Difference: ¥{difference:,.0f} ({difference_percentage:.2f}%)")
                validation_results['is_valid'] = False
                
                # Provide specific recommendations
                if difference_percentage > 10:
                    validation_results['recommended_actions'].append("Large discrepancy detected - verify source data integrity")
                elif difference_percentage > 5:
                    validation_results['recommended_actions'].append("Moderate discrepancy - check for missing or duplicate transactions")
                else:
                    validation_results['recommended_actions'].append("Small discrepancy - verify individual transaction amounts")
        
        # 6. Data quality scoring
        quality_score = 100.0
        
        # Deduct points for issues
        if len(duplicate_mask[duplicate_mask]) > 0:
            quality_score -= 20
        if len(zero_amounts) > 0:
            quality_score -= 10
        if len(invalid_amounts) > 0:
            quality_score -= 30
        
        validation_results['data_quality_score'] = max(0, quality_score)
        
        # 7. Professional recommendations
        if validation_results['data_quality_score'] < 70:
            validation_results['recommended_actions'].append("Data quality is poor - manual review recommended")
        elif validation_results['data_quality_score'] < 90:
            validation_results['recommended_actions'].append("Data quality is acceptable but could be improved")
        else:
            validation_results['recommended_actions'].append("Data quality is excellent")
        
    except Exception as e:
        validation_results['errors'].append(f"Validation error: {str(e)}")
        validation_results['is_valid'] = False
    
    return validation_results

def reconcile_financial_data(df: pd.DataFrame, expected_total: float) -> dict:
    """
    Financial reconciliation system - finds the exact cause of discrepancies.
    This is what professional financial software uses to resolve mismatches.
    """
    reconciliation = {
        'status': 'unreconciled',
        'expected_total': expected_total,
        'actual_total': 0.0,
        'difference': 0.0,
        'difference_percentage': 0.0,
        'root_causes': [],
        'suggested_fixes': [],
        'data_issues': []
    }
    
    try:
        # Calculate actual total
        actual_total = df['amount'].abs().sum()
        reconciliation['actual_total'] = actual_total
        
        # Calculate difference
        difference = abs(actual_total - expected_total)
        reconciliation['difference'] = difference
        reconciliation['difference_percentage'] = (difference / expected_total) * 100
        
        # Root cause analysis
        if difference > 0:
            # Check for common financial data issues
            
            # 1. Duplicate transactions
            try:
                duplicate_mask = df.duplicated(subset=['date', 'description', 'amount'], keep=False)
                if duplicate_mask.any():
                    duplicates = df[duplicate_mask]
                    duplicate_total = duplicates['amount'].abs().sum()
                    reconciliation['root_causes'].append(f"Duplicate transactions: {len(duplicates)} duplicates worth ¥{duplicate_total:,.0f}")
                    reconciliation['suggested_fixes'].append("Remove duplicate transactions")
            except Exception as e:
                # Fallback to string-based duplicate detection
                try:
                    df_string = df.astype(str)
                    duplicate_mask = df_string.duplicated(subset=['date', 'description', 'amount'], keep=False)
                    if duplicate_mask.any():
                        duplicates = df_string[duplicate_mask]
                        reconciliation['root_causes'].append(f"Potential duplicate transactions: {len(duplicates)} potential duplicates detected")
                        reconciliation['suggested_fixes'].append("Review and remove duplicate transactions")
                except Exception as e2:
                    reconciliation['root_causes'].append("Unable to detect duplicates due to data type complexity")
                    reconciliation['suggested_fixes'].append("Manual review of transactions recommended")
            
            # 2. Extreme amounts that might be errors
            extreme_amounts = df[df['amount'].abs() > 100000]
            if len(extreme_amounts) > 0:
                reconciliation['root_causes'].append(f"Extreme amounts: {len(extreme_amounts)} transactions over ¥100,000")
                reconciliation['suggested_fixes'].append("Verify extreme amounts are correct")
            
            # 3. Zero amounts
            zero_amounts = df[df['amount'] == 0]
            if len(zero_amounts) > 0:
                reconciliation['root_causes'].append(f"Zero amounts: {len(zero_amounts)} transactions with ¥0")
                reconciliation['suggested_fixes'].append("Review zero-amount transactions")
            
            # 4. Amount distribution analysis
            positive_amounts = df[df['amount'] > 0]['amount'].sum()
            negative_amounts = abs(df[df['amount'] < 0]['amount'].sum())
            
            reconciliation['data_issues'].append(f"Positive amounts total: ¥{positive_amounts:,.0f}")
            reconciliation['data_issues'].append(f"Negative amounts total: ¥{negative_amounts:,.0f}")
            
            # 5. Check if the issue is in transaction classification
            if 'transaction_type' in df.columns:
                expense_total = df[df['transaction_type'] == 'Expense']['amount'].abs().sum()
                credit_total = df[df['transaction_type'] == 'Credit']['amount'].abs().sum()
                reconciliation['data_issues'].append(f"Expense transactions total: ¥{expense_total:,.0f}")
                reconciliation['data_issues'].append(f"Credit transactions total: ¥{credit_total:,.0f}")
        
        # Determine reconciliation status
        if difference < 100:  # Within ¥100
            reconciliation['status'] = 'reconciled'
        elif difference < 1000:  # Within ¥1,000
            reconciliation['status'] = 'minor_discrepancy'
        else:
            reconciliation['status'] = 'major_discrepancy'
            
    except Exception as e:
        reconciliation['root_causes'].append(f"Reconciliation error: {str(e)}")
    
    return reconciliation

def calculate_running_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate running balance for transactions."""
    df_balance = df.copy()
    
    # Sort by date to ensure chronological order
    df_balance = df_balance.sort_values('date')
    
    # Initialize running balance
    running_balance = 0
    balance_history = []
    
    for idx, row in df_balance.iterrows():
        amount = row['amount']
        
        # Add to running balance
        running_balance += amount
        balance_history.append(running_balance)
    
    # Add running balance column
    df_balance['running_balance'] = balance_history
    
    return df_balance

def detect_transaction_type(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and classify transactions as credit (income) or debit (expense)."""
    
    df = df.copy()
    
    # Initialize transaction type column
    df['transaction_type'] = 'Expense'  # Default to expense
    
    # Check for common credit indicators
    credit_keywords = [
        'refund', 'credit', 'payment', 'adjustment', 'reversal', 'return',
        '返金', 'クレジット', '支払い', '調整', '取消', '返品',
        'statement credit', 'cashback', 'rewards', 'bonus'
    ]
    
    # Check for common debit indicators
    debit_keywords = [
        'purchase', 'charge', 'fee', 'withdrawal', 'cash advance',
        '購入', 'チャージ', '手数料', '引き出し', '現金引き出し'
    ]
    
    for idx, row in df.iterrows():
        description = str(row.get('description', '')).lower()
        original_desc = str(row.get('original_description', '')).lower()
        amount = row.get('amount', 0)
        
        # Check description keywords first
        is_credit = any(keyword in description or keyword in original_desc for keyword in credit_keywords)
        is_debit = any(keyword in description or keyword in original_desc for keyword in debit_keywords)
        
        # Determine transaction type based on keywords and amount sign
        # For Japanese bank statements, most transactions are expenses (outflows)
        if is_credit and not is_debit:
            df.loc[idx, 'transaction_type'] = 'Credit'
        elif is_debit and not is_credit:
            df.loc[idx, 'transaction_type'] = 'Expense'
        else:
            # If no clear keyword match, use amount sign as indicator
            # For Japanese bank statements: negative amounts = expenses (outflows)
            if amount < 0:
                df.loc[idx, 'transaction_type'] = 'Expense'
            else:
                # For Japanese bank statements, default to Expense for positive amounts
                # unless there are clear credit indicators
                df.loc[idx, 'transaction_type'] = 'Expense'
        
        # Force classification for common Japanese transaction patterns
        # Most Japanese bank transactions are expenses, not credits
        if 'visa' in description.lower() or 'visa' in original_desc.lower():
            if 'domestic' in description.lower() or 'domestic' in original_desc.lower():
                # VISA domestic transactions are almost always expenses
                df.loc[idx, 'transaction_type'] = 'Expense'
        
        # Override for obvious credits (refunds, payments, etc.)
        credit_indicators = ['refund', '返金', 'payment', '支払い', 'adjustment', '調整']
        if any(indicator in description.lower() or indicator in original_desc.lower() for indicator in credit_indicators):
            df.loc[idx, 'transaction_type'] = 'Credit'
        
        # Keep original amount value - don't change it
    
    return df

def categorise_transactions_ai(
    df: pd.DataFrame,
    api_key: str,
    categories: list[str],
    subcategories: dict | None = None,
    batch_size: int = 30,
    model: str | None = None,
    base_url: str | None = None,
) -> pd.DataFrame:
    """Categorise transactions using AI (OpenAI or Gemini) for highest accuracy.

    Sends batches of transactions (original Japanese + translated English) to the
    model and asks it to pick the best category and subcategory from the provided
    lists.  Falls back to rule-based categorisation on failure.
    """
    import json as _json

    api_key = api_key or _default_key_for(base_url)
    model = model or _default_model_for(base_url)
    if not api_key:
        st.warning("No API key – falling back to rule-based categorisation.")
        return df  # caller should fall back

    # Build a compact description of valid categories + subcategories for the prompt
    cat_desc_parts: list[str] = []
    for cat in categories:
        subs = list((subcategories or {}).get(cat, {}).keys())
        if subs:
            cat_desc_parts.append(f"- {cat} (subcategories: {', '.join(subs)})")
        else:
            cat_desc_parts.append(f"- {cat}")
    categories_text = "\n".join(cat_desc_parts)

    df = df.copy()
    result_categories: list[str] = ["Uncategorised"] * len(df)
    result_subcategories: list[str] = [""] * len(df)
    result_confidences: list[float] = [0.0] * len(df)

    indices = list(range(len(df)))

    for batch_start in range(0, len(indices), batch_size):
        batch_idx = indices[batch_start:batch_start + batch_size]

        lines: list[str] = []
        for pos, i in enumerate(batch_idx, start=1):
            row = df.iloc[i]
            desc = row.get("description", "")
            orig = row.get("original_description", "")
            amt = row.get("amount", 0)
            lines.append(f'{pos}. "{orig}" / "{desc}" / amount={amt}')

        numbered_block = "\n".join(lines)

        try:
            raw = _ai_generate(
                messages=[
                    {"role": "system", "content": (
                        "You categorise personal finance transactions from Japanese bank "
                        "and credit-card statements.\n\n"
                        "Each transaction has:\n"
                        "- An original Japanese description (abbreviated merchant/payee name)\n"
                        "- An English translation\n"
                        "- An amount (negative = expense, positive = income)\n\n"
                        "IMPORTANT: Use your knowledge of Japanese businesses to identify "
                        "what each merchant actually is. Many entries are abbreviated names "
                        "of real businesses in Japan. For example:\n"
                        "- ｼｮｳｺﾃｲ is a Chinese restaurant (小籠亭)\n"
                        "- ﾄﾘｷｿﾞｸ is Torikizoku (鳥貴族), an izakaya chain\n"
                        "- ﾏﾂﾓﾄｷﾖｼ is Matsumoto Kiyoshi, a drugstore\n"
                        "- ｲｵﾝ is AEON, a supermarket\n"
                        "Identify the actual business and categorise based on what it is "
                        "(restaurant → Food, drugstore → Grooming, etc.), not just the "
                        "literal text translation.\n\n"
                        f"VALID CATEGORIES:\n{categories_text}\n\n"
                        "If no subcategory fits, leave it as an empty string.\n"
                        "Reply with ONLY a JSON array, one object per transaction, in order: "
                        '[{"n":1,"cat":"Food","sub":"Dinner/Eating Out","conf":0.95}, ...]\n'
                        "Return ONLY valid JSON, no markdown fences, no extra text."
                    )},
                    {"role": "user", "content": numbered_block},
                ],
                model=model,
                api_key=api_key,
                max_tokens=max(1500, 120 * len(batch_idx)),
                temperature=0.0,
                base_url=base_url,
                json_mode=True,
            )
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            items = _json.loads(raw)

            for item in items:
                pos = int(item.get("n", 0)) - 1  # 0-based within batch
                if 0 <= pos < len(batch_idx):
                    global_idx = batch_idx[pos]
                    cat = item.get("cat", "Uncategorised")
                    sub = item.get("sub", "") or ""
                    # Validate category is in our list
                    if cat not in categories:
                        cat = "Uncategorised"
                        sub = ""
                    # Validate subcategory belongs to its parent category
                    valid_subs = list((subcategories or {}).get(cat, {}).keys())
                    if sub and valid_subs and sub not in valid_subs:
                        sub = ""
                    result_categories[global_idx] = cat
                    result_subcategories[global_idx] = sub
                    result_confidences[global_idx] = float(item.get("conf", 0.85))

        except Exception as e:
            st.warning(f"AI categorisation batch error: {e}. Affected rows will use rule-based fallback.")
            # Leave those entries as Uncategorised – they'll be caught by the caller

    df["category"] = result_categories
    df["subcategory"] = result_subcategories
    df["confidence"] = result_confidences
    return df


def categorise_transactions(
    df: pd.DataFrame, rules, subcategories = None, 
    uncategorised_label: str = "Uncategorised"
) -> pd.DataFrame:
    """Enhanced categorization with support for main categories and subcategories.

    Matches keywords against BOTH the translated ``description`` and the
    ``original_description`` (Japanese) so that categorization still works
    even when translation quality is imperfect.
    """
    
    # Create patterns for main categories
    patterns = {cat: re.compile("(" + "|".join(map(re.escape, kws)) + ")", re.IGNORECASE) for cat, kws in rules.items()}
    
    # Create patterns for subcategories
    sub_patterns = {}
    if subcategories:
        for main_cat, subs in subcategories.items():
            for sub_cat, keywords in subs.items():
                sub_patterns[f"{main_cat}_{sub_cat}"] = {
                    'main': main_cat,
                    'sub': sub_cat,
                    'pattern': re.compile("(" + "|".join(map(re.escape, keywords)) + ")", re.IGNORECASE)
                }
    
    has_original = "original_description" in df.columns
    
    categories = []
    subcategories_list = []
    
    for idx, row in df.iterrows():
        desc = str(row.get("description", ""))
        orig = str(row.get("original_description", "")) if has_original else ""
        # Combine both descriptions for matching
        combined = f"{desc} {orig}"
        
        assigned_category = uncategorised_label
        assigned_subcategory = ""
        
        # First try to match subcategories for more specific categorization
        for sub_key, sub_info in sub_patterns.items():
            if sub_info['pattern'].search(combined):
                assigned_category = sub_info['main']
                assigned_subcategory = sub_info['sub']
                break
        
        # If no subcategory match, try main categories
        if assigned_category == uncategorised_label:
            for cat, pattern in patterns.items():
                if pattern.search(combined):
                    assigned_category = cat
                    break
        
        categories.append(assigned_category)
        subcategories_list.append(assigned_subcategory)
    
    df = df.copy()
    df["category"] = categories
    df["subcategory"] = subcategories_list
    
    return df


def apply_smart_categorization(df: pd.DataFrame, learning_system, 
                              rules, subcategories) -> pd.DataFrame:
    """Apply smart categorization using the learning system."""
    
    df = df.copy()
    categories = []
    subcategories_list = []
    confidences = []
    prediction_breakdowns = []
    
    for idx, row in df.iterrows():
        # Prepare transaction data for prediction
        transaction_data = {
            'description': row.get('description', ''),
            'original_description': row.get('original_description', ''),
            'amount': row.get('amount', 0),
            'date': row.get('date'),
            'transaction_type': row.get('transaction_type', 'Expense')
        }
        
        # Get smart prediction
        try:
            if learning_system:
                predicted_category, confidence, breakdown = learning_system.predict_category(transaction_data)
            else:
                predicted_category = "Uncategorised"
                confidence = 0.0
                breakdown = {'method': 'fallback', 'confidence': 0.0}
        except Exception as e:
            # Fallback to basic categorization if learning system fails
            predicted_category = "Uncategorised"
            confidence = 0.0
            breakdown = {'method': 'fallback', 'confidence': 0.0, 'error': str(e)}
        
        # Apply subcategory if available
        predicted_subcategory = ""
        if predicted_category in subcategories:
            # Find best matching subcategory
            for sub_cat, keywords in subcategories[predicted_category].items():
                if any(keyword.lower() in str(transaction_data['description']).lower() 
                       for keyword in keywords):
                    predicted_subcategory = sub_cat
                    break
        
        categories.append(predicted_category)
        subcategories_list.append(predicted_subcategory)
        confidences.append(confidence)
        prediction_breakdowns.append(breakdown)
    
    df['category'] = categories
    df['subcategory'] = subcategories_list
    df['confidence'] = confidences
    df['prediction_breakdown'] = prediction_breakdowns
    
    return df


def save_custom_rules(rules, filename: str = "custom_rules.json") -> None:
    """Save custom categorization rules to a JSON file."""
    import json
    try:
        with open(filename, 'w') as f:
            json.dump(rules, f, indent=2)
        st.success(f"Custom rules saved to {filename}")
    except Exception as e:
        st.error(f"Error saving rules: {e}")

def load_custom_rules(filename: str = "custom_rules.json"):
    """Load custom categorization rules from a JSON file."""
    import json
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        st.error(f"Error loading rules: {e}")
        return {}


def main() -> None:
    # Initialize database (ensure all tables exist) with safe fallback if import failed
    try:
        if 'init_db' in globals() and callable(init_db):  # type: ignore[name-defined]
            init_db()  # type: ignore[misc]
        else:
            try:
                import data_store as _ds  # type: ignore
                if hasattr(_ds, 'init_db') and callable(_ds.init_db):
                    _ds.init_db()
                else:
                    st.warning("Database init function not found; proceeding without migration.")
            except Exception:
                st.warning("Database module unavailable; proceeding without DB migration.")
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
    
    # Require authentication (single-user gate if configured)
    if not require_auth():
        return

    st.title("💰 AI Expense Tracker")
    
    # Concise onboarding
    with st.expander("ℹ️ Quick Start Guide", expanded=False):
        st.markdown("""
        **Get started in 3 steps:**
        1. **Upload** your bank statement (CSV/PDF/Image)
        2. **Review** auto-categorized transactions (JPY default)
        3. **Save** to database and export/backup as needed
        
        **Features:**
        - 🌐 Japanese → English translation (AI-powered or free)
        - 🔄 Multi-currency with auto-conversion to JPY
        - 🎯 Smart categorization with learning
        - 🔍 Duplicate detection (configurable)
        - 🔁 Recurring transaction management
        - ☁️ Google Drive backup (optional)
        """)

    # Resume Work Section
    if get_all_active_categorization_sessions and load_session_transactions:
        st.divider()
        st.subheader("🔄 Resume Previous Work")
        
        # Get all active sessions
        active_sessions = get_all_active_categorization_sessions()
        
        if active_sessions:
            st.info(f"📋 **{len(active_sessions)} active categorization session(s) found**")
            
            for session in active_sessions:
                with st.expander(f"📄 {session['file_name']} (Session {session['id']})", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("📊 Total", session['total_transactions'])
                    with col2:
                        st.metric("✅ Reviewed", session['reviewed_count'])
                    with col3:
                        st.metric("🏷️ Categorized", session['categorized_count'])
                    with col4:
                        completion = round((session['reviewed_count'] / session['total_transactions']) * 100, 1) if session['total_transactions'] > 0 else 0
                        st.metric("📈 Progress", f"{completion}%")
                    
                    # Progress bar
                    st.progress(completion / 100)
                    
                    # Session details
                    st.caption(f"**Started:** {session['started_at'][:19]}")
                    st.caption(f"**Last updated:** {session['last_updated'][:19]}")
                    st.caption(f"**Status:** {session['status']}")
                    st.caption(f"**Total transactions:** {session['total_transactions']}")
                    st.caption(f"**Reviewed transactions:** {session['reviewed_transactions']}")
                    
                    # Debug information
                    with st.expander("🔍 Debug Info", expanded=False):
                        st.json(session)
                        
                        # Check if categorization_progress table has data for this session
                        try:
                            from data_store import get_connection
                            import sqlite3
                            
                            conn = get_connection()
                            cur = conn.cursor()
                            cur.execute("SELECT COUNT(*) FROM categorization_progress WHERE session_id = ?", (session['id'],))
                            progress_count = cur.fetchone()[0]
                            st.write(f"**Transactions in categorization_progress:** {progress_count}")
                            
                            if progress_count > 0:
                                cur.execute("SELECT * FROM categorization_progress WHERE session_id = ? LIMIT 3", (session['id'],))
                                sample_data = cur.fetchall()
                                st.write("**Sample data:**")
                                for row in sample_data:
                                    st.write(f"- {row}")
                            
                            conn.close()
                        except Exception as e:
                            st.error(f"Debug error: {e}")
                    
                    # Resume button
                    if st.button(f"🔄 Resume Session {session['id']}", key=f"resume_{session['id']}"):
                        # Load session transactions
                        session_transactions = load_session_transactions(session['id'])
                        
                        if session_transactions:
                            # Convert to DataFrame format
                            df_resume = pd.DataFrame(session_transactions)
                            
                            # Set session state
                            st.session_state['categorization_session_id'] = session['id']
                            st.session_state['resume_mode'] = True
                            st.session_state['resume_data'] = df_resume.to_dict('records')
                            
                            st.success(f"🔄 Resumed session {session['id']} with {len(session_transactions)} transactions!")
                            st.rerun()
                        else:
                            # Provide more detailed error information
                            st.error("❌ No transactions found for this session")
                            st.caption(f"Session ID: {session['id']}, File: {session['file_name']}")
                            st.caption("This might happen if:")
                            st.caption("• The session was created but no transactions were saved")
                            st.caption("• The session data was corrupted")
                            st.caption("• The database was reset")
                            
                            # Offer to delete the empty session
                            if st.button(f"🗑️ Delete Empty Session {session['id']}", key=f"delete_empty_{session['id']}"):
                                try:
                                    if complete_categorization_session:
                                        complete_categorization_session(session['id'])
                                        st.success("✅ Empty session deleted!")
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error deleting session: {e}")
                    
                    # Delete session button
                    if st.button(f"🗑️ Delete Session {session['id']}", key=f"delete_{session['id']}"):
                        # Mark session as abandoned
                        if complete_categorization_session:
                            complete_categorization_session(session['id'])
                            st.success(f"🗑️ Session {session['id']} deleted!")
                            st.rerun()
        else:
            st.info("📭 No active categorization sessions found. Upload a file to start categorizing!")
    
    # Initialize database (first run safe)
    if init_db:
        try:
            init_db()
            st.sidebar.success("🗄️ Local database ready")
        except Exception as e:
            st.sidebar.warning(f"DB init failed: {e}")
    
    # Initialize session state for progress tracking
    if 'categorization_progress' not in st.session_state:
        st.session_state['categorization_progress'] = None
    if 'progress_saved' not in st.session_state:
        st.session_state['progress_saved'] = False
    
    # Initialize Merchant Learning System
    learning_system = MerchantLearningSystem()
    st.sidebar.success("🧠 Merchant Learning System Active")
    
    # Initialize Advanced AI Engines
    if ADVANCED_FEATURES_AVAILABLE:
        if 'ensemble_engine' not in st.session_state:
            st.session_state['ensemble_engine'] = EnsembleCategorizationEngine()
        
        if 'insights_engine' not in st.session_state:
            st.session_state['insights_engine'] = InsightsEngine()
        
        if 'dashboard' not in st.session_state:
            st.session_state['dashboard'] = ModernDashboard()
        
        if 'ml_trainer' not in st.session_state:
            st.session_state['ml_trainer'] = LocalMLTrainer()
        
        st.sidebar.success("✨ Advanced AI Features Active")
    
    # AI Translation Setup
    st.sidebar.header("🤖 AI Translation Settings")

    # ------------------------------------------------------------------
    # Load API keys: secrets.toml → environment variable → manual input
    # ------------------------------------------------------------------
    openai_key = ""
    gemini_key = ""
    try:
        openai_key = st.secrets.get("openai", {}).get("api_key", "")
    except Exception:
        pass
    try:
        gemini_key = st.secrets.get("gemini", {}).get("api_key", "")
    except Exception:
        pass
    if not openai_key:
        openai_key = os.getenv("OPENAI_API_KEY", "")
    if not gemini_key:
        gemini_key = os.getenv("GEMINI_API_KEY", "")

    # Determine which providers are available
    has_openai = bool(openai_key)
    has_gemini = bool(gemini_key)
    has_any_key = has_openai or has_gemini

    # Provider selection
    if has_openai and has_gemini:
        ai_provider = st.sidebar.selectbox(
            "AI Provider",
            ["Gemini (Google)", "OpenAI"],
            index=0,
            help="Gemini has a free tier. OpenAI offers GPT-4o models."
        )
    elif has_gemini:
        ai_provider = "Gemini (Google)"
        st.sidebar.success("✅ Gemini API key loaded")
    elif has_openai:
        ai_provider = "OpenAI"
        st.sidebar.success("✅ OpenAI API key loaded")
    else:
        # Neither key found — let user pick provider and enter key manually
        ai_provider = st.sidebar.selectbox(
            "AI Provider",
            ["Gemini (Google)", "OpenAI"],
            index=0,
            help="Gemini has a free tier (15 req/min). OpenAI offers GPT-4o models."
        )

    # Resolve the active API key and base_url for the chosen provider
    if ai_provider == "Gemini (Google)":
        api_key = gemini_key
        ai_base_url = GEMINI_PROVIDER
        provider_models = ["gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
        model_help = (
            "gemini-3.1-flash-lite: newest, optimized for translation (free). "
            "gemini-2.5-flash: best quality (free, 10 req/min). "
            "gemini-2.5-flash-lite: fast & generous quota (free, 15 req/min, 1k/day)."
        )
        key_url = "https://aistudio.google.com/apikey"
        secrets_section = "gemini"
    else:
        api_key = openai_key
        ai_base_url = None  # default OpenAI endpoint
        provider_models = ["gpt-4o-mini", "gpt-4o"]
        model_help = "gpt-4o-mini: fast & cheap (~$0.01/100 txns). gpt-4o: best quality (~$0.15/100 txns)."
        key_url = "https://platform.openai.com/api-keys"
        secrets_section = "openai"

    # If no key for the selected provider, allow manual entry
    if not api_key:
        api_key = st.sidebar.text_input(
            f"{ai_provider} API Key",
            type="password",
            help=f"Get your key from {key_url}"
        )
        if api_key:
            st.sidebar.success("✅ API key configured for this session")

    # Persist key to env so downstream functions can find it
    if api_key:
        if ai_provider == "OpenAI":
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            os.environ["GEMINI_API_KEY"] = api_key

    # Quick setup guide when no key is configured
    if not api_key:
        with st.sidebar.expander("🚀 Setup: Save your API key permanently"):
            st.write(f"""
            **One-time setup** so you never have to paste it again:

            1. Get your key from [{key_url}]({key_url})
            2. Open the file `.streamlit/secrets.toml` in this project
            3. Paste your key between the quotes:
            ```
            [{secrets_section}]
            api_key = "your-key-here"
            ```
            4. Restart the app — the key loads automatically!
            """)

    # Translation mode and model selection
    if api_key:
        translation_mode = st.sidebar.selectbox(
            "Translation Mode",
            ["AI-Powered", "Free Fallback", "No Translation"],
            index=0,  # Default to AI-Powered when API key is available
            help="AI-Powered uses your selected AI provider for best accuracy, Free Fallback uses Google Translate"
        )
        ai_model = st.sidebar.selectbox(
            "AI Model",
            provider_models,
            index=0,
            help=model_help,
        )
    else:
        translation_mode = "Free Fallback"
        ai_model = "gpt-4o-mini"
        ai_base_url = None
        st.sidebar.info("ℹ️ Using free translation (add an API key for AI accuracy)")
    
    # Smart Learning Dashboard
    if learning_system:
        st.sidebar.header("🧠 Smart Learning Dashboard")
        
        # Show learning statistics
        stats = learning_system.get_learning_stats()
        st.sidebar.write(f"**Total Corrections:** {stats['total_corrections']}")
        st.sidebar.write(f"**Unique Merchants:** {stats['unique_merchants']}")
        st.sidebar.write(f"**Recent Corrections:** {stats['recent_corrections']}")
        
        # Show merchant learning progress
        if stats['merchant_categories']:
            st.sidebar.write("**Learning Progress:**")
            for merchant, categories in list(stats['merchant_categories'].items())[:5]:  # Show first 5
                if merchant != "unknown":
                    category_names = list(categories.keys())
                    st.sidebar.write(f"• {merchant[:20]}: {category_names[0] if category_names else 'None'}")
        
        # Show confidence scores
        if stats['confidence_scores']:
            st.sidebar.write("**Confidence Levels:**")
            high_confidence = sum(1 for merchant_scores in stats['confidence_scores'].values() 
                                for score in merchant_scores.values() if score >= 0.8)
            st.sidebar.write(f"• High Confidence: {high_confidence}")
    
    # View Saved Transactions Section
    st.divider()
    
    # Check if there are any transactions to show a notification
    try:
        if load_all_transactions is not None:
            saved_count = len(load_all_transactions() or [])
            if saved_count > 0:
                st.info(f"📊 **{saved_count} transactions saved** - Expand 'View Saved Transactions' below to analyze your data!")
    except Exception:
        pass
    
    with st.expander("📊 View Saved Transactions", expanded=False):
        if load_all_transactions is not None:
            try:
                saved_transactions = load_all_transactions()
                
                if saved_transactions and len(saved_transactions) > 0:
                    df_saved = pd.DataFrame(saved_transactions)
                    
                    # Summary statistics
                    st.subheader("📈 Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Transactions", len(df_saved))
                    with col2:
                        total_expenses = df_saved[df_saved['transaction_type'] == 'Expense']['amount_jpy'].sum()
                        st.metric("Total Expenses", f"¥{abs(total_expenses):,.0f}")
                    with col3:
                        total_credits = df_saved[df_saved['transaction_type'] == 'Credit']['amount_jpy'].sum()
                        st.metric("Total Credits", f"¥{total_credits:,.0f}")
                    with col4:
                        net = total_credits + total_expenses  # expenses are negative
                        st.metric("Net", f"¥{net:,.0f}", delta_color="normal")
                    
                    # Date range info
                    if 'date' in df_saved.columns:
                        # Handle dates that may have time components
                        df_saved['date'] = pd.to_datetime(df_saved['date'], errors='coerce')
                        min_date = df_saved['date'].min().strftime('%Y-%m-%d')
                        max_date = df_saved['date'].max().strftime('%Y-%m-%d')
                        st.caption(f"📅 Date range: {min_date} to {max_date}")
                    
                    st.divider()
                    
                    # Duplicate Analysis Section
                    st.subheader("🔍 Duplicate Analysis")
                    st.caption("Analyze which transactions were marked as duplicates during import")
                    
                    # Check for duplicates in the current dataset
                    if 'dedupe_hash' in df_saved.columns:
                        # Group by dedupe hash to find duplicates
                        hash_counts = df_saved['dedupe_hash'].value_counts()
                        duplicates = hash_counts[hash_counts > 1]
                        
                        if len(duplicates) > 0:
                            st.warning(f"⚠️ Found {len(duplicates)} duplicate groups in your data")
                            
                            # Show duplicate groups
                            with st.expander(f"View {len(duplicates)} Duplicate Groups", expanded=False):
                                for i, (dedupe_hash, count) in enumerate(duplicates.items(), 1):
                                    duplicate_txs = df_saved[df_saved['dedupe_hash'] == dedupe_hash]
                                    
                                    st.write(f"**Group {i}** ({count} transactions):")
                                    
                                    # Show details of duplicate transactions
                                    for idx, tx in duplicate_txs.iterrows():
                                        st.write(f"  • {tx['date']} | {tx['description']} | ${tx['amount']:.2f} | {tx['category']}")
                                    
                                    # Show why they're considered duplicates
                                    first_tx = duplicate_txs.iloc[0]
                                    st.caption(f"   Hash: {dedupe_hash[:16]}... | Date: {first_tx['date']} | Description: '{first_tx['description']}' | Amount: ${first_tx['amount']:.2f}")
                                    st.divider()
                        else:
                            st.success("✅ No duplicate transactions found in current data")
                    else:
                        st.info("ℹ️ Duplicate analysis not available - dedupe_hash column missing")
                    
                    # Import History and Discarded Duplicates
                    st.subheader("📋 Import History & Discarded Duplicates")

                    try:
                        from data_store import get_import_history, get_discarded_duplicates, restore_discarded_duplicate

                        # Get import history
                        import_history = get_import_history()

                        if import_history:
                            # Top-level totals across shown imports
                            visible_history = import_history[:5]
                            total_inserted = sum(int(h.get('inserted_count', 0)) for h in visible_history)
                            total_discarded = sum(int(h.get('discarded_count', 0)) for h in visible_history)

                            mcol1, mcol2, mcol3 = st.columns([1, 1, 1])
                            with mcol1:
                                st.metric("Imports (latest 5)", len(visible_history))
                            with mcol2:
                                st.metric("Inserted", total_inserted)
                            with mcol3:
                                st.metric("Discarded", total_discarded)

                            st.caption("Click a card to view and restore discarded items")

                            for import_record in visible_history:
                                header = (
                                    f"📁 {import_record['file_name']}  ·  "
                                    f"Imported: {import_record['imported_at']}  ·  "
                                    f"Inserted: {import_record['inserted_count']}  ·  "
                                    f"Discarded: {import_record['discarded_count']}"
                                )

                                with st.expander(header, expanded=False):
                                    if int(import_record.get('discarded_count', 0)) > 0:
                                        discarded = get_discarded_duplicates(import_record['id'])

                                        if discarded:
                                            for i, discard in enumerate(discarded):
                                                col_d1, col_d2 = st.columns([5, 1])

                                                with col_d1:
                                                    currency_symbol = "¥"
                                                    st.write(
                                                        f"• {discard['date']} | {discard['description']} | "
                                                        f"{currency_symbol}{discard['amount']:.2f}"
                                                    )
                                                    st.caption(
                                                        f"Reason: {discard['reason']} | "
                                                        f"Hash: {discard['dedupe_hash'][:16]}..."
                                                    )

                                                with col_d2:
                                                    if st.button("Restore", key=f"restore_{discard['id']}"):
                                                        if restore_discarded_duplicate(discard['id']):
                                                            st.success("✅ Restored")
                                                            st.rerun()
                                                        else:
                                                            st.error("❌ Failed")
                                        else:
                                            st.info("No discarded duplicates for this import")
                                    else:
                                        st.success("No discarded duplicates in this import")
                        else:
                            st.info("No import history found")

                    except Exception as e:
                        st.error(f"Error loading import history: {e}")
                    
                    st.divider()
                    
                    # Filters for saved transactions
                    st.subheader("🔍 Filter Saved Transactions")
                    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1, 1, 1, 1])
                    
                    with filter_col1:
                        saved_categories = st.multiselect(
                            "Categories",
                            options=sorted(df_saved['category'].dropna().unique()) if 'category' in df_saved.columns else [],
                            default=None,
                            key="saved_cat_filter"
                        )
                    
                    with filter_col2:
                        saved_types = st.multiselect(
                            "Type",
                            options=df_saved['transaction_type'].dropna().unique() if 'transaction_type' in df_saved.columns else [],
                            default=None,
                            key="saved_type_filter"
                        )
                    
                    with filter_col3:
                        if 'date' in df_saved.columns:
                            date_from = st.date_input("From Date", value=None, key="saved_date_from")
                            date_to = st.date_input("To Date", value=None, key="saved_date_to")
                        else:
                            date_from = None
                            date_to = None
                    
                    with filter_col4:
                        saved_search = st.text_input(
                            "Search Description",
                            placeholder="Search...",
                            key="saved_search"
                        )
                    
                    # Apply filters
                    df_filtered = df_saved.copy()
                    
                    if saved_categories:
                        df_filtered = df_filtered[df_filtered['category'].isin(saved_categories)]
                    
                    if saved_types:
                        df_filtered = df_filtered[df_filtered['transaction_type'].isin(saved_types)]
                    
                    if date_from and 'date' in df_filtered.columns:
                        # Convert date column to datetime for comparison
                        df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
                        df_filtered = df_filtered[df_filtered['date'] >= pd.Timestamp(date_from)]
                    
                    if date_to and 'date' in df_filtered.columns:
                        # Ensure date column is datetime for comparison
                        if not pd.api.types.is_datetime64_any_dtype(df_filtered['date']):
                            df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
                        df_filtered = df_filtered[df_filtered['date'] <= pd.Timestamp(date_to)]
                    
                    if saved_search:
                        df_filtered = df_filtered[
                            df_filtered['description'].str.contains(saved_search, case=False, na=False)
                        ]
                    
                    st.caption(f"Showing {len(df_filtered)} of {len(df_saved)} transactions")
                    
                    # ========================================================================
                    # ENHANCED DASHBOARD WITH ADVANCED VISUALIZATIONS
                    # ========================================================================
                    if ADVANCED_FEATURES_AVAILABLE and len(df_filtered) > 0:
                        st.divider()
                        st.subheader("📊 Enhanced Analytics Dashboard")
                        
                        # Get dashboard from session state
                        dashboard = st.session_state.get('dashboard')
                        if dashboard:
                            # Create tabs for different views
                            dash_tab1, dash_tab2, dash_tab3, dash_tab4 = st.tabs([
                                "💰 Overview", "📈 Trends", "🎯 Categories", "💡 Insights"
                            ])
                            
                            with dash_tab1:
                                # Hero metrics with trends
                                st.markdown("### Key Metrics")
                                transactions_list = df_filtered.to_dict('records')
                                dashboard.render_hero_metrics(transactions_list)
                                
                                # Quick charts
                                col1, col2 = st.columns(2)
                                with col1:
                                    dashboard.render_category_breakdown_chart(transactions_list)
                                with col2:
                                    dashboard.render_spending_heatmap(transactions_list)
                            
                            with dash_tab2:
                                # Trend analysis
                                st.markdown("### Spending Trends Over Time")
                                dashboard.render_monthly_trend_chart(transactions_list)
                                dashboard.render_comparison_view(transactions_list)
                            
                            with dash_tab3:
                                # Category analysis
                                st.markdown("### Category Deep Dive")
                                dashboard.render_category_trend_chart(transactions_list, top_n=5)
                                dashboard.render_top_merchants_chart(transactions_list, top_n=10)
                            
                            with dash_tab4:
                                # AI-powered insights
                                st.markdown("### AI-Powered Financial Insights")
                                insights_engine = st.session_state.get('insights_engine')
                                if insights_engine:
                                    with st.spinner("🤖 Generating insights..."):
                                        report = insights_engine.generate_comprehensive_report(transactions_list)
                                        dashboard.render_ai_insights_panel(
                                            report.get('insights', []),
                                            report.get('recommendations', [])
                                        )
                                        
                                        # Show forecasts if available
                                        if report.get('forecasts') and report['forecasts'].get('forecasts'):
                                            st.markdown("### 🔮 Spending Forecast")
                                            forecasts = report['forecasts']['forecasts']
                                            forecast_df = pd.DataFrame([
                                                {'Month': k, 'Forecasted Spending': v}
                                                for k, v in forecasts.items()
                                            ])
                                            st.bar_chart(forecast_df.set_index('Month'))
                                            st.caption(f"Confidence: {report['forecasts'].get('confidence', 'unknown').upper()}")
                                else:
                                    st.info("💡 Insights engine not available")
                        else:
                            st.info("Enhanced dashboard not initialized. Refresh the page to enable advanced features.")
                    elif ADVANCED_FEATURES_AVAILABLE:
                        st.info("📊 Enhanced dashboard available - filter some transactions to see advanced analytics!")
                    # ========================================================================
                    
                    # Display transactions
                    st.subheader("📋 Transactions")
                    
                    # Prepare display columns
                    display_columns = ['date', 'description', 'amount_jpy', 'category', 'transaction_type']
                    if 'original_description' in df_filtered.columns:
                        display_columns.insert(2, 'original_description')
                    
                    # Filter to display columns that exist
                    display_columns = [col for col in display_columns if col in df_filtered.columns]
                    
                    # Sort by date descending (most recent first)
                    df_display = df_filtered[display_columns].sort_values('date', ascending=False)
                    
                    # Format the dataframe for display
                    if 'date' in df_display.columns:
                        df_display['date'] = df_display['date'].dt.strftime('%Y-%m-%d')
                    
                    st.dataframe(
                        df_display,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            'date': st.column_config.TextColumn('Date', width='small'),
                            'description': st.column_config.TextColumn('Description', width='medium'),
                            'original_description': st.column_config.TextColumn('Original (JA)', width='medium'),
                            'amount_jpy': st.column_config.NumberColumn('Amount (¥)', format="¥%.0f"),
                            'category': st.column_config.TextColumn('Category', width='small'),
                            'transaction_type': st.column_config.TextColumn('Type', width='small'),
                        }
                    )
                    
                    # Charts
                    st.divider()
                    st.subheader("📊 Analytics")
                    
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        st.write("**Spending by Category**")
                        if 'category' in df_filtered.columns and 'amount_jpy' in df_filtered.columns:
                            # Filter expenses only
                            expenses_df = df_filtered[df_filtered['transaction_type'] == 'Expense'].copy()
                            if not expenses_df.empty:
                                category_spending = expenses_df.groupby('category')['amount_jpy'].sum().abs()
                                st.bar_chart(category_spending)
                            else:
                                st.info("No expense transactions in filtered data")
                    
                    with chart_col2:
                        st.write("**Monthly Trend**")
                        if 'date' in df_filtered.columns and 'amount_jpy' in df_filtered.columns:
                            df_monthly = df_filtered.copy()
                            df_monthly['month'] = pd.to_datetime(df_monthly['date']).dt.to_period('M').astype(str)
                            monthly_totals = df_monthly.groupby('month')['amount_jpy'].sum()
                            st.line_chart(monthly_totals)
                    
                    # Export filtered data
                    st.divider()
                    col_exp1, col_exp2 = st.columns(2)
                    
                    with col_exp1:
                        # Export filtered transactions
                        if st.button("📥 Export Filtered Data"):
                            csv = df_filtered.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=f"filtered_transactions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )
                    
                    with col_exp2:
                        # Quick stats
                        if st.button("📊 Show Statistics"):
                            st.write("**Filtered Data Statistics:**")
                            st.write(f"- Transactions: {len(df_filtered)}")
                            if 'amount_jpy' in df_filtered.columns:
                                st.write(f"- Total Amount: ¥{df_filtered['amount_jpy'].sum():,.0f}")
                                st.write(f"- Average: ¥{df_filtered['amount_jpy'].mean():,.0f}")
                                st.write(f"- Min: ¥{df_filtered['amount_jpy'].min():,.0f}")
                                st.write(f"- Max: ¥{df_filtered['amount_jpy'].max():,.0f}")
                
                else:
                    st.info("📭 No saved transactions yet. Upload and save some transactions to see them here!")
                    st.write("**How to save transactions:**")
                    st.write("1. Upload a CSV/PDF/Image file below")
                    st.write("2. Review the categorized transactions")
                    st.write("3. Click '💾 Save processed transactions to DB'")
                    st.write("4. Come back here to view and analyze your saved data!")
            
            except Exception as e:
                st.error(f"Error loading saved transactions: {e}")
        else:
            st.warning("Database not available. Cannot load saved transactions.")
    
    st.divider()
    
    # Handle resume mode
    if st.session_state.get('resume_mode', False) and st.session_state.get('resume_data'):
        st.success("🔄 **Resume Mode Active** - Working with previously saved transactions")
        st.info("💡 Your previous categorization work has been restored. Continue where you left off!")
        
        # Create DataFrame from resume data
        df = pd.DataFrame(st.session_state['resume_data'])
        
        # Set flag and file info for duplicate analysis section (resume data is typically from CSV)
        st.session_state['csv_file_uploaded'] = True
        st.session_state['csv_file_name'] = f"Resumed Session {st.session_state['categorization_session_id']}"
        st.session_state['csv_file_size'] = len(st.session_state['resume_data'])  # Use data length as size
        # Store dataframe for duplicate analysis
        st.session_state['csv_dataframe'] = st.session_state['resume_data']
        
        # Convert date column to proper format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # Set a dummy uploaded_file for compatibility
        class DummyFile:
            def __init__(self, name):
                self.name = name
        
        uploaded_file = DummyFile(f"Resumed Session {st.session_state['categorization_session_id']}")
        
        # Clear resume mode after loading
        st.session_state['resume_mode'] = False
        
    else:
        # File upload
        uploaded_file = st.file_uploader("Choose a statement file", 
        type=["pdf", "png", "jpg", "jpeg", "csv"])
    
    if uploaded_file is not None:
        # Clear previous CSV file state when new file is uploaded
        st.session_state['csv_file_uploaded'] = False
        st.session_state['csv_file_name'] = None
        st.session_state['csv_file_size'] = None
        st.session_state['csv_dataframe'] = None
        
        # Clear any previous duplicate analysis state to prevent conflicts
        keys_to_remove = []
        for key in st.session_state.keys():
            if key.startswith('duplicate_analysis_') or key.startswith('selected_duplicates_'):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del st.session_state[key]
        
        # Reset debug counter for new file
        st.session_state['duplicate_section_executed'] = 0
        
        # Check if we're in resume mode (df already created above)
        if not st.session_state.get('resume_mode', False):
            try:
                # Determine file type from MIME type or extension
                file_name = (uploaded_file.name or "").lower()
                file_type = uploaded_file.type or ""
                is_pdf = file_type == "application/pdf" or file_name.endswith(".pdf")
                is_csv = file_type == "text/csv" or file_type == "application/csv" or file_name.endswith(".csv")

                if is_pdf:
                    df = extract_transactions_from_pdf(uploaded_file)
                elif is_csv:
                    df = extract_transactions_from_csv(uploaded_file, translation_mode, api_key, ai_model=ai_model, ai_base_url=ai_base_url)
                    # Set flag and file info for duplicate analysis section
                    st.session_state['csv_file_uploaded'] = True
                    st.session_state['csv_file_name'] = uploaded_file.name
                    st.session_state['csv_file_size'] = uploaded_file.size
                    # Store dataframe for duplicate analysis with error handling
                    try:
                        st.session_state['csv_dataframe'] = df.to_dict('records')
                    except Exception as e:
                        st.error(f"Error storing CSV data for duplicate analysis: {e}")
                        # Fallback: store as empty list
                        st.session_state['csv_dataframe'] = []
                else:
                    df = extract_transactions_from_image(uploaded_file)
            except Exception as e:
                st.error(f"Error processing file: {e}")
                return
        
        # Initialize learning system
        learning_system = MerchantLearningSystem()
        
        # Show loading animation during processing
        with st.spinner("🔄 Processing transactions and applying smart categorization..."):
            # MoneyMgr Proven Categorization System (Based on 3,943+ real transactions)
            rules = {
            "Food": [
                # Groceries and Food Stores
                "ローソン", "セブンイレブン", "ファミリーマート", "コンビニ", "lawson", "seven eleven", "family mart",
                "ポプラグループ", "poplar", "スーパー", "supermarket", "grocery", "market", "food", "fresh",
                "イオン", "aeon", "イトーヨーカドー", "itoyokado", "西友", "seiyu", "ライフ", "life",
                # Restaurants and Dining
                "レストラン", "restaurant", "cafe", "dinner", "lunch", "breakfast", "takeaway", "delivery",
                "居酒屋", "izakaya", "バー", "bar", "カフェ", "coffee", "ピザ", "pizza", "寿司", "sushi",
                "マクドナルド", "mcdonalds", "ケンタッキー", "kfc", "スターバックス", "starbucks"
            ],
            "Social Life": [
                # Social Activities
                "飲み会", "drinking", "パーティー", "party", "イベント", "event", "友達", "friend", "同僚", "colleague",
                "会食", "dining", "懇親会", "networking", "歓迎会", "welcome", "送別会", "farewell",
                "カラオケ", "karaoke", "ボーリング", "bowling", "ゲーム", "game", "スポーツ", "sports"
            ],
            "Subscriptions": [
                # Digital Services
                "icloud", "apple music", "amazon prime", "google one", "netflix", "spotify", "hulu", "disney+",
                "アマゾンプライム", "グーグルワン", "アップルミュージック", "アイクラウド",
                "subscription", "membership", "月額", "monthly", "年額", "annual"
            ],
            "Household": [
                # Home and Living
                "家賃", "rent", "光熱費", "utility", "電気", "electric", "ガス", "gas", "水道", "water",
                "家具", "furniture", "家電", "appliance", "日用品", "daily", "掃除", "cleaning",
                "ニトリ", "nitori", "イケア", "ikea", "ホームセンター", "home center"
            ],
            "Transportation": [
                # Public Transport and Travel
                "電車", "train", "バス", "bus", "タクシー", "taxi", "地下鉄", "subway", "モノレール", "monorail",
                "モバイルパス", "mobile pass", "交通費", "transport", "駐車場", "parking", "高速道路", "highway",
                "ＥＴＣ", "etc", "ガソリン", "gasoline", "燃料", "fuel", "車", "car", "バイク", "bike"
            ],
            "Vacation": [
                # Travel and Leisure
                "旅行", "travel", "ホテル", "hotel", "飛行機", "flight", "新幹線", "shinkansen", "観光", "tourism",
                "温泉", "onsen", "リゾート", "resort", "ビーチ", "beach", "山", "mountain", "海", "sea",
                "チケット", "ticket", "ツアー", "tour", "宿泊", "accommodation"
            ],
            "Health": [
                # Healthcare and Wellness
                "病院", "hospital", "クリニック", "clinic", "歯科", "dental", "眼科", "eye", "薬局", "pharmacy",
                "薬", "medicine", "保険", "insurance", "診察", "examination", "治療", "treatment",
                "フィットネス", "fitness", "ジム", "gym", "ヨガ", "yoga", "マッサージ", "massage"
            ],
            "Apparel": [
                # Clothing and Fashion
                "服", "clothing", "靴", "shoes", "バッグ", "bag", "アクセサリー", "accessory", "時計", "watch",
                "ユニクロ", "uniqlo", "zara", "h&m", "gap", "nike", "adidas", "アディダス", "ナイキ",
                "ファッション", "fashion", "スタイル", "style", "ブランド", "brand"
            ],
            "Grooming": [
                # Personal Care
                "美容", "beauty", "化粧品", "cosmetics", "スキンケア", "skincare", "ヘアケア", "haircare",
                "ネイル", "nail", "エステ", "esthetic", "理容", "barber", "美容院", "salon",
                "資生堂", "shiseido", "ポーラ", "pola", "ファンケル", "fancl"
            ],
            "Self-development": [
                # Education and Growth
                "本", "book", "雑誌", "magazine", "新聞", "newspaper", "講座", "course", "セミナー", "seminar",
                "ワークショップ", "workshop", "資格", "certification", "学習", "learning", "スキル", "skill",
                "オンライン", "online", "eラーニング", "elearning", "トレーニング", "training"
            ]
        }
        
        # MoneyMgr Subcategory System for Detailed Breakdown
        subcategories = {
            "Food": {
                "Groceries": ["ローソン", "セブンイレブン", "ファミリーマート", "コンビニ", "スーパー", "ポプラグループ"],
                "Dinner/Eating Out": ["レストラン", "居酒屋", "バー", "dinner", "restaurant", "izakaya"],
                "Lunch/Eating Out": ["lunch", "カフェ", "coffee", "昼食", "ランチ"],
                "Beverages A": ["スターバックス", "コーヒー", "tea", "ジュース", "drink"],
                "Beverages/Non-A": ["アルコール", "酒", "ビール", "wine", "spirits"]
            },
            "Social Life": {
                "Drinking": ["飲み会", "drinking", "パーティー", "party", "カラオケ", "karaoke"],
                "Event": ["イベント", "event", "会食", "dining", "懇親会", "networking"],
                "Friend": ["友達", "friend", "同僚", "colleague", "歓迎会", "送別会"]
            },
            "Transportation": {
                "Subway": ["地下鉄", "subway", "電車", "train", "モノレール", "monorail"],
                "Taxi": ["タクシー", "taxi", "車", "car", "ライドシェア", "rideshare"],
                "Mobile Pass": ["モバイルパス", "mobile pass", "交通費", "transport"],
                "ETC": ["ＥＴＣ", "etc", "高速道路", "highway", "駐車場", "parking"]
            },
            "Household": {
                "Rent": ["家賃", "rent", "住宅費", "housing"],
                "Utilities": ["光熱費", "utility", "電気", "electric", "ガス", "gas", "水道", "water"],
                "Furniture": ["家具", "furniture", "ニトリ", "nitori", "イケア", "ikea"]
            }
        }
        # Currency settings and FX normalization
        st.subheader("💱 Currency & FX")
        default_currency = st.selectbox(
            "Statement currency (applied when missing)",
            options=["JPY", "USD", "EUR", "AUD", "CAD", "GBP", "CNY", "KRW"],
            index=0,
            help="Used to convert amounts to JPY for totals."
        )

        df = enrich_currency_columns(df, default_currency)

        # Detect transaction types (credit vs debit)
        df = detect_transaction_type(df)
        
        # Apply categorization – choose best available method
        category_list = list(rules.keys())
        ai_cat_attempted = False
        if api_key and translation_mode in ("AI-Powered (GPT-4o-mini)", "AI-Powered"):
            # AI-powered categorization (most accurate for Japanese)
            st.info(f"🤖 Using AI-powered categorization ({ai_model})…")
            df_cat = categorise_transactions_ai(df, api_key, category_list, subcategories, model=ai_model, base_url=ai_base_url)
            ai_cat_attempted = True
            # Fill any remaining Uncategorised with rule-based fallback
            uncategorised_mask = df_cat["category"] == "Uncategorised"
            if uncategorised_mask.any():
                fallback = categorise_transactions(
                    df_cat[uncategorised_mask], rules, subcategories
                )
                df_cat.loc[uncategorised_mask, "category"] = fallback["category"]
                df_cat.loc[uncategorised_mask, "subcategory"] = fallback["subcategory"]
        elif learning_system:
            # Use smart learning system for predictions
            df_cat = apply_smart_categorization(df, learning_system, rules, subcategories)
        else:
            # Fallback to basic rule-based categorization
            df_cat = categorise_transactions(df, rules, subcategories)
        
        # Close the loading spinner and show completion status
        st.success(f"✅ Processing complete! Successfully processed {len(df)} transactions.")
        
        # Professional Financial Data Validation
        st.subheader("🏦 Professional Financial Validation")
        
        # Use the new professional validation system (no expected total)
        financial_validation = validate_financial_data(df, expected_total=None)
        
        # Display validation results in a professional format
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if financial_validation['is_valid']:
                st.success("✅ **Validation Passed**")
            else:
                st.error("❌ **Validation Failed**")
        
        with col2:
            quality_score = financial_validation['data_quality_score']
            if quality_score >= 90:
                st.success(f"🟢 **Quality Score:** {quality_score:.0f}%")
            elif quality_score >= 70:
                st.warning(f"🟡 **Quality Score:** {quality_score:.0f}%")
            else:
                st.error(f"🔴 **Quality Score:** {quality_score:.0f}%")
        
        with col3:
            rec_status = financial_validation.get('reconciliation_status', 'unknown')
            if rec_status == 'reconciled':
                st.success("✅ **Reconciled**")
            elif rec_status == 'unknown':
                st.info("ℹ️ **Reconciliation not checked**")
            else:
                st.warning("⚠️ **Reconciliation pending**")
        
        # Remove raw data summary; totals will be presented once below
        # Run reconciliation only if a user-provided expected total is supplied later (skipped)
        
        st.divider()
        
        # Data validation and quality check
        st.subheader("🔍 Data Quality Summary")
        validation_results = validate_transaction_data(df_cat)
        
        # Create a clean summary using columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if validation_results['is_valid']:
                st.success("✅ **Validation Passed**")
            else:
                st.error("❌ **Validation Failed**")
        
        with col2:
            if validation_results['duplicates']:
                st.warning(f"🔍 **{len(validation_results['duplicates'])} Duplicates**")
            else:
                st.success("✅ **No Duplicates**")
        
        with col3:
            if validation_results['anomalies']:
                st.warning(f"🚨 **{len(validation_results['anomalies'])} Anomalies**")
            else:
                st.success("✅ **No Anomalies**")
        
        # Show detailed issues only if they exist
        has_issues = (validation_results['duplicates'] or 
                     validation_results['anomalies'] or 
                     validation_results['warnings'] or 
                     not validation_results['is_valid'])
        
        if has_issues:
            with st.expander("📋 **View Details**", expanded=False):
                # Show errors if any
                if not validation_results['is_valid']:
                    st.error("**Critical Issues:**")
                    for error in validation_results['errors']:
                        st.error(f"• {error}")
                
                # Show duplicates if found
                if validation_results['duplicates']:
                    st.warning(f"**Duplicate Transactions ({len(validation_results['duplicates'])}):**")
                    if st.checkbox("Show duplicate transactions", key="show_duplicates"):
                        duplicates_df = pd.DataFrame(validation_results['duplicates'])
                        st.dataframe(duplicates_df)
                
                # Show anomalies if found
                if validation_results['anomalies']:
                    st.warning(f"**Unusual Amounts ({len(validation_results['anomalies'])}):**")
                    if st.checkbox("Show anomalous transactions", key="show_anomalies"):
                        anomalies_df = pd.DataFrame(validation_results['anomalies'])
                        st.dataframe(anomalies_df)
                
                # Show other warnings
                if validation_results['warnings']:
                    st.info("**Other Warnings:**")
                    for warning in validation_results['warnings']:
                        st.info(f"• {warning}")
        
        st.divider()
        
        # Smart categorization interface
        st.subheader("🎯 Smart Transaction Categorization")
        
        # Get all available categories and subcategories
        all_categories = list(rules.keys()) + ["Uncategorised"]
        all_subcategories = []
        for main_cat, subs in subcategories.items():
            for sub_cat in subs.keys():
                all_subcategories.append(f"{main_cat} - {sub_cat}")
        
        # Show categorization statistics
        category_counts = df_cat['transaction_type'].value_counts()
        
        # Show categorization summary in a clean format
        if 'confidence' in df_cat.columns:
            avg_confidence = df_cat['confidence'].mean()
            high_confidence = len(df_cat[df_cat['confidence'] >= 0.8])
            low_confidence = len(df_cat[df_cat['confidence'] < 0.5])
            
            # Clean summary display
            st.markdown(f"""
            ### 📊 Categorization Summary
            **Total Transactions:** {len(df_cat)}  
            **Average Confidence:** {avg_confidence:.1%}  
            **High Confidence:** {high_confidence} | **Low Confidence:** {low_confidence}
            """)
        else:
            st.markdown(f"### 📊 Categorization Summary\n**Total Transactions:** {len(df_cat)}")
        
        # Display transaction type breakdown
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Transaction Types:**")
            for trans_type, count in category_counts.items():
                if trans_type == 'Credit':
                    st.write(f"🟢 **{trans_type}:** {count} transactions")
                else:
                    st.write(f"🔴 **{trans_type}:** {count} transactions")
        
        with col2:
            # Calculate totals - handle both positive and negative amounts correctly
            expense_df = df_cat[df_cat['transaction_type'] == 'Expense']
            credit_df = df_cat[df_cat['transaction_type'] == 'Credit']
            
            # For Japanese bank statements, expenses are typically negative amounts
            # We want to show the absolute value for display purposes
            expense_amounts = expense_df['amount']
            credit_amounts = credit_df['amount']
            
            # Calculate totals - this is where the issue might be
            total_expenses = abs(expense_amounts.sum())
            total_credits = abs(credit_amounts.sum())
            
            # Net amount calculation (expenses - credits)
            net_amount = total_expenses - total_credits
            
            st.write("**Financial Summary:**")
            st.write(f"🔴 **Total Expenses:** ¥{total_expenses:,.0f}")
            st.write(f"🟢 **Total Credits:** ¥{total_credits:,.0f}")
            st.write(f"💰 **Net Amount:** ¥{net_amount:,.0f}")
            
                        # Check for potential data issues
        # Cleaned UI: remove verbose data quality diagnostics
        
        # Show transaction breakdown by type
        # Cleaned UI: remove transaction type breakdown list
        
        # Add manual correction option
        # Cleaned UI: remove manual correction/alternative calculations/debug blocks
        
        st.divider()
        
        # Cleaned UI: remove category breakdown lists
        expense_df = df_cat[df_cat['transaction_type'] == 'Expense']
        uncategorized_count = int((df_cat['category'] == 'Uncategorised').sum())
        
        # Smart categorization for uncategorized transactions
        if uncategorized_count > 0:
            st.subheader("🚀 Quick Categorization")
            
            # Show progress restoration if available
            if st.session_state['progress_saved'] and st.session_state['categorization_progress']:
                st.info("💾 **Saved Progress Available:** You can restore your previous categorization work.")
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("🔄 Restore Saved Progress"):
                        # Restore the saved progress
                        restored_df = pd.DataFrame(st.session_state['categorization_progress'])
                        df_cat.update(restored_df)
                        st.session_state['progress_saved'] = False
                        st.success("🔄 Progress restored! Your previous work has been loaded.")
                        st.rerun()
                with col2:
                    if st.button("🗑️ Clear Saved Progress"):
                        st.session_state['categorization_progress'] = None
                        st.session_state['progress_saved'] = False
                        st.success("🗑️ Saved progress cleared.")
                        st.rerun()
            
            st.write("Use the dropdowns below to quickly categorize uncategorized transactions:")
            
            # Add navigation hints
            st.info("💡 **Navigation Tips:** Use the buttons below to save progress, skip categorization, or continue to review results.")
            
            # Get uncategorized transactions
            uncategorized_df = df_cat[df_cat['category'] == 'Uncategorised'].copy()
            
            # Create a form for bulk categorization
            with st.form("bulk_categorization"):
                
                # Suggest categories based on description keywords
                for idx, row in uncategorized_df.iterrows():
                    description = str(row['description']).lower()
                    original_desc = str(row.get('original_description', '')).lower()
                    
                    # Smart category suggestions based on keywords
                    suggested_category = "Uncategorised"
                    for category, keywords in rules.items():
                        if any(keyword.lower() in description or keyword.lower() in original_desc for keyword in keywords):
                            suggested_category = category
                            break
                    
                    # Special handling for common Japanese merchants
                    if any(word in original_desc for word in ['ローソン', 'セブンイレブン', 'ファミマ', 'コンビニ']):
                        suggested_category = "Food"
                    elif any(word in original_desc for word in ['ニトリ', 'イケア', '家具']):
                        suggested_category = "Household"
                    elif any(word in original_desc for word in ['アマゾン', 'amazon']):
                        suggested_category = "Subscriptions"
                    elif any(word in original_desc for word in ['モバイルパス', '交通']):
                        suggested_category = "Transportation"
                    
                    # Ensure suggested category is in the list
                    if suggested_category not in all_categories:
                        suggested_category = "Uncategorised"
                    
                    # ── Row 1: Full-width descriptions ──
                    transaction_date = row.get('date', 'Unknown Date')
                    if hasattr(transaction_date, 'strftime'):
                        date_str = transaction_date.strftime('%Y-%m-%d')
                    else:
                        date_str = str(transaction_date)

                    desc_en = str(row.get('description', ''))
                    desc_jp = str(row.get('original_description', ''))
                    trans_type = row.get('transaction_type', 'Expense')
                    type_icon = "🟢" if trans_type == 'Credit' else "🔴"

                    st.markdown(
                        f"**📅 {date_str}** &nbsp; {type_icon} **{trans_type}** &nbsp; **¥{row['amount']:,.0f}**"
                    )
                    st.markdown(f"🇬🇧 {desc_en}")
                    if desc_jp and desc_jp != desc_en:
                        st.markdown(f"🇯🇵 {desc_jp}")

                    # ── Row 2: Category selector ──
                    try:
                        suggested_index = all_categories.index(suggested_category) if suggested_category in all_categories else 0
                        new_category = st.selectbox(
                            f"Category",
                            all_categories,
                            index=suggested_index,
                            key=f"cat_{idx}"
                        )
                    except (ValueError, IndexError):
                        new_category = st.selectbox(
                            f"Category",
                            all_categories,
                            index=0,
                            key=f"cat_{idx}"
                        )
                    st.divider()
                    
                    # Update the category
                    df_cat.loc[idx, 'category'] = new_category
                
                # Add navigation and progress options
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    submitted = st.form_submit_button("✅ Apply All Categorizations")
                
                with col2:
                    save_progress = st.form_submit_button("💾 Save Progress & Continue Later")
                
                with col3:
                    skip_categorization = st.form_submit_button("⏭️ Skip & Review Results")
                
                # Handle form submissions
                if submitted:
                    # Learn from user corrections if smart learning is available
                    if learning_system:
                        for idx, row in uncategorized_df.iterrows():
                            original_category = row.get('category', 'Uncategorised')
                            if original_category != new_category:
                                # Collect feedback for learning
                                transaction_data = {
                                    'description': row.get('description', ''),
                                    'original_description': row.get('original_description', ''),
                                    'amount': row.get('amount', 0),
                                    'date': row.get('date'),
                                    'transaction_type': row.get('transaction_type', 'Expense')
                                }
                                learning_system.learn_from_user_feedback(
                                    str(idx), original_category, new_category, transaction_data
                                )
                    
                    st.success("🎉 All categories updated! Scroll down to see the results.")
                    
                    # Show learning feedback
                    if learning_system:
                        st.success("🧠 Smart Learning System has learned from your corrections!")
                    
                    # Show next steps
                    st.info("🚀 **Next Steps:** Scroll down to review your categorized transactions and see the financial analysis!")
                
                elif save_progress:
                    # Save current progress to session state
                    st.session_state['categorization_progress'] = df_cat.to_dict('records')
                    st.session_state['progress_saved'] = True
                    st.success("💾 Progress saved! You can continue later.")
                
                elif skip_categorization:
                    st.info("⏭️ Skipping categorization. Scroll down to review results.")
        
        # Advanced Categorization Interface
        st.divider()
        st.subheader("🎯 Advanced Categorization Tools")
        
        # Initialize categorization session if needed
        if uploaded_file and create_categorization_session:
            file_name = uploaded_file.name if hasattr(uploaded_file, 'name') else 'uploaded_file'
            
            # Check for existing session
            active_session = get_active_categorization_session(file_name) if get_active_categorization_session else None
            
            if not active_session:
                # Create new session
                session_id = create_categorization_session(file_name, len(df_cat))
                st.success(f"🆕 Created new categorization session (ID: {session_id})")
                st.session_state['categorization_session_id'] = session_id
            else:
                # Resume existing session
                session_id = active_session['id']
                st.info(f"🔄 Resuming categorization session (ID: {session_id})")
                st.session_state['categorization_session_id'] = session_id
                
                # Load existing progress
                if load_categorization_progress:
                    progress_data = load_categorization_progress(session_id)
                    if progress_data:
                        st.success(f"📊 Loaded {len(progress_data)} previously reviewed transactions")
                        
                        # Apply progress to current dataframe
                        for progress in progress_data:
                            mask = (df_cat['description'] == progress['description']) & \
                                   (df_cat['date'] == progress['date']) & \
                                   (df_cat['amount'] == progress['amount'])
                            if mask.any():
                                df_cat.loc[mask, 'category'] = progress['category']
                                df_cat.loc[mask, 'subcategory'] = progress['subcategory']
                                df_cat.loc[mask, 'transaction_type'] = progress['transaction_type']
            
            # Show session statistics
            if get_categorization_session_stats and 'categorization_session_id' in st.session_state:
                session_id = st.session_state['categorization_session_id']
                stats = get_categorization_session_stats(session_id)
                
                if stats:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📊 Total Transactions", stats['total_transactions'])
                    with col2:
                        st.metric("✅ Reviewed", stats['reviewed_transactions'])
                    with col3:
                        st.metric("🏷️ Categorized", stats['categorized_transactions'])
                    with col4:
                        progress = stats['completion_percentage']
                        st.metric("📈 Progress", f"{progress}%")
                        
                        # Progress bar
                        st.progress(progress / 100)
            
            # Bulk categorization tools
            with st.expander("🔧 Bulk Categorization Tools", expanded=True):
                st.write("**Smart categorization tools to speed up your workflow:**")
                
                # Get merchant suggestions
                if get_merchant_categorization_suggestions:
                    suggestions = get_merchant_categorization_suggestions(10)
                    if suggestions:
                        st.write("**📚 Common merchant patterns from your history:**")
                        for suggestion in suggestions[:5]:
                            st.write(f"• **{suggestion['description']}** → {suggestion['category']} ({suggestion['frequency']} times)")
                
                # Learning statistics
                if get_learning_statistics:
                    learning_stats = get_learning_statistics()
                    if learning_stats['total_merchant_patterns'] > 0:
                        st.write("**🧠 Learning System Stats:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📊 Learned Merchants", learning_stats['unique_merchants'])
                        with col2:
                            st.metric("🎯 Total Patterns", learning_stats['total_patterns'])
                        with col3:
                            st.metric("🔄 Recent Learning", learning_stats['recent_learning'])
                
                # Bulk rule creation
                st.write("**🎯 Create bulk categorization rules:**")
                
                rule_col1, rule_col2, rule_col3, rule_col4 = st.columns([2, 1, 1, 1])
                with rule_col1:
                    bulk_pattern = st.text_input("Pattern (e.g., 'amazon', 'starbucks')", key="bulk_pattern")
                with rule_col2:
                    bulk_category = st.selectbox("Category", 
                        ["Food", "Shopping & Retail", "Transportation", "Entertainment", 
                         "Subscriptions", "Utilities", "Healthcare", "Income", "Other"], 
                        key="bulk_category")
                with rule_col3:
                    bulk_subcategory = st.text_input("Subcategory", key="bulk_subcategory")
                with rule_col4:
                    bulk_type = st.selectbox("Type", ["Expense", "Credit"], key="bulk_type")
                
                if st.button("🚀 Apply Rule to Uncategorized") and bulk_pattern and bulk_category:
                    if 'categorization_session_id' in st.session_state:
                        session_id = st.session_state['categorization_session_id']
                        
                        # Apply the rule
                        rule = {
                            'pattern': bulk_pattern,
                            'category': bulk_category,
                            'subcategory': bulk_subcategory,
                            'transaction_type': bulk_type
                        }
                        
                        updated_count = apply_bulk_categorization_rules(session_id, [rule])
                        
                        if updated_count > 0:
                            st.success(f"✅ Applied rule to {updated_count} transactions!")
                            st.rerun()
                        else:
                            st.warning("⚠️ No uncategorized transactions matched this pattern")
                    else:
                        st.error("❌ No active categorization session")
            
            # Smart review interface
            with st.expander("🧠 Smart Review Interface", expanded=False):
                st.write("**Review transactions efficiently with smart suggestions:**")
                
                # Get uncategorized transactions
                uncategorized_mask = df_cat['category'].isna() | (df_cat['category'] == 'Uncategorised')
                uncategorized_df = df_cat[uncategorized_mask].copy()
                
                if len(uncategorized_df) > 0:
                    st.write(f"**Found {len(uncategorized_df)} uncategorized transactions:**")
                    
                    # Group by similar descriptions for batch review
                    description_groups = uncategorized_df.groupby('description').size().sort_values(ascending=False)
                    
                    st.write("**📊 Transactions by description (most common first):**")
                    for desc, count in description_groups.head(10).items():
                        st.write(f"• **{desc}** ({count} transactions)")
                    
                    # Quick categorization for most common merchants
                    if len(description_groups) > 0:
                        most_common = description_groups.index[0]
                        st.write(f"**🎯 Quick categorize: '{most_common}'**")
                        
                        # Get smart suggestions for the most common transaction
                        smart_suggestions = []
                        if get_learning_suggestions:
                            # Get a sample transaction for suggestions
                            sample_transaction = uncategorized_df[uncategorized_df['description'] == most_common].iloc[0]
                            smart_suggestions = get_learning_suggestions(
                                description=most_common,
                                amount=sample_transaction.get('amount'),
                                date=sample_transaction.get('date')
                            )
                        
                        # Show smart suggestions
                        if smart_suggestions:
                            st.write("**🧠 Smart suggestions based on your patterns:**")
                            for i, suggestion in enumerate(smart_suggestions[:3]):
                                confidence = suggestion['confidence']
                                reason = suggestion['reason']
                                st.write(f"• **{suggestion['category']}** (Confidence: {confidence:.1%}) - {reason}")
                        
                        quick_col1, quick_col2, quick_col3 = st.columns([1, 1, 1])
                        with quick_col1:
                            # Pre-populate with best suggestion if available
                            default_category = smart_suggestions[0]['category'] if smart_suggestions else "Food"
                            quick_category = st.selectbox("Category", 
                                ["Food", "Shopping & Retail", "Transportation", "Entertainment", 
                                 "Subscriptions", "Utilities", "Healthcare", "Income", "Other"], 
                                index=["Food", "Shopping & Retail", "Transportation", "Entertainment", 
                                      "Subscriptions", "Utilities", "Healthcare", "Income", "Other"].index(default_category) if default_category in ["Food", "Shopping & Retail", "Transportation", "Entertainment", "Subscriptions", "Utilities", "Healthcare", "Income", "Other"] else 0,
                                key="quick_category")
                        with quick_col2:
                            # Pre-populate with best suggestion if available
                            default_subcategory = smart_suggestions[0]['subcategory'] if smart_suggestions and smart_suggestions[0]['subcategory'] else ""
                            quick_subcategory = st.text_input("Subcategory", value=default_subcategory, key="quick_subcategory")
                        with quick_col3:
                            quick_type = st.selectbox("Type", ["Expense", "Credit"], key="quick_type")
                        
                        if st.button(f"🏷️ Apply to all '{most_common}' transactions"):
                            # Apply to all matching transactions in the current dataframe
                            mask = df_cat['description'] == most_common
                            df_cat.loc[mask, 'category'] = quick_category
                            df_cat.loc[mask, 'subcategory'] = quick_subcategory
                            df_cat.loc[mask, 'transaction_type'] = quick_type
                            
                            # Save progress to database
                            if save_categorization_progress and 'categorization_session_id' in st.session_state:
                                session_id = st.session_state['categorization_session_id']
                                for idx, row in df_cat[mask].iterrows():
                                    save_categorization_progress(session_id, row.to_dict())
                            
                            st.success(f"✅ Categorized {mask.sum()} transactions!")
                            st.rerun()
                else:
                    st.success("🎉 All transactions are categorized!")
            
            # Auto-save progress button
            if st.button("💾 Save Progress to Database") and save_categorization_progress:
                if 'categorization_session_id' in st.session_state:
                    session_id = st.session_state['categorization_session_id']
                    
                    # Save all current progress and learn from categorizations
                    learned_count = 0
                    for idx, row in df_cat.iterrows():
                        save_categorization_progress(session_id, row.to_dict())
                        
                        # Learn from categorization decisions
                        if learn_from_categorization and row.get('category') and row.get('category') != 'Uncategorised':
                            learn_from_categorization(
                                description=row.get('description', ''),
                                category=row.get('category'),
                                subcategory=row.get('subcategory'),
                                amount=row.get('amount'),
                                date=row.get('date')
                            )
                            learned_count += 1
                    
                    success_msg = f"💾 Progress saved to database! Learned from {learned_count} categorizations."
                    if learned_count > 0:
                        success_msg += " 🧠 Your patterns will improve future suggestions!"
                    
                    st.success(success_msg)
                else:
                    st.error("❌ No active categorization session")
            
            # Complete session button
            if st.button("✅ Complete Categorization Session") and complete_categorization_session:
                if 'categorization_session_id' in st.session_state:
                    session_id = st.session_state['categorization_session_id']
                    complete_categorization_session(session_id)
                    st.success("🎉 Categorization session completed!")
                    del st.session_state['categorization_session_id']
                else:
                    st.error("❌ No active categorization session")
        
        # Category filter & review
        st.subheader("🔎 Category Filter & Review")
        try:
            available_categories = sorted([c for c in df_cat['category'].dropna().unique().tolist()])
        except Exception:
            available_categories = []
        selected_categories = st.multiselect(
            "Filter by category",
            options=available_categories,
            default=available_categories
        )

        filtered_df = df_cat[df_cat['category'].isin(selected_categories)].copy() if selected_categories else df_cat.copy()

        # Attach row id for safe updates
        filtered_df['_row_id'] = filtered_df.index

        # Compact summary for the filtered view
        st.write(
            f"Rows: {len(filtered_df)} | Total (abs): ¥{filtered_df['amount'].abs().sum():,.0f}"
        )

        # Column width control for long text
        desc_width_choice = st.select_slider(
            "Description column width",
            options=["small", "medium", "large"],
            value="large",
            help="Adjust how wide description columns render in the table"
        )

        # Editable view for quick verification and corrections
        edited_filtered = st.data_editor(
            filtered_df[
                ['date', 'description', 'original_description', 'amount', 'transaction_type', 'category', 'subcategory', '_row_id']
            ],
            num_rows="dynamic",
            key="category_filter_editor",
            disabled=[False, False, False, True, True, False, False, True],
            use_container_width=True,
            column_config={
                'description': st.column_config.TextColumn('Description', width=desc_width_choice),
                'original_description': st.column_config.TextColumn('Original (Japanese)', width=desc_width_choice)
            }
        )

        # Optional wide read-only view with manual column resizing (drag column edges)
        if st.checkbox("Open wide read-only view (manual column resizing)"):
            st.dataframe(
                filtered_df[['date','description','original_description','amount','transaction_type','category','subcategory']]
                .rename(columns={'original_description':'Original (Japanese)'}),
                use_container_width=True
            )

        # Quick full-text preview for selected row
        with st.expander("Preview full text for a specific row"):
            try:
                row_options = filtered_df.index.astype(str).tolist()
            except Exception:
                row_options = []
            selected_row = st.selectbox("Select row index", options=row_options) if row_options else None
            if selected_row is not None:
                idx = int(selected_row)
                st.write("Description:")
                st.code(str(filtered_df.loc[idx, 'description']))
                if 'original_description' in filtered_df.columns:
                    st.write("Original (Japanese):")
                    st.code(str(filtered_df.loc[idx, 'original_description']))

        # Apply edits back to the master dataframe
        if st.button("✅ Apply Filtered Edits"):
            try:
                for _, r in edited_filtered.iterrows():
                    row_id = r.get('_row_id')
                    if row_id in df_cat.index:
                        if 'category' in r:
                            df_cat.loc[row_id, 'category'] = r['category']
                        if 'subcategory' in r:
                            df_cat.loc[row_id, 'subcategory'] = r['subcategory']
                st.success("✔️ Applied edits to selected rows.")
            except Exception as e:
                st.error(f"Failed to apply edits: {e}")

        # Show the final categorized data
        st.subheader("📋 Review All Transactions")
        
        # Add quick navigation and progress indicator
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write("Final categorized transactions (you can still edit individual categories):")
        
        with col2:
            if st.button("📊 View Financial Summary"):
                st.info("📊 Scroll down to see the financial summary and charts!")
        
        with col3:
            if st.button("💾 Export Data"):
                # Create downloadable CSV
                csv = df_cat.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"categorized_transactions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
        
        # Show categorization progress
        categorized_count = len(df_cat[df_cat['category'] != 'Uncategorised'])
        total_count = len(df_cat)
        progress_percentage = (categorized_count / total_count) * 100
        
        st.progress(progress_percentage / 100)
        st.write(f"📈 **Categorization Progress:** {categorized_count}/{total_count} transactions categorized ({progress_percentage:.1f}%)")
        
        if progress_percentage < 100:
            st.info(f"💡 **Tip:** You can go back to the categorization section above to complete the remaining {total_count - categorized_count} transactions.")
        
        st.divider()
        
        # Save to database section
        st.subheader("💾 Save to Database")
        
        # Check if insert_transactions is available
        try:
            insert_transactions_available = insert_transactions is not None
        except NameError:
            insert_transactions_available = False
        
        can_save = all(
            col in df_cat.columns
            for col in ["date", "description", "original_description", "amount", "currency", "fx_rate", "amount_jpy", "category", "subcategory", "transaction_type"]
        ) and insert_transactions_available

        # Save section with clear instructions
        st.subheader("💾 Save Transactions")
        st.info("**Important:** After categorizing, you MUST click 'Save to Database' below to permanently save your transactions. Use 'Save Progress' above only for temporary backup.")
        
        col_save1, col_save2 = st.columns([1, 1])
        with col_save1:
            save_button_text = "💾 Save processed transactions to DB"
            if can_save:
                save_button_text += " ✅"
            else:
                save_button_text += " ❌ (Data not ready)"
            
            if st.button(save_button_text, type="primary" if can_save else "secondary") and can_save:
                try:
                    batch_id = create_import_record(uploaded_file.name if hasattr(uploaded_file, "name") else "upload", len(df_cat)) if create_import_record else None
                    # Potential duplicate review
                    review_rows = []
                    for _, r in df_cat.iterrows():
                        review_rows.append({
                            "date": r.get("date"),
                            "description": r.get("description"),
                            "original_description": r.get("original_description"),
                            "amount": float(r.get("amount", 0.0)),
                            "currency": r.get("currency", default_currency),
                            "fx_rate": float(r.get("fx_rate", 1.0)),
                            "amount_jpy": float(r.get("amount_jpy", r.get("amount", 0.0))),
                            "category": r.get("category"),
                            "subcategory": r.get("subcategory"),
                            "transaction_type": r.get("transaction_type"),
                        })
                    # Lightweight potential duplicate detection: same date & amount
                    pot_dupes = []
                    try:
                        existing = pd.DataFrame(load_all_transactions() or []) if load_all_transactions else pd.DataFrame()
                        if not existing.empty:
                            existing['date'] = pd.to_datetime(existing['date']).dt.strftime('%Y-%m-%d')
                            for row in review_rows:
                                date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
                                match = existing[(existing['date'] == date_str) & (existing['amount'].round(2) == round(row['amount'], 2))]
                                if len(match) > 0:
                                    pot_dupes.append({
                                        "date": date_str,
                                        "amount": row['amount'],
                                        "new_description": row['description'],
                                        "existing_count": int(len(match))
                                    })
                    except Exception:
                        pass

                    if pot_dupes:
                        st.warning("Potential duplicates detected. Review below; choose whether to insert them.")
                        st.dataframe(pd.DataFrame(pot_dupes))
                        insert_suspected = st.checkbox("Insert suspected duplicates too", value=False)
                    else:
                        insert_suspected = True

                    if not insert_suspected and 'existing' in locals() and not existing.empty:
                        filtered_rows = []
                        for row in review_rows:
                            date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
                            match = existing[(existing['date'] == date_str) & (existing['amount'].round(2) == round(row['amount'], 2))]
                            if len(match) == 0:
                                filtered_rows.append(row)
                        review_rows = filtered_rows

                    inserted, dupes, _ = insert_transactions(review_rows, batch_id)  # type: ignore
                    
                    # Show prominent success message
                    if inserted > 0:
                        st.success(f"🎉 **SUCCESS!** Inserted {inserted} transactions to database. Skipped {dupes} duplicates.")
                        st.balloons()  # Celebration animation
                        st.info("💡 **Next step:** Scroll to the top and expand '📊 View Saved Transactions' to see your data!")
                    else:
                        st.warning(f"No new transactions inserted. Skipped {dupes} duplicates.")
                except Exception as e:
                    st.error(f"Save failed: {e}")
            elif not can_save:
                st.caption("Cannot save yet: data layer not ready or required columns missing.")
        

        with col_save2:
            sanitize = st.checkbox("Sanitize exports (remove Original/Japanese text)", value=False)
            if st.button("🧾 Export DB to CSV") and export_transactions_to_csv is not None:
                try:
                    if sanitize and load_all_transactions is not None:
                        rows = load_all_transactions()
                        df_all = pd.DataFrame(rows)
                        if 'original_description' in df_all.columns:
                            df_all = df_all.drop(columns=['original_description'])
                        ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                        out_path = os.path.join('exports', f'transactions_sanitized_{ts}.csv')
                        os.makedirs('exports', exist_ok=True)
                        df_all.to_csv(out_path, index=False)
                        st.success(f"Exported sanitized CSV: {out_path}")
                    else:
                        csv_path = export_transactions_to_csv()
                        st.success(f"Exported to {csv_path}")
                except Exception as e:
                    st.error(f"Export failed: {e}")

        col_bkp1, col_bkp2 = st.columns([1, 1])
        with col_bkp1:
            if st.button("🗄️ Backup DB (local)") and backup_database is not None:
                try:
                    backup_path = backup_database()
                    st.success(f"Backup created: {backup_path}")
                except Exception as e:
                    st.error(f"Backup failed: {e}")

        with col_bkp2:
            if load_all_transactions is not None and st.button("📚 Show DB count"):
                try:
                    rows = load_all_transactions()
                    st.info(f"DB has {len(rows)} transactions.")
                except Exception as e:
                    st.error(f"Count failed: {e}")

        st.divider()

        # Recurring transactions UI
        st.subheader("🔁 Recurring Transactions")
        st.caption("Mark a transaction as recurring and auto-generate future instances.")

        if list_recurring_rules and upsert_recurring_rule:
            with st.expander("Create / Update recurring rule"):
                merchant_pattern = st.text_input("Merchant contains (pattern)")
                freq = st.selectbox("Frequency", ["monthly", "weekly"], index=0)
                next_date = st.date_input("Next date")
                fixed_amount = st.number_input("Amount (optional)", value=0.0, help="Leave 0 for variable amount detected from transactions")
                cat = st.text_input("Category (optional)")
                subcat = st.text_input("Subcategory (optional)")
                ccy = st.selectbox("Currency", ["JPY", "USD", "EUR", "AUD", "CAD", "GBP", "CNY", "KRW"], index=0)
                if st.button("💾 Save recurring rule"):
                    try:
                        rule_id = upsert_recurring_rule({
                            "merchant_pattern": merchant_pattern.strip(),
                            "frequency": freq,
                            "next_date": next_date.strftime("%Y-%m-%d"),
                            "amount": (None if abs(fixed_amount) < 1e-9 else float(fixed_amount)),
                            "category": (cat.strip() or None),
                            "subcategory": (subcat.strip() or None),
                            "currency": ccy,
                            "active": 1,
                        })
                        st.success(f"Rule saved (id={rule_id})")
                    except Exception as e:
                        st.error(f"Failed to save rule: {e}")

            existing = list_recurring_rules()
            if existing:
                st.write("Active rules:")
                st.dataframe(pd.DataFrame(existing))
            else:
                st.caption("No active recurring rules yet.")

            with st.expander("Preview next generated instances"):
                import datetime as _dt
                preview_rows = []
                for r in existing or []:
                    next_dt = _dt.date.fromisoformat(r["next_date"]) if r.get("next_date") else None
                    if not next_dt:
                        continue
                    # Predict following date
                    if r["frequency"] == "weekly":
                        following = next_dt + _dt.timedelta(days=7)
                    else:
                        # monthly: naive add 30 days
                        following = next_dt + _dt.timedelta(days=30)
                    preview_rows.append({
                        "merchant_pattern": r["merchant_pattern"],
                        "next_date": str(next_dt),
                        "predicted_following": str(following),
                        "amount": r.get("amount"),
                        "category": r.get("category"),
                        "currency": r.get("currency"),
                    })
                if preview_rows:
                    st.dataframe(pd.DataFrame(preview_rows))
                else:
                    st.caption("Nothing to preview yet.")

            # Generate and insert next instances (preview-confirm)
            with st.form("generate_recurring"):
                st.write("Generate next instances for due rules (today or earlier)?")
                gen = st.form_submit_button("➕ Generate Next Instances")
                if gen:
                    try:
                        to_insert = []
                        today = pd.Timestamp.today().date()
                        for r in existing or []:
                            if not r.get("next_date"):
                                continue
                            next_dt = pd.to_datetime(r["next_date"]).date()
                            if next_dt > today:
                                continue
                            desc = f"Recurring: {r['merchant_pattern']}"
                            amt = float(r["amount"]) if r.get("amount") is not None else 0.0
                            ccy = r.get("currency", "JPY")
                            rate = get_fx_rate_to_jpy(ccy, next_dt.strftime("%Y-%m-%d"))
                            to_insert.append({
                                "date": next_dt,
                                "description": desc,
                                "original_description": desc,
                                "amount": amt,
                                "currency": ccy,
                                "fx_rate": rate,
                                "amount_jpy": amt * rate,
                                "category": r.get("category"),
                                "subcategory": r.get("subcategory"),
                                "transaction_type": "Expense" if amt <= 0 else "Credit",
                            })
                        if to_insert and insert_transactions:
                            ins, dups, _ = insert_transactions(to_insert, None)
                            st.success(f"Generated {ins} next instances (skipped {dups} duplicates)")
                        elif not to_insert:
                            st.info("No rules due today.")
                    except Exception as e:
                        st.error(f"Generation failed: {e}")

        # Dedupe Settings UI
        st.divider()
        st.subheader("⚙️ Duplicate Detection Settings")
        st.caption("Configure how the system detects potential duplicate transactions.")
        
        if get_dedupe_settings and save_dedupe_settings:
            with st.expander("🔧 Configure Duplicate Detection"):
                current = get_dedupe_settings()
                
                col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
                with col_d1:
                    similarity = st.slider(
                        "Merchant Name Similarity (%)",
                        min_value=50,
                        max_value=100,
                        value=int(current['similarity_threshold'] * 100),
                        step=5,
                        help="Higher = stricter matching. 85% means merchant names must be 85% similar to be considered duplicates."
                    )
                
                with col_d2:
                    date_range = st.number_input(
                        "Date Range Tolerance (days)",
                        min_value=0,
                        max_value=7,
                        value=current['check_date_range_days'],
                        help="0 = exact date match only. 1-7 = check transactions within ±N days."
                    )
                
                with col_d3:
                    amount_tol = st.slider(
                        "Amount Tolerance (%)",
                        min_value=0.0,
                        max_value=10.0,
                        value=current['check_amount_tolerance'] * 100,
                        step=0.5,
                        format="%.1f",
                        help="0 = exact amount only. 5 = ±5% amount variation allowed."
                    )
                
                if st.button("💾 Save Dedupe Settings"):
                    try:
                        save_dedupe_settings(
                            similarity_threshold=similarity / 100.0,
                            date_range_days=int(date_range),
                            amount_tolerance=amount_tol / 100.0
                        )
                        st.success("✅ Dedupe settings saved!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to save settings: {e}")
                
                # Show current settings summary
                st.caption(f"**Current:** Similarity ≥{int(current['similarity_threshold']*100)}%, Date ±{current['check_date_range_days']} days, Amount ±{current['check_amount_tolerance']*100:.1f}%")
            
            # Fuzzy duplicate checker tool
            with st.expander("🔍 Find Potential Duplicates"):
                st.write("Check for potential duplicates of a specific transaction:")
                check_date = st.date_input("Transaction Date", value=pd.Timestamp.today())
                check_desc = st.text_input("Description")
                check_amount = st.number_input("Amount", value=0.0, format="%.2f")
                
                if st.button("🔎 Find Duplicates") and find_potential_duplicates_fuzzy:
                    if check_desc and check_amount != 0:
                        try:
                            dupes = find_potential_duplicates_fuzzy(
                                date=check_date.strftime("%Y-%m-%d"),
                                description=check_desc,
                                amount=float(check_amount)
                            )
                            if dupes:
                                st.warning(f"Found {len(dupes)} potential duplicate(s):")
                                df_dupes = pd.DataFrame(dupes)
                                st.dataframe(df_dupes, use_container_width=True)
                            else:
                                st.success("✅ No duplicates found with current settings.")
                        except Exception as e:
                            st.error(f"Duplicate check failed: {e}")
                    else:
                        st.warning("Please enter a description and non-zero amount.")

        # Google Drive backup UI
        st.divider()
        st.subheader("☁️ Google Drive Backup")
        st.caption("Authorize once, then upload latest DB backup and CSV export to Drive.")
        try:
            from drive_backup import ensure_oauth, upload_bytes
            creds = ensure_oauth()
            if creds:
                col_g1, col_g2 = st.columns([1, 1])
                with col_g1:
                    if st.button("🔐 Backup DB to Drive"):
                        try:
                            # Create a DB backup file in memory via backup_database path
                            bkp_path = backup_database() if backup_database else None
                            if bkp_path:
                                with open(bkp_path, "rb") as f:
                                    data = f.read()
                                folder_id = st.secrets.get("google", {}).get("drive_folder_id")
                                link = upload_bytes(creds, folder_id, os.path.basename(bkp_path), data, mime="application/x-sqlite3")
                                st.success(f"DB backup uploaded: {link}")
                            else:
                                st.warning("Local backup function not available.")
                        except Exception as e:
                            st.error(f"Drive DB backup failed: {e}")
                with col_g2:
                    if st.button("📤 Export CSV to Drive"):
                        try:
                            csv_path = export_transactions_to_csv() if export_transactions_to_csv else None
                            if csv_path:
                                with open(csv_path, "rb") as f:
                                    data = f.read()
                                folder_id = st.secrets.get("google", {}).get("drive_folder_id")
                                link = upload_bytes(creds, folder_id, os.path.basename(csv_path), data, mime="text/csv")
                                st.success(f"CSV uploaded: {link}")
                            else:
                                st.warning("CSV export function not available.")
                        except Exception as e:
                            st.error(f"Drive CSV upload failed: {e}")
        except Exception as e:
            st.info("Google Drive not configured. Add [google] secrets to enable.")

        # Prepare data for display with better formatting
        display_df = df_cat.copy()
        
        # Format timestamp column for better display
        if 'timestamp' in display_df.columns:
            display_df['timestamp'] = display_df['timestamp'].apply(
                lambda x: x.strftime('%H:%M:%S') if pd.notna(x) and hasattr(x, 'strftime') else x
            )
        
        # Format date column for better display
        if 'date' in display_df.columns:
            display_df['date'] = display_df['date'].apply(
                lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) and hasattr(x, 'strftime') else x
            )
        
        # Calculate and display running balance
        st.subheader("💰 Running Balance Analysis")
        balance_df = calculate_running_balance(df_cat)
        
        # Show balance trend
        if len(balance_df) > 1:
            balance_chart = balance_df[['date', 'running_balance']].copy()
            balance_chart['date'] = pd.to_datetime(balance_chart['date'])
            balance_chart = balance_chart.sort_values('date')
            
            st.line_chart(balance_chart.set_index('date'))
            
            # Show current balance
            current_balance = balance_df['running_balance'].iloc[-1]
            st.info(f"💳 **Current Balance:** ¥{current_balance:,.0f}")
            
            # Show balance statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📈 Highest Balance", f"¥{balance_df['running_balance'].max():,.0f}")
            with col2:
                st.metric("📉 Lowest Balance", f"¥{balance_df['running_balance'].min():,.0f}")
            with col3:
                st.metric("📊 Average Balance", f"¥{balance_df['running_balance'].mean():,.0f}")
        
        # Add quick filters for transactions
        st.divider()
        st.subheader("🔍 Filter Transactions")
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1, 1, 1, 1])
        
        with filter_col1:
            filter_categories = st.multiselect(
                "Categories",
                options=sorted(display_df['category'].unique()) if 'category' in display_df.columns else [],
                default=None,
                help="Select categories to filter (empty = show all)"
            )
        
        with filter_col2:
            filter_type = st.multiselect(
                "Type",
                options=display_df['transaction_type'].unique() if 'transaction_type' in display_df.columns else [],
                default=None,
                help="Filter by transaction type"
            )
        
        with filter_col3:
            filter_currency = st.multiselect(
                "Currency",
                options=sorted(display_df['currency'].unique()) if 'currency' in display_df.columns else [],
                default=None,
                help="Filter by currency"
            )
        
        with filter_col4:
            filter_search = st.text_input(
                "Search Description",
                placeholder="Search...",
                help="Search in transaction descriptions"
            )
        
        # Apply filters
        filtered_df = display_df.copy()
        if filter_categories:
            filtered_df = filtered_df[filtered_df['category'].isin(filter_categories)]
        if filter_type:
            filtered_df = filtered_df[filtered_df['transaction_type'].isin(filter_type)]
        if filter_currency:
            filtered_df = filtered_df[filtered_df['currency'].isin(filter_currency)]
        if filter_search:
            filtered_df = filtered_df[
                filtered_df['description'].str.contains(filter_search, case=False, na=False)
            ]
        
        st.caption(f"Showing {len(filtered_df)} of {len(display_df)} transactions")
        
        # Use data editor for final review
        final_desc_width = st.select_slider(
            "Description column width (final table)",
            options=["small", "medium", "large"],
            value="large"
        )
        edited_df = st.data_editor(
            filtered_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                'description': st.column_config.TextColumn('Description', width=final_desc_width),
                'original_description': st.column_config.TextColumn('Original (Japanese)', width=final_desc_width)
            }
        )
        
        if not edited_df.empty:
            edited_df["month"] = pd.to_datetime(edited_df["date"]).dt.to_period("M").astype(str)
            summary = (
                edited_df.groupby(["month", "category"])["amount"].sum().reset_index(name="total_amount")
            )
            st.subheader("📊 Monthly Summary")
            st.write(summary)
            pivot = summary.pivot(index="month", columns="category", values="total_amount").fillna(0)
            st.bar_chart(pivot)
        
        # Show learning system statistics
        if learning_system:
            st.subheader("🧠 Smart Learning System Statistics")
            learning_stats = learning_system.get_learning_stats()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📚 Total Corrections", learning_stats['total_corrections'])
            with col2:
                st.metric("🏪 Unique Merchants", learning_stats['unique_merchants'])
            with col3:
                st.metric("🔄 Recent Corrections", learning_stats['recent_corrections'])
            
            if learning_stats['merchant_categories']:
                st.write("**Merchant Learning Database:**")
                for merchant, categories in learning_stats['merchant_categories'].items():
                    if merchant != "unknown":
                        st.write(f"• **{merchant}**: {list(categories.keys())}")

    # ========================================================================
    # INDEPENDENT DUPLICATE ANALYSIS SECTION - COMPLETELY SEPARATE FROM MAIN PROCESSING
    # ========================================================================
    
    # Debug: Track when this section is executed
    if 'duplicate_section_executed' not in st.session_state:
        st.session_state['duplicate_section_executed'] = 0
    st.session_state['duplicate_section_executed'] += 1
    
    # Only show duplicate analysis if we have a CSV file and it's not resume mode
    # Use session state to track if we have a CSV file to avoid re-evaluation
    if 'csv_file_uploaded' not in st.session_state:
        st.session_state['csv_file_uploaded'] = False
    
    if st.session_state['csv_file_uploaded'] and not st.session_state.get('resume_mode', False):
        st.divider()
        st.subheader("🔍 Duplicate Analysis & Selective Import")
        st.caption("This section is completely independent from the main processing flow")
        st.caption(f"🔍 Debug: This section has been executed {st.session_state['duplicate_section_executed']} times")
        
        # Create a unique key for this file's duplicate analysis
        # Safely handle None values and sanitize file name
        raw_file_name = st.session_state.get('csv_file_name', 'unknown')
        file_size = st.session_state.get('csv_file_size', 0)
        
        # Sanitize file name for session state key (remove special characters)
        import re
        sanitized_file_name = re.sub(r'[^a-zA-Z0-9_-]', '_', str(raw_file_name))
        file_key = f"duplicate_analysis_{sanitized_file_name}_{file_size}"
        
        # Only run duplicate analysis if not already done for this specific file
        if file_key not in st.session_state or not st.session_state[file_key]:
            st.info("🔍 Analyzing CSV file for potential duplicates...")
            
            try:
                from data_store import compute_dedupe_hash
                from collections import defaultdict
                
                # Create a copy of the dataframe for analysis from session state
                csv_data = st.session_state.get('csv_dataframe', [])
                if not csv_data:
                    st.error("❌ No CSV data available for duplicate analysis")
                    st.session_state[file_key] = True
                    return
                
                try:
                    df_analysis = pd.DataFrame(csv_data)
                except Exception as e:
                    st.error(f"❌ Error creating dataframe from CSV data: {e}")
                    st.session_state[file_key] = True
                    return
                
                # Find potential duplicates
                hash_groups = defaultdict(list)
                
                for idx, row in df_analysis.iterrows():
                    try:
                        date = str(row.get('date', ''))
                        desc = str(row.get('description', ''))
                        amount = float(row.get('amount', 0))
                        
                        # Compute dedupe hash
                        dedupe_hash = compute_dedupe_hash(date, desc, amount)
                        hash_groups[dedupe_hash].append({
                            'row': idx + 1,
                            'date': date,
                            'description': desc,
                            'amount': amount
                        })
                    except Exception as e:
                        st.warning(f"⚠️ Skipping row {idx + 1} due to data error: {e}")
                        continue
                
                # Find duplicates
                duplicates = {h: group for h, group in hash_groups.items() if len(group) > 1}
                
                # Store results in session state with file-specific key
                st.session_state[f'{file_key}_groups'] = duplicates
                st.session_state[f'{file_key}_df'] = df_analysis
                st.session_state[file_key] = True
                
                if duplicates:
                    st.warning(f"⚠️ Found {len(duplicates)} potential duplicate groups in your CSV file")
                else:
                    st.success("✅ No duplicate transactions found in your CSV file")
                    
            except Exception as e:
                st.error(f"Error analyzing duplicates: {e}")
                # Clean up any partial state and mark as failed
                st.session_state[file_key] = True
                # Remove any partial analysis data
                if f'{file_key}_groups' in st.session_state:
                    del st.session_state[f'{file_key}_groups']
                if f'{file_key}_df' in st.session_state:
                    del st.session_state[f'{file_key}_df']
        
        # Show duplicate analysis UI (completely independent)
        if f'{file_key}_groups' in st.session_state and st.session_state[f'{file_key}_groups']:
            duplicates = st.session_state[f'{file_key}_groups']
            df_analysis = st.session_state.get(f'{file_key}_df')
            
            if df_analysis is None:
                st.error("❌ Analysis data not available")
                return
            
            st.write("**Requirements Met:**")
            st.write("✅ Check duplicates marked by system")
            st.write("✅ Manually select valid transactions")
            st.write("✅ Add/save them to database")
            
            with st.expander(f"View {len(duplicates)} Duplicate Groups", expanded=True):
                # Clear previous selection widget states safely BEFORE rendering widgets
                clear_key = f'clear_dup_selections_{file_key}'
                if st.session_state.get(clear_key):
                    keys_to_clear = [k for k in st.session_state.keys() if k.startswith(f"select_{file_key}_")]
                    for k in keys_to_clear:
                        st.session_state.pop(k, None)
                    st.session_state.pop(f"select_all_{file_key}", None)
                    st.session_state[clear_key] = False

                # Build/version indicator for deployment freshness verification
                st.caption("Build: 2025-10-26 15:55Z · dupe-badge-ascii · history-cards")

                # Use a form so checkbox changes do NOT trigger reruns; only the submit does
                with st.form(f"dup_form_{file_key}"):
                    # Optional: select all toggle inside the form
                    select_all = st.checkbox("Select all", key=f"select_all_{file_key}")
                    allow_dupes = st.toggle("Allow importing flagged duplicates", value=True, help="When on, selected items are inserted even if they already exist.")
                    
                    for i, (dedupe_hash, group) in enumerate(duplicates.items(), 1):
                        st.write(f"**Group {i}** ({len(group)} transactions):")
                        
                        # Show each transaction in the group
                        for j, tx in enumerate(group):
                            col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
                            
                            with col1:
                                currency_symbol = "¥"
                                # Badge if exists already
                                try:
                                    from data_store import get_existing_dedupe_hashes, compute_dedupe_hash
                                    dh = compute_dedupe_hash(str(tx['date']), str(tx['description']), float(tx['amount']))
                                    existing = st.session_state.get(f"existing_hashes_{file_key}")
                                    if existing is None:
                                        # Build once per render
                                        all_hashes = [compute_dedupe_hash(str(t['date']), str(t['description']), float(t['amount'])) for g in duplicates.values() for t in g]
                                        existing = get_existing_dedupe_hashes(all_hashes)
                                        st.session_state[f"existing_hashes_{file_key}"] = existing
                                    # Plain ASCII to avoid any encoding/parsing issues in some environments
                                    badge = ('  [saved]') if dh in existing else ''
                                except Exception:
                                    badge = ""
                                st.write(f"  • Row {tx['row']}: {tx['date']} | {tx['description']} | {currency_symbol}{tx['amount']:.2f}{badge}")
                            
                            with col2:
                                tx_key = f"{file_key}_{i}_{j}_{tx['row']}"
                                # Checkbox state will be read on submit; no side-effects here
                                default_checked = bool(select_all)
                                st.checkbox("Select", key=f"select_{tx_key}", value=default_checked)
                            
                            with col3:
                                st.caption("Duplicate")
                            
                            with col4:
                                st.caption("Group " + str(i))
                        
                        st.caption(f"   Hash: {dedupe_hash[:16]}...")
                        st.divider()
                    
                    # Bulk import submit
                    st.subheader("📥 Bulk Import Selected Transactions")
                    submitted = st.form_submit_button("Import Selected", type="primary")
                    
                    if submitted:
                        try:
                            from data_store import insert_transactions, create_import_record
                            
                            # Build selected keys from checkbox states
                            selected_tx_keys = []
                            for i, group in enumerate(duplicates.values(), 1):
                                for j, tx in enumerate(group):
                                    tx_key = f"{file_key}_{i}_{j}_{tx['row']}"
                                    if st.session_state.get(f"select_{tx_key}", False):
                                        selected_tx_keys.append(tx_key)
                            
                            selected_count = len(selected_tx_keys)
                            if selected_count == 0:
                                st.warning("No transactions selected.")
                                st.stop()
                            
                            # Create import record
                            safe_file_name = st.session_state.get('csv_file_name', 'unknown_file')
                            import_batch_id = create_import_record(f"{safe_file_name}_duplicates", selected_count)
                            
                            # Prepare selected transactions for import
                            csv_data = st.session_state.get('csv_dataframe', [])
                            if not csv_data:
                                st.error("❌ No CSV data available for import")
                                st.stop()
                            try:
                                df_analysis = pd.DataFrame(csv_data)
                            except Exception as e:
                                st.error(f"❌ Error creating dataframe from CSV data: {e}")
                                st.stop()
                            
                            transactions_to_import = []
                            for tx_key in selected_tx_keys:
                                parts = tx_key.split('_')
                                row_num = int(parts[-1]) - 1
                                if row_num < 0 or row_num >= len(df_analysis):
                                    continue
                                try:
                                    row_data = df_analysis.iloc[row_num].to_dict()
                                except Exception:
                                    continue
                                try:
                                    tx_data = {
                                        'date': str(row_data.get('date', '')),
                                        'description': str(row_data.get('description', '')),
                                        'original_description': str(row_data.get('original_description', '')),
                                        'amount': float(row_data.get('amount', 0)),
                                        'currency': str(row_data.get('currency', 'JPY')),
                                        'fx_rate': float(row_data.get('fx_rate', 1.0)),
                                        'amount_jpy': float(row_data.get('amount_jpy', row_data.get('amount', 0))),
                                        'category': str(row_data.get('category', '')),
                                        'subcategory': str(row_data.get('subcategory', '')),
                                        'transaction_type': str(row_data.get('transaction_type', 'Expense'))
                                    }
                                except Exception:
                                    continue
                                transactions_to_import.append(tx_data)
                            
                            # Allow intentional import of duplicates based on toggle
                            inserted, dupes, errors = insert_transactions(
                                transactions_to_import,
                                import_batch_id,
                                override_duplicates=bool(allow_dupes)
                            )
                            if inserted > 0:
                                st.success(f"✅ Successfully imported {inserted} transactions!")
                                # Defer clearing widget state to next run; avoid modifying widget keys post-instantiation
                                st.session_state[f'clear_dup_selections_{file_key}'] = True
                                st.rerun()
                            else:
                                st.error("❌ No transactions were imported")
                        except Exception as e:
                            st.error(f"❌ Error importing transactions: {e}")
            
            st.info(f"💡 **Independent Operation:** This section runs completely separately from the main processing. {sum(len(group) - 1 for group in duplicates.values())} transactions will be skipped by default.")


if __name__ == "__main__":
    main()

