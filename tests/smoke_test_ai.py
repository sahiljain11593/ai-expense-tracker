"""End-to-end smoke test for the AI translation + categorization pipeline.

Runs a sample Japanese bank-statement CSV through the full AI pipeline and
prints a pass/fail report. Designed so you (or an agent) can verify a code
change without manually uploading a file in the Streamlit UI.

Usage:
    python tests/smoke_test_ai.py                 # runs against gemini if key is present
    python tests/smoke_test_ai.py --provider openai
    python tests/smoke_test_ai.py --provider both
    python tests/smoke_test_ai.py --model gemini-2.5-flash

API keys are loaded in this order:
    1. .streamlit/secrets.toml ([gemini].api_key, [openai].api_key)
    2. Environment variables (GEMINI_API_KEY, OPENAI_API_KEY)

Exit code:
    0 = all selected providers passed
    1 = at least one provider failed
    2 = no providers had a usable API key
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

# Allow imports from project root when run as `python tests/smoke_test_ai.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress noisy Streamlit warnings about missing ScriptRunContext when we
# call helpers from a plain Python script.
os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
import logging  # noqa: E402

logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.CRITICAL
)


SAMPLE_CSV = PROJECT_ROOT / "tests" / "data" / "sample_japanese_statement.csv"


# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _color(text: str, color: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{RESET}"


def _section(title: str) -> None:
    print()
    print(_color(f"=== {title} ===", BOLD + CYAN))


def _pass(msg: str) -> None:
    print(_color("[PASS] ", GREEN) + msg)


def _fail(msg: str) -> None:
    print(_color("[FAIL] ", RED) + msg)


def _warn(msg: str) -> None:
    print(_color("[WARN] ", YELLOW) + msg)


def _info(msg: str) -> None:
    print(_color("[INFO] ", DIM) + msg)


# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------
def _load_keys_from_secrets() -> dict[str, str]:
    """Best-effort parse of .streamlit/secrets.toml for [openai] and [gemini] keys."""
    secrets = {}
    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return secrets

    try:
        text = secrets_path.read_text()
    except Exception:
        return secrets

    for section in ("openai", "gemini"):
        m = re.search(
            rf"\[{section}\][^\[]*?api_key\s*=\s*\"([^\"]+)\"", text, flags=re.DOTALL
        )
        if m:
            secrets[section] = m.group(1)
    return secrets


def _resolve_api_keys() -> dict[str, str]:
    keys = _load_keys_from_secrets()
    keys.setdefault("openai", os.getenv("OPENAI_API_KEY", ""))
    keys.setdefault("gemini", os.getenv("GEMINI_API_KEY", ""))
    return {k: v for k, v in keys.items() if v}


# ---------------------------------------------------------------------------
# Test runner for a single provider
# ---------------------------------------------------------------------------
EXPECTED_MERCHANT_KEYWORDS = {
    # Lowercased substrings expected in the English translation. We test that AT LEAST
    # half of these appear, to avoid being too brittle across model versions.
    "ｽﾀｰﾊﾞｯｸｽｺｰﾋｰ ｼﾌﾞﾔ": ["starbucks"],
    "ﾄﾘｷｿﾞｸ ｼﾝｼﾞｭｸ": ["torikizoku", "izakaya"],
    "ｾﾌﾞﾝｲﾚﾌﾞﾝ ｱｵﾔﾏ": ["7-eleven", "seven", "eleven"],
    "ｲｵﾝ ﾓｰﾙ ﾐｿﾞﾉｸﾞﾁ": ["aeon", "ion"],
    "ﾏﾂﾓﾄｷﾖｼ ｼﾌﾞﾔ": ["matsumoto", "drugstore"],
    "ﾄﾞﾝｷﾎｰﾃ ﾐｿﾞﾉｸﾞﾁ": ["don quijote", "donki"],
    "ﾈｯﾄﾌﾘｯｸｽ": ["netflix"],
    "ﾕﾆｸﾛ ｼﾌﾞﾔ": ["uniqlo"],
    "LAWSON ﾐｿﾞﾉｸﾞﾁｴｷﾏｴ": ["lawson"],
    "ﾌｧﾐﾘｰﾏｰﾄ ｱｵﾔﾏ": ["familymart", "family mart"],
}


def run_for_provider(
    provider: str, api_key: str, model: str | None, sample_path: Path
) -> bool:
    """Run translation + categorization end-to-end for one provider.

    Returns True on pass, False on fail.
    """
    _section(f"Provider: {provider} (model={model or 'default'})")

    # Lazy import so that even an import error gets caught nicely
    try:
        import pandas as pd
        import transaction_web_app as twa
    except Exception as e:
        _fail(f"failed to import transaction_web_app: {e}")
        return False

    base_url = twa.GEMINI_PROVIDER if provider == "gemini" else None

    # Load sample CSV (already has Date,Description,Amount columns)
    df = pd.read_csv(sample_path)
    _info(f"loaded {len(df)} rows from {sample_path.name}")

    # ---- 1. Batch translation ----
    descriptions = df["Description"].tolist()
    t0 = time.time()
    try:
        translations = twa.translate_batch_ai(
            descriptions,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )
    except Exception as e:
        _fail(f"translate_batch_ai raised: {e}")
        return False
    elapsed = time.time() - t0

    if not translations:
        _fail("translate_batch_ai returned empty mapping")
        return False
    _pass(f"batch translation completed in {elapsed:.1f}s ({len(translations)} entries)")

    # Show a few sample translations
    print()
    print(_color("  Sample translations:", DIM))
    sample_pairs = list(translations.items())[:8]
    for jp, en in sample_pairs:
        print(f"    {jp[:30]:<30} -> {en}")

    # Quality check: at least half the expected merchants should match
    matches, total = 0, 0
    for jp_text, expected_kws in EXPECTED_MERCHANT_KEYWORDS.items():
        if jp_text not in translations:
            continue
        total += 1
        en = (translations[jp_text] or "").lower()
        if any(kw in en for kw in expected_kws):
            matches += 1
    if total > 0:
        ratio = matches / total
        msg = f"merchant recognition: {matches}/{total} ({ratio:.0%})"
        if ratio >= 0.5:
            _pass(msg)
        else:
            _warn(msg + " (model may be misidentifying merchants)")
            # Don't fail the whole test — quality varies by model

    # ---- 2. Categorization ----
    # Build a DataFrame in the shape categorise_transactions_ai expects
    cat_df = pd.DataFrame(
        {
            "description": [translations.get(d, d) for d in descriptions],
            "original_description": descriptions,
            "amount": df["Amount"].astype(float).tolist(),
        }
    )

    # Use the same category schema the app uses
    categories = [
        "Food",
        "Social Life",
        "Subscriptions",
        "Household",
        "Transportation",
        "Vacation",
        "Health",
        "Apparel",
        "Grooming",
        "Self-development",
        "Income",
    ]
    subcategories = {}  # let the model emit free-form sub; validation will null-out invalid ones

    t0 = time.time()
    try:
        result = twa.categorise_transactions_ai(
            cat_df,
            api_key=api_key,
            categories=categories,
            subcategories=subcategories,
            model=model,
            base_url=base_url,
        )
    except Exception as e:
        _fail(f"categorise_transactions_ai raised: {e}")
        return False
    elapsed = time.time() - t0

    cats_assigned = (result["category"] != "Uncategorised").sum()
    _pass(
        f"categorization completed in {elapsed:.1f}s ({cats_assigned}/{len(result)} rows categorized)"
    )

    # Summary table
    print()
    print(_color("  Final results (first 15 rows):", DIM))
    print(f"    {'Original':<28} {'English':<35} {'Category':<18} {'Conf':>6}")
    for _, row in result.head(15).iterrows():
        orig = row.get("original_description", "")[:28]
        eng = row.get("description", "")[:35]
        cat = row.get("category", "")[:18]
        conf = row.get("confidence", 0.0)
        print(f"    {orig:<28} {eng:<35} {cat:<18} {conf:>6.2f}")

    # Hard-fail if the LLM crashed without crashing (i.e., everything Uncategorised)
    if cats_assigned == 0:
        _fail("zero rows were categorized — the AI categorizer effectively failed")
        return False

    # Soft-warn if many rows are Uncategorised (could indicate JSON parse issues)
    if cats_assigned < len(result) * 0.5:
        _warn(
            f"only {cats_assigned}/{len(result)} rows categorized — investigate JSON parsing"
        )

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="AI pipeline smoke test")
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "both"],
        default="gemini",
        help="Which provider(s) to test (default: gemini)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model name (e.g. gpt-4o-mini, gemini-2.5-flash). "
        "Defaults to the provider's default.",
    )
    parser.add_argument(
        "--csv",
        default=str(SAMPLE_CSV),
        help=f"Path to sample CSV (default: {SAMPLE_CSV.relative_to(PROJECT_ROOT)})",
    )
    args = parser.parse_args()

    sample_path = Path(args.csv).resolve()
    if not sample_path.exists():
        _fail(f"sample CSV not found: {sample_path}")
        return 2

    keys = _resolve_api_keys()

    providers_to_test: list[str] = []
    if args.provider == "both":
        providers_to_test = ["gemini", "openai"]
    else:
        providers_to_test = [args.provider]

    # Filter to providers that actually have a key
    runnable = [(p, keys[p]) for p in providers_to_test if keys.get(p)]
    if not runnable:
        _fail(
            "no API keys found for the selected provider(s). "
            f"Looked in .streamlit/secrets.toml and env vars. Selected: {providers_to_test}"
        )
        return 2

    # Some providers are skipped (no key) — note that
    skipped = [p for p in providers_to_test if not keys.get(p)]
    for p in skipped:
        _warn(f"skipping {p} (no API key configured)")

    _section("AI Pipeline Smoke Test")
    _info(f"sample file: {sample_path.relative_to(PROJECT_ROOT)}")
    _info(f"providers to test: {[p for p, _ in runnable]}")

    results: dict[str, bool] = {}
    for provider, key in runnable:
        try:
            ok = run_for_provider(provider, key, args.model, sample_path)
        except KeyboardInterrupt:
            _warn("interrupted by user")
            return 1
        except Exception as e:
            _fail(f"unexpected error in {provider}: {e}")
            ok = False
        results[provider] = ok

    # Summary
    _section("Summary")
    for provider, ok in results.items():
        if ok:
            _pass(f"{provider}: passed")
        else:
            _fail(f"{provider}: failed")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
