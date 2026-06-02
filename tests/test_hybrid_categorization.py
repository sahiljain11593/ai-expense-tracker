"""Tests for hybrid categorization pipeline."""

import pandas as pd

from categorization_config import get_categorization_rules, get_subcategories
from categorization_engine import apply_hybrid_categorization, categorise_transactions
from transaction_web_app import MerchantLearningSystem


def test_rules_match_japanese_original_description():
    df = pd.DataFrame(
        [
            {
                "date": "2026-04-30",
                "description": "Yakiniku shop",
                "original_description": "焼肉ホルモンアポロン溝の口",
                "amount": 14190,
            },
            {
                "date": "2026-04-23",
                "description": "Hotel booking",
                "original_description": "HOTEL AT BOOKING.COM利用国NL",
                "amount": 79211,
            },
            {
                "date": "2026-04-08",
                "description": "Rakuten electricity",
                "original_description": "楽天でんき",
                "amount": 11843,
            },
        ]
    )
    rules = get_categorization_rules()
    subcats = get_subcategories()
    out = categorise_transactions(df, rules, subcats)
    assert out.loc[0, "category"] == "Food"
    assert out.loc[1, "category"] == "Vacation"
    assert out.loc[2, "category"] == "Household"


def test_hybrid_uses_rules_before_learning():
    df = pd.DataFrame(
        [
            {
                "date": "2026-04-26",
                "description": "convenience",
                "original_description": "セブン−イレブン",
                "amount": 192,
            }
        ]
    )
    learning = MerchantLearningSystem()
    out = apply_hybrid_categorization(
        df, learning, get_categorization_rules(), get_subcategories()
    )
    assert out.loc[0, "category"] == "Food"
    assert out.loc[0, "confidence"] >= 0.5
