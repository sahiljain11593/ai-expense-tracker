"""
Hybrid and rule-based transaction categorization.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from categorization_config import CATEGORY_MATCH_ORDER


def transaction_match_text(row) -> str:
    parts = [
        str(row.get("description", "") or ""),
        str(row.get("original_description", "") or ""),
    ]
    return " ".join(parts)


def categorize_text_with_rules(
    text: str,
    rules: Dict[str, List[str]],
    subcategories: Optional[Dict[str, Dict[str, List[str]]]] = None,
    uncategorised_label: str = "Uncategorised",
) -> Tuple[str, str]:
    sub_patterns = {}
    if subcategories:
        for main_cat, subs in subcategories.items():
            for sub_cat, keywords in subs.items():
                sub_patterns[f"{main_cat}_{sub_cat}"] = {
                    "main": main_cat,
                    "sub": sub_cat,
                    "pattern": re.compile(
                        "(" + "|".join(map(re.escape, keywords)) + ")", re.IGNORECASE
                    ),
                }

    for sub_info in sub_patterns.values():
        if sub_info["pattern"].search(text):
            return sub_info["main"], sub_info["sub"]

    patterns = {
        cat: re.compile("(" + "|".join(map(re.escape, kws)) + ")", re.IGNORECASE)
        for cat, kws in rules.items()
    }
    for cat in CATEGORY_MATCH_ORDER:
        if cat in patterns and patterns[cat].search(text):
            return cat, ""
    for cat, pattern in patterns.items():
        if cat not in CATEGORY_MATCH_ORDER and pattern.search(text):
            return cat, ""

    return uncategorised_label, ""


def categorise_transactions(
    df: pd.DataFrame,
    rules: Dict[str, List[str]],
    subcategories: Optional[Dict[str, Dict[str, List[str]]]] = None,
    uncategorised_label: str = "Uncategorised",
) -> pd.DataFrame:
    categories = []
    subcategories_list = []
    for _, row in df.iterrows():
        cat, sub = categorize_text_with_rules(
            transaction_match_text(row), rules, subcategories, uncategorised_label
        )
        categories.append(cat)
        subcategories_list.append(sub)
    out = df.copy()
    out["category"] = categories
    out["subcategory"] = subcategories_list
    return out


def apply_hybrid_categorization(
    df: pd.DataFrame,
    learning_system,
    rules: Dict[str, List[str]],
    subcategories: Optional[Dict[str, Dict[str, List[str]]]],
    ensemble_engine=None,
    historical_data=None,
    *,
    learning_min_confidence: float = 0.5,
    ensemble_min_confidence: float = 0.35,
) -> pd.DataFrame:
    df = df.copy()
    categories = []
    subcategories_list = []
    confidences = []
    prediction_breakdowns = []

    for _, row in df.iterrows():
        text = transaction_match_text(row)
        cat, sub = categorize_text_with_rules(text, rules, subcategories)
        confidence = 0.85 if cat != "Uncategorised" else 0.0
        breakdown = {"method": "rules", "confidence": confidence, "category": cat}

        transaction_data = {
            "description": row.get("description", ""),
            "original_description": row.get("original_description", ""),
            "amount": row.get("amount", 0),
            "date": row.get("date"),
            "transaction_type": row.get("transaction_type", "Expense"),
        }

        if cat == "Uncategorised" and learning_system:
            learned_cat, learned_conf, learned_breakdown = learning_system.predict_category(
                transaction_data
            )
            if learned_cat != "Uncategorised" and learned_conf >= learning_min_confidence:
                cat = learned_cat
                confidence = learned_conf
                breakdown = {**learned_breakdown, "method": "merchant_learning"}

        if cat == "Uncategorised" and ensemble_engine is not None:
            try:
                ens_cat, ens_sub, ens_conf, ens_expl = ensemble_engine.predict(
                    {
                        "description": transaction_data["description"],
                        "amount": -abs(float(transaction_data.get("amount") or 0)),
                        "date": str(transaction_data.get("date") or ""),
                    },
                    historical_data=historical_data,
                )
                if ens_cat != "Uncategorised" and ens_conf >= ensemble_min_confidence:
                    cat = ens_cat
                    sub = ens_sub or sub
                    confidence = ens_conf
                    breakdown = {
                        "method": "ensemble",
                        "confidence": ens_conf,
                        "explanation": ens_expl,
                    }
            except Exception as exc:
                breakdown["ensemble_error"] = str(exc)

        if cat != "Uncategorised" and not sub and subcategories and cat in subcategories:
            for sub_cat, keywords in subcategories[cat].items():
                if any(kw.lower() in text.lower() for kw in keywords):
                    sub = sub_cat
                    break

        categories.append(cat)
        subcategories_list.append(sub)
        confidences.append(confidence)
        prediction_breakdowns.append(breakdown)

    df["category"] = categories
    df["subcategory"] = subcategories_list
    df["confidence"] = confidences
    df["prediction_breakdown"] = prediction_breakdowns
    return df
