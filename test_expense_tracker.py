#!/usr/bin/env python3
"""
Pytest tests for AI Expense Tracker

Tests cover:
- Data store operations (import, dedupe, recurring rules)
- Duplicate detection with fuzzy matching
- Settings persistence
- Recurring rule generation

Run with: pytest test_expense_tracker.py -v
"""

import os
import tempfile
import pytest
from datetime import datetime, date, timedelta
from unittest.mock import patch, MagicMock

# Import the modules to test
from data_store import (
    init_db, 
    insert_transactions, 
    load_all_transactions,
    get_dedupe_settings,
    set_dedupe_settings,
    find_potential_duplicates,
    upsert_recurring_rule,
    list_recurring_rules,
    get_setting,
    set_setting
)


class TestDataStore:
    """Test data store operations with temporary database."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    def test_init_db(self, temp_db):
        """Test database initialization."""
        init_db(temp_db)
        # Database should be created and tables should exist
        assert os.path.exists(temp_db)
    
    def test_insert_and_load_transactions(self, temp_db):
        """Test inserting and loading transactions."""
        init_db(temp_db)
        
        # Test data
        transactions = [
            {
                "date": "2025-01-01",
                "description": "Test Store",
                "original_description": "テストストア",
                "amount": 1000.0,
                "currency": "JPY",
                "fx_rate": 1.0,
                "amount_jpy": 1000.0,
                "category": "Food",
                "subcategory": "Groceries",
                "transaction_type": "Expense"
            },
            {
                "date": "2025-01-02",
                "description": "Another Store",
                "original_description": "アナザーストア",
                "amount": 2000.0,
                "currency": "JPY",
                "fx_rate": 1.0,
                "amount_jpy": 2000.0,
                "category": "Shopping",
                "subcategory": "Clothing",
                "transaction_type": "Expense"
            }
        ]
        
        # Insert transactions
        inserted, dupes, _ = insert_transactions(transactions, 1, temp_db)
        assert inserted == 2
        assert dupes == 0
        
        # Load transactions
        loaded = load_all_transactions(temp_db)
        assert len(loaded) == 2
        assert loaded[0]['description'] == "Test Store"
        assert loaded[1]['description'] == "Another Store"
    
    def test_duplicate_detection(self, temp_db):
        """Test exact duplicate detection."""
        init_db(temp_db)
        
        # Insert first transaction
        transactions1 = [{
            "date": "2025-01-01",
            "description": "Test Store",
            "original_description": "テストストア",
            "amount": 1000.0,
            "currency": "JPY",
            "fx_rate": 1.0,
            "amount_jpy": 1000.0,
            "category": "Food",
            "subcategory": "Groceries",
            "transaction_type": "Expense"
        }]
        
        inserted, dupes, _ = insert_transactions(transactions1, 1, temp_db)
        assert inserted == 1
        assert dupes == 0
        
        # Try to insert the same transaction again
        inserted, dupes, _ = insert_transactions(transactions1, 2, temp_db)
        assert inserted == 0
        assert dupes == 1
    
    def test_dedupe_settings(self, temp_db):
        """Test dedupe settings persistence."""
        init_db(temp_db)
        
        # Test default settings
        settings = get_dedupe_settings(temp_db)
        assert settings["merchant_similarity_threshold"] == 0.8
        assert settings["amount_tolerance_percent"] == 1.0
        assert settings["date_tolerance_days"] == 1
        
        # Test setting new values
        new_settings = {
            "merchant_similarity_threshold": 0.9,
            "amount_tolerance_percent": 2.0,
            "date_tolerance_days": 2
        }
        set_dedupe_settings(new_settings, temp_db)
        
        # Verify settings were saved
        updated_settings = get_dedupe_settings(temp_db)
        assert updated_settings["merchant_similarity_threshold"] == 0.9
        assert updated_settings["amount_tolerance_percent"] == 2.0
        assert updated_settings["date_tolerance_days"] == 2
    
    def test_fuzzy_duplicate_detection(self, temp_db):
        """Test fuzzy duplicate detection."""
        init_db(temp_db)
        
        # Insert existing transaction
        existing_transaction = [{
            "date": "2025-01-01",
            "description": "Amazon Japan",
            "original_description": "アマゾンジャパン",
            "amount": 1500.0,
            "currency": "JPY",
            "fx_rate": 1.0,
            "amount_jpy": 1500.0,
            "category": "Shopping",
            "subcategory": "Online",
            "transaction_type": "Expense"
        }]
        
        inserted, _, _ = insert_transactions(existing_transaction, 1, temp_db)
        assert inserted == 1
        
        # Test similar transactions
        similar_transactions = [
            {
                "date": "2025-01-01",  # Same date
                "description": "Amazon JP",  # Similar description
                "original_description": "アマゾンJP",
                "amount": 1500.0,  # Same amount
                "currency": "JPY",
                "fx_rate": 1.0,
                "amount_jpy": 1500.0,
                "category": "Shopping",
                "subcategory": "Online",
                "transaction_type": "Expense"
            },
            {
                "date": "2025-01-02",  # Different date
                "description": "Amazon Japan",  # Same description
                "original_description": "アマゾンジャパン",
                "amount": 1500.0,  # Same amount
                "currency": "JPY",
                "fx_rate": 1.0,
                "amount_jpy": 1500.0,
                "category": "Shopping",
                "subcategory": "Online",
                "transaction_type": "Expense"
            }
        ]
        
        # Find potential duplicates
        potential_dupes = find_potential_duplicates(similar_transactions, temp_db)
        
        # Should find at least one potential duplicate (the first one)
        assert len(potential_dupes) >= 1
        
        # Check the similarity score
        if potential_dupes:
            assert potential_dupes[0]["similarity_score"] > 0.6  # Reasonable similarity
            # The first transaction should have same date (0 days diff)
            same_date_dupes = [dup for dup in potential_dupes if dup["date_diff_days"] == 0]
            if same_date_dupes:
                assert same_date_dupes[0]["amount_diff"] == 0  # Same amount
    
    def test_recurring_rules(self, temp_db):
        """Test recurring rule operations."""
        init_db(temp_db)
        
        # Create a recurring rule
        rule = {
            "merchant_pattern": "Amazon",
            "frequency": "monthly",
            "next_date": "2025-02-01",
            "amount": 1500.0,
            "category": "Shopping",
            "subcategory": "Online",
            "currency": "JPY",
            "active": 1
        }
        
        rule_id = upsert_recurring_rule(rule, temp_db)
        assert rule_id > 0
        
        # List recurring rules
        rules = list_recurring_rules(temp_db)
        assert len(rules) == 1
        assert rules[0]["merchant_pattern"] == "Amazon"
        assert rules[0]["frequency"] == "monthly"
        
        # Update the rule
        rule["id"] = rule_id
        rule["amount"] = 2000.0
        updated_id = upsert_recurring_rule(rule, temp_db)
        assert updated_id == rule_id
        
        # Verify update
        rules = list_recurring_rules(temp_db)
        assert rules[0]["amount"] == 2000.0
    
    def test_settings_persistence(self, temp_db):
        """Test general settings persistence."""
        init_db(temp_db)
        
        # Test setting and getting a value
        set_setting("test_key", "test_value", temp_db)
        value = get_setting("test_key", temp_db)
        assert value == "test_value"
        
        # Test updating a value
        set_setting("test_key", "updated_value", temp_db)
        value = get_setting("test_key", temp_db)
        assert value == "updated_value"
        
        # Test non-existent key
        value = get_setting("non_existent", temp_db)
        assert value is None


class TestDedupeLogic:
    """Test dedupe logic with various scenarios."""
    
    def test_merchant_similarity(self):
        """Test merchant name similarity detection."""
        from difflib import SequenceMatcher
        
        # Test cases
        test_cases = [
            ("Amazon Japan", "Amazon JP", 0.7),  # High similarity
            ("Amazon Japan", "Amazon", 0.6),     # Medium similarity
            ("Amazon Japan", "eBay", 0.1),       # Low similarity
            ("Starbucks", "Starbucks Coffee", 0.7),  # High similarity
            ("McDonald's", "McDonalds", 0.8),    # Very high similarity
        ]
        
        for desc1, desc2, expected_min_similarity in test_cases:
            similarity = SequenceMatcher(None, desc1.lower(), desc2.lower()).ratio()
            assert similarity >= expected_min_similarity, f"Similarity between '{desc1}' and '{desc2}' should be >= {expected_min_similarity}, got {similarity}"
    
    def test_amount_tolerance(self):
        """Test amount tolerance calculations."""
        # Test cases: (amount1, amount2, tolerance_percent, should_match)
        test_cases = [
            (1000.0, 1000.0, 1.0, True),    # Exact match
            (1000.0, 1010.0, 1.0, True),    # Within 1%
            (1000.0, 1020.0, 1.0, False),   # Outside 1%
            (1000.0, 990.0, 1.0, True),     # Within 1% (negative)
            (1000.0, 980.0, 1.0, False),    # Outside 1% (negative)
            (0.0, 0.0, 1.0, True),          # Zero amounts
        ]
        
        for amount1, amount2, tolerance_percent, should_match in test_cases:
            amount_diff = abs(amount1 - amount2)
            if amount1 != 0:
                actual_tolerance = amount_diff / abs(amount1)
                matches = actual_tolerance <= (tolerance_percent / 100.0)
            else:
                matches = amount2 == 0
            
            assert matches == should_match, f"Amount tolerance test failed: {amount1} vs {amount2} with {tolerance_percent}% tolerance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])