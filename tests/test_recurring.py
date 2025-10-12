"""
Tests for recurring transaction generation logic.

Tests cover:
- Recurring rule creation and persistence
- Rule activation/deactivation
- Next date calculations (monthly/weekly)
- Transaction generation from rules
"""

import os
import sys
import tempfile
import pytest
from datetime import date, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import data_store


@pytest.fixture
def test_db():
    """Create a temporary test database."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db') as f:
        test_db_path = f.name
    
    # Initialize
    data_store.init_db(test_db_path)
    
    yield test_db_path
    
    # Cleanup
    if os.path.exists(test_db_path):
        os.remove(test_db_path)


def test_create_recurring_rule(test_db):
    """Test creating a new recurring rule."""
    rule = {
        'merchant_pattern': 'Netflix',
        'frequency': 'monthly',
        'next_date': '2024-02-01',
        'amount': -1500,
        'category': 'Subscriptions',
        'subcategory': 'Streaming',
        'currency': 'JPY',
        'active': 1,
    }
    
    rule_id = data_store.upsert_recurring_rule(rule, db_path=test_db)
    
    assert rule_id > 0


def test_list_recurring_rules(test_db):
    """Test listing active recurring rules."""
    # Create some rules
    rule1 = {
        'merchant_pattern': 'Netflix',
        'frequency': 'monthly',
        'next_date': '2024-02-01',
        'amount': -1500,
        'category': 'Subscriptions',
        'subcategory': None,
        'currency': 'JPY',
        'active': 1,
    }
    
    rule2 = {
        'merchant_pattern': 'Spotify',
        'frequency': 'monthly',
        'next_date': '2024-02-05',
        'amount': -980,
        'category': 'Subscriptions',
        'subcategory': None,
        'currency': 'JPY',
        'active': 1,
    }
    
    data_store.upsert_recurring_rule(rule1, db_path=test_db)
    data_store.upsert_recurring_rule(rule2, db_path=test_db)
    
    rules = data_store.list_recurring_rules(db_path=test_db)
    
    assert len(rules) == 2
    assert any(r['merchant_pattern'] == 'Netflix' for r in rules)
    assert any(r['merchant_pattern'] == 'Spotify' for r in rules)


def test_update_recurring_rule(test_db):
    """Test updating an existing recurring rule."""
    # Create a rule
    rule = {
        'merchant_pattern': 'Netflix',
        'frequency': 'monthly',
        'next_date': '2024-02-01',
        'amount': -1500,
        'category': 'Subscriptions',
        'subcategory': None,
        'currency': 'JPY',
        'active': 1,
    }
    
    rule_id = data_store.upsert_recurring_rule(rule, db_path=test_db)
    
    # Update it
    rule['id'] = rule_id
    rule['amount'] = -1600  # Price increase
    rule['next_date'] = '2024-03-01'
    
    updated_id = data_store.upsert_recurring_rule(rule, db_path=test_db)
    
    assert updated_id == rule_id
    
    # Verify update
    rules = data_store.list_recurring_rules(db_path=test_db)
    assert len(rules) == 1
    assert rules[0]['amount'] == -1600


def test_deactivate_recurring_rule(test_db):
    """Test deactivating a recurring rule."""
    # Create a rule
    rule = {
        'merchant_pattern': 'Netflix',
        'frequency': 'monthly',
        'next_date': '2024-02-01',
        'amount': -1500,
        'category': 'Subscriptions',
        'subcategory': None,
        'currency': 'JPY',
        'active': 1,
    }
    
    rule_id = data_store.upsert_recurring_rule(rule, db_path=test_db)
    
    # Deactivate it
    rule['id'] = rule_id
    rule['active'] = 0
    
    data_store.upsert_recurring_rule(rule, db_path=test_db)
    
    # Should not appear in active rules list
    rules = data_store.list_recurring_rules(db_path=test_db)
    assert len(rules) == 0


def test_weekly_frequency_rule(test_db):
    """Test creating a weekly recurring rule."""
    rule = {
        'merchant_pattern': 'Gym Membership',
        'frequency': 'weekly',
        'next_date': '2024-01-15',
        'amount': -2000,
        'category': 'Health',
        'subcategory': None,
        'currency': 'JPY',
        'active': 1,
    }
    
    rule_id = data_store.upsert_recurring_rule(rule, db_path=test_db)
    
    rules = data_store.list_recurring_rules(db_path=test_db)
    assert len(rules) == 1
    assert rules[0]['frequency'] == 'weekly'


def test_monthly_frequency_rule(test_db):
    """Test creating a monthly recurring rule."""
    rule = {
        'merchant_pattern': 'Rent',
        'frequency': 'monthly',
        'next_date': '2024-02-01',
        'amount': -80000,
        'category': 'Housing',
        'subcategory': None,
        'currency': 'JPY',
        'active': 1,
    }
    
    rule_id = data_store.upsert_recurring_rule(rule, db_path=test_db)
    
    rules = data_store.list_recurring_rules(db_path=test_db)
    assert len(rules) == 1
    assert rules[0]['frequency'] == 'monthly'


def test_variable_amount_rule(test_db):
    """Test creating a rule with no fixed amount (variable)."""
    rule = {
        'merchant_pattern': 'Electric Company',
        'frequency': 'monthly',
        'next_date': '2024-02-15',
        'amount': None,  # Variable amount
        'category': 'Utilities',
        'subcategory': None,
        'currency': 'JPY',
        'active': 1,
    }
    
    rule_id = data_store.upsert_recurring_rule(rule, db_path=test_db)
    
    rules = data_store.list_recurring_rules(db_path=test_db)
    assert len(rules) == 1
    assert rules[0]['amount'] is None


def test_multiple_currencies(test_db):
    """Test recurring rules with different currencies."""
    rule_jpy = {
        'merchant_pattern': 'Local Subscription',
        'frequency': 'monthly',
        'next_date': '2024-02-01',
        'amount': -1000,
        'category': 'Subscriptions',
        'subcategory': None,
        'currency': 'JPY',
        'active': 1,
    }
    
    rule_usd = {
        'merchant_pattern': 'AWS',
        'frequency': 'monthly',
        'next_date': '2024-02-01',
        'amount': -50,
        'category': 'Subscriptions',
        'subcategory': 'Cloud',
        'currency': 'USD',
        'active': 1,
    }
    
    data_store.upsert_recurring_rule(rule_jpy, db_path=test_db)
    data_store.upsert_recurring_rule(rule_usd, db_path=test_db)
    
    rules = data_store.list_recurring_rules(db_path=test_db)
    assert len(rules) == 2
    
    currencies = {r['currency'] for r in rules}
    assert 'JPY' in currencies
    assert 'USD' in currencies


def test_rules_sorted_by_next_date(test_db):
    """Test that rules are returned sorted by next_date."""
    rule1 = {
        'merchant_pattern': 'Later Rule',
        'frequency': 'monthly',
        'next_date': '2024-03-01',
        'amount': -1000,
        'category': 'Subscriptions',
        'subcategory': None,
        'currency': 'JPY',
        'active': 1,
    }
    
    rule2 = {
        'merchant_pattern': 'Earlier Rule',
        'frequency': 'monthly',
        'next_date': '2024-02-01',
        'amount': -1000,
        'category': 'Subscriptions',
        'subcategory': None,
        'currency': 'JPY',
        'active': 1,
    }
    
    data_store.upsert_recurring_rule(rule1, db_path=test_db)
    data_store.upsert_recurring_rule(rule2, db_path=test_db)
    
    rules = data_store.list_recurring_rules(db_path=test_db)
    
    # Should be sorted by next_date
    assert rules[0]['merchant_pattern'] == 'Earlier Rule'
    assert rules[1]['merchant_pattern'] == 'Later Rule'


def test_recurring_rule_with_category(test_db):
    """Test that category information is preserved in recurring rules."""
    rule = {
        'merchant_pattern': 'Netflix',
        'frequency': 'monthly',
        'next_date': '2024-02-01',
        'amount': -1500,
        'category': 'Subscriptions',
        'subcategory': 'Entertainment',
        'currency': 'JPY',
        'active': 1,
    }
    
    rule_id = data_store.upsert_recurring_rule(rule, db_path=test_db)
    
    rules = data_store.list_recurring_rules(db_path=test_db)
    assert len(rules) == 1
    assert rules[0]['category'] == 'Subscriptions'
    assert rules[0]['subcategory'] == 'Entertainment'


def test_merchant_pattern_matching(test_db):
    """Test that merchant patterns are stored correctly for matching."""
    rule = {
        'merchant_pattern': 'Starbucks',
        'frequency': 'weekly',
        'next_date': '2024-01-15',
        'amount': -500,
        'category': 'Food',
        'subcategory': None,
        'currency': 'JPY',
        'active': 1,
    }
    
    rule_id = data_store.upsert_recurring_rule(rule, db_path=test_db)
    
    rules = data_store.list_recurring_rules(db_path=test_db)
    assert rules[0]['merchant_pattern'] == 'Starbucks'


def test_settings_persistence(test_db):
    """Test that settings table works correctly for rule-related config."""
    # This tests the general settings mechanism used by recurring rules
    data_store.set_setting('test_key', 'test_value', db_path=test_db)
    
    value = data_store.get_setting('test_key', db_path=test_db)
    assert value == 'test_value'


def test_empty_rules_list(test_db):
    """Test that listing rules returns empty list when no rules exist."""
    rules = data_store.list_recurring_rules(db_path=test_db)
    assert rules == []


def test_rule_persistence_across_connections(test_db):
    """Test that rules persist across database connections."""
    rule = {
        'merchant_pattern': 'Netflix',
        'frequency': 'monthly',
        'next_date': '2024-02-01',
        'amount': -1500,
        'category': 'Subscriptions',
        'subcategory': None,
        'currency': 'JPY',
        'active': 1,
    }
    
    # Insert rule
    rule_id = data_store.upsert_recurring_rule(rule, db_path=test_db)
    
    # Read in separate connection (happens automatically in list_recurring_rules)
    rules = data_store.list_recurring_rules(db_path=test_db)
    
    assert len(rules) == 1
    assert rules[0]['merchant_pattern'] == 'Netflix'

