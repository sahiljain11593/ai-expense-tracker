"""
Pytest configuration and shared fixtures.

This file contains pytest fixtures that are available to all test files.
"""

import os
import sys
import tempfile
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(scope='session')
def test_data_dir():
    """Provide a temporary directory for test data that persists for the session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope='function')
def clean_env(monkeypatch):
    """Ensure tests run with clean environment variables."""
    # Remove sensitive env vars during tests
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('GOOGLE_APPLICATION_CREDENTIALS', raising=False)
    yield


@pytest.fixture
def sample_transaction():
    """Provide a sample transaction dict for testing."""
    return {
        'date': '2024-01-15',
        'description': 'Test Merchant',
        'original_description': 'Test Merchant Original',
        'amount': -1000,
        'currency': 'JPY',
        'fx_rate': 1.0,
        'amount_jpy': -1000,
        'category': 'Food',
        'subcategory': 'Restaurants',
        'transaction_type': 'Expense',
    }


@pytest.fixture
def sample_recurring_rule():
    """Provide a sample recurring rule dict for testing."""
    return {
        'merchant_pattern': 'Netflix',
        'frequency': 'monthly',
        'next_date': '2024-02-01',
        'amount': -1500,
        'category': 'Subscriptions',
        'subcategory': 'Entertainment',
        'currency': 'JPY',
        'active': 1,
    }

