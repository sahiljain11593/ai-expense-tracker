#!/usr/bin/env python3
"""
Test script to verify the save process works correctly.
This simulates what happens when a user uploads and saves transactions.
"""

import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_store import (
    init_db, 
    create_import_record, 
    insert_transactions, 
    load_all_transactions
)


def test_save_process():
    print("🧪 Testing Save Process...")
    
    # Initialize database
    init_db()
    
    # Get current state
    current_transactions = len(load_all_transactions() or [])
    
    print(f"📊 Current state: {current_transactions} transactions")
    
    # Create test transactions
    test_transactions = [
        {
            'date': '2025-10-12',
            'description': 'Test Transaction 1',
            'original_description': 'テスト取引1',
            'amount': -1000,
            'currency': 'JPY',
            'fx_rate': 1.0,
            'amount_jpy': -1000,
            'category': 'Test Category',
            'subcategory': 'Test Subcategory',
            'transaction_type': 'Expense',
        },
        {
            'date': '2025-10-12',
            'description': 'Test Transaction 2',
            'original_description': 'テスト取引2',
            'amount': -500,
            'currency': 'JPY',
            'fx_rate': 1.0,
            'amount_jpy': -500,
            'category': 'Test Category',
            'subcategory': 'Test Subcategory',
            'transaction_type': 'Expense',
        }
    ]
    
    # Test 1: Create import record
    print("\n1️⃣ Testing import record creation...")
    batch_id = create_import_record("test_save_process.csv", len(test_transactions))
    print(f"✅ Created import record with batch_id: {batch_id}")
    
    # Test 2: Insert transactions
    print("\n2️⃣ Testing transaction insertion...")
    inserted, dupes, errors = insert_transactions(test_transactions, batch_id)
    print(f"✅ Inserted: {inserted}, Duplicates: {dupes}, Errors: {errors}")
    
    # Test 3: Verify data was saved
    print("\n3️⃣ Verifying saved data...")
    final_transactions = load_all_transactions() or []
    
    new_transactions = len(final_transactions) - current_transactions
    
    print(f"📈 New transactions added: {new_transactions}")
    
    # Test 4: Check specific data
    print("\n4️⃣ Checking specific transaction data...")
    test_transaction = next((t for t in final_transactions if 'Test Transaction 1' in t['description']), None)
    if test_transaction:
        print(f"✅ Found test transaction: {test_transaction['description']} - ¥{test_transaction['amount_jpy']}")
        print(f"   Category: {test_transaction['category']}")
        print(f"   Batch ID: {test_transaction['import_batch_id']}")
    else:
        print("❌ Test transaction not found!")
    
    # Test 5: Test duplicate detection
    print("\n5️⃣ Testing duplicate detection...")
    inserted2, dupes2, errors2 = insert_transactions(test_transactions, batch_id)
    print(f"✅ Duplicate test - Inserted: {inserted2}, Duplicates: {dupes2}, Errors: {errors2}")
    
    if dupes2 == 2:
        print("✅ Duplicate detection working correctly!")
    else:
        print("❌ Duplicate detection may have issues")
    
    print("\n🎉 Save process test completed!")
    print(f"📊 Final state: {len(final_transactions)} transactions")


if __name__ == "__main__":
    test_save_process()
