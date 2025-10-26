#!/usr/bin/env python3
"""
Analyze duplicate transactions to understand why they were marked as duplicates.
"""

import sqlite3
import sys
import os
from datetime import datetime
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_store import get_connection, compute_dedupe_hash

def analyze_duplicates(db_path="data/expense_tracker.db"):
    """Analyze what transactions were marked as duplicates and why."""
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}")
        return
    
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        
        # Get all transactions with their dedupe hashes
        cur.execute("""
            SELECT id, date, description, amount, dedupe_hash, created_at, import_batch_id
            FROM transactions 
            ORDER BY created_at DESC
        """)
        
        transactions = cur.fetchall()
        print(f"📊 Total transactions in database: {len(transactions)}")
        print()
        
        # Group by dedupe hash to find duplicates
        hash_groups = defaultdict(list)
        for tx in transactions:
            hash_groups[tx[4]].append(tx)  # tx[4] is dedupe_hash
        
        # Find groups with more than one transaction
        duplicates = {h: group for h, group in hash_groups.items() if len(group) > 1}
        
        if not duplicates:
            print("✅ No duplicate transactions found in database")
            return
        
        print(f"🔍 Found {len(duplicates)} duplicate groups:")
        print("=" * 80)
        
        for i, (dedupe_hash, group) in enumerate(duplicates.items(), 1):
            print(f"\n📋 Duplicate Group {i}:")
            print(f"   Hash: {dedupe_hash[:16]}...")
            print(f"   Count: {len(group)} transactions")
            print()
            
            for j, tx in enumerate(group):
                tx_id, date, desc, amount, _, created_at, batch_id = tx
                print(f"   {j+1}. ID: {tx_id} | {date} | {desc} | ${amount:.2f}")
                print(f"      Created: {created_at} | Batch: {batch_id}")
            
            # Analyze why they're considered duplicates
            first_tx = group[0]
            print(f"\n   🔍 Analysis:")
            print(f"   - Date: {first_tx[1]}")
            print(f"   - Description: '{first_tx[2]}'")
            print(f"   - Amount: {first_tx[3]}")
            
            # Check if amounts are exactly the same
            amounts = [tx[3] for tx in group]
            if len(set(amounts)) == 1:
                print(f"   - ✅ All amounts identical: ${amounts[0]}")
            else:
                print(f"   - ⚠️  Amounts differ: {amounts}")
            
            # Check if descriptions are the same (case-insensitive)
            descs = [tx[2].lower().strip() for tx in group]
            if len(set(descs)) == 1:
                print(f"   - ✅ All descriptions identical: '{descs[0]}'")
            else:
                print(f"   - ⚠️  Descriptions differ: {descs}")
            
            print("-" * 60)
        
        # Check recent imports
        print(f"\n📥 Recent imports:")
        cur.execute("""
            SELECT id, file_name, imported_at, num_rows 
            FROM imports 
            ORDER BY imported_at DESC 
            LIMIT 5
        """)
        
        imports = cur.fetchall()
        for imp in imports:
            imp_id, file_name, imported_at, num_rows = imp
            print(f"   - {file_name} ({num_rows} rows) - {imported_at}")
        
    finally:
        conn.close()

def check_dedupe_settings(db_path="data/expense_tracker.db"):
    """Check current deduplication settings."""
    from data_store import get_dedupe_settings
    
    try:
        settings = get_dedupe_settings(db_path)
        print(f"\n⚙️  Current Deduplication Settings:")
        print(f"   - Similarity Threshold: {settings.get('similarity_threshold', 'Not set')}")
        print(f"   - Date Range Days: {settings.get('date_range_days', 'Not set')}")
        print(f"   - Amount Tolerance: {settings.get('amount_tolerance', 'Not set')}")
    except Exception as e:
        print(f"   - Error reading settings: {e}")

if __name__ == "__main__":
    print("🔍 Analyzing Duplicate Transactions")
    print("=" * 50)
    
    analyze_duplicates()
    check_dedupe_settings()
    
    print(f"\n💡 Tips:")
    print(f"   - If amounts are identical, these might be legitimate duplicates")
    print(f"   - If amounts differ slightly, check for rounding issues")
    print(f"   - If descriptions differ, check for formatting differences")
    print(f"   - Consider adjusting deduplication settings if needed")
