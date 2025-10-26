#!/usr/bin/env python3
"""
Check if transactions would be marked as duplicates before importing.
"""

import sys
import os
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_store import compute_dedupe_hash

def analyze_csv_for_duplicates(csv_file):
    """Analyze a CSV file to find potential duplicates before importing."""
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return
    
    print(f"🔍 Analyzing {csv_file} for potential duplicates...")
    print("=" * 60)
    
    # Read CSV file
    import pandas as pd
    try:
        df = pd.read_csv(csv_file)
        print(f"📊 Total rows in CSV: {len(df)}")
        print(f"📋 Columns: {list(df.columns)}")
        print()
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    # Find potential duplicates
    hash_groups = defaultdict(list)
    
    for idx, row in df.iterrows():
        # Extract date, description, amount
        date = str(row.get('date', row.get('Date', '')))
        desc = str(row.get('description', row.get('Description', row.get('Transaction', ''))))
        amount = float(row.get('amount', row.get('Amount', row.get('Debit', row.get('Credit', 0)))))
        
        # Compute dedupe hash
        dedupe_hash = compute_dedupe_hash(date, desc, amount)
        hash_groups[dedupe_hash].append({
            'row': idx + 1,
            'date': date,
            'description': desc,
            'amount': amount
        })
    
    # Find duplicates
    duplicates = {h: group for h, group in hash_groups.items() if len(group) > 1}
    
    if not duplicates:
        print("✅ No duplicate transactions found in CSV")
        return
    
    print(f"🔍 Found {len(duplicates)} duplicate groups in CSV:")
    print()
    
    for i, (dedupe_hash, group) in enumerate(duplicates.items(), 1):
        print(f"📋 Duplicate Group {i}:")
        print(f"   Hash: {dedupe_hash[:16]}...")
        print(f"   Count: {len(group)} transactions")
        print()
        
        for j, tx in enumerate(group):
            print(f"   {j+1}. Row {tx['row']}: {tx['date']} | {tx['description']} | ${tx['amount']:.2f}")
        
        print("-" * 40)
    
    print(f"\n💡 Summary:")
    print(f"   - Total rows: {len(df)}")
    print(f"   - Unique transactions: {len(hash_groups)}")
    print(f"   - Duplicate groups: {len(duplicates)}")
    print(f"   - Would be skipped: {sum(len(group) - 1 for group in duplicates.values())}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 check_duplicates.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    analyze_csv_for_duplicates(csv_file)
