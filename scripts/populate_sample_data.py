#!/usr/bin/env python3
"""
Quick script to populate the database with sample transactions
so you can test the "View Saved Transactions" feature.
"""

import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_store import init_db, create_import_record, insert_transactions


def main():
    print("🔄 Populating database with sample transactions...")
    
    # Initialize database
    init_db()
    
    # Sample transactions (based on your CSV data)
    sample_transactions = [
        {
            'date': '2025-09-03',
            'description': 'Rakuten Gold Card Annual Fee',
            'original_description': '楽天ゴールドカード年会費／２６年０７月迄',
            'amount': -2200,
            'currency': 'JPY',
            'fx_rate': 1.0,
            'amount_jpy': -2200,
            'category': 'Subscriptions',
            'subcategory': 'Credit Card',
            'transaction_type': 'Expense',
        },
        {
            'date': '2025-08-31',
            'description': 'Amazon.com',
            'original_description': 'AMAZON.CO.JP',
            'amount': -4980,
            'currency': 'JPY',
            'fx_rate': 1.0,
            'amount_jpy': -4980,
            'category': 'Shopping & Retail',
            'subcategory': 'Online',
            'transaction_type': 'Expense',
        },
        {
            'date': '2025-08-31',
            'description': 'Amazon.com',
            'original_description': 'AMAZON.CO.JP',
            'amount': -2232,
            'currency': 'JPY',
            'fx_rate': 1.0,
            'amount_jpy': -2232,
            'category': 'Shopping & Retail',
            'subcategory': 'Online',
            'transaction_type': 'Expense',
        },
        {
            'date': '2025-08-31',
            'description': 'Don Quijote',
            'original_description': 'ドンキホーテミゾノグチエキマ',
            'amount': -540,
            'currency': 'JPY',
            'fx_rate': 1.0,
            'amount_jpy': -540,
            'category': 'Shopping & Retail',
            'subcategory': 'General',
            'transaction_type': 'Expense',
        },
        {
            'date': '2025-08-30',
            'description': 'Salary Deposit',
            'original_description': '給与',
            'amount': 250000,
            'currency': 'JPY',
            'fx_rate': 1.0,
            'amount_jpy': 250000,
            'category': 'Income',
            'subcategory': 'Salary',
            'transaction_type': 'Credit',
        },
        {
            'date': '2025-08-29',
            'description': 'Starbucks Coffee',
            'original_description': 'スターバックスコーヒー',
            'amount': -650,
            'currency': 'JPY',
            'fx_rate': 1.0,
            'amount_jpy': -650,
            'category': 'Food',
            'subcategory': 'Coffee',
            'transaction_type': 'Expense',
        },
        {
            'date': '2025-08-28',
            'description': 'JR Train Fare',
            'original_description': 'JR運賃',
            'amount': -180,
            'currency': 'JPY',
            'fx_rate': 1.0,
            'amount_jpy': -180,
            'category': 'Transportation',
            'subcategory': 'Train',
            'transaction_type': 'Expense',
        },
        {
            'date': '2025-08-27',
            'description': 'Convenience Store',
            'original_description': 'セブンイレブン',
            'amount': -320,
            'currency': 'JPY',
            'fx_rate': 1.0,
            'amount_jpy': -320,
            'category': 'Food',
            'subcategory': 'Convenience Store',
            'transaction_type': 'Expense',
        }
    ]
    
    # Create import record
    batch_id = create_import_record("sample_data.csv", len(sample_transactions))
    
    # Insert transactions
    inserted, dupes, _ = insert_transactions(sample_transactions, batch_id)
    
    print(f"✅ Successfully inserted {inserted} transactions")
    print(f"⚠️ Skipped {dupes} duplicates")
    print(f"📊 Total transactions now in database")
    
    # Show summary
    print("\n📈 Summary:")
    total_expenses = sum(t['amount_jpy'] for t in sample_transactions if t['transaction_type'] == 'Expense')
    total_credits = sum(t['amount_jpy'] for t in sample_transactions if t['transaction_type'] == 'Credit')
    print(f"   Total Expenses: ¥{abs(total_expenses):,}")
    print(f"   Total Credits: ¥{total_credits:,}")
    print(f"   Net: ¥{total_credits + total_expenses:,}")
    
    print("\n🎉 Done! Now you can test the 'View Saved Transactions' feature in the app!")
    print("   Go to: https://sahiljain11593-ai-expense-tracker-transaction-web-app-lu0rff.streamlit.app/")


if __name__ == "__main__":
    main()
