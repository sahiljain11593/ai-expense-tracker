#!/usr/bin/env python3
"""
Excel Data Analyzer for MoneyMgr Data
Analyzes the structure and content of the MoneyMgr Excel export
"""

import pandas as pd
import os
from datetime import datetime

def analyze_excel_file(filename):
    """Analyze the Excel file structure and content."""
    
    print("🔍 Analyzing MoneyMgr Excel file...")
    print("=" * 50)
    
    try:
        # Read the Excel file
        df = pd.read_excel(filename)
        
        print(f"📊 **File Information:**")
        print(f"   • Filename: {filename}")
        print(f"   • Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"   • File size: {os.path.getsize(filename) / 1024:.1f} KB")
        print()
        
        print(f"📋 **Column Structure:**")
        for i, col in enumerate(df.columns):
            print(f"   {i+1:2d}. {col}")
        print()
        
        print(f"📅 **Data Range:**")
        # Try to find date columns
        date_columns = []
        for col in df.columns:
            if 'date' in col.lower() or '日' in col:
                date_columns.append(col)
        
        if date_columns:
            for col in date_columns:
                try:
                    # Convert to datetime and find range
                    dates = pd.to_datetime(df[col], errors='coerce')
                    valid_dates = dates.dropna()
                    if len(valid_dates) > 0:
                        min_date = valid_dates.min()
                        max_date = valid_dates.max()
                        print(f"   • {col}: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                except:
                    pass
        print()
        
        print(f"💰 **Amount Information:**")
        # Try to find amount columns
        amount_columns = []
        for col in df.columns:
            if 'amount' in col.lower() or '金額' in col or 'amount' in col:
                amount_columns.append(col)
        
        if amount_columns:
            for col in amount_columns:
                try:
                    amounts = pd.to_numeric(df[col], errors='coerce')
                    valid_amounts = amounts.dropna()
                    if len(valid_amounts) > 0:
                        print(f"   • {col}:")
                        print(f"     - Min: ¥{valid_amounts.min():,.0f}")
                        print(f"     - Max: ¥{valid_amounts.max():,.0f}")
                        print(f"     - Total: ¥{valid_amounts.sum():,.0f}")
                        print(f"     - Count: {len(valid_amounts)} transactions")
                except:
                    pass
        print()
        
        print(f"🏷️ **Category Analysis:**")
        # Try to find category columns
        category_columns = []
        for col in df.columns:
            if 'category' in col.lower() or 'カテゴリ' in col or 'category' in col:
                category_columns.append(col)
        
        if category_columns:
            for col in category_columns:
                print(f"   • {col}:")
                category_counts = df[col].value_counts()
                print(f"     - Total categories: {len(category_counts)}")
                print(f"     - Top 10 categories:")
                for cat, count in category_counts.head(10).items():
                    print(f"       {cat}: {count} transactions")
        else:
            print("   • No category column found")
        print()
        
        print(f"🏪 **Merchant/Description Analysis:**")
        # Try to find description columns
        desc_columns = []
        for col in df.columns:
            if 'description' in col.lower() or 'merchant' in col.lower() or '店' in col or '商品' in col:
                desc_columns.append(col)
        
        if desc_columns:
            for col in desc_columns:
                print(f"   • {col}:")
                # Show some sample values
                sample_values = df[col].dropna().head(5).tolist()
                print(f"     - Sample values:")
                for val in sample_values:
                    print(f"       • {str(val)[:50]}...")
        else:
            print("   • No description/merchant column found")
        print()
        
        print(f"📝 **Sample Data (First 5 rows):**")
        print(df.head().to_string())
        print()
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return None

if __name__ == "__main__":
    filename = "2025-08-31 2.xlsx"
    df = analyze_excel_file(filename)
    
    if df is not None:
        print("✅ Analysis complete! Check the output above for data structure details.")
    else:
        print("❌ Failed to analyze the Excel file.")
