#!/usr/bin/env python3
import csv

csv_path = "/workspace/temp_repo/bank/statements/enavi202509(5734) (2).csv"

print("Debugging CSV file...")
print("=" * 50)

with open(csv_path, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    print("Column names:", reader.fieldnames)
    print()
    
    for i, row in enumerate(reader):
        if i >= 5:  # Only show first 5 rows
            break
        
        print(f"Row {i+1}:")
        for key, value in row.items():
            print(f"  {key}: '{value}'")
        print()
        
        # Check if this row has valid data
        if row.get('利用日') and row.get('利用店名・商品名') and row.get('利用金額'):
            print("  ✅ This row has valid data")
        else:
            print("  ❌ This row is missing required fields")
        print()