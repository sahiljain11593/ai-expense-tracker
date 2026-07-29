#!/usr/bin/env python3
"""
Analyze uncategorised transactions to identify patterns and create better rules.
"""

import csv
from collections import Counter, defaultdict

def analyze_uncategorised_transactions():
    """Analyze uncategorised transactions to find patterns."""
    
    # Read the enhanced categorized data
    uncategorised = []
    all_transactions = []
    
    with open('/workspace/enhanced_categorized_transactions.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            all_transactions.append(row)
            if row['enhanced_category'] == 'Uncategorised':
                uncategorised.append(row)
    
    print(f"📊 Analyzing {len(uncategorised)} uncategorised transactions...")
    print(f"Total transactions: {len(all_transactions)}")
    print(f"Uncategorised rate: {len(uncategorised)/len(all_transactions)*100:.1f}%")
    print()
    
    # Analyze by amount ranges
    print("💰 Analysis by Amount Ranges:")
    amount_ranges = [
        (0, 1000, "¥0 - ¥1,000"),
        (1000, 5000, "¥1,000 - ¥5,000"),
        (5000, 10000, "¥5,000 - ¥10,000"),
        (10000, 50000, "¥10,000 - ¥50,000"),
        (50000, float('inf'), "¥50,000+")
    ]
    
    for min_amt, max_amt, label in amount_ranges:
        if max_amt == float('inf'):
            group = [t for t in uncategorised if float(t['amount']) >= min_amt]
        else:
            group = [t for t in uncategorised if min_amt <= float(t['amount']) < max_amt]
        
        if group:
            total_amount = sum(float(t['amount']) for t in group)
            print(f"  {label}: {len(group)} transactions, ¥{total_amount:,.0f}")
    
    print()
    
    # Analyze by description patterns
    print("🔍 Analysis by Description Patterns:")
    
    # Common Japanese merchants
    japanese_merchants = [
        "ローソン", "セブンイレブン", "ファミリーマート", "ポプラグループ", "ライフ",
        "イオン", "ニトリ", "ドンキホーテ", "バーガーキング", "イージーズカフェ",
        "サカドヤ", "エヌシーデイ", "ハナマルキ", "ハナコウジョウ", "サウナ東京",
        "マルエツ", "マツノヤヨウガ", "ガンソズシ", "マルガメセイメン", "カブシキガイシャ",
        "サントラック", "バブルハウス", "チケットジャム", "ロコハロイ", "ノクテイプラザ",
        "マルイファ", "東京ミッドタウン", "新マルノウチビル", "タイムズカー", "イデミツ",
        "アポロステーション", "イベント", "ＥＴＣ", "楽天", "AMAZON"
    ]
    
    merchant_counts = Counter()
    merchant_amounts = defaultdict(float)
    
    for transaction in uncategorised:
        description = transaction['description']
        amount = float(transaction['amount'])
        
        for merchant in japanese_merchants:
            if merchant in description:
                merchant_counts[merchant] += 1
                merchant_amounts[merchant] += amount
                break
    
    print("  Top merchants by transaction count:")
    for merchant, count in merchant_counts.most_common(10):
        total_amount = merchant_amounts[merchant]
        print(f"    {merchant}: {count} transactions, ¥{total_amount:,.0f}")
    
    print()
    
    # Analyze by specific patterns
    print("🎯 Specific Pattern Analysis:")
    
    patterns = {
        "Convenience Stores": ["ローソン", "セブンイレブン", "ファミリーマート", "ポプラグループ", "ライフ"],
        "ETC/Highway": ["ＥＴＣ", "etc", "高速道路", "highway"],
        "Events": ["イベント", "event", "コンサート", "concert"],
        "Pharmacies": ["薬局", "pharmacy", "ドラッグストア", "drugstore"],
        "Restaurants": ["レストラン", "restaurant", "居酒屋", "izakaya", "カフェ", "cafe"],
        "Online Shopping": ["AMAZON", "amazon", "アマゾン", "楽天", "rakuten"],
        "Furniture": ["ニトリ", "nitori", "家具", "furniture"],
        "Entertainment": ["チケット", "ticket", "映画", "movie", "ゲーム", "game"]
    }
    
    for pattern_name, keywords in patterns.items():
        matching_transactions = []
        for transaction in uncategorised:
            description = transaction['description']
            if any(keyword in description for keyword in keywords):
                matching_transactions.append(transaction)
        
        if matching_transactions:
            total_amount = sum(float(t['amount']) for t in matching_transactions)
            print(f"  {pattern_name}: {len(matching_transactions)} transactions, ¥{total_amount:,.0f}")
    
    print()
    
    # Show sample uncategorised transactions
    print("📋 Sample Uncategorised Transactions:")
    print("  (First 20 by amount)")
    
    sorted_uncategorised = sorted(uncategorised, key=lambda x: float(x['amount']), reverse=True)
    
    for i, transaction in enumerate(sorted_uncategorised[:20]):
        amount = float(transaction['amount'])
        description = transaction['description']
        date = transaction['date']
        print(f"    {i+1:2d}. ¥{amount:8,.0f} | {date} | {description[:50]}...")
    
    print()
    
    # Generate improved categorization rules
    print("🔧 Recommended Improved Categorization Rules:")
    print()
    
    improved_rules = {
        "Convenience Stores": [
            "ローソン", "セブンイレブン", "ファミリーマート", "コンビニ", "lawson", 
            "seven eleven", "family mart", "ポプラグループ", "poplar", "スーパー", 
            "supermarket", "grocery", "ライフ", "life", "イオン", "aeon"
        ],
        "ETC & Transportation": [
            "ＥＴＣ", "etc", "高速道路", "highway", "モバイルパス", "mobile pass",
            "交通費", "transport", "駐車場", "parking", "ガソリン", "gasoline",
            "ＥＮＥＯＳ", "ENEOS", "ガソリンスタンド", "gas station"
        ],
        "Events & Entertainment": [
            "イベント", "event", "コンサート", "concert", "映画", "movie", "ゲーム", "game",
            "チケット", "ticket", "ツアー", "tour", "イベントゴリョウブン", "ゾゾマリンスタジアム",
            "ハナマルウドン", "ミニストップ"
        ],
        "Health & Beauty": [
            "薬局", "pharmacy", "ドラッグストア", "drugstore", "美容", "beauty",
            "化粧品", "cosmetics", "病院", "hospital", "クリニック", "clinic"
        ],
        "Food & Dining": [
            "レストラン", "restaurant", "居酒屋", "izakaya", "カフェ", "cafe", "バー", "bar",
            "ピザ", "pizza", "寿司", "sushi", "マクドナルド", "mcdonalds", "ケンタッキー", "kfc",
            "スターバックス", "starbucks", "ドンキホーテ", "バーガーキング", "イージーズカフェ",
            "サカドヤ", "エヌシーデイ", "ハナマルキ", "ハナコウジョウ", "サウナ東京",
            "マルエツ", "マツノヤヨウガ", "ガンソズシ", "マルガメセイメン", "カブシキガイシャ",
            "サントラック", "バブルハウス"
        ],
        "Shopping & Retail": [
            "AMAZON", "amazon", "アマゾン", "楽天", "rakuten", "ヤフー", "yahoo",
            "ニトリ", "nitori", "イケア", "ikea", "ホームセンター", "home center",
            "チケットジャム", "ロコハロイ", "ノクテイプラザ", "マルイファ", "東京ミッドタウン",
            "新マルノウチビル", "タイムズカー", "イデミツ", "アポロステーション"
        ]
    }
    
    for category, keywords in improved_rules.items():
        print(f"  {category}:")
        for keyword in keywords[:10]:  # Show first 10 keywords
            print(f"    - {keyword}")
        if len(keywords) > 10:
            print(f"    ... and {len(keywords) - 10} more")
        print()

if __name__ == "__main__":
    analyze_uncategorised_transactions()