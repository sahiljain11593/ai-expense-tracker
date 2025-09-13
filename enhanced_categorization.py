#!/usr/bin/env python3
"""
Enhanced Categorization System for Japanese Bank Statements

This system provides improved categorization rules specifically designed
for Japanese bank statements and merchants.
"""

import csv
import re
from datetime import datetime
from collections import defaultdict

class EnhancedCategorizer:
    """Enhanced categorization system for Japanese transactions."""
    
    def __init__(self):
        self.categorization_rules = {
            "Home & Furniture": [
                # Furniture and home goods
                "ニトリ", "nitori", "家具", "furniture", "ホームセンター", "home center",
                "イケア", "ikea", "無印良品", "muji", "ニトリネット"
            ],
            "Convenience Stores": [
                # Convenience stores and supermarkets
                "ローソン", "セブンイレブン", "ファミリーマート", "コンビニ", "lawson", 
                "seven eleven", "family mart", "ポプラグループ", "poplar", "スーパー", 
                "supermarket", "grocery", "ライフ", "life", "イオン", "aeon",
                "イトーヨーカドー", "itoyokado", "西友", "seiyu"
            ],
            "Food & Dining": [
                # Restaurants and food
                "レストラン", "restaurant", "cafe", "dinner", "lunch", "breakfast", 
                "居酒屋", "izakaya", "バー", "bar", "カフェ", "coffee", "ピザ", "pizza",
                "マクドナルド", "mcdonalds", "ケンタッキー", "kfc", "スターバックス", "starbucks",
                "ドンキホーテ", "バーガーキング", "イージーズカフェ", "サカドヤ", "エヌシーデイ",
                "ハナマルキ", "ハナコウジョウ", "サウナ東京", "マルエツ", "マツノヤヨウガ",
                "ガンソズシ", "マルガメセイメン", "カブシキガイシャ", "サントラック", "バブルハウス"
            ],
            "Transportation": [
                # Public transport and travel
                "電車", "train", "バス", "bus", "タクシー", "taxi", "地下鉄", "subway", 
                "モノレール", "monorail", "モバイルパス", "mobile pass", "交通費", "transport",
                "駐車場", "parking", "高速道路", "highway", "ＥＴＣ", "etc", "ガソリン", 
                "gasoline", "燃料", "fuel", "車", "car", "バイク", "bike", "ＥＮＥＯＳ", "ENEOS"
            ],
            "Shopping & Retail": [
                # Online shopping and retail
                "AMAZON", "amazon", "アマゾン", "楽天", "rakuten", "ヤフー", "yahoo",
                "ユニクロ", "uniqlo", "zara", "h&m", "gap", "nike", "adidas", "アディダス", "ナイキ",
                "ファッション", "fashion", "スタイル", "style", "ブランド", "brand",
                "チケットジャム", "ロコハロイ", "ノクテイプラザ", "マルイファ", "東京ミッドタウン",
                "新マルノウチビル", "タイムズカー", "イデミツ", "アポロステーション"
            ],
            "Entertainment & Events": [
                # Entertainment and events
                "イベント", "event", "コンサート", "concert", "映画", "movie", "ゲーム", "game",
                "スポーツ", "sports", "カラオケ", "karaoke", "ボーリング", "bowling",
                "ゾゾマリンスタジアム", "ハナマルウドン", "ミニストップ", "イベントゴリョウブン",
                "チケット", "ticket", "ツアー", "tour", "宿泊", "accommodation"
            ],
            "Health & Beauty": [
                # Healthcare and personal care
                "病院", "hospital", "クリニック", "clinic", "歯科", "dental", "眼科", "eye", 
                "薬局", "pharmacy", "薬", "medicine", "保険", "insurance", "診察", "examination",
                "治療", "treatment", "フィットネス", "fitness", "ジム", "gym", "ヨガ", "yoga",
                "マッサージ", "massage", "美容", "beauty", "化粧品", "cosmetics"
            ],
            "Subscriptions & Services": [
                # Digital services and utilities
                "OPENAI", "CHATGPT", "icloud", "apple music", "amazon prime", "google one",
                "netflix", "spotify", "hulu", "disney+", "アマゾンプライム", "グーグルワン",
                "アップルミュージック", "アイクラウド", "subscription", "membership", "月額", "monthly",
                "楽天モバイル", "rakuten mobile", "通信料", "communication", "電話", "phone",
                "楽天でんき", "rakuten electricity", "楽天ガス", "rakuten gas", "電気", "electric", "ガス", "gas"
            ],
            "Fees & Charges": [
                # Bank fees and charges
                "手数料", "fee", "年会費", "annual fee", "利用料", "usage fee", "ＡＴＭ", "ATM",
                "支払手数料", "payment fee", "現地利用額", "local usage", "変換レート", "exchange rate",
                "楽天ゴールドカード年会費", "credit card annual fee"
            ],
            "Cash & ATM": [
                # Cash withdrawals
                "ローソン銀行", "lawson bank", "ＣＤ", "CD", "現金", "cash", "引き出し", "withdrawal"
            ]
        }
        
        # Subcategories for more detailed analysis
        self.subcategories = {
            "Convenience Stores": {
                "Lawson": ["ローソン", "lawson"],
                "Seven Eleven": ["セブンイレブン", "seven eleven"],
                "Family Mart": ["ファミリーマート", "family mart"],
                "Supermarkets": ["スーパー", "supermarket", "ライフ", "life", "イオン", "aeon"]
            },
            "Transportation": {
                "ETC/Highway": ["ＥＴＣ", "etc", "高速道路", "highway"],
                "Public Transport": ["電車", "train", "バス", "bus", "地下鉄", "subway"],
                "Mobile Pass": ["モバイルパス", "mobile pass"],
                "Fuel": ["ガソリン", "gasoline", "ＥＮＥＯＳ", "ENEOS"]
            },
            "Shopping & Retail": {
                "Online Shopping": ["AMAZON", "amazon", "アマゾン", "楽天", "rakuten"],
                "Furniture": ["ニトリ", "nitori", "イケア", "ikea"],
                "Fashion": ["ユニクロ", "uniqlo", "zara", "h&m", "gap"]
            }
        }
    
    def categorize_transaction(self, description, amount=None):
        """Categorize a transaction with enhanced rules and confidence scoring."""
        description_lower = description.lower()
        
        # Check each category with confidence scoring
        best_category = "Uncategorised"
        best_confidence = 0.0
        matched_keywords = []
        
        for category, keywords in self.categorization_rules.items():
            matches = []
            for keyword in keywords:
                if keyword.lower() in description_lower:
                    matches.append(keyword)
            
            if matches:
                # Calculate confidence based on number of matches and keyword specificity
                confidence = min(0.95, 0.3 + (len(matches) * 0.1))
                
                # Boost confidence for exact merchant matches
                for match in matches:
                    if match.lower() in ["amazon", "ニトリ", "ローソン", "セブンイレブン", "ファミリーマート"]:
                        confidence = min(0.95, confidence + 0.2)
                
                if confidence > best_confidence:
                    best_category = category
                    best_confidence = confidence
                    matched_keywords = matches
        
        # Special rules for amount-based categorization
        if amount and amount > 50000 and best_category == "Uncategorised":
            # Large amounts are likely furniture or major purchases
            if any(word in description_lower for word in ["ニトリ", "nitori", "家具", "furniture"]):
                best_category = "Home & Furniture"
                best_confidence = 0.8
        
        # Determine subcategory
        subcategory = self.get_subcategory(best_category, description_lower)
        
        return {
            'category': best_category,
            'subcategory': subcategory,
            'confidence': best_confidence,
            'matched_keywords': matched_keywords
        }
    
    def get_subcategory(self, category, description_lower):
        """Get subcategory for a transaction."""
        if category in self.subcategories:
            for sub_cat, keywords in self.subcategories[category].items():
                if any(keyword.lower() in description_lower for keyword in keywords):
                    return sub_cat
        return ""
    
    def process_transactions(self, input_file, output_file):
        """Process transactions with enhanced categorization."""
        print(f"🔄 Processing transactions with enhanced categorization...")
        
        transactions = []
        categorized_count = 0
        
        # Read transactions
        with open(input_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    amount = float(row['amount'])
                    description = row['description']
                    
                    # Get enhanced categorization
                    categorization = self.categorize_transaction(description, amount)
                    
                    # Add categorization data to transaction
                    transaction = dict(row)
                    transaction.update({
                        'enhanced_category': categorization['category'],
                        'subcategory': categorization['subcategory'],
                        'confidence': categorization['confidence'],
                        'matched_keywords': '|'.join(categorization['matched_keywords'])
                    })
                    
                    transactions.append(transaction)
                    
                    if categorization['category'] != 'Uncategorised':
                        categorized_count += 1
                        
                except (ValueError, KeyError) as e:
                    print(f"⚠️ Skipping invalid transaction: {e}")
                    continue
        
        # Write enhanced data
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            if transactions:
                fieldnames = transactions[0].keys()
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(transactions)
        
        print(f"✅ Enhanced categorization complete!")
        print(f"📊 Categorized: {categorized_count}/{len(transactions)} transactions ({categorized_count/len(transactions)*100:.1f}%)")
        
        return transactions
    
    def generate_categorization_report(self, transactions):
        """Generate detailed categorization report."""
        report = []
        report.append("# Enhanced Categorization Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Category summary
        category_stats = defaultdict(lambda: {'count': 0, 'total_amount': 0, 'avg_confidence': 0})
        
        for transaction in transactions:
            category = transaction['enhanced_category']
            amount = float(transaction['amount'])
            confidence = float(transaction['confidence'])
            
            category_stats[category]['count'] += 1
            category_stats[category]['total_amount'] += amount
            category_stats[category]['avg_confidence'] += confidence
        
        # Calculate averages
        for category, stats in category_stats.items():
            if stats['count'] > 0:
                stats['avg_confidence'] /= stats['count']
        
        # Sort by total amount
        sorted_categories = sorted(category_stats.items(), key=lambda x: x[1]['total_amount'], reverse=True)
        
        report.append("## Category Summary")
        report.append("")
        report.append("| Category | Transactions | Total Amount | Avg Confidence |")
        report.append("|----------|-------------|--------------|----------------|")
        
        for category, stats in sorted_categories:
            report.append(f"| {category} | {stats['count']} | ¥{stats['total_amount']:,.0f} | {stats['avg_confidence']:.1%} |")
        
        report.append("")
        
        # Subcategory analysis
        report.append("## Subcategory Analysis")
        report.append("")
        
        subcategory_stats = defaultdict(lambda: {'count': 0, 'total_amount': 0})
        
        for transaction in transactions:
            if transaction['subcategory']:
                subcategory = transaction['subcategory']
                amount = float(transaction['amount'])
                
                subcategory_stats[subcategory]['count'] += 1
                subcategory_stats[subcategory]['total_amount'] += amount
        
        if subcategory_stats:
            sorted_subcategories = sorted(subcategory_stats.items(), key=lambda x: x[1]['total_amount'], reverse=True)
            
            report.append("| Subcategory | Transactions | Total Amount |")
            report.append("|-------------|-------------|--------------|")
            
            for subcategory, stats in sorted_subcategories:
                report.append(f"| {subcategory} | {stats['count']} | ¥{stats['total_amount']:,.0f} |")
        
        report.append("")
        
        # Uncategorised analysis
        uncategorised = [t for t in transactions if t['enhanced_category'] == 'Uncategorised']
        if uncategorised:
            report.append("## Uncategorised Transactions Analysis")
            report.append("")
            report.append(f"**Total Uncategorised:** {len(uncategorised)} transactions, ¥{sum(float(t['amount']) for t in uncategorised):,.0f}")
            report.append("")
            
            # Group by common patterns
            pattern_groups = defaultdict(list)
            for transaction in uncategorised:
                description = transaction['description']
                # Look for common patterns
                if 'ＥＴＣ' in description:
                    pattern_groups['ETC Charges'].append(transaction)
                elif any(word in description for word in ['イベント', 'event']):
                    pattern_groups['Events'].append(transaction)
                elif any(word in description for word in ['薬局', 'pharmacy']):
                    pattern_groups['Pharmacies'].append(transaction)
                else:
                    pattern_groups['Other'].append(transaction)
            
            for pattern, group in pattern_groups.items():
                if group:
                    total_amount = sum(float(t['amount']) for t in group)
                    report.append(f"### {pattern}")
                    report.append(f"- **Count:** {len(group)} transactions")
                    report.append(f"- **Total:** ¥{total_amount:,.0f}")
                    report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        categorized_count = len([t for t in transactions if t['enhanced_category'] != 'Uncategorised'])
        total_count = len(transactions)
        categorization_rate = categorized_count / total_count * 100
        
        if categorization_rate < 80:
            report.append(f"- ⚠️ **Categorization rate is {categorization_rate:.1f}%** - Add more rules for Japanese merchants")
        else:
            report.append(f"- ✅ **Categorization rate is {categorization_rate:.1f}%** - Good coverage")
        
        report.append("- Review uncategorised transactions and add specific rules")
        report.append("- Consider implementing machine learning for better accuracy")
        report.append("- Set up automated categorization for recurring merchants")
        
        return "\n".join(report)


def main():
    """Main function to run enhanced categorization."""
    categorizer = EnhancedCategorizer()
    
    # Process transactions
    input_file = "/workspace/categorized_transactions.csv"
    output_file = "/workspace/enhanced_categorized_transactions.csv"
    
    transactions = categorizer.process_transactions(input_file, output_file)
    
    # Generate report
    report = categorizer.generate_categorization_report(transactions)
    
    # Save report
    with open("/workspace/enhanced_categorization_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"📄 Enhanced categorization report saved to: /workspace/enhanced_categorization_report.md")
    print(f"📊 Enhanced data saved to: {output_file}")


if __name__ == "__main__":
    main()