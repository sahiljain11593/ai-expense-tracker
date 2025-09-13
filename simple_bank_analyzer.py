#!/usr/bin/env python3
"""
Simple Bank Statement Analyzer

A lightweight version that works with basic Python libraries to analyze
and cross-check bank statements without external dependencies.
"""

import csv
import json
import re
from datetime import datetime
from collections import defaultdict, Counter
import os

class SimpleBankAnalyzer:
    """Simple bank statement analyzer using only standard library."""
    
    def __init__(self):
        self.csv_data = []
        self.pdf_data = []
        self.categorization_rules = {
            "Food & Groceries": [
                "ローソン", "セブンイレブン", "ファミリーマート", "コンビニ", "lawson", "seven eleven", "family mart",
                "ポプラグループ", "poplar", "スーパー", "supermarket", "grocery", "market", "food", "fresh",
                "イオン", "aeon", "イトーヨーカドー", "itoyokado", "西友", "seiyu", "ライフ", "life",
                "ドンキホーテ", "バーガーキング", "イージーズカフェ", "サカドヤ", "エヌシーデイ",
                "ハナマルキ", "ハナコウジョウ", "サウナ東京", "マルエツ", "マツノヤヨウガ",
                "ガンソズシ", "マルガメセイメン", "カブシキガイシャ", "サントラック", "バブルハウス"
            ],
            "Transportation": [
                "電車", "train", "バス", "bus", "タクシー", "taxi", "地下鉄", "subway", "モノレール", "monorail",
                "モバイルパス", "mobile pass", "交通費", "transport", "駐車場", "parking", "高速道路", "highway",
                "ＥＴＣ", "etc", "ガソリン", "gasoline", "燃料", "fuel", "車", "car", "バイク", "bike",
                "ＥＮＥＯＳ", "ENEOS", "ガソリンスタンド", "gas station"
            ],
            "Shopping & Retail": [
                "AMAZON", "amazon", "アマゾン", "楽天", "rakuten", "ヤフー", "yahoo",
                "ニトリ", "nitori", "イケア", "ikea", "ホームセンター", "home center", "家具", "furniture",
                "ユニクロ", "uniqlo", "zara", "h&m", "gap", "nike", "adidas", "アディダス", "ナイキ",
                "ファッション", "fashion", "スタイル", "style", "ブランド", "brand",
                "チケットジャム", "ロコハロイ", "ノクテイプラザ", "マルイファ", "東京ミッドタウン",
                "新マルノウチビル", "タイムズカー", "イデミツ", "アポロステーション"
            ],
            "Subscriptions & Services": [
                "OPENAI", "CHATGPT", "icloud", "apple music", "amazon prime", "google one", 
                "netflix", "spotify", "hulu", "disney+", "アマゾンプライム", "グーグルワン",
                "アップルミュージック", "アイクラウド", "subscription", "membership", "月額", "monthly", "年額", "annual",
                "楽天モバイル", "rakuten mobile", "通信料", "communication", "電話", "phone",
                "楽天でんき", "rakuten electricity", "楽天ガス", "rakuten gas", "電気", "electric", "ガス", "gas"
            ],
            "Entertainment & Recreation": [
                "イベントゴリョウブン", "event", "イベント", "コンサート", "concert", "映画", "movie",
                "ゲーム", "game", "スポーツ", "sports", "カラオケ", "karaoke", "ボーリング", "bowling",
                "ゾゾマリンスタジアム", "ハナマルウドン", "ミニストップ", "イベントゴリョウブン"
            ],
            "Health & Wellness": [
                "病院", "hospital", "クリニック", "clinic", "歯科", "dental", "眼科", "eye", "薬局", "pharmacy",
                "薬", "medicine", "保険", "insurance", "診察", "examination", "治療", "treatment",
                "フィットネス", "fitness", "ジム", "gym", "ヨガ", "yoga", "マッサージ", "massage"
            ],
            "Fees & Charges": [
                "手数料", "fee", "年会費", "annual fee", "利用料", "usage fee", "ＡＴＭ", "ATM",
                "支払手数料", "payment fee", "現地利用額", "local usage", "変換レート", "exchange rate"
            ],
            "Cash & ATM": [
                "ローソン銀行", "lawson bank", "ＣＤ", "CD", "現金", "cash", "引き出し", "withdrawal"
            ]
        }
    
    def parse_csv_file(self, file_path):
        """Parse CSV bank statement file."""
        print(f"📄 Parsing CSV file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8-sig') as file:  # Use utf-8-sig to handle BOM
            reader = csv.DictReader(file)
            
            # Debug: print column names
            print(f"Column names: {reader.fieldnames}")
            
            for row_num, row in enumerate(reader, 1):
                # Handle the BOM issue with the first column
                date_key = None
                for key in row.keys():
                    if '利用日' in key:
                        date_key = key
                        break
                
                if not date_key or not row.get('利用店名・商品名') or not row.get('利用金額'):
                    continue
                
                try:
                    # Parse date - remove quotes and strip
                    date_str = row[date_key].strip().strip('"')
                    if re.match(r'\d{4}/\d{2}/\d{2}', date_str):
                        date = datetime.strptime(date_str, '%Y/%m/%d')
                    else:
                        continue
                    
                    # Parse amount - remove quotes and clean
                    amount_str = row['利用金額'].strip().strip('"')
                    amount = float(re.sub(r'[^\d.-]', '', amount_str))
                    
                    # Skip zero amounts
                    if amount == 0:
                        continue
                    
                    # Create transaction record
                    transaction = {
                        'date': date,
                        'description': row['利用店名・商品名'].strip().strip('"'),
                        'amount': amount,
                        'user': row.get('利用者', '').strip().strip('"'),
                        'payment_method': row.get('支払方法', '').strip().strip('"'),
                        'fee': float(re.sub(r'[^\d.-]', '', row.get('支払手数料', '0').strip().strip('"'))),
                        'total_amount': float(re.sub(r'[^\d.-]', '', row.get('支払総額', '0').strip().strip('"'))),
                        'source': 'CSV'
                    }
                    
                    self.csv_data.append(transaction)
                    
                except (ValueError, KeyError) as e:
                    print(f"⚠️ Skipping invalid row {row_num}: {e}")
                    continue
        
        print(f"✅ Parsed {len(self.csv_data)} transactions from CSV")
        return self.csv_data
    
    def categorize_transaction(self, description):
        """Categorize a transaction based on description."""
        description_lower = description.lower()
        
        for category, keywords in self.categorization_rules.items():
            for keyword in keywords:
                if keyword.lower() in description_lower:
                    return category
        
        return "Uncategorised"
    
    def analyze_data(self, data):
        """Analyze transaction data and return summary statistics."""
        if not data:
            return {}
        
        # Basic statistics
        total_amount = sum(t['amount'] for t in data)
        transaction_count = len(data)
        
        # Date range
        dates = [t['date'] for t in data]
        min_date = min(dates)
        max_date = max(dates)
        
        # Categorization
        categories = defaultdict(list)
        for transaction in data:
            category = self.categorize_transaction(transaction['description'])
            categories[category].append(transaction)
        
        # Category summaries
        category_summary = {}
        for category, transactions in categories.items():
            category_summary[category] = {
                'count': len(transactions),
                'total_amount': sum(t['amount'] for t in transactions),
                'avg_amount': sum(t['amount'] for t in transactions) / len(transactions) if transactions else 0
            }
        
        return {
            'transaction_count': transaction_count,
            'total_amount': total_amount,
            'date_range': f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}",
            'categories': category_summary,
            'duplicates': self.find_duplicates(data)
        }
    
    def find_duplicates(self, data):
        """Find duplicate transactions."""
        seen = set()
        duplicates = []
        
        for transaction in data:
            key = (transaction['date'], transaction['description'], transaction['amount'])
            if key in seen:
                duplicates.append(transaction)
            else:
                seen.add(key)
        
        return duplicates
    
    def cross_check_data(self, csv_data, pdf_data):
        """Cross-check CSV and PDF data for discrepancies."""
        print("🔍 Cross-checking CSV and PDF data...")
        
        # Create comparison keys
        csv_keys = set()
        for t in csv_data:
            key = f"{t['date'].strftime('%Y-%m-%d')}|{t['description']}|{t['amount']}"
            csv_keys.add(key)
        
        pdf_keys = set()
        for t in pdf_data:
            key = f"{t['date'].strftime('%Y-%m-%d')}|{t['description']}|{t['amount']}"
            pdf_keys.add(key)
        
        # Find differences
        only_in_csv = csv_keys - pdf_keys
        only_in_pdf = pdf_keys - csv_keys
        common = csv_keys & pdf_keys
        
        # Calculate totals
        csv_total = sum(t['amount'] for t in csv_data)
        pdf_total = sum(t['amount'] for t in pdf_data)
        difference = abs(csv_total - pdf_total)
        
        return {
            'csv_total': csv_total,
            'pdf_total': pdf_total,
            'difference': difference,
            'only_in_csv': len(only_in_csv),
            'only_in_pdf': len(only_in_pdf),
            'common': len(common),
            'reconciliation_status': 'reconciled' if difference < 1000 else 'discrepancy'
        }
    
    def generate_report(self, csv_analysis, pdf_analysis, cross_check):
        """Generate comprehensive analysis report."""
        report = []
        report.append("# Bank Statement Analysis Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- CSV Transactions: {csv_analysis['transaction_count']} | Total: ¥{csv_analysis['total_amount']:,.0f}")
        if pdf_analysis:
            report.append(f"- PDF Transactions: {pdf_analysis['transaction_count']} | Total: ¥{pdf_analysis['total_amount']:,.0f}")
        report.append(f"- Reconciliation Status: {cross_check['reconciliation_status']}")
        report.append(f"- Total Difference: ¥{cross_check['difference']:,.0f}")
        report.append("")
        
        # CSV Analysis
        report.append("## CSV Statement Analysis")
        report.append(f"- **Total Transactions:** {csv_analysis['transaction_count']}")
        report.append(f"- **Total Amount:** ¥{csv_analysis['total_amount']:,.0f}")
        report.append(f"- **Date Range:** {csv_analysis['date_range']}")
        report.append(f"- **Duplicates Found:** {len(csv_analysis['duplicates'])}")
        report.append("")
        
        # Category breakdown
        report.append("### Category Breakdown")
        for category, stats in csv_analysis['categories'].items():
            report.append(f"- **{category}**: {stats['count']} transactions, ¥{stats['total_amount']:,.0f} (avg: ¥{stats['avg_amount']:,.0f})")
        report.append("")
        
        # Cross-check results
        if cross_check:
            report.append("## Cross-Check Results")
            report.append(f"- **CSV Total:** ¥{cross_check['csv_total']:,.0f}")
            if pdf_analysis:
                report.append(f"- **PDF Total:** ¥{cross_check['pdf_total']:,.0f}")
            report.append(f"- **Difference:** ¥{cross_check['difference']:,.0f}")
            report.append(f"- **Only in CSV:** {cross_check['only_in_csv']} transactions")
            if pdf_analysis:
                report.append(f"- **Only in PDF:** {cross_check['only_in_pdf']} transactions")
            report.append(f"- **Common:** {cross_check['common']} transactions")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        if cross_check['reconciliation_status'] == 'discrepancy':
            report.append("- ⚠️ **Discrepancy detected** - Review both statements carefully")
            report.append("- Check for missing transactions in either source")
            report.append("- Verify transaction amounts and dates")
        else:
            report.append("- ✅ **Statements are reconciled** - Data integrity is good")
        
        if csv_analysis['duplicates']:
            report.append(f"- ⚠️ **{len(csv_analysis['duplicates'])} duplicate transactions found** - Review and remove if necessary")
        
        report.append("- Review uncategorized transactions and add rules if needed")
        report.append("- Consider setting up automated reconciliation for future statements")
        
        return "\n".join(report)
    
    def export_categorized_data(self, data, filename):
        """Export categorized data to CSV."""
        print(f"📥 Exporting categorized data to {filename}")
        
        # Add categories to data
        for transaction in data:
            transaction['category'] = self.categorize_transaction(transaction['description'])
        
        # Write to CSV
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            if data:
                fieldnames = data[0].keys()
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
        
        print(f"✅ Exported {len(data)} transactions to {filename}")


def main():
    """Main function to run the bank statement analyzer."""
    analyzer = SimpleBankAnalyzer()
    
    # File paths
    csv_path = "/workspace/temp_repo/bank/statements/enavi202509(5734) (2).csv"
    
    print("🏦 Bank Statement Analyzer")
    print("=" * 50)
    
    try:
        # Parse CSV data
        csv_data = analyzer.parse_csv_file(csv_path)
        
        if not csv_data:
            print("❌ No data found in CSV file")
            return
        
        # Analyze CSV data
        print("\n📊 Analyzing CSV data...")
        csv_analysis = analyzer.analyze_data(csv_data)
        
        # Cross-check (simplified - no PDF parsing for now)
        print("\n🔍 Running analysis...")
        cross_check = {
            'csv_total': csv_analysis['total_amount'],
            'pdf_total': 0,  # No PDF data for now
            'difference': csv_analysis['total_amount'],
            'only_in_csv': csv_analysis['transaction_count'],
            'only_in_pdf': 0,
            'common': 0,
            'reconciliation_status': 'csv_only'
        }
        
        # Generate report
        print("\n📝 Generating report...")
        report = analyzer.generate_report(csv_analysis, None, cross_check)
        
        # Save report
        with open("/workspace/bank_analysis_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        print("✅ Analysis complete!")
        print(f"📄 Report saved to: /workspace/bank_analysis_report.md")
        
        # Export categorized data
        analyzer.export_categorized_data(csv_data, "/workspace/categorized_transactions.csv")
        print(f"📊 Categorized data saved to: /workspace/categorized_transactions.csv")
        
        # Print summary
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"CSV Transactions: {csv_analysis['transaction_count']} | Total: ¥{csv_analysis['total_amount']:,.0f}")
        print(f"Date Range: {csv_analysis['date_range']}")
        print(f"Duplicates: {len(csv_analysis['duplicates'])}")
        print(f"Categories: {len(csv_analysis['categories'])}")
        
        # Show top categories
        print("\nTop Categories by Amount:")
        sorted_categories = sorted(csv_analysis['categories'].items(), 
                                 key=lambda x: x[1]['total_amount'], reverse=True)
        for category, stats in sorted_categories[:5]:
            print(f"  {category}: ¥{stats['total_amount']:,.0f} ({stats['count']} transactions)")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()