#!/usr/bin/env python3
"""
Bank Statement Analyzer and Cross-Checker

This tool provides comprehensive analysis and cross-checking of bank statements
from CSV and PDF sources, with intelligent categorization and reconciliation.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BankStatementAnalyzer:
    """Comprehensive bank statement analysis and cross-checking system."""
    
    def __init__(self):
        self.csv_data = None
        self.pdf_data = None
        self.reconciliation_results = {}
        self.categorization_rules = self._load_categorization_rules()
        
    def _load_categorization_rules(self) -> Dict[str, List[str]]:
        """Load comprehensive categorization rules for Japanese bank statements."""
        return {
            "Food & Groceries": [
                # Convenience stores
                "ローソン", "セブンイレブン", "ファミリーマート", "コンビニ", "lawson", "seven eleven", "family mart",
                "ポプラグループ", "poplar", "スーパー", "supermarket", "grocery", "market", "food", "fresh",
                "イオン", "aeon", "イトーヨーカドー", "itoyokado", "西友", "seiyu", "ライフ", "life",
                # Restaurants and dining
                "レストラン", "restaurant", "cafe", "dinner", "lunch", "breakfast", "takeaway", "delivery",
                "居酒屋", "izakaya", "バー", "bar", "カフェ", "coffee", "ピザ", "pizza", "寿司", "sushi",
                "マクドナルド", "mcdonalds", "ケンタッキー", "kfc", "スターバックス", "starbucks",
                # Specific merchants
                "ドンキホーテ", "バーガーキング", "イージーズカフェ", "サカドヤ", "エヌシーデイ",
                "ハナマルキ", "ハナコウジョウ", "サウナ東京", "マルエツ", "マツノヤヨウガ",
                "ガンソズシ", "マルガメセイメン", "カブシキガイシャ", "サントラック", "バブルハウス"
            ],
            "Transportation": [
                # Public transport
                "電車", "train", "バス", "bus", "タクシー", "taxi", "地下鉄", "subway", "モノレール", "monorail",
                "モバイルパス", "mobile pass", "交通費", "transport", "駐車場", "parking", "高速道路", "highway",
                "ＥＴＣ", "etc", "ガソリン", "gasoline", "燃料", "fuel", "車", "car", "バイク", "bike",
                # Specific merchants
                "ＥＮＥＯＳ", "ENEOS", "ガソリンスタンド", "gas station"
            ],
            "Shopping & Retail": [
                # Online shopping
                "AMAZON", "amazon", "アマゾン", "楽天", "rakuten", "ヤフー", "yahoo",
                # Department stores and retail
                "ニトリ", "nitori", "イケア", "ikea", "ホームセンター", "home center", "家具", "furniture",
                "ユニクロ", "uniqlo", "zara", "h&m", "gap", "nike", "adidas", "アディダス", "ナイキ",
                "ファッション", "fashion", "スタイル", "style", "ブランド", "brand",
                # Specific merchants
                "チケットジャム", "ロコハロイ", "ノクテイプラザ", "マルイファ", "東京ミッドタウン",
                "新マルノウチビル", "タイムズカー", "イデミツ", "アポロステーション"
            ],
            "Subscriptions & Services": [
                # Digital services
                "OPENAI", "CHATGPT", "icloud", "apple music", "amazon prime", "google one", 
                "netflix", "spotify", "hulu", "disney+", "アマゾンプライム", "グーグルワン",
                "アップルミュージック", "アイクラウド", "subscription", "membership", "月額", "monthly", "年額", "annual",
                # Communication services
                "楽天モバイル", "rakuten mobile", "通信料", "communication", "電話", "phone",
                # Utilities
                "楽天でんき", "rakuten electricity", "楽天ガス", "rakuten gas", "電気", "electric", "ガス", "gas"
            ],
            "Entertainment & Recreation": [
                # Entertainment venues
                "イベントゴリョウブン", "event", "イベント", "コンサート", "concert", "映画", "movie",
                "ゲーム", "game", "スポーツ", "sports", "カラオケ", "karaoke", "ボーリング", "bowling",
                # Specific venues
                "ゾゾマリンスタジアム", "ハナマルウドン", "ミニストップ", "イベントゴリョウブン"
            ],
            "Health & Wellness": [
                # Healthcare
                "病院", "hospital", "クリニック", "clinic", "歯科", "dental", "眼科", "eye", "薬局", "pharmacy",
                "薬", "medicine", "保険", "insurance", "診察", "examination", "治療", "treatment",
                "フィットネス", "fitness", "ジム", "gym", "ヨガ", "yoga", "マッサージ", "massage"
            ],
            "Fees & Charges": [
                # Bank fees
                "手数料", "fee", "年会費", "annual fee", "利用料", "usage fee", "ＡＴＭ", "ATM",
                "支払手数料", "payment fee", "現地利用額", "local usage", "変換レート", "exchange rate"
            ],
            "Cash & ATM": [
                # Cash withdrawals
                "ローソン銀行", "lawson bank", "ＣＤ", "CD", "現金", "cash", "引き出し", "withdrawal"
            ]
        }
    
    def parse_csv_statement(self, csv_path: str) -> pd.DataFrame:
        """Parse Japanese bank statement CSV file."""
        try:
            # Read CSV with proper encoding
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Map Japanese columns to English
            column_mapping = {
                '利用日': 'date',
                '利用店名・商品名': 'description',
                '利用者': 'user',
                '支払方法': 'payment_method',
                '利用金額': 'amount',
                '支払手数料': 'fee',
                '支払総額': 'total_amount',
                '9月支払金額': 'september_payment',
                '10月繰越残高': 'october_balance',
                '新規サイン': 'new_sign'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Clean and convert data
            df = self._clean_transaction_data(df)
            
            logger.info(f"Successfully parsed CSV: {len(df)} transactions")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            raise
    
    def parse_pdf_statement(self, pdf_path: str) -> pd.DataFrame:
        """Parse bank statement PDF file."""
        try:
            import pdfplumber
            
            transactions = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:10]:  # Process first 10 pages
                    tables = page.extract_tables()
                    for table in tables:
                        if len(table) > 1 and len(table[0]) >= 3:
                            # Try to identify transaction rows
                            for row in table[1:]:
                                if len(row) >= 3 and row[0] and row[1] and row[2]:
                                    try:
                                        # Parse date
                                        date_str = str(row[0]).strip()
                                        if re.match(r'\d{4}/\d{2}/\d{2}', date_str):
                                            date = datetime.strptime(date_str, '%Y/%m/%d')
                                            
                                            # Parse description
                                            description = str(row[1]).strip()
                                            
                                            # Parse amount
                                            amount_str = str(row[2]).strip()
                                            amount = float(re.sub(r'[^\d.-]', '', amount_str))
                                            
                                            transactions.append({
                                                'date': date,
                                                'description': description,
                                                'amount': amount,
                                                'source': 'PDF'
                                            })
                                    except Exception as e:
                                        continue
            
            df = pd.DataFrame(transactions)
            if not df.empty:
                df = self._clean_transaction_data(df)
            
            logger.info(f"Successfully parsed PDF: {len(df)} transactions")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            raise
    
    def _clean_transaction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize transaction data."""
        # Remove empty rows
        df = df.dropna(subset=['date', 'description', 'amount'])
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Clean amounts
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove rows with invalid data
        df = df.dropna(subset=['date', 'amount'])
        
        # Add transaction type
        df['transaction_type'] = 'Expense'  # Most Japanese bank transactions are expenses
        
        # Add unique ID
        df['transaction_id'] = range(len(df))
        
        return df
    
    def cross_check_statements(self, csv_data: pd.DataFrame, pdf_data: pd.DataFrame) -> Dict:
        """Cross-check CSV and PDF statements for accuracy."""
        results = {
            'csv_summary': self._get_data_summary(csv_data, 'CSV'),
            'pdf_summary': self._get_data_summary(pdf_data, 'PDF'),
            'discrepancies': [],
            'reconciliation_status': 'unknown'
        }
        
        # Find transactions only in CSV
        csv_only = self._find_unique_transactions(csv_data, pdf_data, 'CSV')
        results['csv_only'] = csv_only
        
        # Find transactions only in PDF
        pdf_only = self._find_unique_transactions(pdf_data, csv_data, 'PDF')
        results['pdf_only'] = pdf_only
        
        # Calculate reconciliation status
        csv_total = csv_data['amount'].sum()
        pdf_total = pdf_data['amount'].sum()
        difference = abs(csv_total - pdf_total)
        
        if difference < 1000:  # Within ¥1,000
            results['reconciliation_status'] = 'reconciled'
        elif difference < 10000:  # Within ¥10,000
            results['reconciliation_status'] = 'minor_discrepancy'
        else:
            results['reconciliation_status'] = 'major_discrepancy'
        
        results['total_difference'] = difference
        results['csv_total'] = csv_total
        results['pdf_total'] = pdf_total
        
        return results
    
    def _get_data_summary(self, df: pd.DataFrame, source: str) -> Dict:
        """Get summary statistics for a dataset."""
        if df.empty:
            return {'source': source, 'count': 0, 'total': 0, 'date_range': 'N/A'}
        
        return {
            'source': source,
            'count': len(df),
            'total': df['amount'].sum(),
            'abs_total': df['amount'].abs().sum(),
            'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
            'duplicates': len(df) - len(df.drop_duplicates(subset=['date', 'description', 'amount']))
        }
    
    def _find_unique_transactions(self, df1: pd.DataFrame, df2: pd.DataFrame, source: str) -> List[Dict]:
        """Find transactions unique to one dataset."""
        if df1.empty or df2.empty:
            return df1.to_dict('records') if not df1.empty else []
        
        # Create comparison keys
        df1['key'] = df1['date'].astype(str) + '|' + df1['description'] + '|' + df1['amount'].astype(str)
        df2['key'] = df2['date'].astype(str) + '|' + df2['description'] + '|' + df2['amount'].astype(str)
        
        # Find unique transactions
        unique_mask = ~df1['key'].isin(df2['key'])
        unique_df = df1[unique_mask].copy()
        
        # Remove the key column
        unique_df = unique_df.drop('key', axis=1)
        
        return unique_df.to_dict('records')
    
    def categorize_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize transactions using intelligent rules."""
        df = df.copy()
        categories = []
        confidences = []
        
        for _, row in df.iterrows():
            description = str(row['description']).lower()
            category, confidence = self._categorize_transaction(description)
            categories.append(category)
            confidences.append(confidence)
        
        df['category'] = categories
        df['confidence'] = confidences
        
        return df
    
    def _categorize_transaction(self, description: str) -> Tuple[str, float]:
        """Categorize a single transaction with confidence score."""
        description = description.lower()
        
        # Check each category
        for category, keywords in self.categorization_rules.items():
            matches = sum(1 for keyword in keywords if keyword.lower() in description)
            if matches > 0:
                confidence = min(0.9, 0.5 + (matches * 0.1))
                return category, confidence
        
        return "Uncategorised", 0.0
    
    def generate_analysis_report(self, csv_data: pd.DataFrame, pdf_data: pd.DataFrame) -> str:
        """Generate comprehensive analysis report."""
        report = []
        report.append("# Bank Statement Analysis Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Cross-check results
        cross_check = self.cross_check_statements(csv_data, pdf_data)
        
        report.append("## Summary")
        report.append(f"- CSV Transactions: {cross_check['csv_summary']['count']} | Total: ¥{cross_check['csv_summary']['total']:,.0f}")
        report.append(f"- PDF Transactions: {cross_check['pdf_summary']['count']} | Total: ¥{cross_check['pdf_summary']['total']:,.0f}")
        report.append(f"- Reconciliation Status: {cross_check['reconciliation_status']}")
        report.append(f"- Total Difference: ¥{cross_check['total_difference']:,.0f}")
        report.append("")
        
        # Categorization analysis
        if not csv_data.empty:
            categorized_csv = self.categorize_transactions(csv_data)
            report.append("## Categorization Analysis (CSV)")
            
            category_summary = categorized_csv.groupby('category').agg({
                'amount': ['count', 'sum'],
                'confidence': 'mean'
            }).round(2)
            
            for category in category_summary.index:
                count = category_summary.loc[category, ('amount', 'count')]
                total = category_summary.loc[category, ('amount', 'sum')]
                avg_confidence = category_summary.loc[category, ('confidence', 'mean')]
                report.append(f"- **{category}**: {count} transactions, ¥{total:,.0f} (avg confidence: {avg_confidence:.1%})")
            
            report.append("")
        
        # Discrepancies
        if cross_check['csv_only'] or cross_check['pdf_only']:
            report.append("## Discrepancies Found")
            
            if cross_check['csv_only']:
                report.append(f"### Transactions only in CSV ({len(cross_check['csv_only'])}):")
                for i, tx in enumerate(cross_check['csv_only'][:10]):  # Show first 10
                    report.append(f"- {tx['date'].strftime('%Y-%m-%d')} | {tx['description'][:30]}... | ¥{tx['amount']:,.0f}")
                if len(cross_check['csv_only']) > 10:
                    report.append(f"... and {len(cross_check['csv_only']) - 10} more")
                report.append("")
            
            if cross_check['pdf_only']:
                report.append(f"### Transactions only in PDF ({len(cross_check['pdf_only'])}):")
                for i, tx in enumerate(cross_check['pdf_only'][:10]):  # Show first 10
                    report.append(f"- {tx['date'].strftime('%Y-%m-%d')} | {tx['description'][:30]}... | ¥{tx['amount']:,.0f}")
                if len(cross_check['pdf_only']) > 10:
                    report.append(f"... and {len(cross_check['pdf_only']) - 10} more")
                report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        if cross_check['reconciliation_status'] == 'major_discrepancy':
            report.append("- ⚠️ **Major discrepancy detected** - Review both statements carefully")
            report.append("- Check for missing transactions in either source")
            report.append("- Verify transaction amounts and dates")
        elif cross_check['reconciliation_status'] == 'minor_discrepancy':
            report.append("- ⚠️ **Minor discrepancy detected** - Likely due to timing differences")
            report.append("- Check for transactions near statement cut-off dates")
        else:
            report.append("- ✅ **Statements are reconciled** - Data integrity is good")
        
        report.append("- Review uncategorized transactions and add rules if needed")
        report.append("- Consider setting up automated reconciliation for future statements")
        
        return "\n".join(report)
    
    def export_categorized_data(self, df: pd.DataFrame, output_path: str) -> None:
        """Export categorized data to CSV."""
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Exported categorized data to {output_path}")


def main():
    """Main function to run the bank statement analyzer."""
    analyzer = BankStatementAnalyzer()
    
    # File paths
    csv_path = "/workspace/temp_repo/bank/statements/enavi202509(5734) (2).csv"
    pdf_path = "/workspace/temp_repo/bank/statements/statement_202509.pdf"
    
    try:
        # Parse statements
        print("📊 Parsing bank statements...")
        csv_data = analyzer.parse_csv_statement(csv_path)
        pdf_data = analyzer.parse_pdf_statement(pdf_path)
        
        # Cross-check
        print("🔍 Cross-checking statements...")
        cross_check = analyzer.cross_check_statements(csv_data, pdf_data)
        
        # Generate report
        print("📝 Generating analysis report...")
        report = analyzer.generate_analysis_report(csv_data, pdf_data)
        
        # Save report
        with open("/workspace/bank_analysis_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        print("✅ Analysis complete!")
        print(f"📄 Report saved to: /workspace/bank_analysis_report.md")
        
        # Export categorized data
        if not csv_data.empty:
            categorized_data = analyzer.categorize_transactions(csv_data)
            analyzer.export_categorized_data(categorized_data, "/workspace/categorized_transactions.csv")
            print(f"📊 Categorized data saved to: /workspace/categorized_transactions.csv")
        
        # Print summary
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"CSV Transactions: {cross_check['csv_summary']['count']} | Total: ¥{cross_check['csv_summary']['total']:,.0f}")
        print(f"PDF Transactions: {cross_check['pdf_summary']['count']} | Total: ¥{cross_check['pdf_summary']['total']:,.0f}")
        print(f"Reconciliation: {cross_check['reconciliation_status']}")
        print(f"Difference: ¥{cross_check['total_difference']:,.0f}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()