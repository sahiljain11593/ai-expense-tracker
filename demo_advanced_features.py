"""
demo_advanced_features.py

Demonstration script showcasing the new advanced AI and dashboard features.
This script provides examples of how to use all the new functionality.

Run with: python demo_advanced_features.py
"""

import sys
from datetime import datetime, timedelta
from typing import List, Dict
import random

# Import new modules
from ml_engine import EnsembleCategorizationEngine, LocalMLTrainer
from insights_engine import InsightsEngine, SpendingAnalytics
from dashboard import ModernDashboard


def generate_sample_transactions(num_transactions: int = 100) -> List[Dict]:
    """Generate sample transactions for demonstration."""
    categories = {
        'Food': ['STARBUCKS', 'MCDONALDS', 'SEVEN ELEVEN', 'LAWSON', 'SUSHI RESTAURANT'],
        'Transportation': ['UBER', 'JR RAILWAY', 'METRO', 'TAXI SERVICE'],
        'Shopping': ['AMAZON', 'UNIQLO', 'NIKE STORE', 'ZARA'],
        'Entertainment': ['NETFLIX', 'SPOTIFY', 'CINEMA', 'GYM MEMBERSHIP'],
        'Bills': ['ELECTRICITY BILL', 'INTERNET SERVICE', 'PHONE BILL']
    }
    
    transactions = []
    start_date = datetime.now() - timedelta(days=90)
    
    for i in range(num_transactions):
        # Pick random category and merchant
        category = random.choice(list(categories.keys()))
        merchant = random.choice(categories[category])
        
        # Generate transaction
        date = start_date + timedelta(days=random.randint(0, 90))
        amount = -random.randint(100, 10000)
        
        transactions.append({
            'id': i + 1,
            'date': date.strftime('%Y-%m-%d'),
            'description': f"VISA DOMESTIC USE VS {merchant}",
            'amount': amount,
            'category': category,
            'subcategory': None,
            'transaction_type': 'Expense'
        })
    
    return transactions


def demo_ensemble_categorization():
    """Demonstrate ensemble categorization engine."""
    print("\n" + "="*80)
    print("🤖 DEMO: Ensemble Categorization Engine")
    print("="*80)
    
    # Initialize engine
    engine = EnsembleCategorizationEngine()
    
    # Generate sample data
    historical_transactions = generate_sample_transactions(50)
    
    # Test transaction
    test_transaction = {
        'description': 'VISA DOMESTIC USE VS STARBUCKS COFFEE',
        'amount': -450,
        'date': '2025-10-12'
    }
    
    print(f"\n📝 Test Transaction:")
    print(f"   Description: {test_transaction['description']}")
    print(f"   Amount: ¥{abs(test_transaction['amount']):,}")
    print(f"   Date: {test_transaction['date']}")
    
    # Get prediction
    category, subcategory, confidence, explanation = engine.predict(
        transaction=test_transaction,
        historical_data=historical_transactions
    )
    
    print(f"\n🎯 Prediction Results:")
    print(f"   Category: {category}")
    print(f"   Subcategory: {subcategory or 'N/A'}")
    print(f"   Confidence: {confidence:.1%}")
    
    print(f"\n📊 Model Agreement:")
    print(f"   Agreement Level: {explanation['agreement_level']:.1%}")
    
    print(f"\n🔍 Individual Model Predictions:")
    for model_name, pred in explanation['model_predictions'].items():
        agreed = "✓" if pred['agreed'] else "✗"
        print(f"   {agreed} {model_name.replace('_', ' ').title()}: "
              f"{pred['category']} ({pred['confidence']:.1%})")
    
    print(f"\n💡 Reasoning:")
    for reason in explanation['reasoning']:
        print(f"   • {reason}")
    
    # Simulate learning from correction
    print(f"\n📚 Learning from User Correction:")
    actual_category = "Food"
    engine.learn_from_correction(
        transaction=test_transaction,
        predicted_category=category,
        actual_category=actual_category,
        model_predictions=explanation['model_predictions']
    )
    print(f"   ✅ Updated models based on correction to '{actual_category}'")
    
    # Show updated statistics
    stats = engine.get_model_stats()
    print(f"\n📈 Model Performance:")
    for model_name, metrics in stats.items():
        print(f"   {model_name.replace('_', ' ').title()}:")
        print(f"      Accuracy: {metrics.get('accuracy', 0):.1%}")
        print(f"      Weight: {metrics.get('current_weight', 0):.1%}")


def demo_insights_engine():
    """Demonstrate comprehensive insights engine."""
    print("\n" + "="*80)
    print("💡 DEMO: Comprehensive Insights Engine")
    print("="*80)
    
    # Initialize engine
    engine = InsightsEngine()
    
    # Generate sample data
    transactions = generate_sample_transactions(100)
    
    print(f"\n📊 Analyzing {len(transactions)} transactions...")
    
    # Generate report
    report = engine.generate_comprehensive_report(transactions)
    
    # Display summary
    print(f"\n📝 Summary:")
    summary = report['summary']
    print(f"   Total Transactions: {summary['total_transactions']}")
    print(f"   Total Expenses: ¥{summary['total_expenses']:,.0f}")
    print(f"   Total Income: ¥{summary['total_income']:,.0f}")
    print(f"   Net Cashflow: ¥{summary['net_cashflow']:,.0f}")
    print(f"   Average Transaction: ¥{abs(summary['average_transaction']):,.0f}")
    print(f"   Number of Categories: {summary['number_of_categories']}")
    
    # Display insights
    print(f"\n💡 Key Insights:")
    for i, insight in enumerate(report['insights'], 1):
        print(f"   {i}. {insight}")
    
    # Display recommendations
    print(f"\n🎯 Top Recommendations:")
    for i, rec in enumerate(report['recommendations'][:5], 1):
        priority = rec.get('priority', 'low').upper()
        print(f"\n   {i}. [{priority}] {rec.get('message', 'N/A')}")
        print(f"      💡 {rec.get('suggestion', 'N/A')}")
        if 'potential_savings' in rec:
            print(f"      💰 Potential Savings: ¥{rec['potential_savings']:,.0f}")
    
    # Display forecasts
    if 'forecasts' in report and report['forecasts'].get('forecasts'):
        print(f"\n🔮 Spending Forecasts:")
        for month, amount in report['forecasts']['forecasts'].items():
            print(f"   {month}: ¥{amount:,.0f}")
        print(f"   Confidence: {report['forecasts'].get('confidence', 'unknown').upper()}")
    
    # Display anomalies
    if report['anomalies']:
        print(f"\n⚠️  Anomalies Detected:")
        for i, anomaly in enumerate(report['anomalies'][:3], 1):
            print(f"\n   {i}. Type: {anomaly['type']}")
            print(f"      Severity: {anomaly['severity'].upper()}")
            print(f"      Reason: {anomaly['reason']}")
    
    # Display category breakdown
    if report['category_breakdown']:
        print(f"\n📊 Category Breakdown:")
        for category, stats in list(report['category_breakdown'].items())[:5]:
            print(f"\n   {category}:")
            print(f"      Total: ¥{stats['total_spent']:,.0f}")
            print(f"      Average: ¥{stats['average_transaction']:,.0f}")
            print(f"      Count: {stats['transaction_count']}")
            print(f"      % of Total: {stats['percentage_of_total']:.1f}%")


def demo_local_ml_trainer():
    """Demonstrate local ML training capabilities."""
    print("\n" + "="*80)
    print("🧠 DEMO: Local ML Training")
    print("="*80)
    
    # Initialize trainer
    trainer = LocalMLTrainer()
    
    print(f"\n📚 Adding Training Examples:")
    
    # Add training examples
    examples = [
        ({'description': 'STARBUCKS', 'amount': -450}, 'Food'),
        ({'description': 'UBER RIDE', 'amount': -1200}, 'Transportation'),
        ({'description': 'AMAZON PURCHASE', 'amount': -3500}, 'Shopping'),
        ({'description': 'GYM MEMBERSHIP', 'amount': -8000}, 'Entertainment'),
        ({'description': 'ELECTRICITY BILL', 'amount': -5000}, 'Bills'),
    ]
    
    for transaction, category in examples:
        trainer.add_training_example(transaction, category)
        print(f"   ✓ Added: {transaction['description']} → {category}")
    
    # Get training stats
    stats = trainer.get_training_stats()
    
    print(f"\n📊 Training Statistics:")
    print(f"   Total Examples: {stats['total_examples']}")
    print(f"   Ready to Train: {'✅ Yes' if stats['ready_to_train'] else '❌ No (need 10+ examples)'}")
    
    if stats['categories']:
        print(f"\n   Categories:")
        for category, count in stats['categories'].items():
            print(f"      {category}: {count} examples")
    
    if stats.get('most_common_category'):
        print(f"\n   Most Common: {stats['most_common_category'][0]} ({stats['most_common_category'][1]} examples)")
    
    print(f"\n🔒 Privacy Note:")
    print(f"   ✓ All training happens locally on your device")
    print(f"   ✓ No data is sent to external services")
    print(f"   ✓ Models are stored locally")


def demo_spending_analytics():
    """Demonstrate spending analytics."""
    print("\n" + "="*80)
    print("📈 DEMO: Spending Analytics")
    print("="*80)
    
    # Generate sample data
    transactions = generate_sample_transactions(100)
    import pandas as pd
    df = pd.DataFrame(transactions)
    
    # Prepare chart data
    print(f"\n📊 Preparing Chart Data:")
    
    # Category chart
    category_data = SpendingAnalytics.prepare_category_chart_data(df)
    if category_data:
        print(f"\n   📊 Category Chart Data:")
        print(f"      Type: {category_data['type']}")
        print(f"      Categories: {len(category_data['labels'])}")
        for label, value in zip(category_data['labels'][:5], category_data['values'][:5]):
            print(f"         {label}: ¥{value:,.0f}")
    
    # Monthly trend
    monthly_data = SpendingAnalytics.prepare_monthly_trend_data(df)
    if monthly_data:
        print(f"\n   📈 Monthly Trend Data:")
        print(f"      Type: {monthly_data['type']}")
        print(f"      Months: {len(monthly_data['labels'])}")
        print(f"      Latest Month:")
        if monthly_data['labels']:
            print(f"         Expenses: ¥{monthly_data['expenses'][-1]:,.0f}")
            print(f"         Income: ¥{monthly_data['income'][-1]:,.0f}")
    
    # Key metrics
    metrics = SpendingAnalytics.calculate_key_metrics(df)
    if metrics:
        print(f"\n   📊 Key Metrics:")
        print(f"      Total Expenses: ¥{metrics['total_expenses']:,.0f}")
        print(f"      Total Income: ¥{metrics['total_income']:,.0f}")
        print(f"      Net Savings: ¥{metrics['net_savings']:,.0f}")
        print(f"      Savings Rate: {metrics['savings_rate']:.1f}%")
        print(f"      Avg Daily Spending: ¥{metrics['average_daily_spending']:,.0f}")


def demo_all_features():
    """Run all feature demonstrations."""
    print("\n" + "="*80)
    print("🎉 ADVANCED FEATURES DEMONSTRATION")
    print("="*80)
    print("\nThis demo showcases all the new advanced AI and analytics features.")
    print("Each section demonstrates a different capability of the system.")
    
    # Run all demos
    demo_ensemble_categorization()
    demo_insights_engine()
    demo_local_ml_trainer()
    demo_spending_analytics()
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE")
    print("="*80)
    print("\nAll features demonstrated successfully!")
    print("\nNext Steps:")
    print("1. Review FEATURE_INTEGRATION_GUIDE.md for integration instructions")
    print("2. Update transaction_web_app.py to use these new features")
    print("3. Test with real transaction data")
    print("4. Customize visualizations and thresholds as needed")
    print("\n")


if __name__ == "__main__":
    try:
        demo_all_features()
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

