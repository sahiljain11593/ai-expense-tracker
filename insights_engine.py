"""
insights_engine.py

Comprehensive Reporting and Insights Engine
Generates AI-powered financial insights, trends, and recommendations.

Features:
- Spending pattern analysis
- Trend detection and forecasting
- Anomaly detection
- Budget recommendations
- Comparative analysis
- Personalized insights
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _amount_column(df: pd.DataFrame) -> Optional[str]:
    if 'amount_jpy' in df.columns:
        return 'amount_jpy'
    if 'amount' in df.columns:
        return 'amount'
    return None


def _with_analysis_amount(df: pd.DataFrame) -> pd.DataFrame:
    amount_col = _amount_column(df)
    result = df.copy()
    if amount_col:
        result['_analysis_amount'] = pd.to_numeric(result[amount_col], errors='coerce').fillna(0)
        result['_spend_amount'] = result['_analysis_amount'].abs()
    return result


def _expense_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    result = _with_analysis_amount(df)
    if '_analysis_amount' not in result.columns:
        return pd.DataFrame()
    if 'transaction_type' in result.columns:
        return result[result['transaction_type'].fillna('').eq('Expense')].copy()
    return result[result['_analysis_amount'] < 0].copy()


def _income_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    result = _with_analysis_amount(df)
    if '_analysis_amount' not in result.columns:
        return pd.DataFrame()
    if 'transaction_type' in result.columns:
        return result[result['transaction_type'].fillna('').eq('Credit')].copy()
    return result[result['_analysis_amount'] > 0].copy()


class InsightsEngine:
    """
    Main insights engine that generates comprehensive financial reports
    and AI-powered recommendations.
    """
    
    def __init__(self):
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.budget_advisor = BudgetAdvisor()
        self.pattern_analyzer = PatternAnalyzer()
    
    def generate_comprehensive_report(
        self, 
        transactions: List[Dict],
        date_range: Optional[Tuple[str, str]] = None
    ) -> Dict:
        """
        Generate comprehensive financial report with all insights.
        
        Returns a dictionary containing:
        - spending_summary
        - category_breakdown
        - trends
        - anomalies
        - recommendations
        - insights
        """
        if not transactions:
            return self._empty_report()
        
        df = pd.DataFrame(transactions)
        
        # Filter by date range if specified
        if date_range:
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= date_range[0]) & (df['date'] <= date_range[1])]
        
        report = {
            'summary': self._generate_summary(df),
            'category_breakdown': self._analyze_categories(df),
            'monthly_trends': self.trend_analyzer.analyze_monthly_trends(df),
            'spending_patterns': self.pattern_analyzer.analyze_patterns(df),
            'anomalies': self.anomaly_detector.detect_anomalies(df),
            'recommendations': self._generate_recommendations(df),
            'insights': self._generate_insights(df),
            'forecasts': self.trend_analyzer.forecast_spending(df),
        }
        
        return report
    
    def _generate_summary(self, df: pd.DataFrame) -> Dict:
        """Generate high-level summary statistics."""
        analysis_df = _with_analysis_amount(df)
        expenses = _expense_rows(df)
        income = _income_rows(df)
        total_expenses = expenses['_spend_amount'].sum() if not expenses.empty else 0
        total_income = income['_spend_amount'].sum() if not income.empty else 0
        
        return {
            'total_transactions': len(df),
            'total_expenses': total_expenses,
            'total_income': total_income,
            'net_cashflow': total_income - total_expenses,
            'average_transaction': analysis_df['_analysis_amount'].mean() if '_analysis_amount' in analysis_df.columns else 0,
            'largest_expense': expenses['_spend_amount'].max() if not expenses.empty else 0,
            'number_of_categories': df['category'].nunique() if 'category' in df.columns else 0,
            'date_range': {
                'start': df['date'].min() if 'date' in df.columns else None,
                'end': df['date'].max() if 'date' in df.columns else None
            }
        }
    
    def _analyze_categories(self, df: pd.DataFrame) -> Dict:
        """Analyze spending by category."""
        if 'category' not in df.columns or _amount_column(df) is None:
            return {}
        
        # Filter expenses only
        expenses = _expense_rows(df)
        
        if expenses.empty:
            return {}
        
        # Group by category
        category_stats = expenses.groupby('category').agg({
            '_spend_amount': ['sum', 'mean', 'count', 'std']
        }).round(2)
        total_spending = expenses['_spend_amount'].sum()
        
        # Convert to dictionary
        result = {}
        for category in category_stats.index:
            result[category] = {
                'total_spent': category_stats.loc[category, ('_spend_amount', 'sum')],
                'average_transaction': category_stats.loc[category, ('_spend_amount', 'mean')],
                'transaction_count': int(category_stats.loc[category, ('_spend_amount', 'count')]),
                'std_deviation': category_stats.loc[category, ('_spend_amount', 'std')],
                'percentage_of_total': (category_stats.loc[category, ('_spend_amount', 'sum')] / total_spending * 100) if total_spending else 0
            }
        
        # Sort by total spent
        result = dict(sorted(result.items(), key=lambda x: x[1]['total_spent'], reverse=True))
        
        return result
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[Dict]:
        """Generate actionable recommendations based on spending patterns."""
        recommendations = []
        
        # Get recommendations from budget advisor
        budget_recs = self.budget_advisor.generate_recommendations(df)
        recommendations.extend(budget_recs)
        
        # Get pattern-based recommendations
        pattern_recs = self.pattern_analyzer.generate_recommendations(df)
        recommendations.extend(pattern_recs)
        
        # Prioritize recommendations by impact
        recommendations.sort(key=lambda x: x.get('potential_savings', 0), reverse=True)
        
        return recommendations[:10]  # Top 10 recommendations
    
    def _generate_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate human-readable insights."""
        insights = []
        
        if df.empty:
            return insights
        
        # Spending velocity
        if 'date' in df.columns and _amount_column(df) is not None:
            df['date'] = pd.to_datetime(df['date'])
            expenses = _expense_rows(df)
            
            if not expenses.empty:
                daily_spending = expenses.groupby('date')['_spend_amount'].sum().mean()
                insights.append(f"💰 Average daily spending: ¥{daily_spending:,.0f}")
        
        # Category insights
        if 'category' in df.columns and _amount_column(df) is not None:
            expenses = _expense_rows(df)
            if not expenses.empty:
                by_category = expenses.groupby('category')['_spend_amount'].sum()
                top_category = by_category.idxmax()
                top_amount = by_category.max()
                total_expenses = expenses['_spend_amount'].sum()
                percentage = (top_amount / abs(total_expenses)) * 100
                
                insights.append(
                    f"📊 Largest spending category: {top_category} "
                    f"(¥{top_amount:,.0f}, {percentage:.1f}% of total)"
                )
        
        # Transaction frequency
        if 'date' in df.columns:
            date_range = (pd.to_datetime(df['date'].max()) - pd.to_datetime(df['date'].min())).days
            if date_range > 0:
                avg_transactions_per_day = len(df) / date_range
                insights.append(f"📈 Average {avg_transactions_per_day:.1f} transactions per day")
        
        return insights
    
    def _empty_report(self) -> Dict:
        """Return empty report structure."""
        return {
            'summary': {},
            'category_breakdown': {},
            'monthly_trends': {},
            'spending_patterns': {},
            'anomalies': [],
            'recommendations': [],
            'insights': [],
            'forecasts': {}
        }


class TrendAnalyzer:
    """Analyzes spending trends over time."""
    
    def analyze_monthly_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze month-over-month spending trends."""
        if 'date' not in df.columns or _amount_column(df) is None:
            return {}
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        
        # Filter expenses
        expenses = _expense_rows(df)
        if not expenses.empty:
            expenses['month'] = pd.to_datetime(expenses['date']).dt.to_period('M')
        
        if expenses.empty:
            return {}
        
        # Monthly totals
        monthly_totals = expenses.groupby('month')['_spend_amount'].sum()
        
        # Calculate trends
        if len(monthly_totals) < 2:
            return {'monthly_totals': monthly_totals.to_dict()}
        
        # Month-over-month change
        mom_change = monthly_totals.pct_change() * 100
        
        # Average monthly spending
        avg_monthly = monthly_totals.mean()
        
        # Trend direction
        recent_months = monthly_totals.tail(3)
        if len(recent_months) >= 2:
            trend = "increasing" if recent_months.iloc[-1] > recent_months.iloc[0] else "decreasing"
        else:
            trend = "stable"
        
        return {
            'monthly_totals': {str(k): v for k, v in monthly_totals.to_dict().items()},
            'average_monthly_spending': avg_monthly,
            'month_over_month_change': {str(k): v for k, v in mom_change.to_dict().items()},
            'trend_direction': trend,
            'highest_month': str(monthly_totals.idxmax()),
            'lowest_month': str(monthly_totals.idxmin())
        }
    
    def forecast_spending(self, df: pd.DataFrame, months_ahead: int = 3) -> Dict:
        """Forecast future spending based on historical trends."""
        if 'date' not in df.columns or _amount_column(df) is None:
            return {}
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        
        expenses = _expense_rows(df)
        if not expenses.empty:
            expenses['month'] = pd.to_datetime(expenses['date']).dt.to_period('M')
        
        if expenses.empty:
            return {}
        
        # Monthly totals
        monthly_totals = expenses.groupby('month')['_spend_amount'].sum()
        
        if len(monthly_totals) < 3:
            # Not enough data for forecasting
            return {
                'message': 'Insufficient data for forecasting',
                'minimum_required_months': 3
            }
        
        # Simple moving average forecast
        window_size = min(3, len(monthly_totals))
        recent_avg = monthly_totals.tail(window_size).mean()
        
        # Linear trend
        x = np.arange(len(monthly_totals))
        y = monthly_totals.values
        slope, intercept = np.polyfit(x, y, 1)
        
        # Forecast next months
        forecasts = {}
        last_month = monthly_totals.index[-1]
        
        for i in range(1, months_ahead + 1):
            forecast_month = last_month + i
            # Combine moving average and trend
            trend_forecast = slope * (len(monthly_totals) + i) + intercept
            forecast_value = (recent_avg + trend_forecast) / 2
            forecasts[str(forecast_month)] = max(0, forecast_value)
        
        return {
            'forecasts': forecasts,
            'method': 'hybrid_moving_average_trend',
            'confidence': 'medium' if len(monthly_totals) >= 6 else 'low'
        }


class AnomalyDetector:
    """Detects unusual spending patterns and transactions."""
    
    def detect_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect anomalous transactions."""
        if _amount_column(df) is None:
            return []
        
        anomalies = []
        
        expenses = _expense_rows(df)
        
        if expenses.empty or len(expenses) < 10:
            return anomalies
        
        # Statistical anomaly detection
        mean_amount = expenses['_spend_amount'].mean()
        std_amount = expenses['_spend_amount'].std()
        
        # Transactions beyond 2 standard deviations
        threshold = mean_amount + (2 * std_amount)
        
        unusual_transactions = expenses[expenses['_spend_amount'] > threshold]
        
        for _, trans in unusual_transactions.iterrows():
            anomalies.append({
                'type': 'unusually_large',
                'transaction': trans.to_dict(),
                'amount': trans['_spend_amount'],
                'severity': 'high' if trans['_spend_amount'] > mean_amount + (3 * std_amount) else 'medium',
                'reason': f"Transaction amount (¥{trans['_spend_amount']:,.0f}) is {((trans['_spend_amount'] / mean_amount) - 1) * 100:.0f}% above average"
            })
        
        # Frequency anomalies
        if 'description' in df.columns:
            merchant_freq = expenses.groupby('description').size()
            
            # Detect one-time large expenses
            for merchant, count in merchant_freq.items():
                if count == 1:
                    trans = expenses[expenses['description'] == merchant].iloc[0]
                    if trans['_spend_amount'] > mean_amount * 1.5:
                        anomalies.append({
                            'type': 'one_time_large_expense',
                            'transaction': trans.to_dict(),
                            'amount': trans['_spend_amount'],
                            'severity': 'low',
                            'reason': f"First-time large expense at {merchant}"
                        })
        
        return anomalies


class BudgetAdvisor:
    """Provides budget recommendations and optimization suggestions."""
    
    def generate_recommendations(self, df: pd.DataFrame) -> List[Dict]:
        """Generate budget recommendations."""
        recommendations = []
        
        if 'category' not in df.columns or _amount_column(df) is None:
            return recommendations
        
        expenses = _expense_rows(df)
        
        if expenses.empty:
            return recommendations
        
        # Analyze spending by category
        category_spending = expenses.groupby('category')['_spend_amount'].sum()
        total_spending = category_spending.sum()
        
        # Identify overspending categories (>30% of budget)
        for category, amount in category_spending.items():
            percentage = (amount / total_spending) * 100
            
            if percentage > 30:
                recommendations.append({
                    'type': 'budget_alert',
                    'priority': 'high',
                    'category': category,
                    'current_spending': amount,
                    'percentage_of_total': percentage,
                    'message': f"{category} spending is {percentage:.1f}% of total expenses",
                    'suggestion': f"Consider reducing {category} expenses by 20% to save ¥{amount * 0.2:,.0f}",
                    'potential_savings': amount * 0.2
                })
        
        # Identify frequent small transactions
        if 'description' in df.columns:
            merchant_stats = expenses.groupby('description').agg({
                '_spend_amount': ['count', 'sum', 'mean']
            })
            
            # Find merchants with frequent small purchases
            frequent_small = merchant_stats[
                (merchant_stats[('_spend_amount', 'count')] >= 10) &
                (merchant_stats[('_spend_amount', 'mean')] < 1000)
            ]
            
            for merchant in frequent_small.head(3).index:
                count = int(frequent_small.loc[merchant, ('_spend_amount', 'count')])
                total = frequent_small.loc[merchant, ('_spend_amount', 'sum')]
                avg = frequent_small.loc[merchant, ('_spend_amount', 'mean')]
                
                recommendations.append({
                    'type': 'frequency_optimization',
                    'priority': 'medium',
                    'merchant': merchant,
                    'transaction_count': count,
                    'total_spent': total,
                    'average_amount': avg,
                    'message': f"{count} transactions at {merchant} totaling ¥{total:,.0f}",
                    'suggestion': f"Reducing frequency by 25% could save ¥{total * 0.25:,.0f}",
                    'potential_savings': total * 0.25
                })
        
        return recommendations


class PatternAnalyzer:
    """Analyzes spending patterns and behaviors."""
    
    def analyze_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze spending patterns."""
        patterns = {}
        
        if df.empty:
            return patterns
        
        # Day of week patterns
        if 'date' in df.columns and _amount_column(df) is not None:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.day_name()
            df['is_weekend'] = df['date'].dt.dayofweek >= 5
            
            expenses = _expense_rows(df)
            if not expenses.empty:
                expenses['date'] = pd.to_datetime(expenses['date'])
                expenses['day_of_week'] = expenses['date'].dt.day_name()
                expenses['is_weekend'] = expenses['date'].dt.dayofweek >= 5
            
            if not expenses.empty:
                # Spending by day of week
                dow_spending = expenses.groupby('day_of_week')['_spend_amount'].sum()
                patterns['day_of_week'] = dow_spending.to_dict()
                
                # Weekend vs weekday
                weekend_spending = expenses[expenses['is_weekend']]['_spend_amount'].sum()
                weekday_spending = expenses[~expenses['is_weekend']]['_spend_amount'].sum()
                total_spending = weekend_spending + weekday_spending
                
                patterns['weekend_vs_weekday'] = {
                    'weekend': weekend_spending,
                    'weekday': weekday_spending,
                    'weekend_percentage': weekend_spending / total_spending * 100 if total_spending else 0
                }
        
        # Time of month patterns
        if 'date' in df.columns and _amount_column(df) is not None:
            expenses = _expense_rows(df)
            if not expenses.empty:
                expenses['day_of_month'] = pd.to_datetime(expenses['date']).dt.day
            
            if not expenses.empty:
                # Early, mid, late month spending
                early_month = expenses[expenses['day_of_month'] <= 10]['_spend_amount'].sum()
                mid_month = expenses[(expenses['day_of_month'] > 10) & (expenses['day_of_month'] <= 20)]['_spend_amount'].sum()
                late_month = expenses[expenses['day_of_month'] > 20]['_spend_amount'].sum()
                
                patterns['time_of_month'] = {
                    'early_month': early_month,
                    'mid_month': mid_month,
                    'late_month': late_month
                }
        
        return patterns
    
    def generate_recommendations(self, df: pd.DataFrame) -> List[Dict]:
        """Generate pattern-based recommendations."""
        recommendations = []
        
        patterns = self.analyze_patterns(df)
        
        # Weekend spending recommendations
        if 'weekend_vs_weekday' in patterns:
            weekend_pct = patterns['weekend_vs_weekday']['weekend_percentage']
            if weekend_pct > 40:
                recommendations.append({
                    'type': 'pattern_alert',
                    'priority': 'medium',
                    'pattern': 'weekend_spending',
                    'message': f"Weekend spending is {weekend_pct:.1f}% of total",
                    'suggestion': "Consider meal prepping or planning weekend activities to reduce costs",
                    'potential_savings': patterns['weekend_vs_weekday']['weekend'] * 0.15
                })
        
        return recommendations


class SpendingAnalytics:
    """Advanced spending analytics and visualizations data."""
    
    @staticmethod
    def prepare_category_chart_data(df: pd.DataFrame) -> Dict:
        """Prepare data for category spending chart."""
        if 'category' not in df.columns or _amount_column(df) is None:
            return {}
        
        expenses = _expense_rows(df)
        
        if expenses.empty:
            return {}
        
        category_totals = expenses.groupby('category')['_spend_amount'].sum()
        
        return {
            'labels': category_totals.index.tolist(),
            'values': category_totals.values.tolist(),
            'type': 'pie'
        }
    
    @staticmethod
    def prepare_monthly_trend_data(df: pd.DataFrame) -> Dict:
        """Prepare data for monthly trend chart."""
        if 'date' not in df.columns or _amount_column(df) is None:
            return {}
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M').astype(str)
        
        expenses = _expense_rows(df)
        income = _income_rows(df)
        if not expenses.empty:
            expenses['month'] = pd.to_datetime(expenses['date']).dt.to_period('M').astype(str)
        if not income.empty:
            income['month'] = pd.to_datetime(income['date']).dt.to_period('M').astype(str)
        
        monthly_expenses = expenses.groupby('month')['_spend_amount'].sum() if not expenses.empty else pd.Series()
        monthly_income = income.groupby('month')['_spend_amount'].sum() if not income.empty else pd.Series()
        
        # Combine all months
        all_months = sorted(set(monthly_expenses.index.tolist() + monthly_income.index.tolist()))
        
        return {
            'labels': all_months,
            'expenses': [monthly_expenses.get(m, 0) for m in all_months],
            'income': [monthly_income.get(m, 0) for m in all_months],
            'type': 'line'
        }
    
    @staticmethod
    def calculate_key_metrics(df: pd.DataFrame) -> Dict:
        """Calculate key financial metrics."""
        if df.empty:
            return {}
        
        expenses = _expense_rows(df)
        income = _income_rows(df)
        
        total_expenses = expenses['_spend_amount'].sum() if not expenses.empty else 0
        total_income = income['_spend_amount'].sum() if not income.empty else 0
        
        return {
            'total_expenses': total_expenses,
            'total_income': total_income,
            'net_savings': total_income - total_expenses,
            'savings_rate': (total_income - total_expenses) / total_income * 100 if total_income > 0 else 0,
            'average_daily_spending': total_expenses / 30 if total_expenses > 0 else 0,
            'transaction_count': len(df)
        }

