"""
dashboard.py

Enhanced Dashboard with Modern Metrics and Interactive Charts
Provides rich visualizations and interactive components for the expense tracker.

Features:
- Modern metric cards with trends
- Interactive Plotly charts
- Real-time analytics
- Comparative views
- Drill-down capabilities
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from mobile_ui import layout_columns, use_compact_layout
except ImportError:
    def use_compact_layout():
        return False
    def layout_columns(count, *, weights=None):
        return st.columns(count)
from plotly.subplots import make_subplots


class ModernDashboard:
    """Modern dashboard with enhanced visualizations and metrics."""
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#1f77b4',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17becf',
            'expense': '#e74c3c',
            'income': '#27ae60',
            'categories': px.colors.qualitative.Set3
        }
    
    def render_hero_metrics(self, transactions: List[Dict]):
        """Render hero metrics at the top of dashboard."""
        if not transactions:
            st.info("📊 No transaction data available yet")
            return
        
        df = pd.DataFrame(transactions)
        
        # Calculate metrics
        expenses = df[df['amount'] < 0] if 'amount' in df.columns else pd.DataFrame()
        income = df[df['amount'] > 0] if 'amount' in df.columns else pd.DataFrame()
        
        total_expenses = abs(expenses['amount'].sum()) if not expenses.empty else 0
        total_income = income['amount'].sum() if not income.empty else 0
        net_balance = total_income - total_expenses
        
        # Calculate month-over-month changes
        expense_change = self._calculate_mom_change(expenses)
        income_change = self._calculate_mom_change(income)
        
        col1, col2, col3, col4 = layout_columns(4)
        
        with col1:
            self._render_metric_card(
                "💰 Total Expenses",
                f"¥{total_expenses:,.0f}",
                expense_change,
                "decrease"
            )
        
        with col2:
            self._render_metric_card(
                "💵 Total Income",
                f"¥{total_income:,.0f}",
                income_change,
                "increase"
            )
        
        with col3:
            net_change = income_change - expense_change if income_change and expense_change else None
            self._render_metric_card(
                "📊 Net Balance",
                f"¥{net_balance:,.0f}",
                net_change,
                "increase" if net_balance > 0 else "decrease"
            )
        
        with col4:
            savings_rate = (net_balance / total_income * 100) if total_income > 0 else 0
            self._render_metric_card(
                "🎯 Savings Rate",
                f"{savings_rate:.1f}%",
                None,
                "neutral"
            )
    
    def _render_metric_card(
        self, 
        title: str, 
        value: str, 
        change: Optional[float],
        trend_direction: str
    ):
        """Render a single metric card."""
        # Determine delta color
        if change is not None:
            if trend_direction == "increase":
                delta_color = "normal" if change > 0 else "inverse"
            elif trend_direction == "decrease":
                delta_color = "normal" if change < 0 else "inverse"
            else:
                delta_color = "off"
            
            st.metric(
                label=title,
                value=value,
                delta=f"{change:+.1f}%" if change else None,
                delta_color=delta_color
            )
        else:
            st.metric(label=title, value=value)
    
    def render_category_breakdown_chart(self, transactions: List[Dict]):
        """Render interactive category breakdown chart."""
        if not transactions:
            return
        
        df = pd.DataFrame(transactions)
        expenses = df[df['amount'] < 0].copy() if 'amount' in df.columns else pd.DataFrame()
        
        if expenses.empty or 'category' not in expenses.columns:
            st.info("No categorized expenses to display")
            return
        
        # Aggregate by category
        category_totals = expenses.groupby('category')['amount'].sum().abs().sort_values(ascending=False)
        
        # Create interactive pie chart
        fig = go.Figure(data=[go.Pie(
            labels=category_totals.index,
            values=category_totals.values,
            hole=0.4,
            marker=dict(colors=self.color_scheme['categories']),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>¥%{value:,.0f}<br>%{percent}<extra></extra>'
        )])
        
        legend_cfg = (
            dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
            if use_compact_layout()
            else dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
        )
        fig.update_layout(
            title="Spending by Category",
            height=320 if use_compact_layout() else 400,
            showlegend=True,
            legend=legend_cfg,
            margin=dict(b=80 if use_compact_layout() else 40),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_monthly_trend_chart(self, transactions: List[Dict]):
        """Render monthly spending trend chart."""
        if not transactions:
            return
        
        df = pd.DataFrame(transactions)
        
        if 'date' not in df.columns or 'amount' not in df.columns:
            return
        
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M').astype(str)
        
        # Separate expenses and income
        expenses = df[df['amount'] < 0].copy()
        income = df[df['amount'] > 0].copy()
        
        # Monthly aggregates
        monthly_expenses = expenses.groupby('month')['amount'].sum().abs() if not expenses.empty else pd.Series()
        monthly_income = income.groupby('month')['amount'].sum() if not income.empty else pd.Series()
        
        # Get all months
        all_months = sorted(set(list(monthly_expenses.index) + list(monthly_income.index)))
        
        # Create subplots
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        # Add expense trace
        fig.add_trace(
            go.Scatter(
                x=all_months,
                y=[monthly_expenses.get(m, 0) for m in all_months],
                name="Expenses",
                mode='lines+markers',
                line=dict(color=self.color_scheme['expense'], width=3),
                marker=dict(size=8),
                hovertemplate='<b>Expenses</b><br>%{x}<br>¥%{y:,.0f}<extra></extra>'
            )
        )
        
        # Add income trace
        fig.add_trace(
            go.Scatter(
                x=all_months,
                y=[monthly_income.get(m, 0) for m in all_months],
                name="Income",
                mode='lines+markers',
                line=dict(color=self.color_scheme['income'], width=3),
                marker=dict(size=8),
                hovertemplate='<b>Income</b><br>%{x}<br>¥%{y:,.0f}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title="Monthly Income vs Expenses",
            xaxis_title="Month",
            yaxis_title="Amount (¥)",
            height=400,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_category_trend_chart(self, transactions: List[Dict], top_n: int = 5):
        """Render trend chart for top categories over time."""
        if not transactions:
            return
        
        df = pd.DataFrame(transactions)
        
        if 'date' not in df.columns or 'amount' not in df.columns or 'category' not in df.columns:
            return
        
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M').astype(str)
        
        expenses = df[df['amount'] < 0].copy()
        
        if expenses.empty:
            return
        
        # Get top N categories by total spending
        top_categories = expenses.groupby('category')['amount'].sum().abs().nlargest(top_n).index
        
        # Filter for top categories
        top_expenses = expenses[expenses['category'].isin(top_categories)]
        
        # Monthly spending by category
        category_monthly = top_expenses.groupby(['month', 'category'])['amount'].sum().abs().unstack(fill_value=0)
        
        # Create line chart
        fig = go.Figure()
        
        for category in category_monthly.columns:
            fig.add_trace(go.Scatter(
                x=category_monthly.index,
                y=category_monthly[category],
                name=category,
                mode='lines+markers',
                hovertemplate=f'<b>{category}</b><br>%{{x}}<br>¥%{{y:,.0f}}<extra></extra>'
            ))
        
        legend_cfg = (
            dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            if use_compact_layout()
            else dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        fig.update_layout(
            title=f"Top {top_n} Categories - Monthly Trend",
            xaxis_title="Month",
            yaxis_title="Amount (¥)",
            height=320 if use_compact_layout() else 400,
            hovermode='x unified',
            legend=legend_cfg,
            margin=dict(b=100 if use_compact_layout() else 40),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_spending_heatmap(self, transactions: List[Dict]):
        """Render spending heatmap by day of week and hour."""
        if not transactions:
            return
        
        df = pd.DataFrame(transactions)
        
        if 'date' not in df.columns or 'amount' not in df.columns:
            return
        
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.day_name()
        
        expenses = df[df['amount'] < 0].copy()
        
        if expenses.empty:
            return
        
        # Aggregate by day of week
        dow_spending = expenses.groupby('day_of_week')['amount'].sum().abs()
        
        # Order days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_spending = dow_spending.reindex(day_order, fill_value=0)
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=dow_spending.index,
                y=dow_spending.values,
                marker_color=self.color_scheme['primary'],
                hovertemplate='<b>%{x}</b><br>¥%{y:,.0f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Spending by Day of Week",
            xaxis_title="Day of Week",
            yaxis_title="Total Spending (¥)",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_top_merchants_chart(self, transactions: List[Dict], top_n: int = 10):
        """Render top merchants by spending."""
        if not transactions:
            return
        
        df = pd.DataFrame(transactions)
        
        if 'description' not in df.columns or 'amount' not in df.columns:
            return
        
        expenses = df[df['amount'] < 0].copy()
        
        if expenses.empty:
            return
        
        # Top merchants
        top_merchants = expenses.groupby('description')['amount'].sum().abs().nlargest(top_n)
        
        # Create horizontal bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=top_merchants.values,
                y=top_merchants.index,
                orientation='h',
                marker_color=self.color_scheme['warning'],
                hovertemplate='<b>%{y}</b><br>¥%{x:,.0f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=f"Top {top_n} Merchants by Spending",
            xaxis_title="Total Spending (¥)",
            yaxis_title="Merchant",
            height=400,
            yaxis=dict(autorange="reversed")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_ai_insights_panel(self, insights: List[str], recommendations: List[Dict]):
        """Render AI-powered insights and recommendations panel."""
        st.subheader("🤖 AI-Powered Insights")
        
        # Display insights
        if insights:
            for insight in insights:
                st.info(insight)
        
        # Display recommendations
        if recommendations:
            st.markdown("### 💡 Recommendations")
            
            for i, rec in enumerate(recommendations[:5], 1):
                priority_emoji = {
                    'high': '🔴',
                    'medium': '🟡',
                    'low': '🟢'
                }.get(rec.get('priority', 'low'), '⚪')
                
                with st.expander(f"{priority_emoji} {rec.get('message', 'Recommendation')}"):
                    st.write(f"**Suggestion:** {rec.get('suggestion', 'N/A')}")
                    
                    if 'potential_savings' in rec:
                        st.write(f"**Potential Savings:** ¥{rec['potential_savings']:,.0f}")
                    
                    if 'category' in rec:
                        st.write(f"**Category:** {rec['category']}")
    
    def render_comparison_view(self, transactions: List[Dict]):
        """Render comparison view for different time periods."""
        if not transactions:
            return
        
        df = pd.DataFrame(transactions)
        
        if 'date' not in df.columns or 'amount' not in df.columns:
            return
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Get current and previous month
        current_month = df['date'].max().to_period('M')
        previous_month = current_month - 1
        
        current_data = df[df['date'].dt.to_period('M') == current_month]
        previous_data = df[df['date'].dt.to_period('M') == previous_month]
        
        if current_data.empty or previous_data.empty:
            st.info("Need at least 2 months of data for comparison")
            return
        
        # Calculate metrics
        current_expenses = abs(current_data[current_data['amount'] < 0]['amount'].sum())
        previous_expenses = abs(previous_data[previous_data['amount'] < 0]['amount'].sum())
        
        change_pct = ((current_expenses - previous_expenses) / previous_expenses * 100) if previous_expenses > 0 else 0
        
        st.subheader(f"📊 Comparison: {previous_month} vs {current_month}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                f"Previous ({previous_month})",
                f"¥{previous_expenses:,.0f}"
            )
        
        with col2:
            st.metric(
                f"Current ({current_month})",
                f"¥{current_expenses:,.0f}",
                f"{change_pct:+.1f}%"
            )
        
        with col3:
            difference = current_expenses - previous_expenses
            st.metric(
                "Difference",
                f"¥{abs(difference):,.0f}",
                "More" if difference > 0 else "Less"
            )
    
    def _calculate_mom_change(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate month-over-month percentage change."""
        if df.empty or 'date' not in df.columns or 'amount' not in df.columns:
            return None
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        
        monthly_totals = df.groupby('month')['amount'].sum()
        
        if len(monthly_totals) < 2:
            return None
        
        # Get last two months
        last_month = monthly_totals.iloc[-1]
        previous_month = monthly_totals.iloc[-2]
        
        if previous_month == 0:
            return None
        
        change = ((last_month - previous_month) / abs(previous_month)) * 100
        return change


class InteractiveFilters:
    """Interactive filtering components for dashboard."""
    
    @staticmethod
    def render_date_range_filter() -> Tuple[Optional[datetime], Optional[datetime]]:
        """Render date range filter."""
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date", value=None)
        
        with col2:
            end_date = st.date_input("End Date", value=None)
        
        return start_date, end_date
    
    @staticmethod
    def render_category_filter(categories: List[str]) -> List[str]:
        """Render category multi-select filter."""
        if not categories:
            return []
        
        selected = st.multiselect(
            "Filter by Categories",
            options=categories,
            default=categories
        )
        
        return selected
    
    @staticmethod
    def render_amount_filter() -> Tuple[Optional[float], Optional[float]]:
        """Render amount range filter."""
        col1, col2 = st.columns(2)
        
        with col1:
            min_amount = st.number_input("Min Amount", value=None, step=100.0)
        
        with col2:
            max_amount = st.number_input("Max Amount", value=None, step=100.0)
        
        return min_amount, max_amount

