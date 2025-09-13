#!/usr/bin/env python3
"""
Enhanced Transaction Web App with Bank Statement Cross-Checking

This enhanced version of the transaction web app includes:
- Bank statement cross-checking and reconciliation
- Advanced categorization with confidence scoring
- Financial data validation and quality checks
- Comprehensive reporting and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import io
import re

# Import our bank statement analyzer
from bank_statement_analyzer import BankStatementAnalyzer

# Page configuration
st.set_page_config(
    page_title="Enhanced Expense Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EnhancedTransactionApp:
    """Enhanced transaction application with bank statement analysis."""
    
    def __init__(self):
        self.analyzer = BankStatementAnalyzer()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'csv_data' not in st.session_state:
            st.session_state.csv_data = None
        if 'pdf_data' not in st.session_state:
            st.session_state.pdf_data = None
        if 'categorized_data' not in st.session_state:
            st.session_state.categorized_data = None
        if 'reconciliation_results' not in st.session_state:
            st.session_state.reconciliation_results = None
        if 'analysis_report' not in st.session_state:
            st.session_state.analysis_report = None
    
    def run(self):
        """Main application runner."""
        st.title("🏦 Enhanced Bank Statement Analyzer")
        st.markdown("Cross-check your bank statements and get intelligent expense categorization")
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Choose Analysis Type",
            ["📊 Bank Statement Cross-Check", "📈 Expense Analysis", "🔍 Data Quality Check", "📋 Reports"]
        )
        
        if page == "📊 Bank Statement Cross-Check":
            self.bank_statement_cross_check()
        elif page == "📈 Expense Analysis":
            self.expense_analysis()
        elif page == "🔍 Data Quality Check":
            self.data_quality_check()
        elif page == "📋 Reports":
            self.reports_page()
    
    def bank_statement_cross_check(self):
        """Bank statement cross-checking interface."""
        st.header("📊 Bank Statement Cross-Check")
        st.markdown("Upload your CSV and PDF bank statements to cross-check for accuracy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 CSV Statement")
            csv_file = st.file_uploader(
                "Upload CSV bank statement",
                type=['csv'],
                help="Upload your bank's CSV export file"
            )
            
            if csv_file is not None:
                try:
                    csv_data = self.analyzer.parse_csv_statement(csv_file)
                    st.session_state.csv_data = csv_data
                    
                    st.success(f"✅ CSV loaded: {len(csv_data)} transactions")
                    
                    # Show preview
                    with st.expander("📋 CSV Preview"):
                        st.dataframe(csv_data.head(10))
                        
                        # Summary stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Transactions", len(csv_data))
                        with col2:
                            st.metric("Total Amount", f"¥{csv_data['amount'].sum():,.0f}")
                        with col3:
                            st.metric("Date Range", f"{csv_data['date'].min().strftime('%m/%d')} - {csv_data['date'].max().strftime('%m/%d')}")
                
                except Exception as e:
                    st.error(f"Error parsing CSV: {e}")
        
        with col2:
            st.subheader("📄 PDF Statement")
            pdf_file = st.file_uploader(
                "Upload PDF bank statement",
                type=['pdf'],
                help="Upload your bank's PDF statement"
            )
            
            if pdf_file is not None:
                try:
                    # Save uploaded file temporarily
                    with open("/tmp/uploaded_statement.pdf", "wb") as f:
                        f.write(pdf_file.getbuffer())
                    
                    pdf_data = self.analyzer.parse_pdf_statement("/tmp/uploaded_statement.pdf")
                    st.session_state.pdf_data = pdf_data
                    
                    st.success(f"✅ PDF loaded: {len(pdf_data)} transactions")
                    
                    # Show preview
                    with st.expander("📋 PDF Preview"):
                        st.dataframe(pdf_data.head(10))
                        
                        # Summary stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Transactions", len(pdf_data))
                        with col2:
                            st.metric("Total Amount", f"¥{pdf_data['amount'].sum():,.0f}")
                        with col3:
                            st.metric("Date Range", f"{pdf_data['date'].min().strftime('%m/%d')} - {pdf_data['date'].max().strftime('%m/%d')}")
                
                except Exception as e:
                    st.error(f"Error parsing PDF: {e}")
        
        # Cross-check analysis
        if st.session_state.csv_data is not None and st.session_state.pdf_data is not None:
            st.subheader("🔍 Cross-Check Analysis")
            
            if st.button("🔄 Run Cross-Check Analysis"):
                with st.spinner("Analyzing statements..."):
                    reconciliation_results = self.analyzer.cross_check_statements(
                        st.session_state.csv_data, 
                        st.session_state.pdf_data
                    )
                    st.session_state.reconciliation_results = reconciliation_results
                    
                    # Generate analysis report
                    analysis_report = self.analyzer.generate_analysis_report(
                        st.session_state.csv_data,
                        st.session_state.pdf_data
                    )
                    st.session_state.analysis_report = analysis_report
            
            if st.session_state.reconciliation_results:
                self.display_reconciliation_results()
    
    def display_reconciliation_results(self):
        """Display reconciliation results."""
        results = st.session_state.reconciliation_results
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if results['reconciliation_status'] == 'reconciled':
                st.success("✅ **Reconciled**")
            elif results['reconciliation_status'] == 'minor_discrepancy':
                st.warning("⚠️ **Minor Discrepancy**")
            else:
                st.error("❌ **Major Discrepancy**")
        
        with col2:
            st.metric("CSV Total", f"¥{results['csv_total']:,.0f}")
        
        with col3:
            st.metric("PDF Total", f"¥{results['pdf_total']:,.0f}")
        
        with col4:
            st.metric("Difference", f"¥{results['total_difference']:,.0f}")
        
        # Detailed analysis
        st.subheader("📊 Detailed Analysis")
        
        # Transaction counts
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**CSV Summary:**")
            st.write(f"- Transactions: {results['csv_summary']['count']}")
            st.write(f"- Total Amount: ¥{results['csv_summary']['total']:,.0f}")
            st.write(f"- Date Range: {results['csv_summary']['date_range']}")
            st.write(f"- Duplicates: {results['csv_summary']['duplicates']}")
        
        with col2:
            st.write("**PDF Summary:**")
            st.write(f"- Transactions: {results['pdf_summary']['count']}")
            st.write(f"- Total Amount: ¥{results['pdf_summary']['total']:,.0f}")
            st.write(f"- Date Range: {results['pdf_summary']['date_range']}")
            st.write(f"- Duplicates: {results['pdf_summary']['duplicates']}")
        
        # Discrepancies
        if results['csv_only'] or results['pdf_only']:
            st.subheader("🔍 Discrepancies Found")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if results['csv_only']:
                    st.write(f"**Only in CSV ({len(results['csv_only'])} transactions):**")
                    csv_df = pd.DataFrame(results['csv_only'])
                    st.dataframe(csv_df.head(10))
            
            with col2:
                if results['pdf_only']:
                    st.write(f"**Only in PDF ({len(results['pdf_only'])} transactions):**")
                    pdf_df = pd.DataFrame(results['pdf_only'])
                    st.dataframe(pdf_df.head(10))
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if results['reconciliation_status'] == 'major_discrepancy':
            st.error("**Major discrepancy detected!**")
            st.write("- Review both statements carefully")
            st.write("- Check for missing transactions")
            st.write("- Verify transaction amounts and dates")
        elif results['reconciliation_status'] == 'minor_discrepancy':
            st.warning("**Minor discrepancy detected**")
            st.write("- Likely due to timing differences")
            st.write("- Check transactions near statement cut-off dates")
        else:
            st.success("**Statements are reconciled!**")
            st.write("- Data integrity is good")
            st.write("- Proceed with expense analysis")
    
    def expense_analysis(self):
        """Expense analysis and categorization."""
        st.header("📈 Expense Analysis")
        
        if st.session_state.csv_data is None:
            st.warning("Please upload a CSV statement first in the Cross-Check page")
            return
        
        st.subheader("🧠 Smart Categorization")
        
        if st.button("🎯 Run Smart Categorization"):
            with st.spinner("Categorizing transactions..."):
                categorized_data = self.analyzer.categorize_transactions(st.session_state.csv_data)
                st.session_state.categorized_data = categorized_data
                st.success("✅ Categorization complete!")
        
        if st.session_state.categorized_data is not None:
            self.display_categorization_results()
    
    def display_categorization_results(self):
        """Display categorization results."""
        df = st.session_state.categorized_data
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", len(df))
        
        with col2:
            categorized_count = len(df[df['category'] != 'Uncategorised'])
            st.metric("Categorized", f"{categorized_count}/{len(df)}")
        
        with col3:
            avg_confidence = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        with col4:
            high_confidence = len(df[df['confidence'] >= 0.8])
            st.metric("High Confidence", high_confidence)
        
        # Category breakdown
        st.subheader("📊 Category Breakdown")
        
        category_summary = df.groupby('category').agg({
            'amount': ['count', 'sum'],
            'confidence': 'mean'
        }).round(2)
        
        category_summary.columns = ['Count', 'Total Amount', 'Avg Confidence']
        category_summary = category_summary.sort_values('Total Amount', ascending=False)
        
        # Display as table
        st.dataframe(category_summary)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart of categories
            fig = px.pie(
                category_summary.reset_index(),
                values='Total Amount',
                names='category',
                title="Expense Distribution by Category"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart of transaction counts
            fig = px.bar(
                category_summary.reset_index(),
                x='category',
                y='Count',
                title="Transaction Count by Category"
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed transaction view
        st.subheader("📋 Detailed Transactions")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_category = st.selectbox(
                "Filter by Category",
                ['All'] + list(df['category'].unique())
            )
        
        with col2:
            min_confidence = st.slider(
                "Minimum Confidence",
                0.0, 1.0, 0.0, 0.1
            )
        
        with col3:
            min_amount = st.number_input(
                "Minimum Amount",
                min_value=0, value=0
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        
        filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
        filtered_df = filtered_df[filtered_df['amount'] >= min_amount]
        
        # Display filtered results
        st.write(f"Showing {len(filtered_df)} transactions")
        st.dataframe(filtered_df)
        
        # Export options
        if st.button("📥 Export Categorized Data"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"categorized_transactions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    def data_quality_check(self):
        """Data quality validation and checks."""
        st.header("🔍 Data Quality Check")
        
        if st.session_state.csv_data is None:
            st.warning("Please upload a CSV statement first")
            return
        
        df = st.session_state.csv_data
        
        st.subheader("📊 Data Quality Metrics")
        
        # Quality checks
        quality_issues = []
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['date', 'description', 'amount'])
        if duplicates.any():
            quality_issues.append(f"Found {duplicates.sum()} duplicate transactions")
        
        # Check for missing data
        missing_dates = df['date'].isna().sum()
        if missing_dates > 0:
            quality_issues.append(f"Found {missing_dates} transactions with missing dates")
        
        missing_amounts = df['amount'].isna().sum()
        if missing_amounts > 0:
            quality_issues.append(f"Found {missing_amounts} transactions with missing amounts")
        
        # Check for zero amounts
        zero_amounts = (df['amount'] == 0).sum()
        if zero_amounts > 0:
            quality_issues.append(f"Found {zero_amounts} transactions with zero amounts")
        
        # Check for extreme amounts
        extreme_amounts = (df['amount'].abs() > 100000).sum()
        if extreme_amounts > 0:
            quality_issues.append(f"Found {extreme_amounts} transactions with amounts over ¥100,000")
        
        # Display quality status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not quality_issues:
                st.success("✅ **Data Quality: Excellent**")
            elif len(quality_issues) <= 2:
                st.warning("⚠️ **Data Quality: Good**")
            else:
                st.error("❌ **Data Quality: Needs Attention**")
        
        with col2:
            st.metric("Total Transactions", len(df))
        
        with col3:
            st.metric("Data Completeness", f"{((len(df) - missing_dates - missing_amounts) / len(df) * 100):.1f}%")
        
        # Show issues
        if quality_issues:
            st.subheader("⚠️ Quality Issues Found")
            for issue in quality_issues:
                st.warning(issue)
        
        # Data distribution
        st.subheader("📈 Data Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Amount distribution
            fig = px.histogram(
                df, x='amount', 
                title="Transaction Amount Distribution",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Daily transaction count
            daily_counts = df.groupby(df['date'].dt.date).size()
            fig = px.line(
                x=daily_counts.index,
                y=daily_counts.values,
                title="Daily Transaction Count"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Outliers detection
        st.subheader("🔍 Outlier Detection")
        
        # Statistical outliers
        Q1 = df['amount'].quantile(0.25)
        Q3 = df['amount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df['amount'] < lower_bound) | (df['amount'] > upper_bound)]
        
        if len(outliers) > 0:
            st.write(f"Found {len(outliers)} statistical outliers:")
            st.dataframe(outliers[['date', 'description', 'amount']])
        else:
            st.success("No statistical outliers found")
    
    def reports_page(self):
        """Reports and export functionality."""
        st.header("📋 Reports & Export")
        
        if st.session_state.analysis_report:
            st.subheader("📄 Analysis Report")
            st.markdown(st.session_state.analysis_report)
            
            # Download report
            st.download_button(
                label="📥 Download Analysis Report",
                data=st.session_state.analysis_report,
                file_name=f"bank_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown"
            )
        
        if st.session_state.categorized_data is not None:
            st.subheader("📊 Categorized Data Export")
            
            # Export options
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Export All Data"):
                    csv = st.session_state.categorized_data.to_csv(index=False)
                    st.download_button(
                        label="Download Full Dataset",
                        data=csv,
                        file_name=f"full_categorized_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📥 Export Summary"):
                    summary = st.session_state.categorized_data.groupby('category').agg({
                        'amount': ['count', 'sum'],
                        'confidence': 'mean'
                    }).round(2)
                    summary.columns = ['Count', 'Total Amount', 'Avg Confidence']
                    
                    csv = summary.to_csv()
                    st.download_button(
                        label="Download Summary",
                        data=csv,
                        file_name=f"category_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
        
        # Custom report generation
        st.subheader("🔧 Custom Report Generator")
        
        if st.session_state.categorized_data is not None:
            df = st.session_state.categorized_data
            
            # Report options
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input("Start Date", df['date'].min().date())
                end_date = st.date_input("End Date", df['date'].max().date())
            
            with col2:
                selected_categories = st.multiselect(
                    "Categories to Include",
                    df['category'].unique(),
                    default=df['category'].unique()
                )
            
            if st.button("📊 Generate Custom Report"):
                # Filter data
                filtered_df = df[
                    (df['date'].dt.date >= start_date) & 
                    (df['date'].dt.date <= end_date) &
                    (df['category'].isin(selected_categories))
                ]
                
                # Generate report
                report = self.generate_custom_report(filtered_df, start_date, end_date)
                st.markdown(report)
    
    def generate_custom_report(self, df: pd.DataFrame, start_date, end_date) -> str:
        """Generate custom report for filtered data."""
        report = []
        report.append(f"# Custom Expense Report")
        report.append(f"**Period:** {start_date} to {end_date}")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- Total Transactions: {len(df)}")
        report.append(f"- Total Amount: ¥{df['amount'].sum():,.0f}")
        report.append(f"- Average Transaction: ¥{df['amount'].mean():,.0f}")
        report.append("")
        
        # Category breakdown
        report.append("## Category Breakdown")
        category_summary = df.groupby('category').agg({
            'amount': ['count', 'sum'],
            'confidence': 'mean'
        }).round(2)
        
        for category in category_summary.index:
            count = category_summary.loc[category, ('amount', 'count')]
            total = category_summary.loc[category, ('amount', 'sum')]
            avg_confidence = category_summary.loc[category, ('confidence', 'mean')]
            report.append(f"- **{category}**: {count} transactions, ¥{total:,.0f} (avg confidence: {avg_confidence:.1%})")
        
        return "\n".join(report)


def main():
    """Main application entry point."""
    app = EnhancedTransactionApp()
    app.run()


if __name__ == "__main__":
    main()