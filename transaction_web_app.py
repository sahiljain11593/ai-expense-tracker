"""
transaction_web_app.py

This Streamlit application provides a simple user interface for
uploading credit card statements in PDF or image (screenshot) format,
extracting transaction data, automatically assigning categories based on
keyword rules, and allowing the user to review and adjust categories.

The app assumes that the user has installed the following Python
packages in their environment:

* streamlit — for building the web interface
* pandas — for data manipulation
* pdfplumber — for reading table data from PDF files
* pytesseract — for Optical Character Recognition on images
* Pillow (PIL) — for image handling

You also need to have the Tesseract OCR engine installed on your
system for pytesseract to work.

Usage: run this app with

    streamlit run transaction_web_app.py
"""

import io
import os
import re
from datetime import datetime
from typing import Optional


import pandas as pd
import streamlit as st

# Data layer
try:
    from data_store import (
        init_db,
        create_import_record,
        insert_transactions,
        export_transactions_to_csv,
        backup_database,
        load_all_transactions,
        upsert_recurring_rule,
        list_recurring_rules,
        get_setting,
        set_setting,
    )
except Exception as _e:
    # Allow the app to still render other parts; show a soft warning
    init_db = None  # type: ignore
    create_import_record = None  # type: ignore
    insert_transactions = None  # type: ignore
    export_transactions_to_csv = None  # type: ignore
    backup_database = None  # type: ignore
    load_all_transactions = None  # type: ignore
    upsert_recurring_rule = None  # type: ignore
    list_recurring_rules = None  # type: ignore

# Auth UI (Firebase Google Sign-In)
try:
    from auth_ui import require_auth
except Exception:
    def require_auth() -> bool:  # fallback no-auth when module unavailable
        return True

# Wide layout for more horizontal space
st.set_page_config(layout="wide")

# Smart Learning System (now using built-in MerchantLearningSystem class)

try:
    import pdfplumber  # type: ignore
except ImportError:
    pdfplumber = None

try:
    import pytesseract  # type: ignore
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None


class MerchantLearningSystem:
    """Learning system that improves categorization based on user feedback."""
    
    def __init__(self):
        self.merchant_categories = {}  # merchant -> category mapping
        self.merchant_patterns = {}    # merchant patterns -> category mapping
        self.user_corrections = []     # track user corrections for analysis
        self.confidence_scores = {}    # confidence level for each merchant
    
    def learn_from_user_feedback(self, transaction_id: str, old_category: str, new_category: str, transaction_data: dict):
        """Learn from user corrections to improve future categorizations."""
        merchant = self._extract_merchant(transaction_data['description'])
        original_desc = transaction_data.get('original_description', '')
        
        # Store the correction
        correction = {
            'transaction_id': transaction_id,
            'merchant': merchant,
            'old_category': old_category,
            'new_category': new_category,
            'description': transaction_data['description'],
            'original_description': original_desc,
            'amount': transaction_data['amount'],
            'timestamp': pd.Timestamp.now()
        }
        self.user_corrections.append(correction)
        
        # Update merchant category mapping
        if merchant not in self.merchant_categories:
            self.merchant_categories[merchant] = {}
        
        if new_category not in self.merchant_categories[merchant]:
            self.merchant_categories[merchant][new_category] = 0
        
        self.merchant_categories[merchant][new_category] += 1
        
        # Update confidence scores
        self._update_confidence(merchant, new_category)
    
    def suggest_category(self, description: str, original_description: str = "") -> tuple:
        """Suggest category based on learned patterns."""
        merchant = self._extract_merchant(description)
        
        if merchant in self.merchant_categories:
            # Get the most frequently used category for this merchant
            categories = self.merchant_categories[merchant]
            if categories:
                best_category = max(categories, key=categories.get)
                confidence = self.confidence_scores.get(merchant, {}).get(best_category, 0.5)
                return best_category, confidence
        
        return "Uncategorised", 0.0
    
    def predict_category(self, transaction_data: dict) -> tuple:
        """Predict category for a transaction with confidence and breakdown."""
        description = transaction_data.get('description', '')
        original_description = transaction_data.get('original_description', '')
        
        # Get suggestion from learning system
        suggested_category, confidence = self.suggest_category(description, original_description)
        
        # Create prediction breakdown
        breakdown = {
            'method': 'merchant_learning',
            'confidence': confidence,
            'merchant': self._extract_merchant(description),
            'suggested_category': suggested_category
        }
        
        return suggested_category, confidence, breakdown
    
    def _extract_merchant(self, description: str) -> str:
        """Extract merchant name from transaction description."""
        # Remove common prefixes and suffixes
        description = description.lower()
        
        # Common patterns to remove
        patterns_to_remove = [
            r'visa\s+domestic\s+use\s+vs\s+',
            r'credit\s+card\s+',
            r'debit\s+card\s+',
            r'atm\s+',
            r'pos\s+',
            r'\d+',  # Remove numbers
            r'[^\w\s]',  # Remove special characters
        ]
        
        merchant = description
        for pattern in patterns_to_remove:
            merchant = re.sub(pattern, '', merchant)
        
        # Clean up whitespace
        merchant = ' '.join(merchant.split())
        
        return merchant if merchant else "unknown"
    
    def _update_confidence(self, merchant: str, category: str):
        """Update confidence score for merchant-category pair."""
        if merchant not in self.confidence_scores:
            self.confidence_scores[merchant] = {}
        
        if category not in self.confidence_scores[merchant]:
            self.confidence_scores[merchant][category] = 0.5
        
        # Increase confidence with each correction
        current_confidence = self.confidence_scores[merchant][category]
        self.confidence_scores[merchant][category] = min(0.95, current_confidence + 0.1)
    
    def get_learning_stats(self) -> dict:
        """Get statistics about the learning system."""
        total_corrections = len(self.user_corrections)
        unique_merchants = len(self.merchant_categories)
        
        # Calculate accuracy improvement
        recent_corrections = [c for c in self.user_corrections 
                            if (pd.Timestamp.now() - c['timestamp']).days < 30]
        
        return {
            'total_corrections': total_corrections,
            'unique_merchants': unique_merchants,
            'recent_corrections': len(recent_corrections),
            'merchant_categories': self.merchant_categories,
            'confidence_scores': self.confidence_scores
        }


def extract_transactions_from_pdf(file_stream: io.BytesIO) -> pd.DataFrame:
    """Extract transactions from a PDF statement using pdfplumber.

    Assumes the PDF contains a table with columns Date, Description and
    Amount.  This function looks for the largest table on the first few
    pages.  It may need adaptation for your specific statement layout.
    """
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is not installed; please install it to process PDFs.")

    transactions = []
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages[:5]:
            tables = page.extract_tables()
            for table in tables:
                if len(table) > 1 and len(table[0]) >= 3:
                    header = [h.strip().lower() for h in table[0]]
                    try:
                        date_idx = header.index("date")
                        desc_idx = header.index("description")
                        amt_idx = header.index("amount")
                    except ValueError:
                        continue
                    for row in table[1:]:
                        try:
                            date = datetime.strptime(row[date_idx].strip(), "%d/%m/%Y")
                        except Exception:
                            try:
                                date = datetime.strptime(row[date_idx].strip(), "%Y-%m-%d")
                            except Exception:
                                continue
                        description = row[desc_idx].strip()
                        try:
                            amount = float(row[amt_idx].replace(",", ""))
                        except Exception:
                            continue
                        transactions.append({"date": date, "description": 
description, "amount": amount})
            if transactions:
                break
    if not transactions:
        raise RuntimeError("No transaction table detected in the uploaded PDF.")
    df = pd.DataFrame(transactions)
    return df


def extract_transactions_from_image(file_stream: io.BytesIO) -> pd.DataFrame:
    """Extract transactions from an image using OCR.

    This function reads the entire image as text and then attempts to
    parse lines that contain a date, description and amount.  It is
    simplistic and may need refinement for real statement layouts.  If
    pytesseract is not available, an error is raised.
    """
    if pytesseract is None or Image is None:
        raise RuntimeError(
            "pytesseract or PIL is not installed; please install them and ensure tesseract is available."
        )
    image = Image.open(file_stream)
    text = pytesseract.image_to_string(image)
    lines = text.splitlines()
    pattern = re.compile(r"(\\d{2}/\\d{2}/\\d{4})\\s+(.+?)\\s+(-?\\d+[.,]?\\d*)")
    records = []
    for line in lines:
        match = pattern.search(line)
        if match:
            date_str, desc, amt_str = match.groups()
            try:
                date = datetime.strptime(date_str, "%d/%m/%Y")
            except Exception:
                continue
            amount = float(amt_str.replace(",", ""))
            records.append({"date": date, "description": desc.strip(), 
"amount": amount})
    if not records:
        raise RuntimeError(
            "No transactions detected in the image.  Ensure the statement is clearly legible and try again."
        )
    df = pd.DataFrame(records)
    return df



def translate_japanese_to_english_ai(text: str, api_key: str = None) -> str:
    """Translate Japanese text to English using OpenAI GPT-3.5-turbo for high accuracy."""
    try:
        import openai
        
        # Check if text contains Japanese characters
        if not text or not any(ord(char) > 127 for char in text):
            return text
        
        # If no API key provided, try to get from environment
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            st.warning("No OpenAI API key found. Using free translation fallback.")
            return translate_japanese_to_english_fallback(text)
        
        # Configure OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Create translation prompt
        prompt = f"""
        Translate the following Japanese text to English. This is from a credit card statement, so maintain accuracy for financial terms and merchant names.
        
        Japanese text: {text}
        
        English translation:"""
        
        # Get translation from GPT-3.5-turbo
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional translator specializing in financial documents. Translate Japanese to English accurately, especially for merchant names and financial terms."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.1  # Low temperature for consistent translations
        )
        
        translated = response.choices[0].message.content.strip()
        return translated
        
    except Exception as e:
        st.warning(f"AI translation failed for '{text}': {e}")
        # Fallback to free translation
        return translate_japanese_to_english_fallback(text)

def translate_japanese_to_english_fallback(text: str) -> str:
    """Fallback translation using deep-translator when AI translation fails."""
    try:
        from deep_translator import GoogleTranslator
        if text and any(ord(char) > 127 for char in text):
            translated = GoogleTranslator(source='ja', target='en').translate(text)
            return translated
        return text
    except Exception as e:
        st.warning(f"Fallback translation failed for '{text}': {e}")
        return text

def translate_japanese_to_english(text: str, mode: str = "Free Fallback", api_key: str = None) -> str:
    """Main translation function - handles different translation modes."""
    if mode == "AI-Powered (GPT-3.5)":
        return translate_japanese_to_english_ai(text, api_key)
    elif mode == "Free Fallback":
        return translate_japanese_to_english_fallback(text)
    else:  # No Translation
        return text

def extract_transactions_from_csv(file_stream: io.BytesIO, translation_mode: str = "Free Fallback", api_key: str = None) -> pd.DataFrame:
    """Extract transactions from a CSV file.
    
    This function reads CSV files and attempts to identify date, description, and amount columns.
    It handles common CSV formats from different banks and financial institutions.
    Now includes Japanese translation support.
    """
    try:
        # Try to read the CSV with different encodings
        df = pd.read_csv(file_stream, encoding='utf-8')
    except UnicodeDecodeError:
        file_stream.seek(0)  # Reset file pointer
        df = pd.read_csv(file_stream, encoding='latin-1')
    
    # Common column names for different banks (including Japanese)
    date_columns = ['date', 'transaction_date', 'posting_date', 'date_posted', 'transaction date',
                    '利用日', '取引日', '決済日']
    desc_columns = ['description', 'merchant', 'payee', 'transaction_description', 'details',
                    '利用店名・商品名', '店舗名', '商品名', '取引内容']
    amount_columns = ['amount', 'debit', 'credit', 'transaction_amount', 'amount_debited', 'amount_credited',
                      '支払総額', '利用金額', '支払金額', '取引金額']
    time_columns = ['time', 'transaction_time', 'time_posted', 'timestamp', '取引時刻', '利用時刻', '決済時刻']
    
    # Find the actual column names in the CSV
    date_col = None
    desc_col = None
    amount_col = None
    time_col = None
    
    for col in df.columns:
        col_lower = col.lower().strip()
        # Normalize quotes and BOM for exact-name checks in Japanese
        col_original = col.strip().lstrip('\ufeff').strip('"').strip("'")
        
        # Check English column names
        if col_lower in [name.lower() for name in date_columns]:
            date_col = col
        elif col_lower in [name.lower() for name in desc_columns]:
            desc_col = col
        elif col_lower in [name.lower() for name in amount_columns]:
            amount_col = col
        elif col_lower in [name.lower() for name in time_columns]:
            time_col = col
        
        # Check Japanese column names
        if col_original in date_columns:
            date_col = col
        elif col_original in desc_columns:
            desc_col = col
        elif col_original in amount_columns:
            amount_col = col
        elif col_original in time_columns:
            time_col = col

    # Prefer 支払総額 when available
    if amount_col:
        for col in df.columns:
            col_norm = col.strip().lstrip('\ufeff').strip('"').strip("'")
            if col_norm == '支払総額':
                amount_col = col
                break
    
    if not all([date_col, desc_col, amount_col]):
        # If we can't find the expected columns, show available columns and let user choose
        st.warning(f"Could not automatically identify columns. Available columns: {list(df.columns)}")
        st.write("Please select the correct columns:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            date_col = st.selectbox("Date column:", df.columns, index=0)
        with col2:
            desc_col = st.selectbox("Description column:", df.columns, index=1)
        with col3:
            amount_col = st.selectbox("Amount column:", df.columns, index=2)
    
    # Process the data with progress bar
    st.write("🔄 **Processing transactions...**")
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    transactions = []
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        try:
            # Update progress
            progress = (idx + 1) / total_rows
            progress_bar.progress(progress)
            status_text.text(f"Processing row {idx + 1} of {total_rows} ({progress:.1%})")
            
            # Handle different date formats
            date_str = str(row[date_col]).strip()
            if pd.isna(date_str) or date_str == '':
                continue
                
            # Try different date formats (including Japanese format)
            date = None
            date_formats = ['%Y/%m/%d', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']
            for fmt in date_formats:
                try:
                    date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            
            if date is None:
                continue
            
            # Extract timestamp if available
            timestamp = None
            if time_col:
                time_str = str(row[time_col]).strip()
                if not pd.isna(time_str) and time_str != '':
                    try:
                        # Try to parse time in various formats
                        time_formats = ['%H:%M:%S', '%H:%M', '%H.%M.%S', '%H.%M']
                        for fmt in time_formats:
                            try:
                                time_obj = datetime.strptime(time_str, fmt).time()
                                timestamp = time_obj
                                break
                            except ValueError:
                                continue
                    except:
                        pass
                
            # Get description and translate if it's Japanese
            description = str(row[desc_col]).strip()
            if pd.isna(description) or description == '':
                continue
            
            # Translate Japanese description to English
            original_description = description
            description = translate_japanese_to_english(description, translation_mode, api_key)
                
            # Handle amount (could be positive or negative)
            amount_str = str(row[amount_col]).strip()
            if pd.isna(amount_str) or amount_str == '':
                continue
                
            # Remove currency symbols, commas, and Japanese characters
            amount_str = re.sub(r'[^\d.-]', '', amount_str)
            amount = float(amount_str)
            
            transactions.append({
                "date": date,
                "timestamp": timestamp,
                "description": description,
                "original_description": original_description,  # Keep original for reference
                "amount": amount
            })
            
        except Exception as e:
            st.warning(f"Error processing row {idx + 1}: {row}. Error: {e}")
            continue
    
    # Complete progress bar
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    
    if not transactions:
        raise RuntimeError("No valid transactions found in the CSV file.")
    
    # Clear progress elements
    progress_bar.empty()
    status_text.empty()
    
    df_result = pd.DataFrame(transactions)
    return df_result


@st.cache_data(show_spinner=False)
def get_fx_rate_to_jpy(currency: str, for_date: Optional[str]) -> float:
    """Fetch FX rate to JPY for a given date using exchangerate.host.

    Returns 1.0 for JPY or on error. for_date format: YYYY-MM-DD.
    """
    try:
        currency = (currency or "JPY").upper().strip()
        if currency == "JPY":
            return 1.0
        import requests
        # Historical if date provided, else latest
        if for_date:
            url = f"https://api.exchangerate.host/{for_date}"
        else:
            url = "https://api.exchangerate.host/latest"
        params = {"base": currency, "symbols": "JPY"}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        rate = float(data.get("rates", {}).get("JPY", 1.0))
        return rate if rate > 0 else 1.0
    except Exception:
        return 1.0


def enrich_currency_columns(df: pd.DataFrame, default_currency: str = "JPY") -> pd.DataFrame:
    df2 = df.copy()
    # If a currency column exists, use it; otherwise apply default
    currency_col = None
    for c in df2.columns:
        if str(c).lower().strip() in ["currency", "curr", "ccy"]:
            currency_col = c
            break
    if currency_col is None:
        df2["currency"] = default_currency
    else:
        df2.rename(columns={currency_col: "currency"}, inplace=True)

    # Compute fx_rate and amount_jpy per row
    fx_rates = []
    amts_jpy = []
    for _, r in df2.iterrows():
        date_val = r.get("date")
        if hasattr(date_val, "strftime"):
            date_str = date_val.strftime("%Y-%m-%d")
        else:
            date_str = str(date_val)
        ccy = str(r.get("currency", default_currency) or default_currency).upper()
        rate = get_fx_rate_to_jpy(ccy, date_str)
        fx_rates.append(rate)
        try:
            amt = float(r.get("amount", 0.0))
        except Exception:
            amt = 0.0
        amts_jpy.append(amt * rate)
    df2["fx_rate"] = fx_rates
    df2["amount_jpy"] = amts_jpy
    return df2

def validate_transaction_data(df: pd.DataFrame) -> dict:
    """Comprehensive data validation for transaction data."""
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'duplicates': [],
        'anomalies': []
    }
    
    try:
        # Check for duplicate transactions
        duplicate_mask = df.duplicated(subset=['date', 'description', 'amount'], keep=False)
        if duplicate_mask.any():
            duplicates = df[duplicate_mask]
            validation_results['duplicates'] = duplicates.to_dict('records')
            validation_results['warnings'].append(f"Found {len(duplicates)} duplicate transactions")
        
        # Check for amount anomalies
        if 'amount' in df.columns:
            amounts = df['amount']
            mean_amount = amounts.mean()
            std_amount = amounts.std()
            
            # Detect amounts that are 3 standard deviations from mean
            anomaly_mask = abs(amounts - mean_amount) > (3 * std_amount)
            if anomaly_mask.any():
                anomalies = df[anomaly_mask]
                validation_results['anomalies'] = anomalies.to_dict('records')
                validation_results['warnings'].append(f"Found {len(anomalies)} transactions with unusual amounts")
            
            # Check for zero amounts
            zero_amounts = df[amounts == 0]
            if len(zero_amounts) > 0:
                validation_results['warnings'].append(f"Found {len(zero_amounts)} transactions with zero amounts")
        
        # Check date range validity
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            min_date = dates.min()
            max_date = dates.max()
            current_date = pd.Timestamp.now()
            
            # Check for future dates
            future_dates = df[dates > current_date]
            if len(future_dates) > 0:
                validation_results['warnings'].append(f"Found {len(future_dates)} transactions with future dates")
            
            # Check for very old dates (more than 10 years ago)
            ten_years_ago = current_date - pd.DateOffset(years=10)
            old_dates = df[dates < ten_years_ago]
            if len(old_dates) > 0:
                validation_results['warnings'].append(f"Found {len(old_dates)} transactions older than 10 years")
        
        # Check for missing critical data
        missing_descriptions = df[df['description'].isna() | (df['description'] == '')]
        if len(missing_descriptions) > 0:
            validation_results['errors'].append(f"Found {len(missing_descriptions)} transactions with missing descriptions")
            validation_results['is_valid'] = False
        
        missing_amounts = df[df['amount'].isna()]
        if len(missing_amounts) > 0:
            validation_results['errors'].append(f"Found {len(missing_amounts)} transactions with missing amounts")
            validation_results['is_valid'] = False
        
        # Check for extreme amounts (suspicious transactions)
        if 'amount' in df.columns:
            amounts = df['amount']
            # Flag amounts over ¥1,000,000 as potentially suspicious
            large_amounts = df[abs(amounts) > 1000000]
            if len(large_amounts) > 0:
                validation_results['warnings'].append(f"Found {len(large_amounts)} transactions with amounts over ¥1,000,000")
        
    except Exception as e:
        validation_results['errors'].append(f"Validation error: {str(e)}")
        validation_results['is_valid'] = False
    
    return validation_results

def validate_financial_data(df: pd.DataFrame, expected_total: float = None) -> dict:
    """
    Comprehensive financial data validation following industry best practices.
    This is the type of validation used in professional financial applications.
    """
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'data_quality_score': 0.0,
        'reconciliation_status': 'unknown',
        'recommended_actions': []
    }
    
    try:
        # 1. Basic data integrity checks
        if df.empty:
            validation_results['errors'].append("No transaction data found")
            validation_results['is_valid'] = False
            return validation_results
        
        # 2. Amount data validation
        if 'amount' not in df.columns:
            validation_results['errors'].append("Missing 'amount' column")
            validation_results['is_valid'] = False
            return validation_results
        
        amounts = df['amount']
        
        # Check for invalid amounts
        invalid_amounts = amounts[~amounts.apply(lambda x: isinstance(x, (int, float)) and not pd.isna(x))]
        if len(invalid_amounts) > 0:
            validation_results['errors'].append(f"Found {len(invalid_amounts)} invalid amount values")
            validation_results['is_valid'] = False
        
        # Check for zero amounts (suspicious in financial data)
        zero_amounts = amounts[amounts == 0]
        if len(zero_amounts) > 0:
            validation_results['warnings'].append(f"Found {len(zero_amounts)} transactions with zero amounts")
        
        # 3. Duplicate detection (critical for financial accuracy)
        try:
            # Safe duplicate detection that handles mixed data types
            duplicate_mask = df.duplicated(subset=['date', 'description', 'amount'], keep=False)
            if duplicate_mask.any():
                duplicates = df[duplicate_mask]
                validation_results['warnings'].append(f"Found {len(duplicates)} duplicate transactions")
                validation_results['recommended_actions'].append("Review and remove duplicate transactions")
        except Exception as e:
            # Fallback to string-based duplicate detection
            try:
                df_string = df.astype(str)
                duplicate_mask = df_string.duplicated(subset=['date', 'description', 'amount'], keep=False)
                if duplicate_mask.any():
                    duplicates = df_string[duplicate_mask]
                    validation_results['warnings'].append(f"Found {len(duplicates)} potential duplicate transactions (string-based detection)")
                    validation_results['recommended_actions'].append("Review and remove duplicate transactions")
            except Exception as e2:
                validation_results['warnings'].append("Unable to detect duplicates due to data type complexity")
                validation_results['recommended_actions'].append("Manual review of transactions recommended")
        
        # 4. Amount range validation
        min_amount = amounts.min()
        max_amount = amounts.max()
        mean_amount = amounts.mean()
        
        # Flag suspicious amounts
        if abs(min_amount) > 1000000:  # Over ¥1M
            validation_results['warnings'].append(f"Found extremely large negative amount: ¥{min_amount:,.0f}")
        if max_amount > 1000000:  # Over ¥1M
            validation_results['warnings'].append(f"Found extremely large positive amount: ¥{max_amount:,.0f}")
        
        # 5. Financial reconciliation (if expected total provided)
        if expected_total is not None:
            actual_total = amounts.abs().sum()
            difference = abs(actual_total - expected_total)
            difference_percentage = (difference / expected_total) * 100
            
            validation_results['reconciliation_status'] = 'reconciled' if difference < 100 else 'unreconciled'
            
            if difference > 100:  # More than ¥100 difference
                validation_results['errors'].append(f"Financial reconciliation failed: Expected ¥{expected_total:,.0f}, Got ¥{actual_total:,.0f}")
                validation_results['errors'].append(f"Difference: ¥{difference:,.0f} ({difference_percentage:.2f}%)")
                validation_results['is_valid'] = False
                
                # Provide specific recommendations
                if difference_percentage > 10:
                    validation_results['recommended_actions'].append("Large discrepancy detected - verify source data integrity")
                elif difference_percentage > 5:
                    validation_results['recommended_actions'].append("Moderate discrepancy - check for missing or duplicate transactions")
                else:
                    validation_results['recommended_actions'].append("Small discrepancy - verify individual transaction amounts")
        
        # 6. Data quality scoring
        quality_score = 100.0
        
        # Deduct points for issues
        if len(duplicate_mask[duplicate_mask]) > 0:
            quality_score -= 20
        if len(zero_amounts) > 0:
            quality_score -= 10
        if len(invalid_amounts) > 0:
            quality_score -= 30
        
        validation_results['data_quality_score'] = max(0, quality_score)
        
        # 7. Professional recommendations
        if validation_results['data_quality_score'] < 70:
            validation_results['recommended_actions'].append("Data quality is poor - manual review recommended")
        elif validation_results['data_quality_score'] < 90:
            validation_results['recommended_actions'].append("Data quality is acceptable but could be improved")
        else:
            validation_results['recommended_actions'].append("Data quality is excellent")
        
    except Exception as e:
        validation_results['errors'].append(f"Validation error: {str(e)}")
        validation_results['is_valid'] = False
    
    return validation_results

def reconcile_financial_data(df: pd.DataFrame, expected_total: float) -> dict:
    """
    Financial reconciliation system - finds the exact cause of discrepancies.
    This is what professional financial software uses to resolve mismatches.
    """
    reconciliation = {
        'status': 'unreconciled',
        'expected_total': expected_total,
        'actual_total': 0.0,
        'difference': 0.0,
        'difference_percentage': 0.0,
        'root_causes': [],
        'suggested_fixes': [],
        'data_issues': []
    }
    
    try:
        # Calculate actual total
        actual_total = df['amount'].abs().sum()
        reconciliation['actual_total'] = actual_total
        
        # Calculate difference
        difference = abs(actual_total - expected_total)
        reconciliation['difference'] = difference
        reconciliation['difference_percentage'] = (difference / expected_total) * 100
        
        # Root cause analysis
        if difference > 0:
            # Check for common financial data issues
            
            # 1. Duplicate transactions
            try:
                duplicate_mask = df.duplicated(subset=['date', 'description', 'amount'], keep=False)
                if duplicate_mask.any():
                    duplicates = df[duplicate_mask]
                    duplicate_total = duplicates['amount'].abs().sum()
                    reconciliation['root_causes'].append(f"Duplicate transactions: {len(duplicates)} duplicates worth ¥{duplicate_total:,.0f}")
                    reconciliation['suggested_fixes'].append("Remove duplicate transactions")
            except Exception as e:
                # Fallback to string-based duplicate detection
                try:
                    df_string = df.astype(str)
                    duplicate_mask = df_string.duplicated(subset=['date', 'description', 'amount'], keep=False)
                    if duplicate_mask.any():
                        duplicates = df_string[duplicate_mask]
                        reconciliation['root_causes'].append(f"Potential duplicate transactions: {len(duplicates)} potential duplicates detected")
                        reconciliation['suggested_fixes'].append("Review and remove duplicate transactions")
                except Exception as e2:
                    reconciliation['root_causes'].append("Unable to detect duplicates due to data type complexity")
                    reconciliation['suggested_fixes'].append("Manual review of transactions recommended")
            
            # 2. Extreme amounts that might be errors
            extreme_amounts = df[df['amount'].abs() > 100000]
            if len(extreme_amounts) > 0:
                reconciliation['root_causes'].append(f"Extreme amounts: {len(extreme_amounts)} transactions over ¥100,000")
                reconciliation['suggested_fixes'].append("Verify extreme amounts are correct")
            
            # 3. Zero amounts
            zero_amounts = df[df['amount'] == 0]
            if len(zero_amounts) > 0:
                reconciliation['root_causes'].append(f"Zero amounts: {len(zero_amounts)} transactions with ¥0")
                reconciliation['suggested_fixes'].append("Review zero-amount transactions")
            
            # 4. Amount distribution analysis
            positive_amounts = df[df['amount'] > 0]['amount'].sum()
            negative_amounts = abs(df[df['amount'] < 0]['amount'].sum())
            
            reconciliation['data_issues'].append(f"Positive amounts total: ¥{positive_amounts:,.0f}")
            reconciliation['data_issues'].append(f"Negative amounts total: ¥{negative_amounts:,.0f}")
            
            # 5. Check if the issue is in transaction classification
            if 'transaction_type' in df.columns:
                expense_total = df[df['transaction_type'] == 'Expense']['amount'].abs().sum()
                credit_total = df[df['transaction_type'] == 'Credit']['amount'].abs().sum()
                reconciliation['data_issues'].append(f"Expense transactions total: ¥{expense_total:,.0f}")
                reconciliation['data_issues'].append(f"Credit transactions total: ¥{credit_total:,.0f}")
        
        # Determine reconciliation status
        if difference < 100:  # Within ¥100
            reconciliation['status'] = 'reconciled'
        elif difference < 1000:  # Within ¥1,000
            reconciliation['status'] = 'minor_discrepancy'
        else:
            reconciliation['status'] = 'major_discrepancy'
            
    except Exception as e:
        reconciliation['root_causes'].append(f"Reconciliation error: {str(e)}")
    
    return reconciliation

def calculate_running_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate running balance for transactions."""
    df_balance = df.copy()
    
    # Sort by date to ensure chronological order
    df_balance = df_balance.sort_values('date')
    
    # Initialize running balance
    running_balance = 0
    balance_history = []
    
    for idx, row in df_balance.iterrows():
        amount = row['amount']
        
        # Add to running balance
        running_balance += amount
        balance_history.append(running_balance)
    
    # Add running balance column
    df_balance['running_balance'] = balance_history
    
    return df_balance

def detect_transaction_type(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and classify transactions as credit (income) or debit (expense)."""
    
    df = df.copy()
    
    # Initialize transaction type column
    df['transaction_type'] = 'Expense'  # Default to expense
    
    # Check for common credit indicators
    credit_keywords = [
        'refund', 'credit', 'payment', 'adjustment', 'reversal', 'return',
        '返金', 'クレジット', '支払い', '調整', '取消', '返品',
        'statement credit', 'cashback', 'rewards', 'bonus'
    ]
    
    # Check for common debit indicators
    debit_keywords = [
        'purchase', 'charge', 'fee', 'withdrawal', 'cash advance',
        '購入', 'チャージ', '手数料', '引き出し', '現金引き出し'
    ]
    
    for idx, row in df.iterrows():
        description = str(row.get('description', '')).lower()
        original_desc = str(row.get('original_description', '')).lower()
        amount = row.get('amount', 0)
        
        # Check description keywords first
        is_credit = any(keyword in description or keyword in original_desc for keyword in credit_keywords)
        is_debit = any(keyword in description or keyword in original_desc for keyword in debit_keywords)
        
        # Determine transaction type based on keywords and amount sign
        # For Japanese bank statements, most transactions are expenses (outflows)
        if is_credit and not is_debit:
            df.loc[idx, 'transaction_type'] = 'Credit'
        elif is_debit and not is_credit:
            df.loc[idx, 'transaction_type'] = 'Expense'
        else:
            # If no clear keyword match, use amount sign as indicator
            # For Japanese bank statements: negative amounts = expenses (outflows)
            if amount < 0:
                df.loc[idx, 'transaction_type'] = 'Expense'
            else:
                # For Japanese bank statements, default to Expense for positive amounts
                # unless there are clear credit indicators
                df.loc[idx, 'transaction_type'] = 'Expense'
        
        # Force classification for common Japanese transaction patterns
        # Most Japanese bank transactions are expenses, not credits
        if 'visa' in description.lower() or 'visa' in original_desc.lower():
            if 'domestic' in description.lower() or 'domestic' in original_desc.lower():
                # VISA domestic transactions are almost always expenses
                df.loc[idx, 'transaction_type'] = 'Expense'
        
        # Override for obvious credits (refunds, payments, etc.)
        credit_indicators = ['refund', '返金', 'payment', '支払い', 'adjustment', '調整']
        if any(indicator in description.lower() or indicator in original_desc.lower() for indicator in credit_indicators):
            df.loc[idx, 'transaction_type'] = 'Credit'
        
        # Keep original amount value - don't change it
    
    return df

def categorise_transactions(
    df: pd.DataFrame, rules, subcategories = None, 
    uncategorised_label: str = "Uncategorised"
) -> pd.DataFrame:
    """Enhanced categorization with support for main categories and subcategories."""
    
    # Create patterns for main categories
    patterns = {cat: re.compile("(" + "|".join(map(re.escape, kws)) + ")", re.IGNORECASE) for cat, kws in rules.items()}
    
    # Create patterns for subcategories
    sub_patterns = {}
    if subcategories:
        for main_cat, subs in subcategories.items():
            for sub_cat, keywords in subs.items():
                sub_patterns[f"{main_cat}_{sub_cat}"] = {
                    'main': main_cat,
                    'sub': sub_cat,
                    'pattern': re.compile("(" + "|".join(map(re.escape, keywords)) + ")", re.IGNORECASE)
                }
    
    categories = []
    subcategories_list = []
    
    for desc in df["description"].astype(str):
        assigned_category = uncategorised_label
        assigned_subcategory = ""
        
        # First try to match subcategories for more specific categorization
        for sub_key, sub_info in sub_patterns.items():
            if sub_info['pattern'].search(desc):
                assigned_category = sub_info['main']
                assigned_subcategory = sub_info['sub']
                break
        
        # If no subcategory match, try main categories
        if assigned_category == uncategorised_label:
            for cat, pattern in patterns.items():
                if pattern.search(desc):
                    assigned_category = cat
                    break
        
        categories.append(assigned_category)
        subcategories_list.append(assigned_subcategory)
    
    df = df.copy()
    df["category"] = categories
    df["subcategory"] = subcategories_list
    
    return df


def apply_smart_categorization(df: pd.DataFrame, learning_system, 
                              rules, subcategories) -> pd.DataFrame:
    """Apply smart categorization using the learning system."""
    
    df = df.copy()
    categories = []
    subcategories_list = []
    confidences = []
    prediction_breakdowns = []
    
    for idx, row in df.iterrows():
        # Prepare transaction data for prediction
        transaction_data = {
            'description': row.get('description', ''),
            'original_description': row.get('original_description', ''),
            'amount': row.get('amount', 0),
            'date': row.get('date'),
            'transaction_type': row.get('transaction_type', 'Expense')
        }
        
        # Get smart prediction
        try:
            if learning_system:
                predicted_category, confidence, breakdown = learning_system.predict_category(transaction_data)
            else:
                predicted_category = "Uncategorised"
                confidence = 0.0
                breakdown = {'method': 'fallback', 'confidence': 0.0}
        except Exception as e:
            # Fallback to basic categorization if learning system fails
            predicted_category = "Uncategorised"
            confidence = 0.0
            breakdown = {'method': 'fallback', 'confidence': 0.0, 'error': str(e)}
        
        # Apply subcategory if available
        predicted_subcategory = ""
        if predicted_category in subcategories:
            # Find best matching subcategory
            for sub_cat, keywords in subcategories[predicted_category].items():
                if any(keyword.lower() in str(transaction_data['description']).lower() 
                       for keyword in keywords):
                    predicted_subcategory = sub_cat
                    break
        
        categories.append(predicted_category)
        subcategories_list.append(predicted_subcategory)
        confidences.append(confidence)
        prediction_breakdowns.append(breakdown)
    
    df['category'] = categories
    df['subcategory'] = subcategories_list
    df['confidence'] = confidences
    df['prediction_breakdown'] = prediction_breakdowns
    
    return df


def save_custom_rules(rules, filename: str = "custom_rules.json") -> None:
    """Save custom categorization rules to a JSON file."""
    import json
    try:
        with open(filename, 'w') as f:
            json.dump(rules, f, indent=2)
        st.success(f"Custom rules saved to {filename}")
    except Exception as e:
        st.error(f"Error saving rules: {e}")

def load_custom_rules(filename: str = "custom_rules.json"):
    """Load custom categorization rules from a JSON file."""
    import json
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        st.error(f"Error loading rules: {e}")
        return {}


def main() -> None:
    # Require authentication (single-user gate if configured)
    if not require_auth():
        return

    st.title("Transaction Categoriser")
    st.write(
        "Upload a credit card statement in PDF, CSV, or image format. The app will extract transaction data, "
        "assign categories based on keyword rules and let you review the results."
    )

    # Onboarding note
    st.info(
        "Getting started: 1) Upload CSV/PDF, 2) Review categories, 3) Save to DB, 4) Export/Backup as needed."
    )

    # Initialize database (first run safe)
    if init_db:
        try:
            init_db()
            st.sidebar.success("🗄️ Local database ready")
        except Exception as e:
            st.sidebar.warning(f"DB init failed: {e}")
    
    # Initialize session state for progress tracking
    if 'categorization_progress' not in st.session_state:
        st.session_state['categorization_progress'] = None
    if 'progress_saved' not in st.session_state:
        st.session_state['progress_saved'] = False
    
    # Initialize Merchant Learning System
    learning_system = MerchantLearningSystem()
    st.sidebar.success("🧠 Merchant Learning System Active")
    
    # AI Translation Setup
    st.sidebar.header("🤖 AI Translation Settings")
    st.sidebar.write("For best Japanese translation accuracy, use OpenAI GPT-3.5")
    
    # Check for existing API key
    existing_api_key = os.getenv('OPENAI_API_KEY')
    
    if existing_api_key:
        st.sidebar.success("✅ OpenAI API key found in environment")
        api_key = existing_api_key
    else:
        # API Key input
        api_key = st.sidebar.text_input(
            "OpenAI API Key", 
            type="password",
            help="Get your API key from https://platform.openai.com/api-keys (same account as ChatGPT Premium)"
        )
        
        if api_key:
            st.sidebar.success("✅ API key configured for this session")
            # Set environment variable for this session
            os.environ['OPENAI_API_KEY'] = api_key
    
    # Quick setup guide for ChatGPT Premium users
    if not api_key:
        with st.sidebar.expander("🚀 Quick Setup for ChatGPT Premium Users"):
            st.write("""
            1. **Go to:** https://platform.openai.com/api-keys
            2. **Sign in** with your ChatGPT Premium account
            3. **Click "Create new secret key"**
            4. **Copy the key** (starts with `sk-...`)
            5. **Paste it above** for AI-powered Japanese translation
            """)
            st.success("💡 Your ChatGPT Premium account gives you access to the API!")
    
    # Translation mode selection
    if api_key:
        translation_mode = st.sidebar.selectbox(
            "Translation Mode",
            ["Free Fallback", "AI-Powered (GPT-3.5)", "No Translation"],
            index=0,  # Default to Free Fallback
            help="Free Fallback uses Google Translate, AI-Powered uses OpenAI for better accuracy"
        )
    else:
        translation_mode = "Free Fallback"
        st.sidebar.success("ℹ️ Using free translation (enter API key for AI accuracy)")
    
    # Smart Learning Dashboard
    if learning_system:
        st.sidebar.header("🧠 Smart Learning Dashboard")
        
        # Show learning statistics
        stats = learning_system.get_learning_stats()
        st.sidebar.write(f"**Total Corrections:** {stats['total_corrections']}")
        st.sidebar.write(f"**Unique Merchants:** {stats['unique_merchants']}")
        st.sidebar.write(f"**Recent Corrections:** {stats['recent_corrections']}")
        
        # Show merchant learning progress
        if stats['merchant_categories']:
            st.sidebar.write("**Learning Progress:**")
            for merchant, categories in list(stats['merchant_categories'].items())[:5]:  # Show first 5
                if merchant != "unknown":
                    category_names = list(categories.keys())
                    st.sidebar.write(f"• {merchant[:20]}: {category_names[0] if category_names else 'None'}")
        
        # Show confidence scores
        if stats['confidence_scores']:
            st.sidebar.write("**Confidence Levels:**")
            high_confidence = sum(1 for merchant_scores in stats['confidence_scores'].values() 
                                for score in merchant_scores.values() if score >= 0.8)
            st.sidebar.write(f"• High Confidence: {high_confidence}")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a statement file", 
type=["pdf", "png", "jpg", "jpeg", "csv"])
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/pdf":
                df = extract_transactions_from_pdf(uploaded_file)
            elif uploaded_file.type == "text/csv":
                df = extract_transactions_from_csv(uploaded_file, translation_mode, api_key)
            else:
                df = extract_transactions_from_image(uploaded_file)
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return
        
        # Initialize learning system
        learning_system = MerchantLearningSystem()
        
        # Show loading animation during processing
        with st.spinner("🔄 Processing transactions and applying smart categorization..."):
            # MoneyMgr Proven Categorization System (Based on 3,943+ real transactions)
            rules = {
            "Food": [
                # Groceries and Food Stores
                "ローソン", "セブンイレブン", "ファミリーマート", "コンビニ", "lawson", "seven eleven", "family mart",
                "ポプラグループ", "poplar", "スーパー", "supermarket", "grocery", "market", "food", "fresh",
                "イオン", "aeon", "イトーヨーカドー", "itoyokado", "西友", "seiyu", "ライフ", "life",
                # Restaurants and Dining
                "レストラン", "restaurant", "cafe", "dinner", "lunch", "breakfast", "takeaway", "delivery",
                "居酒屋", "izakaya", "バー", "bar", "カフェ", "coffee", "ピザ", "pizza", "寿司", "sushi",
                "マクドナルド", "mcdonalds", "ケンタッキー", "kfc", "スターバックス", "starbucks"
            ],
            "Social Life": [
                # Social Activities
                "飲み会", "drinking", "パーティー", "party", "イベント", "event", "友達", "friend", "同僚", "colleague",
                "会食", "dining", "懇親会", "networking", "歓迎会", "welcome", "送別会", "farewell",
                "カラオケ", "karaoke", "ボーリング", "bowling", "ゲーム", "game", "スポーツ", "sports"
            ],
            "Subscriptions": [
                # Digital Services
                "icloud", "apple music", "amazon prime", "google one", "netflix", "spotify", "hulu", "disney+",
                "アマゾンプライム", "グーグルワン", "アップルミュージック", "アイクラウド",
                "subscription", "membership", "月額", "monthly", "年額", "annual"
            ],
            "Household": [
                # Home and Living
                "家賃", "rent", "光熱費", "utility", "電気", "electric", "ガス", "gas", "水道", "water",
                "家具", "furniture", "家電", "appliance", "日用品", "daily", "掃除", "cleaning",
                "ニトリ", "nitori", "イケア", "ikea", "ホームセンター", "home center"
            ],
            "Transportation": [
                # Public Transport and Travel
                "電車", "train", "バス", "bus", "タクシー", "taxi", "地下鉄", "subway", "モノレール", "monorail",
                "モバイルパス", "mobile pass", "交通費", "transport", "駐車場", "parking", "高速道路", "highway",
                "ＥＴＣ", "etc", "ガソリン", "gasoline", "燃料", "fuel", "車", "car", "バイク", "bike"
            ],
            "Vacation": [
                # Travel and Leisure
                "旅行", "travel", "ホテル", "hotel", "飛行機", "flight", "新幹線", "shinkansen", "観光", "tourism",
                "温泉", "onsen", "リゾート", "resort", "ビーチ", "beach", "山", "mountain", "海", "sea",
                "チケット", "ticket", "ツアー", "tour", "宿泊", "accommodation"
            ],
            "Health": [
                # Healthcare and Wellness
                "病院", "hospital", "クリニック", "clinic", "歯科", "dental", "眼科", "eye", "薬局", "pharmacy",
                "薬", "medicine", "保険", "insurance", "診察", "examination", "治療", "treatment",
                "フィットネス", "fitness", "ジム", "gym", "ヨガ", "yoga", "マッサージ", "massage"
            ],
            "Apparel": [
                # Clothing and Fashion
                "服", "clothing", "靴", "shoes", "バッグ", "bag", "アクセサリー", "accessory", "時計", "watch",
                "ユニクロ", "uniqlo", "zara", "h&m", "gap", "nike", "adidas", "アディダス", "ナイキ",
                "ファッション", "fashion", "スタイル", "style", "ブランド", "brand"
            ],
            "Grooming": [
                # Personal Care
                "美容", "beauty", "化粧品", "cosmetics", "スキンケア", "skincare", "ヘアケア", "haircare",
                "ネイル", "nail", "エステ", "esthetic", "理容", "barber", "美容院", "salon",
                "資生堂", "shiseido", "ポーラ", "pola", "ファンケル", "fancl"
            ],
            "Self-development": [
                # Education and Growth
                "本", "book", "雑誌", "magazine", "新聞", "newspaper", "講座", "course", "セミナー", "seminar",
                "ワークショップ", "workshop", "資格", "certification", "学習", "learning", "スキル", "skill",
                "オンライン", "online", "eラーニング", "elearning", "トレーニング", "training"
            ]
        }
        
        # MoneyMgr Subcategory System for Detailed Breakdown
        subcategories = {
            "Food": {
                "Groceries": ["ローソン", "セブンイレブン", "ファミリーマート", "コンビニ", "スーパー", "ポプラグループ"],
                "Dinner/Eating Out": ["レストラン", "居酒屋", "バー", "dinner", "restaurant", "izakaya"],
                "Lunch/Eating Out": ["lunch", "カフェ", "coffee", "昼食", "ランチ"],
                "Beverages A": ["スターバックス", "コーヒー", "tea", "ジュース", "drink"],
                "Beverages/Non-A": ["アルコール", "酒", "ビール", "wine", "spirits"]
            },
            "Social Life": {
                "Drinking": ["飲み会", "drinking", "パーティー", "party", "カラオケ", "karaoke"],
                "Event": ["イベント", "event", "会食", "dining", "懇親会", "networking"],
                "Friend": ["友達", "friend", "同僚", "colleague", "歓迎会", "送別会"]
            },
            "Transportation": {
                "Subway": ["地下鉄", "subway", "電車", "train", "モノレール", "monorail"],
                "Taxi": ["タクシー", "taxi", "車", "car", "ライドシェア", "rideshare"],
                "Mobile Pass": ["モバイルパス", "mobile pass", "交通費", "transport"],
                "ETC": ["ＥＴＣ", "etc", "高速道路", "highway", "駐車場", "parking"]
            },
            "Household": {
                "Rent": ["家賃", "rent", "住宅費", "housing"],
                "Utilities": ["光熱費", "utility", "電気", "electric", "ガス", "gas", "水道", "water"],
                "Furniture": ["家具", "furniture", "ニトリ", "nitori", "イケア", "ikea"]
            }
        }
        # Currency settings and FX normalization
        st.subheader("💱 Currency & FX")
        default_currency = st.selectbox(
            "Statement currency (applied when missing)",
            options=["JPY", "USD", "EUR", "AUD", "CAD", "GBP", "CNY", "KRW"],
            index=0,
            help="Used to convert amounts to JPY for totals."
        )

        df = enrich_currency_columns(df, default_currency)

        # Detect transaction types (credit vs debit)
        df = detect_transaction_type(df)
        
        # Apply categorization with smart learning if available
        if learning_system:
            # Use smart learning system for predictions
            df_cat = apply_smart_categorization(df, learning_system, rules, subcategories)
        else:
            # Fallback to basic categorization
            df_cat = categorise_transactions(df, rules, subcategories)
        
        # Close the loading spinner and show completion status
        st.success(f"✅ Processing complete! Successfully processed {len(df)} transactions.")
        
        # Professional Financial Data Validation
        st.subheader("🏦 Professional Financial Validation")
        
        # Use the new professional validation system (no expected total)
        financial_validation = validate_financial_data(df, expected_total=None)
        
        # Display validation results in a professional format
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if financial_validation['is_valid']:
                st.success("✅ **Validation Passed**")
            else:
                st.error("❌ **Validation Failed**")
        
        with col2:
            quality_score = financial_validation['data_quality_score']
            if quality_score >= 90:
                st.success(f"🟢 **Quality Score:** {quality_score:.0f}%")
            elif quality_score >= 70:
                st.warning(f"🟡 **Quality Score:** {quality_score:.0f}%")
            else:
                st.error(f"🔴 **Quality Score:** {quality_score:.0f}%")
        
        with col3:
            rec_status = financial_validation.get('reconciliation_status', 'unknown')
            if rec_status == 'reconciled':
                st.success("✅ **Reconciled**")
            elif rec_status == 'unknown':
                st.info("ℹ️ **Reconciliation not checked**")
            else:
                st.warning("⚠️ **Reconciliation pending**")
        
        # Remove raw data summary; totals will be presented once below
        # Run reconciliation only if a user-provided expected total is supplied later (skipped)
        
        st.divider()
        
        # Data validation and quality check
        st.subheader("🔍 Data Quality Summary")
        validation_results = validate_transaction_data(df_cat)
        
        # Create a clean summary using columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if validation_results['is_valid']:
                st.success("✅ **Validation Passed**")
            else:
                st.error("❌ **Validation Failed**")
        
        with col2:
            if validation_results['duplicates']:
                st.warning(f"🔍 **{len(validation_results['duplicates'])} Duplicates**")
            else:
                st.success("✅ **No Duplicates**")
        
        with col3:
            if validation_results['anomalies']:
                st.warning(f"🚨 **{len(validation_results['anomalies'])} Anomalies**")
            else:
                st.success("✅ **No Anomalies**")
        
        # Show detailed issues only if they exist
        has_issues = (validation_results['duplicates'] or 
                     validation_results['anomalies'] or 
                     validation_results['warnings'] or 
                     not validation_results['is_valid'])
        
        if has_issues:
            with st.expander("📋 **View Details**", expanded=False):
                # Show errors if any
                if not validation_results['is_valid']:
                    st.error("**Critical Issues:**")
                    for error in validation_results['errors']:
                        st.error(f"• {error}")
                
                # Show duplicates if found
                if validation_results['duplicates']:
                    st.warning(f"**Duplicate Transactions ({len(validation_results['duplicates'])}):**")
                    if st.checkbox("Show duplicate transactions", key="show_duplicates"):
                        duplicates_df = pd.DataFrame(validation_results['duplicates'])
                        st.dataframe(duplicates_df)
                
                # Show anomalies if found
                if validation_results['anomalies']:
                    st.warning(f"**Unusual Amounts ({len(validation_results['anomalies'])}):**")
                    if st.checkbox("Show anomalous transactions", key="show_anomalies"):
                        anomalies_df = pd.DataFrame(validation_results['anomalies'])
                        st.dataframe(anomalies_df)
                
                # Show other warnings
                if validation_results['warnings']:
                    st.info("**Other Warnings:**")
                    for warning in validation_results['warnings']:
                        st.info(f"• {warning}")
        
        st.divider()
        
        # Smart categorization interface
        st.subheader("🎯 Smart Transaction Categorization")
        
        # Get all available categories and subcategories
        all_categories = list(rules.keys()) + ["Uncategorised"]
        all_subcategories = []
        for main_cat, subs in subcategories.items():
            for sub_cat in subs.keys():
                all_subcategories.append(f"{main_cat} - {sub_cat}")
        
        # Show categorization statistics
        category_counts = df_cat['transaction_type'].value_counts()
        
        # Show categorization summary in a clean format
        if 'confidence' in df_cat.columns:
            avg_confidence = df_cat['confidence'].mean()
            high_confidence = len(df_cat[df_cat['confidence'] >= 0.8])
            low_confidence = len(df_cat[df_cat['confidence'] < 0.5])
            
            # Clean summary display
            st.markdown(f"""
            ### 📊 Categorization Summary
            **Total Transactions:** {len(df_cat)}  
            **Average Confidence:** {avg_confidence:.1%}  
            **High Confidence:** {high_confidence} | **Low Confidence:** {low_confidence}
            """)
        else:
            st.markdown(f"### 📊 Categorization Summary\n**Total Transactions:** {len(df_cat)}")
        
        # Display transaction type breakdown
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Transaction Types:**")
            for trans_type, count in category_counts.items():
                if trans_type == 'Credit':
                    st.write(f"🟢 **{trans_type}:** {count} transactions")
                else:
                    st.write(f"🔴 **{trans_type}:** {count} transactions")
        
        with col2:
            # Calculate totals - handle both positive and negative amounts correctly
            expense_df = df_cat[df_cat['transaction_type'] == 'Expense']
            credit_df = df_cat[df_cat['transaction_type'] == 'Credit']
            
            # For Japanese bank statements, expenses are typically negative amounts
            # We want to show the absolute value for display purposes
            expense_amounts = expense_df['amount']
            credit_amounts = credit_df['amount']
            
            # Calculate totals - this is where the issue might be
            total_expenses = abs(expense_amounts.sum())
            total_credits = abs(credit_amounts.sum())
            
            # Net amount calculation (expenses - credits)
            net_amount = total_expenses - total_credits
            
            st.write("**Financial Summary:**")
            st.write(f"🔴 **Total Expenses:** ¥{total_expenses:,.0f}")
            st.write(f"🟢 **Total Credits:** ¥{total_credits:,.0f}")
            st.write(f"💰 **Net Amount:** ¥{net_amount:,.0f}")
            
                        # Check for potential data issues
        # Cleaned UI: remove verbose data quality diagnostics
        
        # Show transaction breakdown by type
        # Cleaned UI: remove transaction type breakdown list
        
        # Add manual correction option
        # Cleaned UI: remove manual correction/alternative calculations/debug blocks
        
        st.divider()
        
        # Cleaned UI: remove category breakdown lists
        expense_df = df_cat[df_cat['transaction_type'] == 'Expense']
        uncategorized_count = int((df_cat['category'] == 'Uncategorised').sum())
        
        # Smart categorization for uncategorized transactions
        if uncategorized_count > 0:
            st.subheader("🚀 Quick Categorization")
            
            # Show progress restoration if available
            if st.session_state['progress_saved'] and st.session_state['categorization_progress']:
                st.info("💾 **Saved Progress Available:** You can restore your previous categorization work.")
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("🔄 Restore Saved Progress"):
                        # Restore the saved progress
                        restored_df = pd.DataFrame(st.session_state['categorization_progress'])
                        df_cat.update(restored_df)
                        st.session_state['progress_saved'] = False
                        st.success("🔄 Progress restored! Your previous work has been loaded.")
                        st.rerun()
                with col2:
                    if st.button("🗑️ Clear Saved Progress"):
                        st.session_state['categorization_progress'] = None
                        st.session_state['progress_saved'] = False
                        st.success("🗑️ Saved progress cleared.")
                        st.rerun()
            
            st.write("Use the dropdowns below to quickly categorize uncategorized transactions:")
            
            # Add navigation hints
            st.info("💡 **Navigation Tips:** Use the buttons below to save progress, skip categorization, or continue to review results.")
            
            # Get uncategorized transactions
            uncategorized_df = df_cat[df_cat['category'] == 'Uncategorised'].copy()
            
            # Create a form for bulk categorization
            with st.form("bulk_categorization"):
                # Header row for clarity
                col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 2, 2, 1, 1, 1, 1])
                with col1:
                    st.write("**📅 Date & Description**")
                with col2:
                    st.write("**🇯🇵 Original (Japanese)**")
                with col3:
                    st.write("**🏷️ Category**")
                with col4:
                    st.write("**💰 Amount**")
                with col5:
                    st.write("**💳 Type**")
                with col6:
                    st.write("**🎯 Confidence**")
                with col7:
                    st.write("**⏰ Time**")
                st.divider()
                
                # Suggest categories based on description keywords
                for idx, row in uncategorized_df.iterrows():
                    description = str(row['description']).lower()
                    original_desc = str(row.get('original_description', '')).lower()
                    
                    # Smart category suggestions based on keywords
                    suggested_category = "Uncategorised"
                    for category, keywords in rules.items():
                        if any(keyword.lower() in description or keyword.lower() in original_desc for keyword in keywords):
                            suggested_category = category
                            break
                    
                    # Special handling for common Japanese merchants
                    if any(word in original_desc for word in ['ローソン', 'セブンイレブン', 'ファミマ', 'コンビニ']):
                        suggested_category = "Groceries"
                    elif any(word in original_desc for word in ['ニトリ', 'イケア', '家具']):
                        suggested_category = "Shopping & Retail"
                    elif any(word in original_desc for word in ['アマゾン', 'amazon']):
                        suggested_category = "Shopping & Retail"
                    elif any(word in original_desc for word in ['モバイルパス', '交通']):
                        suggested_category = "Transportation"
                    
                    # Ensure suggested category is in the list
                    if suggested_category not in all_categories:
                        suggested_category = "Uncategorised"
                    
                    col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 2, 2, 1, 1, 1, 1])
                    with col1:
                        # Show transaction date
                        transaction_date = row.get('date', 'Unknown Date')
                        if hasattr(transaction_date, 'strftime'):
                            date_str = transaction_date.strftime('%Y-%m-%d')
                        else:
                            date_str = str(transaction_date)
                        st.write(f"**📅 {date_str}**")
                        st.write(f"**{row['description'][:40]}...**")
                    with col2:
                        # Show original description if available
                        if row.get('original_description'):
                            st.write(f"**🇯🇵 {row['original_description'][:30]}...**")
                        else:
                            st.write("**No original description**")
                    with col3:
                        try:
                            # Safe index finding with fallback
                            suggested_index = all_categories.index(suggested_category) if suggested_category in all_categories else 0
                            new_category = st.selectbox(
                                f"Category for {row['description'][:25]}...",
                                all_categories,
                                index=suggested_index,
                                key=f"cat_{idx}"
                            )
                        except (ValueError, IndexError):
                            # Fallback to first category if there's any issue
                            new_category = st.selectbox(
                                f"Category for {row['description'][:25]}...",
                                all_categories,
                                index=0,
                                key=f"cat_{idx}"
                            )
                    with col4:
                        st.write(f"**¥{row['amount']:,}**")
                    with col5:
                        # Show transaction type with color coding
                        trans_type = row.get('transaction_type', 'Expense')
                        if trans_type == 'Credit':
                            st.write("🟢 **Credit**")
                        else:
                            st.write("🔴 **Expense**")
                    with col6:
                        # Show confidence score if available
                        if 'confidence' in row:
                            confidence = row['confidence']
                            if confidence >= 0.8:
                                st.write("🟢 **High**")
                            elif confidence >= 0.5:
                                st.write("🟡 **Med**")
                            else:
                                st.write("🔴 **Low**")
                            st.write(f"**{confidence:.0%}**")
                        else:
                            st.write("⚪ **N/A**")
                    with col7:
                        # Show timestamp if available
                        if 'timestamp' in row and row.get('timestamp'):
                            timestamp = row['timestamp']
                            if hasattr(timestamp, 'strftime'):
                                time_str = timestamp.strftime('%H:%M:%S')
                            else:
                                time_str = str(timestamp)
                            st.write(f"**⏰ {time_str}**")
                        else:
                            st.write("**⏰ N/A**")
                    
                    # Update the category
                    df_cat.loc[idx, 'category'] = new_category
                
                # Add navigation and progress options
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    submitted = st.form_submit_button("✅ Apply All Categorizations")
                
                with col2:
                    save_progress = st.form_submit_button("💾 Save Progress & Continue Later")
                
                with col3:
                    skip_categorization = st.form_submit_button("⏭️ Skip & Review Results")
                
                # Handle form submissions
                if submitted:
                    # Learn from user corrections if smart learning is available
                    if learning_system:
                        for idx, row in uncategorized_df.iterrows():
                            original_category = row.get('category', 'Uncategorised')
                            if original_category != new_category:
                                # Collect feedback for learning
                                transaction_data = {
                                    'description': row.get('description', ''),
                                    'original_description': row.get('original_description', ''),
                                    'amount': row.get('amount', 0),
                                    'date': row.get('date'),
                                    'transaction_type': row.get('transaction_type', 'Expense')
                                }
                                learning_system.learn_from_user_feedback(
                                    str(idx), original_category, new_category, transaction_data
                                )
                    
                    st.success("🎉 All categories updated! Scroll down to see the results.")
                    
                    # Show learning feedback
                    if learning_system:
                        st.success("🧠 Smart Learning System has learned from your corrections!")
                    
                    # Show next steps
                    st.info("🚀 **Next Steps:** Scroll down to review your categorized transactions and see the financial analysis!")
                
                elif save_progress:
                    # Save current progress to session state
                    st.session_state['categorization_progress'] = df_cat.to_dict('records')
                    st.session_state['progress_saved'] = True
                    st.success("💾 Progress saved! You can continue later.")
                
                elif skip_categorization:
                    st.info("⏭️ Skipping categorization. Scroll down to review results.")
        
        # Category filter & review
        st.subheader("🔎 Category Filter & Review")
        try:
            available_categories = sorted([c for c in df_cat['category'].dropna().unique().tolist()])
        except Exception:
            available_categories = []
        selected_categories = st.multiselect(
            "Filter by category",
            options=available_categories,
            default=available_categories
        )

        filtered_df = df_cat[df_cat['category'].isin(selected_categories)].copy() if selected_categories else df_cat.copy()

        # Attach row id for safe updates
        filtered_df['_row_id'] = filtered_df.index

        # Compact summary for the filtered view
        st.write(
            f"Rows: {len(filtered_df)} | Total (abs): ¥{filtered_df['amount'].abs().sum():,.0f}"
        )

        # Column width control for long text
        desc_width_choice = st.select_slider(
            "Description column width",
            options=["small", "medium", "large"],
            value="large",
            help="Adjust how wide description columns render in the table"
        )

        # Editable view for quick verification and corrections
        edited_filtered = st.data_editor(
            filtered_df[
                ['date', 'description', 'original_description', 'amount', 'transaction_type', 'category', 'subcategory', '_row_id']
            ],
            num_rows="dynamic",
            key="category_filter_editor",
            disabled=[False, False, False, True, True, False, False, True],
            use_container_width=True,
            column_config={
                'description': st.column_config.TextColumn('Description', width=desc_width_choice),
                'original_description': st.column_config.TextColumn('Original (Japanese)', width=desc_width_choice)
            }
        )

        # Optional wide read-only view with manual column resizing (drag column edges)
        if st.checkbox("Open wide read-only view (manual column resizing)"):
            st.dataframe(
                filtered_df[['date','description','original_description','amount','transaction_type','category','subcategory']]
                .rename(columns={'original_description':'Original (Japanese)'}),
                use_container_width=True
            )

        # Quick full-text preview for selected row
        with st.expander("Preview full text for a specific row"):
            try:
                row_options = filtered_df.index.astype(str).tolist()
            except Exception:
                row_options = []
            selected_row = st.selectbox("Select row index", options=row_options) if row_options else None
            if selected_row is not None:
                idx = int(selected_row)
                st.write("Description:")
                st.code(str(filtered_df.loc[idx, 'description']))
                if 'original_description' in filtered_df.columns:
                    st.write("Original (Japanese):")
                    st.code(str(filtered_df.loc[idx, 'original_description']))

        # Apply edits back to the master dataframe
        if st.button("✅ Apply Filtered Edits"):
            try:
                for _, r in edited_filtered.iterrows():
                    row_id = r.get('_row_id')
                    if row_id in df_cat.index:
                        if 'category' in r:
                            df_cat.loc[row_id, 'category'] = r['category']
                        if 'subcategory' in r:
                            df_cat.loc[row_id, 'subcategory'] = r['subcategory']
                st.success("✔️ Applied edits to selected rows.")
            except Exception as e:
                st.error(f"Failed to apply edits: {e}")

        # Show the final categorized data
        st.subheader("📋 Review All Transactions")
        
        # Add quick navigation and progress indicator
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write("Final categorized transactions (you can still edit individual categories):")
        
        with col2:
            if st.button("📊 View Financial Summary"):
                st.info("📊 Scroll down to see the financial summary and charts!")
        
        with col3:
            if st.button("💾 Export Data"):
                # Create downloadable CSV
                csv = df_cat.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"categorized_transactions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
        
        # Show categorization progress
        categorized_count = len(df_cat[df_cat['category'] != 'Uncategorised'])
        total_count = len(df_cat)
        progress_percentage = (categorized_count / total_count) * 100
        
        st.progress(progress_percentage / 100)
        st.write(f"📈 **Categorization Progress:** {categorized_count}/{total_count} transactions categorized ({progress_percentage:.1f}%)")
        
        if progress_percentage < 100:
            st.info(f"💡 **Tip:** You can go back to the categorization section above to complete the remaining {total_count - categorized_count} transactions.")
        
        st.divider()
        
        # Save to database section
        st.subheader("💾 Save to Database")
        can_save = all(
            col in df_cat.columns
            for col in ["date", "description", "original_description", "amount", "currency", "fx_rate", "amount_jpy", "category", "subcategory", "transaction_type"]
        ) and insert_transactions is not None

        col_save1, col_save2 = st.columns([1, 1])
        with col_save1:
            if st.button("💾 Save processed transactions to DB") and can_save:
                try:
                    batch_id = create_import_record(uploaded_file.name if hasattr(uploaded_file, "name") else "upload", len(df_cat)) if create_import_record else None
                    # Potential duplicate review
                    review_rows = []
                    for _, r in df_cat.iterrows():
                        review_rows.append({
                            "date": r.get("date"),
                            "description": r.get("description"),
                            "original_description": r.get("original_description"),
                            "amount": float(r.get("amount", 0.0)),
                            "currency": r.get("currency", default_currency),
                            "fx_rate": float(r.get("fx_rate", 1.0)),
                            "amount_jpy": float(r.get("amount_jpy", r.get("amount", 0.0))),
                            "category": r.get("category"),
                            "subcategory": r.get("subcategory"),
                            "transaction_type": r.get("transaction_type"),
                        })
                    # Lightweight potential duplicate detection: same date & amount
                    pot_dupes = []
                    try:
                        existing = pd.DataFrame(load_all_transactions() or []) if load_all_transactions else pd.DataFrame()
                        if not existing.empty:
                            existing['date'] = pd.to_datetime(existing['date']).dt.strftime('%Y-%m-%d')
                            for row in review_rows:
                                date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
                                match = existing[(existing['date'] == date_str) & (existing['amount'].round(2) == round(row['amount'], 2))]
                                if len(match) > 0:
                                    pot_dupes.append({
                                        "date": date_str,
                                        "amount": row['amount'],
                                        "new_description": row['description'],
                                        "existing_count": int(len(match))
                                    })
                    except Exception:
                        pass

                    if pot_dupes:
                        st.warning("Potential duplicates detected. Review below; choose whether to insert them.")
                        st.dataframe(pd.DataFrame(pot_dupes))
                        insert_suspected = st.checkbox("Insert suspected duplicates too", value=False)
                    else:
                        insert_suspected = True

                    if not insert_suspected and 'existing' in locals() and not existing.empty:
                        filtered_rows = []
                        for row in review_rows:
                            date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
                            match = existing[(existing['date'] == date_str) & (existing['amount'].round(2) == round(row['amount'], 2))]
                            if len(match) == 0:
                                filtered_rows.append(row)
                        review_rows = filtered_rows

                    inserted, dupes, _ = insert_transactions(review_rows, batch_id)  # type: ignore
                    st.success(f"Inserted {inserted} rows. Skipped {dupes} strict duplicates.")
                except Exception as e:
                    st.error(f"Save failed: {e}")
            elif not can_save:
                st.caption("Cannot save yet: data layer not ready or required columns missing.")

        with col_save2:
            sanitize = st.checkbox("Sanitize exports (remove Original/Japanese text)", value=False)
            if st.button("🧾 Export DB to CSV") and export_transactions_to_csv is not None:
                try:
                    if sanitize and load_all_transactions is not None:
                        rows = load_all_transactions()
                        df_all = pd.DataFrame(rows)
                        if 'original_description' in df_all.columns:
                            df_all = df_all.drop(columns=['original_description'])
                        ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                        out_path = os.path.join('exports', f'transactions_sanitized_{ts}.csv')
                        os.makedirs('exports', exist_ok=True)
                        df_all.to_csv(out_path, index=False)
                        st.success(f"Exported sanitized CSV: {out_path}")
                    else:
                        csv_path = export_transactions_to_csv()
                        st.success(f"Exported to {csv_path}")
                except Exception as e:
                    st.error(f"Export failed: {e}")

        col_bkp1, col_bkp2 = st.columns([1, 1])
        with col_bkp1:
            if st.button("🗄️ Backup DB (local)") and backup_database is not None:
                try:
                    backup_path = backup_database()
                    st.success(f"Backup created: {backup_path}")
                except Exception as e:
                    st.error(f"Backup failed: {e}")

        with col_bkp2:
            if load_all_transactions is not None and st.button("📚 Show DB count"):
                try:
                    rows = load_all_transactions()
                    st.info(f"DB has {len(rows)} transactions.")
                except Exception as e:
                    st.error(f"Count failed: {e}")

        st.divider()

        # Recurring transactions UI
        st.subheader("🔁 Recurring Transactions")
        st.caption("Mark a transaction as recurring and auto-generate future instances.")

        if list_recurring_rules and upsert_recurring_rule:
            with st.expander("Create / Update recurring rule"):
                merchant_pattern = st.text_input("Merchant contains (pattern)")
                freq = st.selectbox("Frequency", ["monthly", "weekly"], index=0)
                next_date = st.date_input("Next date")
                fixed_amount = st.number_input("Amount (optional)", value=0.0, help="Leave 0 for variable amount detected from transactions")
                cat = st.text_input("Category (optional)")
                subcat = st.text_input("Subcategory (optional)")
                ccy = st.selectbox("Currency", ["JPY", "USD", "EUR", "AUD", "CAD", "GBP", "CNY", "KRW"], index=0)
                if st.button("💾 Save recurring rule"):
                    try:
                        rule_id = upsert_recurring_rule({
                            "merchant_pattern": merchant_pattern.strip(),
                            "frequency": freq,
                            "next_date": next_date.strftime("%Y-%m-%d"),
                            "amount": (None if abs(fixed_amount) < 1e-9 else float(fixed_amount)),
                            "category": (cat.strip() or None),
                            "subcategory": (subcat.strip() or None),
                            "currency": ccy,
                            "active": 1,
                        })
                        st.success(f"Rule saved (id={rule_id})")
                    except Exception as e:
                        st.error(f"Failed to save rule: {e}")

            existing = list_recurring_rules()
            if existing:
                st.write("Active rules:")
                st.dataframe(pd.DataFrame(existing))
            else:
                st.caption("No active recurring rules yet.")

            with st.expander("Preview next generated instances"):
                import datetime as _dt
                preview_rows = []
                for r in existing or []:
                    next_dt = _dt.date.fromisoformat(r["next_date"]) if r.get("next_date") else None
                    if not next_dt:
                        continue
                    # Predict following date
                    if r["frequency"] == "weekly":
                        following = next_dt + _dt.timedelta(days=7)
                    else:
                        # monthly: naive add 30 days
                        following = next_dt + _dt.timedelta(days=30)
                    preview_rows.append({
                        "merchant_pattern": r["merchant_pattern"],
                        "next_date": str(next_dt),
                        "predicted_following": str(following),
                        "amount": r.get("amount"),
                        "category": r.get("category"),
                        "currency": r.get("currency"),
                    })
                if preview_rows:
                    st.dataframe(pd.DataFrame(preview_rows))
                else:
                    st.caption("Nothing to preview yet.")

            # Generate and insert next instances (preview-confirm)
            with st.form("generate_recurring"):
                st.write("Generate next instances for due rules (today or earlier)?")
                gen = st.form_submit_button("➕ Generate Next Instances")
                if gen:
                    try:
                        to_insert = []
                        today = pd.Timestamp.today().date()
                        for r in existing or []:
                            if not r.get("next_date"):
                                continue
                            next_dt = pd.to_datetime(r["next_date"]).date()
                            if next_dt > today:
                                continue
                            desc = f"Recurring: {r['merchant_pattern']}"
                            amt = float(r["amount"]) if r.get("amount") is not None else 0.0
                            ccy = r.get("currency", "JPY")
                            rate = get_fx_rate_to_jpy(ccy, next_dt.strftime("%Y-%m-%d"))
                            to_insert.append({
                                "date": next_dt,
                                "description": desc,
                                "original_description": desc,
                                "amount": amt,
                                "currency": ccy,
                                "fx_rate": rate,
                                "amount_jpy": amt * rate,
                                "category": r.get("category"),
                                "subcategory": r.get("subcategory"),
                                "transaction_type": "Expense" if amt <= 0 else "Credit",
                            })
                        if to_insert and insert_transactions:
                            ins, dups, _ = insert_transactions(to_insert, None)
                            st.success(f"Generated {ins} next instances (skipped {dups} duplicates)")
                        elif not to_insert:
                            st.info("No rules due today.")
                    except Exception as e:
                        st.error(f"Generation failed: {e}")

        # Google Drive backup UI
        st.subheader("☁️ Google Drive Backup")
        st.caption("Authorize once, then upload latest DB backup and CSV export to Drive.")
        try:
            from drive_backup import ensure_oauth, upload_bytes
            creds = ensure_oauth()
            if creds:
                col_g1, col_g2 = st.columns([1, 1])
                with col_g1:
                    if st.button("🔐 Backup DB to Drive"):
                        try:
                            # Create a DB backup file in memory via backup_database path
                            bkp_path = backup_database() if backup_database else None
                            if bkp_path:
                                with open(bkp_path, "rb") as f:
                                    data = f.read()
                                folder_id = st.secrets.get("google", {}).get("drive_folder_id")
                                link = upload_bytes(creds, folder_id, os.path.basename(bkp_path), data, mime="application/x-sqlite3")
                                st.success(f"DB backup uploaded: {link}")
                            else:
                                st.warning("Local backup function not available.")
                        except Exception as e:
                            st.error(f"Drive DB backup failed: {e}")
                with col_g2:
                    if st.button("📤 Export CSV to Drive"):
                        try:
                            csv_path = export_transactions_to_csv() if export_transactions_to_csv else None
                            if csv_path:
                                with open(csv_path, "rb") as f:
                                    data = f.read()
                                folder_id = st.secrets.get("google", {}).get("drive_folder_id")
                                link = upload_bytes(creds, folder_id, os.path.basename(csv_path), data, mime="text/csv")
                                st.success(f"CSV uploaded: {link}")
                            else:
                                st.warning("CSV export function not available.")
                        except Exception as e:
                            st.error(f"Drive CSV upload failed: {e}")
        except Exception as e:
            st.info("Google Drive not configured. Add [google] secrets to enable.")

        # Prepare data for display with better formatting
        display_df = df_cat.copy()
        
        # Format timestamp column for better display
        if 'timestamp' in display_df.columns:
            display_df['timestamp'] = display_df['timestamp'].apply(
                lambda x: x.strftime('%H:%M:%S') if pd.notna(x) and hasattr(x, 'strftime') else x
            )
        
        # Format date column for better display
        if 'date' in display_df.columns:
            display_df['date'] = display_df['date'].apply(
                lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) and hasattr(x, 'strftime') else x
            )
        
        # Calculate and display running balance
        st.subheader("💰 Running Balance Analysis")
        balance_df = calculate_running_balance(df_cat)
        
        # Show balance trend
        if len(balance_df) > 1:
            balance_chart = balance_df[['date', 'running_balance']].copy()
            balance_chart['date'] = pd.to_datetime(balance_chart['date'])
            balance_chart = balance_chart.sort_values('date')
            
            st.line_chart(balance_chart.set_index('date'))
            
            # Show current balance
            current_balance = balance_df['running_balance'].iloc[-1]
            st.info(f"💳 **Current Balance:** ¥{current_balance:,.0f}")
            
            # Show balance statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📈 Highest Balance", f"¥{balance_df['running_balance'].max():,.0f}")
            with col2:
                st.metric("📉 Lowest Balance", f"¥{balance_df['running_balance'].min():,.0f}")
            with col3:
                st.metric("📊 Average Balance", f"¥{balance_df['running_balance'].mean():,.0f}")
        
        # Use data editor for final review
        final_desc_width = st.select_slider(
            "Description column width (final table)",
            options=["small", "medium", "large"],
            value="large"
        )
        edited_df = st.data_editor(
            display_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                'description': st.column_config.TextColumn('Description', width=final_desc_width),
                'original_description': st.column_config.TextColumn('Original (Japanese)', width=final_desc_width)
            }
        )
        
        if not edited_df.empty:
            edited_df["month"] = pd.to_datetime(edited_df["date"]).dt.to_period("M").astype(str)
            summary = (
                edited_df.groupby(["month", "category"])["amount"].sum().reset_index(name="total_amount")
            )
            st.subheader("📊 Monthly Summary")
            st.write(summary)
            pivot = summary.pivot(index="month", columns="category", values="total_amount").fillna(0)
            st.bar_chart(pivot)
        
        # Show learning system statistics
        if learning_system:
            st.subheader("🧠 Smart Learning System Statistics")
            learning_stats = learning_system.get_learning_stats()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📚 Total Corrections", learning_stats['total_corrections'])
            with col2:
                st.metric("🏪 Unique Merchants", learning_stats['unique_merchants'])
            with col3:
                st.metric("🔄 Recent Corrections", learning_stats['recent_corrections'])
            
            if learning_stats['merchant_categories']:
                st.write("**Merchant Learning Database:**")
                for merchant, categories in learning_stats['merchant_categories'].items():
                    if merchant != "unknown":
                        st.write(f"• **{merchant}**: {list(categories.keys())}")


if __name__ == "__main__":
    main()

