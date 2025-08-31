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
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

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
    """Translate Japanese text to English using OpenAI GPT-4 for high accuracy."""
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
        
        # Get translation from GPT-4
        response = client.chat.completions.create(
            model="gpt-4",
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

def translate_japanese_to_english(text: str, mode: str = "AI-Powered (GPT-4)", api_key: str = None) -> str:
    """Main translation function - handles different translation modes."""
    if mode == "AI-Powered (GPT-4)":
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
                      '利用金額', '支払金額', '取引金額']
    
    # Find the actual column names in the CSV
    date_col = None
    desc_col = None
    amount_col = None
    
    for col in df.columns:
        col_lower = col.lower().strip()
        col_original = col.strip()
        
        # Check English column names
        if col_lower in [name.lower() for name in date_columns]:
            date_col = col
        elif col_lower in [name.lower() for name in desc_columns]:
            desc_col = col
        elif col_lower in [name.lower() for name in amount_columns]:
            amount_col = col
        
        # Check Japanese column names
        if col_original in date_columns:
            date_col = col
        elif col_original in desc_columns:
            desc_col = col
        elif col_original in amount_columns:
            amount_col = col
    
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
    
    # Show column mapping
    st.info(f"Column mapping: Date='{date_col}', Description='{desc_col}', Amount='{amount_col}'")
    
    # Process the data
    transactions = []
    for idx, row in df.iterrows():
        try:
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
                
            # Get description and translate if it's Japanese
            description = str(row[desc_col]).strip()
            if pd.isna(description) or description == '':
                continue
            
            # Translate Japanese description to English
            original_description = description
            description = translate_japanese_to_english(description, translation_mode, api_key)
            
            # Show translation progress
            if original_description != description:
                st.info(f"Translated: '{original_description}' → '{description}'")
                
            # Handle amount (could be positive or negative)
            amount_str = str(row[amount_col]).strip()
            if pd.isna(amount_str) or amount_str == '':
                continue
                
            # Remove currency symbols, commas, and Japanese characters
            amount_str = re.sub(r'[^\d.-]', '', amount_str)
            amount = float(amount_str)
            
            transactions.append({
                "date": date,
                "description": description,
                "original_description": original_description,  # Keep original for reference
                "amount": amount
            })
            
        except Exception as e:
            st.warning(f"Error processing row {idx + 1}: {row}. Error: {e}")
            continue
    
    if not transactions:
        raise RuntimeError("No valid transactions found in the CSV file.")
    
    st.success(f"Successfully processed {len(transactions)} transactions with Japanese translation!")
    df_result = pd.DataFrame(transactions)
    return df_result
def categorise_transactions(
    df: pd.DataFrame, rules: Dict[str, List[str]], uncategorised_label: str = "Uncategorised"
) -> pd.DataFrame:
    patterns = {cat: re.compile("(" + "|".join(map(re.escape, kws)) + ")", 
re.IGNORECASE) for cat, kws in rules.items()}
    categories: List[str] = []
    for desc in df["description"].astype(str):
        assigned: Optional[str] = None
        for cat, pattern in patterns.items():
            if pattern.search(desc):
                assigned = cat
                break
        if assigned is None:
            assigned = uncategorised_label
        categories.append(assigned)
    df = df.copy()
    df["category"] = categories
    return df


def save_custom_rules(rules: Dict[str, List[str]], filename: str = "custom_rules.json") -> None:
    """Save custom categorization rules to a JSON file."""
    import json
    try:
        with open(filename, 'w') as f:
            json.dump(rules, f, indent=2)
        st.success(f"Custom rules saved to {filename}")
    except Exception as e:
        st.error(f"Error saving rules: {e}")

def load_custom_rules(filename: str = "custom_rules.json") -> Dict[str, List[str]]:
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
    st.title("Transaction Categoriser")
    st.write(
        "Upload a credit card statement in PDF, CSV, or image format. The app will extract transaction data, "
        "assign categories based on keyword rules and let you review the results."
    )
    
    # AI Translation Setup
    st.sidebar.header("🤖 AI Translation Settings")
    st.sidebar.write("For best Japanese translation accuracy, use OpenAI GPT-4")
    
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
            st.info("💡 Your ChatGPT Premium account gives you access to the API!")
    
    # Translation mode selection
    if api_key:
        translation_mode = st.sidebar.selectbox(
            "Translation Mode",
            ["AI-Powered (GPT-4)", "Free Fallback", "No Translation"],
            index=0,  # Default to AI-powered
            help="AI translation provides the best accuracy for Japanese financial terms"
        )
    else:
        translation_mode = "Free Fallback"
        st.sidebar.info("ℹ️ Using free translation (enter API key for AI accuracy)")
    
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
        st.success(f"Loaded {len(df)} transactions.")
        # Enhanced categorization rules with comprehensive keywords
        rules = {
            "Groceries": [
                "supermarket", "grocery", "market", "food", "fresh", "organic", 
                "whole foods", "trader joe", "kroger", "safeway", "walmart", "target",
                "costco", "sam's club", "aldi", "lidl", "publix", "meijer"
            ],
            "Dining & Restaurants": [
                "restaurant", "cafe", "dinner", "lunch", "breakfast", "burger", 
                "takeaway", "delivery", "pizza", "mcdonalds", "burger king", "kfc",
                "subway", "chipotle", "starbucks", "dunkin", "panera",
                "olive garden", "applebees", "tgi fridays", "buffalo wild wings"
            ],
            "Transportation": [
                "taxi", "uber", "lyft", "bus", "fuel", "gas", "train", "subway",
                "metro", "parking", "toll", "ezpass", "fastrak", "shell", "exxon",
                "bp", "chevron", "mobil", "valero", "marathon", "speedway"
            ],
            "Subscriptions & Services": [
                "subscription", "membership", "netflix", "spotify", "amazon prime",
                "hulu", "disney+", "hbo max", "apple one", "icloud", "dropbox",
                "adobe", "microsoft", "google", "zoom", "slack", "asana"
            ],
            "Shopping & Retail": [
                "amazon", "ebay", "etsy", "target", "walmart", "best buy", "home depot",
                "lowes", "macy's", "nordstrom", "gap", "old navy", "h&m", "zara",
                "nike", "adidas", "apple store", "microsoft store"
            ],
            "Entertainment": [
                "movie", "theater", "cinema", "concert", "show", "ticket", "event",
                "amusement", "park", "museum", "zoo", "aquarium", "bowling", "golf",
                "fitness", "gym", "yoga", "pilates", "tennis", "swimming"
            ],
            "Healthcare": [
                "pharmacy", "cvs", "walgreens", "rite aid", "doctor", "hospital",
                "clinic", "medical", "dental", "vision", "insurance", "copay",
                "deductible", "prescription", "medicine", "drug"
            ],
            "Utilities & Bills": [
                "electric", "gas", "water", "internet", "phone", "cable", "tv",
                "electricity", "utility", "bill", "payment", "at&t", "verizon",
                "comcast", "xfinity", "spectrum", "cox", "optimum"
            ],
            "Books & Education": [
                "bookstore", "book", "amazon kindle", "barnes & noble", "library",
                "course", "class", "training", "workshop", "seminar", "conference",
                "university", "college", "school", "tuition", "textbook"
            ],
            "Travel": [
                "hotel", "airbnb", "flight", "airline", "delta", "american", "united",
                "southwest", "jetblue", "spirit", "frontier", "booking", "expedia",
                "hotels.com", "marriott", "hilton", "hyatt", "car rental", "hertz"
            ]
        }
        df_cat = categorise_transactions(df, rules)
        st.subheader("Review transactions")
        st.write("You can edit the category column below to correct misclassifications or fill in Uncategorised items.")
        edited_df = st.experimental_data_editor(df_cat, num_rows="dynamic")
        if not edited_df.empty:
            edited_df["month"] = pd.to_datetime(edited_df["date"]).dt.to_period("M").astype(str)
            summary = (
                edited_df.groupby(["month", "category"])["amount"].sum().reset_index(name="total_amount")
            )
            st.subheader("Monthly summary")
            st.write(summary)
            pivot = summary.pivot(index="month", columns="category", values="total_amount").fillna(0)
            st.bar_chart(pivot)


if __name__ == "__main__":
    main()

