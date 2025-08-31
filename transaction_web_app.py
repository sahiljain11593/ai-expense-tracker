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
    df: pd.DataFrame, rules: Dict[str, List[str]], subcategories: Dict[str, Dict[str, List[str]]] = None, 
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
            st.info("💡 Your ChatGPT Premium account gives you access to the API!")
    
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
        df_cat = categorise_transactions(df, rules, subcategories)
        
        # Smart categorization interface
        st.subheader("🎯 Smart Transaction Categorization")
        
        # Get all available categories and subcategories
        all_categories = list(rules.keys()) + ["Uncategorised"]
        all_subcategories = []
        for main_cat, subs in subcategories.items():
            for sub_cat in subs.keys():
                all_subcategories.append(f"{main_cat} - {sub_cat}")
        
        # Show categorization statistics
        category_counts = df_cat['category'].value_counts()
        st.info(f"📊 **Categorization Summary:** {len(df_cat)} total transactions")
        
        # Display category breakdown with subcategories
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Main Categories:**")
            for cat, count in category_counts.items():
                if cat != "Uncategorised":
                    st.write(f"• {cat}: {count}")
                    
                    # Show subcategories for this main category
                    if cat in subcategories:
                        sub_counts = df_cat[df_cat['category'] == cat]['subcategory'].value_counts()
                        for sub_cat, sub_count in sub_counts.items():
                            if sub_cat:  # Only show non-empty subcategories
                                st.write(f"  └─ {sub_cat}: {sub_count}")
        
        with col2:
            uncategorized_count = category_counts.get("Uncategorised", 0)
            st.write(f"**Uncategorized:** {uncategorized_count}")
            
            # Show total transactions
            st.write(f"**Total Transactions:** {len(df_cat)}")
        
        # Smart categorization for uncategorized transactions
        if uncategorized_count > 0:
            st.subheader("🚀 Quick Categorization")
            st.write("Use the dropdowns below to quickly categorize uncategorized transactions:")
            
            # Get uncategorized transactions
            uncategorized_df = df_cat[df_cat['category'] == 'Uncategorised'].copy()
            
            # Create a form for bulk categorization
            with st.form("bulk_categorization"):
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
                    
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.write(f"**{row['description'][:50]}...**")
                    with col2:
                        new_category = st.selectbox(
                            f"Category for {row['description'][:30]}...",
                            all_categories,
                            index=all_categories.index(suggested_category),
                            key=f"cat_{idx}"
                        )
                    with col3:
                        st.write(f"¥{row['amount']:,}")
                    
                    # Update the category
                    df_cat.loc[idx, 'category'] = new_category
                
                submitted = st.form_submit_button("✅ Apply All Categorizations")
                if submitted:
                    st.success("🎉 All categories updated! Scroll down to see the results.")
        
        # Show the final categorized data
        st.subheader("📋 Review All Transactions")
        st.write("Final categorized transactions (you can still edit individual categories):")
        
        # Use data editor for final review
        edited_df = st.data_editor(df_cat, num_rows="dynamic")
        
        if not edited_df.empty:
            edited_df["month"] = pd.to_datetime(edited_df["date"]).dt.to_period("M").astype(str)
            summary = (
                edited_df.groupby(["month", "category"])["amount"].sum().reset_index(name="total_amount")
            )
            st.subheader("📊 Monthly Summary")
            st.write(summary)
            pivot = summary.pivot(index="month", columns="category", values="total_amount").fillna(0)
            st.bar_chart(pivot)


if __name__ == "__main__":
    main()

