# 🤖 AI-Powered Expense Tracker

A smart expense tracking application that automatically categorizes transactions from bank statements, with **AI-powered Japanese translation** support.

## ✨ Features

- **📄 Multi-format Support**: PDF, CSV, and image files
- **🌐 AI Translation**: Japanese → English using OpenAI GPT-4
- **🧠 Smart Categorization**: 10+ expense categories with intelligent keyword matching
- **📊 Data Visualization**: Monthly spending reports and charts
- **🔧 Easy Setup**: Simple configuration for ChatGPT Premium users

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Tesseract OCR (for image processing)
- OpenAI API key (optional, for best translation accuracy)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd expense-tracker
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR**
   - **macOS**: `brew install tesseract`
   - **Ubuntu**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

4. **Run the app**
   ```bash
   streamlit run transaction_web_app.py
   ```

## 🔑 OpenAI API Setup (Optional)

For best Japanese translation accuracy:

1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Enter it in the app's sidebar
3. Choose "AI-Powered (GPT-3.5)" translation mode

## 📁 Supported File Formats

### CSV Files
- **English**: Standard bank export formats
- **Japanese**: Automatic column detection for Japanese bank statements
- **Smart Translation**: Merchant names translated to English

### PDF Files
- Bank statements with table data
- Automatic column detection
- Handles multiple date formats

### Image Files
- Screenshots of bank statements
- OCR processing with Tesseract
- Automatic text extraction

## 🏷️ Expense Categories

The app automatically categorizes expenses into:
- **Groceries** - Supermarkets, food stores
- **Dining & Restaurants** - Restaurants, cafes, delivery
- **Transportation** - Fuel, parking, ride-sharing
- **Subscriptions & Services** - Netflix, Spotify, software
- **Shopping & Retail** - Amazon, department stores
- **Entertainment** - Movies, concerts, activities
- **Healthcare** - Pharmacies, medical services
- **Utilities & Bills** - Electricity, internet, phone
- **Books & Education** - Courses, training, books
- **Travel** - Hotels, flights, car rentals

## 🔧 Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Translation Modes
- **AI-Powered (GPT-3.5)**: Best accuracy, requires API key
- **Free Fallback**: Basic translation, no cost
- **No Translation**: Keep original text

## 📊 Usage

1. **Upload File**: Choose your bank statement file
2. **Review Data**: Check extracted transactions
3. **Edit Categories**: Manually adjust if needed
4. **View Reports**: See monthly spending summaries
5. **Export Data**: Use Export/Backup buttons in the app (CSV and DB backup)
6. **Google Drive Backup**: Use "Backup to Drive" (authorize first) for cloud backups
7. **Recurring**: Create rules (weekly/monthly) and generate next instances

## 🛠️ Development

### Project Structure
```
expense-tracker/
├── transaction_web_app.py    # Main application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── .gitignore              # Git ignore rules
└── venv/                   # Virtual environment
```

### Adding New Features
- **New File Formats**: Add extraction functions
- **Categories**: Update the rules dictionary
- **Translation**: Extend language support

## 🚀 Deployment Options

### Local Development
```bash
streamlit run transaction_web_app.py
```

### Streamlit Cloud Deployment

#### Step 1: Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select this repository: `sahiljain11593/ai-expense-tracker`
4. Set main file: `transaction_web_app.py`
5. Click "Deploy"

#### Step 2: Configure Secrets (Settings → Secrets)

**Firebase Authentication** (Optional - for single-user access control):
```toml
[firebase]
apiKey = "YOUR_WEB_API_KEY"
authDomain = "your-project.firebaseapp.com"
projectId = "your-project"
appId = "YOUR_APP_ID"

[auth]
allowed_email = "your@gmail.com"
```

**Google Drive Backup** (Optional - for cloud persistence):
```toml
[google]
client_id = "YOUR_OAUTH_CLIENT_ID.apps.googleusercontent.com"
client_secret = "YOUR_OAUTH_CLIENT_SECRET"
redirect_uri = "https://your-app-name.streamlit.app"
drive_folder_id = ""  # Leave empty to auto-create folder
```

#### Step 3: Set Up Authentication (Optional)

1. **Create Firebase Project**:
   - Go to [Firebase Console](https://console.firebase.google.com/)
   - Create new project
   - Enable Google Authentication in Authentication → Sign-in method
   - Copy Web API Key and App ID to Streamlit Secrets

2. **Configure Google OAuth**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create OAuth 2.0 credentials
   - Add authorized redirect URI: `https://your-app-name.streamlit.app`
   - Enable Google Drive API
   - Copy Client ID and Secret to Streamlit Secrets

#### Step 4: Using Google Drive Backup

1. In the deployed app, scroll to "☁️ Google Drive Backup"
2. Click "Authorize Drive" → Opens Google OAuth
3. Sign in and authorize access
4. Copy the authorization code from URL (the `code=...` parameter)
5. Paste code back into the app
6. Click "🔐 Backup DB to Drive" or "📤 Export CSV to Drive"
7. Dated backup files are created in your Drive folder

**Monthly Backup Routine** (Recommended for Streamlit Cloud):
- Streamlit Cloud apps have ephemeral storage (resets on rebuild)
- Schedule manual backups monthly or after major data imports
- Alternative: Use GitHub Actions to trigger backups automatically
- Download backup files from Google Drive for local restore if needed

### Local E2E Smoke Test
```bash
# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run smoke test (headless, no auth required)
python scripts/e2e_check.py

# Run full test suite
pytest tests/ -v

# Run specific test file
pytest tests/test_import_dedupe.py -v
```

### Running Tests
The project includes comprehensive pytest tests for import/dedupe and recurring transaction logic:

```bash
# Install test dependencies
pip install -r requirements.txt  # includes pytest

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test categories
pytest tests/test_import_dedupe.py -v
pytest tests/test_recurring.py -v

# Run with coverage (install pytest-cov first)
pip install pytest-cov
pytest --cov=data_store --cov-report=html
```

Test coverage includes:
- ✅ Strict duplicate prevention (same date+desc+amount)
- ✅ Fuzzy duplicate detection with configurable tolerance
- ✅ Import record creation and batch tracking
- ✅ Recurring rule creation and persistence
- ✅ Weekly/monthly frequency calculations
- ✅ Settings persistence and retrieval

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 🔐 Privacy & Security

### Data Protection
- **No PII in Logs**: Application minimizes logging of sensitive transaction data
- **Local-First**: SQLite database stored locally at `data/expenses.db`
- **Sanitized Exports**: Toggle to remove original Japanese text from exports
- **Secrets Management**: All API keys and credentials via Streamlit Secrets (never in code)

### Security Best Practices
1. **Never commit secrets** to the repository
2. **Use Streamlit Secrets** for all sensitive configuration
3. **Enable authentication** for production deployments
4. **Regular backups** to Google Drive with encrypted storage
5. **Review exported files** before sharing (use sanitize toggle)

## 🗄️ Optional: Database Migration to Supabase

For persistent cloud storage beyond Streamlit Cloud's ephemeral filesystem:

### Supabase Setup (Free Tier)
1. Create account at [supabase.com](https://supabase.com)
2. Create new project
3. Run schema migration:
```sql
CREATE TABLE transactions (
  id BIGSERIAL PRIMARY KEY,
  date DATE NOT NULL,
  description TEXT NOT NULL,
  original_description TEXT,
  amount DECIMAL NOT NULL,
  currency TEXT DEFAULT 'JPY',
  fx_rate DECIMAL DEFAULT 1.0,
  amount_jpy DECIMAL NOT NULL,
  category TEXT,
  subcategory TEXT,
  transaction_type TEXT,
  import_batch_id BIGINT,
  dedupe_hash TEXT UNIQUE NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE imports (
  id BIGSERIAL PRIMARY KEY,
  file_name TEXT,
  imported_at TIMESTAMP NOT NULL,
  num_rows INTEGER NOT NULL
);

CREATE TABLE settings (
  key TEXT PRIMARY KEY,
  value TEXT
);

CREATE TABLE recurring_rules (
  id BIGSERIAL PRIMARY KEY,
  merchant_pattern TEXT NOT NULL,
  frequency TEXT NOT NULL,
  next_date DATE NOT NULL,
  amount DECIMAL,
  category TEXT,
  subcategory TEXT,
  currency TEXT DEFAULT 'JPY',
  active INTEGER DEFAULT 1
);
```

4. Update `data_store.py` to use PostgreSQL via psycopg2
5. Add Supabase connection string to Streamlit Secrets

**Alternative**: Continue using SQLite with Google Drive backups as the source of truth. Download backup, restore locally when needed.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit** for the web framework
- **OpenAI** for AI translation capabilities
- **Tesseract** for OCR processing
- **Pandas** for data manipulation

## 📞 Support

If you encounter issues:
1. Check the error messages in the app
2. Verify your file format is supported
3. Ensure all dependencies are installed
4. Check your OpenAI API key (if using AI translation)

---

**Made with ❤️ for easy expense tracking**
