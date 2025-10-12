# 🤖 AI-Powered Expense Tracker

A smart expense tracking application that automatically categorizes transactions from bank statements, with **AI-powered Japanese translation** support.

## ✨ Features

- **📄 Multi-format Support**: PDF, CSV, and image files
- **🌐 AI Translation**: Japanese → English using OpenAI GPT-3.5 (with free fallback)
- **🧠 Smart Categorization**: 10+ expense categories with intelligent keyword matching
- **🔍 Advanced Duplicate Detection**: Configurable fuzzy matching for similar transactions
- **📊 Data Visualization**: Monthly spending reports and interactive charts
- **🔄 Recurring Transactions**: Automatic detection and generation of recurring payments
- **☁️ Cloud Backup**: Google Drive integration for data persistence
- **🔐 Secure**: Google Sign-In authentication with single-user access control
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

1. **Upload File**: Choose your bank statement file (CSV, PDF, or image)
2. **Review Data**: Check extracted transactions with AI translation
3. **Configure Duplicates**: Adjust duplicate detection sensitivity in sidebar
4. **Edit Categories**: Manually adjust categories with confidence scores
5. **Filter & Search**: Use advanced filters to find specific transactions
6. **Save to Database**: Store processed transactions locally
7. **View Reports**: See monthly spending summaries and charts
8. **Export Data**: Download CSV exports anytime
9. **Cloud Backup**: Backup to Google Drive for data persistence
10. **Recurring Rules**: Create and manage recurring transaction patterns

## 💾 Data Management & Backup

### Local Storage
- **SQLite Database**: All data stored in `data/expenses.db`
- **Automatic Backups**: Local backups created in `backups/` folder
- **CSV Exports**: Downloadable exports in `exports/` folder
- **Sanitization**: Option to remove PII from exports

### Google Drive Backup (Recommended for Streamlit Cloud)

#### Setup Google Drive Integration
1. **Create Google Cloud Project**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing
   - Enable Google Drive API

2. **Create OAuth Credentials**:
   - Go to "Credentials" → "Create Credentials" → "OAuth 2.0 Client IDs"
   - Application type: "Web application"
   - Add authorized redirect URI: `https://your-app-name.streamlit.app`
   - Download the JSON file and note `client_id` and `client_secret`

3. **Configure Streamlit Secrets**:
   ```toml
   [google]
   client_id = "your-client-id.apps.googleusercontent.com"
   client_secret = "your-client-secret"
   redirect_uri = "https://your-app-name.streamlit.app"
   drive_folder_id = "optional-folder-id"  # Leave empty to auto-create
   ```

#### Using Google Drive Backup
1. **Authorize**: Click "Authorize Google Drive" in the app
2. **Paste Code**: Copy the authorization code from the popup
3. **Backup**: Use "Backup DB to Drive" and "Export CSV to Drive" buttons
4. **Restore**: Download files from Drive and import back to the app

### Monthly Backup Strategy (Streamlit Cloud)
Since Streamlit Cloud sessions are ephemeral, follow this monthly routine:

1. **Week 1**: Upload new transactions, process, and backup to Drive
2. **Week 2**: Continue adding transactions, backup weekly
3. **Week 3**: Review and categorize, backup before major changes
4. **Week 4**: Full backup of database and CSV export to Drive

### Data Recovery
If you lose access to your Streamlit Cloud app:
1. Download the latest backup from Google Drive
2. Restore the SQLite database to a local installation
3. Continue using the app locally or redeploy

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
1. **Fork this repository** to your GitHub account
2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**
3. **Click "New app"** and connect your GitHub repository
4. **Set the main file path** to `transaction_web_app.py`
5. **Deploy** the application

#### Step 2: Configure Authentication (Optional)
If you want to restrict access to your app:

1. **Set up Firebase Authentication**:
   - Go to [Firebase Console](https://console.firebase.google.com/)
   - Create a new project or select existing
   - Enable Authentication → Sign-in method → Google
   - Get your Firebase config

2. **Set up Google OAuth for Drive backup**:
   - Follow the Google Drive setup steps above
   - Get your OAuth client credentials

3. **Configure Streamlit Secrets**:
   - Go to your app's settings in Streamlit Cloud
   - Click "Secrets" and add:
   ```toml
   [firebase]
   apiKey = "your-firebase-api-key"
   authDomain = "your-project.firebaseapp.com"
   projectId = "your-project-id"
   appId = "your-firebase-app-id"

   [auth]
   allowed_email = "your-email@gmail.com"

   [google]
   client_id = "your-oauth-client-id"
   client_secret = "your-oauth-client-secret"
   redirect_uri = "https://your-app-name.streamlit.app"
   drive_folder_id = "optional-folder-id"
   ```

4. **Reboot the app** to apply secrets

#### Step 3: First-Time Setup
1. **Access your app** - you'll see a Google Sign-In button if auth is configured
2. **Sign in** with your allowed email address
3. **Authorize Google Drive** (if configured) for backup functionality
4. **Start uploading** your transaction files!

#### Step 4: Regular Usage
- **Upload transactions** weekly or monthly
- **Backup to Drive** after each major upload
- **Export CSV** for external analysis
- **Review and categorize** transactions as needed

## 🧪 Testing

### Local E2E Smoke Test (no auth)
```bash
python3 -m venv .venv --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
. .venv/bin/activate && python get-pip.py
pip install -r requirements.txt
python scripts/e2e_check.py
```

### Unit Tests
```bash
# Install pytest
pip install pytest

# Run all tests
python -m pytest test_expense_tracker.py -v

# Run specific test categories
python -m pytest test_expense_tracker.py::TestDataStore -v
python -m pytest test_expense_tracker.py::TestDedupeLogic -v
```

### Test Coverage
The test suite covers:
- ✅ Database operations (insert, load, dedupe)
- ✅ Settings persistence
- ✅ Fuzzy duplicate detection
- ✅ Recurring rule management
- ✅ Merchant similarity algorithms
- ✅ Amount tolerance calculations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

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
