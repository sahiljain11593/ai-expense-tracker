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

1. Deploy this repo on Streamlit Cloud (link your GitHub repo)
2. Set Secrets (Settings → Secrets):
```
[firebase]
apiKey = "YOUR_WEB_API_KEY"
authDomain = "your-project.firebaseapp.com"
projectId = "your-project"
appId = "YOUR_APP_ID"

[auth]
allowed_email = "your@gmail.com"

[google]
client_id = "YOUR_OAUTH_CLIENT_ID"
client_secret = "YOUR_OAUTH_CLIENT_SECRET"
redirect_uri = "https://<your-streamlit-app-url>"
drive_folder_id = "optional-folder-id"
```
3. Reboot the app. You will see a Google Sign-In button; only the allowed email can proceed.
4. For Google Drive backup, authorize using the provided link and paste the `code` back to the app.

### Local E2E Smoke Test (no auth)
```bash
python3 -m venv .venv --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
. .venv/bin/activate && python get-pip.py
pip install -r requirements.txt
python scripts/e2e_check.py
```

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
