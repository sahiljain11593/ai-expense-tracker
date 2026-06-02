# AGENTS.md

## Cursor Cloud specific instructions

### Services

| Service | Purpose | Dev command |
|---------|---------|-------------|
| Streamlit app | Main UI (`transaction_web_app.py`) | `streamlit run transaction_web_app.py --server.port 8501` |
| SQLite | Local data (`data/expenses.db`) | Created on first `init_db()` / app run |

No separate API or Docker stack is required for normal development.

### Commands (see also `README.md`)

- **Dependencies:** `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- **Tests:** `pytest -v` (35 tests including Rakuten parser and hybrid categorization)
- **Demo:** `python demo_advanced_features.py`

### Non-obvious notes

- **Rakuten card PDFs:** Upload uses `extract_transactions_from_pdf`, which tries `extract_rakuten_card_pdf` first (Japanese ご利用明細 text), then English table headers. Generic table parsing guards `None` cells to avoid `.strip()` errors.
- **Categorization:** `get_categorization_rules()` / `apply_hybrid_categorization()` in `categorization_config.py` and `categorization_engine.py` (rules → merchant learning DB → optional ensemble ML). `apply_smart_categorization` in the app is an alias for hybrid.
- **Merchant learning:** `MerchantLearningSystem` loads persisted rows via `data_store.load_merchant_learning()` and writes corrections with `learn_from_categorization`.
- **Streamlit in tests/scripts:** Importing `transaction_web_app` may log harmless “missing ScriptRunContext” warnings outside `streamlit run`.
- **OCR/PDF extras:** Tesseract is optional; PDF parsing needs `pdfplumber` only.
