# AGENTS.md

## Cursor Cloud specific instructions

### Run locally

```bash
source .venv/bin/activate
streamlit run transaction_web_app.py --server.port 8501 --server.headless true
```

One-time VM packages if venv fails: `sudo apt-get install -y python3.12-venv tesseract-ocr`

### Tests

```bash
pytest
python scripts/e2e_check.py
python demo_advanced_features.py
```

### Notes

- `.streamlit/` is gitignored; minimal `secrets.toml` allows running without Firebase.
- Dates from pandas DataFrames are normalized via `normalize_date_for_db()` before SQLite writes.
- No repo linter configured (use `pytest` as the quality gate).
