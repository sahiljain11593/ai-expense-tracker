# AGENTS.md

Guidance for AI agents working in this repository.

## Cursor Cloud specific instructions

### Product

Single **Streamlit** app (`transaction_web_app.py`) with embedded **SQLite** (`data/expenses.db`). No separate API server, Docker Compose, or npm workspace. Optional externals: OpenAI (translation), Firebase (auth), Google Drive (backup), Tesseract (image OCR).

### One-time VM packages (not in update script)

If `python3 -m venv` fails with `ensurepip` missing:

```bash
sudo apt-get install -y python3.12-venv tesseract-ocr
```

Tesseract is only required for **image** statement uploads; CSV/PDF paths work without it.

### Dependencies (automatic on startup)

The VM **update script** creates `.venv` if needed and runs `pip install -r requirements.txt`. Activate with:

```bash
source .venv/bin/activate
```

### Run the app (manual — not in update script)

```bash
source .venv/bin/activate
streamlit run transaction_web_app.py --server.port 8501 --server.headless true
```

Or: `python run_app.py` (same port 8501).

App URL: `http://127.0.0.1:8501/`

### Streamlit secrets

`.streamlit/` is gitignored. For local/cloud VM runs **without** Firebase, a minimal `.streamlit/secrets.toml` is enough (empty sections or placeholder values); the app can run with auth disabled when Firebase is not configured.

Production secrets (Firebase, Google, OpenAI) are documented in `README.md`.

### Tests and smoke checks

| Command | Purpose |
|---------|---------|
| `pytest` | 26 unit tests (`tests/`) — no server required |
| `python scripts/e2e_check.py` | Headless DB import/export/backup smoke test |
| `python demo_advanced_features.py` | ML/insights demo (may hit known unpack errors on learning step) |

There is **no** configured linter (ruff/flake8/black) in the repo.

### Gotchas

- **Hot reload**: Streamlit reloads Python on save; reinstalling packages with `pip` while the server is running may require a browser refresh or server restart.
- **PayPay / multi-column CSVs**: Some bank exports use split amount columns; use a simple 3-column CSV for UI upload tests, or `scripts/e2e_check.py` / `insert_transactions` for programmatic imports.
- **Save from categorization UI**: If save fails with `type 'Timestamp' is not supported`, the `date` field in `save_categorization_progress` needs string/date normalization (pandas `Timestamp` vs SQLite).
- **`data_store.py`**: `insert_transactions` expects date values compatible with SQLite binding; `insert_transactions` normalizes dates internally, but session progress save may not.

### Services summary

| Service | Required? |
|---------|-----------|
| Streamlit on 8501 | Yes for UI E2E |
| SQLite file DB | Yes (auto-created) |
| Tesseract | Optional (images) |
| OpenAI / Firebase / Google | Optional |
