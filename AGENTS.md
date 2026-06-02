# AGENTS.md

## Cursor Cloud specific instructions

See README for install/run. Key commands: `source .venv/bin/activate`, `streamlit run transaction_web_app.py`, `pytest`.

### Mobile browser testing

- Open the sidebar → **📱 Mobile** → enable **Compact layout** on your phone.
- CSS also stacks columns automatically under 768px width.
- Sidebar starts **collapsed** so the main workflow is visible first.
- Sample Streamlit config for LAN testing from a phone: copy `.streamlit/config.toml.example` to `.streamlit/config.toml` and set `server.address = "0.0.0.0"`.

### Gotchas

- Pandas dates are normalized via `normalize_date_for_db()` before SQLite writes.
- `.streamlit/` is gitignored; use `config.toml.example` as a template.
