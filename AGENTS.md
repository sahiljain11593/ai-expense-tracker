# AGENTS.md

## Cursor Cloud specific instructions

See README for install/run. Key commands: `python3 -m pytest`, `streamlit run transaction_web_app.py`.

### Quick start for a new agent

```bash
cd /workspace
git fetch origin
git checkout cursor/stabilize-gemini-roadmap-755d
python3 -m pytest -q          # must be 87 passed
python3 -m py_compile transaction_web_app.py data_store.py services/translation.py services/merchants.py
```

### Current branch

`cursor/stabilize-gemini-roadmap-755d` — open PR #12 on GitHub.
All work for this session lives here. Merge to `main` to deploy to Streamlit Cloud.

### What was completed (session Jun 22 2026)

| Area | Status |
|---|---|
| GitHub Actions CI | ✅ `.github/workflows/ci.yml` — py_compile + pytest on every push |
| Gemini 3.1 Flash-Lite translation | ✅ Default AI provider; OpenAI + free fallback preserved |
| Batch CSV translation | ✅ One API call per upload instead of per-row |
| Persistent translation cache | ✅ `translation_cache` SQLite table |
| Retry/backoff | ✅ 3 attempts, 1s/2s/4s+jitter on 429/5xx |
| Single-prompt batch + JSON recovery | ✅ Gemini sends all uncached texts in one prompt; ≥80% recovery on truncation |
| PDF/image AI translation | ✅ Both extractors accept `translation_mode`/`api_key` |
| Static merchant library | ✅ `services/merchants.py` — 180+ JP merchants, zero-cost lookup |
| User merchant library | ✅ Auto-learns every new AI translation; management UI (add/edit/delete/bulk-import CSV) |
| Token/cost meter | ✅ Sidebar shows calls, tokens in/out, ~$ cost, cache size |
| Persist provider preference | ✅ `settings` DB table, restored on reload |
| Module split | ✅ `services/translation.py` extracted (4103 → 3618 line main file) |
| Auto Drive backup | ✅ Triggers after every `insert_transactions` when Drive is authorized |
| P0 correctness fixes | ✅ Insights category bug, positive-expense analytics, contextual learning |

### What is next (from ROADMAP.md)

- **P1.1** — Persistent translation cache is done. Remaining: check `merchant_learning` DB before calling AI for categorization.
- **P1.3** — Persistent categorization cache: `categorise_transactions_ai` should query `merchant_learning` first.
- **P1.4** — PDF/image translation is done. Remaining: batch mode for PDFs (currently per-row).
- **P2.1** — Token/cost meter is done. Remaining: add retries counter to the meter.
- **T10** — Supabase migration (needs `SUPABASE_URL` + `SUPABASE_KEY` secrets).

### Translation resolution order (cheapest first)

```
Step 0a  services/merchants.py MERCHANT_LIBRARY    180+ built-in JP brands    ~0 μs   $0
Step 0b  user_merchant_library DB                  user-learned merchants      ~1 ms   $0
Step 1   translation_cache DB                      all prior AI translations   ~1 ms   $0
Step 2   AI provider call                          only truly new text         ~1–5 s  tokens
```

### Key files

| File | Purpose |
|---|---|
| `transaction_web_app.py` | Main Streamlit UI (3,618 lines) |
| `data_store.py` | SQLite layer — all DB tables and CRUD |
| `services/translation.py` | AI translation service (Gemini + OpenAI + fallback) |
| `services/merchants.py` | Static + user merchant library |
| `ml_engine.py` | Ensemble categorization (initialized but not yet wired to main flow) |
| `insights_engine.py` | Financial analytics engine |
| `dashboard.py` | Plotly visualizations |
| `mobile_ui.py` | Compact layout helpers |
| `ROADMAP.md` | Source of truth for what is done and what is next |

### Running the smoke test (requires API key)

```bash
python3 tests/smoke_test_ai.py --provider gemini   # needs GEMINI_API_KEY
python3 tests/smoke_test_ai.py --provider openai   # needs OPENAI_API_KEY
```

Set keys via `.streamlit/secrets.toml`:
```toml
[gemini]
api_key = "..."

[openai]
api_key = "..."
```

### Gotchas

- Pandas dates are normalized via `normalize_date_for_db()` before SQLite writes.
- `.streamlit/` is gitignored; use `config.toml.example` as a template.
- `services/merchants.py` uses NFKC normalization — half-width katakana from CC statements is converted to full-width before lookup.
- Short ASCII-only keys (< 4 chars, e.g. `au`, `AWS`) are excluded from substring matching to prevent false positives.
- `user_merchant_library.source = 'user'` is never overwritten by auto-learn updates.
- Sidebar starts **collapsed** so the main workflow is visible first on mobile.

### Mobile browser testing

- Open the sidebar → **📱 Mobile** → enable **Compact layout** on your phone.
- CSS also stacks columns automatically under 768px width.
- Copy `.streamlit/config.toml.example` → `.streamlit/config.toml` and set `server.address = "0.0.0.0"` for LAN testing.
