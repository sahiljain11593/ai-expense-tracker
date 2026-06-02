# AI Expense Tracker — Roadmap & Checklist

**Last updated:** May 8, 2026
**Current state:** Multi-provider AI (OpenAI + Gemini) integrated. P0 urgent fixes from the Opus 4.7 code review are complete.

This document is the source of truth for what's done, what's next, and how to verify each change. Update the checkboxes as items land.

---

## Quick Status

| Phase | Focus | Status |
|---|---|---|
| **P0** | Correctness & urgent fixes | DONE |
| **P1** | Cost & quality | NEXT UP |
| **P2** | Observability | After P1 |
| **P3** | UX polish | After P2 |
| **Test automation** | Smoke-test agent for sample CSV | Set up alongside P1 |

---

## P0 — Correctness (DONE)

These were the must-fixes from the Opus 4.7 review.

- [x] **H1/H2 — Provider-aware key & model defaults.** `_default_key_for()` / `_default_model_for()` resolve based on the selected provider; no more silent OpenAI fallback when Gemini is selected.
- [x] **H3 — Empty Gemini response guard.** `_gemini_call()` disables thinking via `ThinkingConfig(thinking_budget=0)`, raises a clear error if `response.text` is empty, and uses larger token budgets (`max(800, 80*N)` translation, `max(1500, 120*N)` categorization).
- [x] **H4 — Exponential backoff retry.** `_ai_generate()` retries up to 3 times for 429/5xx/network errors with `1s + 2s + 4s + jitter`.
- [x] **M9 — Subcategory validation.** Categorization now drops subcategories that don't belong to their parent.
- [x] **L16 — Repo cleanup.** Stray `https:/aistudio.google.com/` directory removed; `.gitignore` updated.
- [x] **Model upgrade.** Added `gemini-3.1-flash-lite` (Google's newest, free-tier, optimized for translation) as the default Gemini model.
- [x] **Merchant-protection fix.** `[[BRAND_N]]` placeholder tokens caused the model to copy translations across rows in batch mode (every row looked structurally identical). Replaced with inline English-name substitution for batch translation via new helper `protect_known_merchants_inline`. Smoke test now hits **10/10 merchant recognition** vs 4/10 before.

---

## P1 — Cost & Quality (NEXT UP)

The single biggest savings come from caching. Items are roughly ordered by ROI.

### P1.1 — Persistent SQLite translation cache (HIGH IMPACT)
The session-scoped cache in `st.session_state["translation_cache"]` evaporates on Streamlit Cloud restarts and re-burns API budget on every re-upload.

- [ ] Create `translation_cache` table in `data_store.py`:
  ```sql
  CREATE TABLE IF NOT EXISTS translation_cache (
    jp_text     TEXT PRIMARY KEY,
    en_text     TEXT NOT NULL,
    model       TEXT,
    provider    TEXT,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
  );
  ```
- [ ] Add `data_store.get_cached_translations(jp_texts: list[str]) -> dict[str, str]`
- [ ] Add `data_store.save_translations(mapping: dict, model: str, provider: str)`
- [ ] In `extract_transactions_from_csv` (around line ~920), check the SQLite cache **before** the session cache, and write back to SQLite after successful AI calls.
- [ ] Acceptance: re-uploading the same file twice should make zero AI calls on the second upload.

### P1.2 — Per-row JSON recovery in batch translation
Currently a single truncated row dumps the entire batch to the free Google Translate fallback.

- [ ] In `translate_batch_ai` (around line ~675), wrap `_json.loads(raw)` in a recovery block: when it fails, regex-extract `"\d+"\s*:\s*"[^"]*"` pairs and parse what we can.
- [ ] Same pattern in `categorise_transactions_ai` (around line ~1597).
- [ ] Acceptance: a deliberately malformed JSON response should still recover ≥80% of rows.

### P1.3 — Persistent categorization cache (use `merchant_learning`)
The DB already has a `merchant_learning` table. `categorise_transactions_ai` doesn't consult it before calling the LLM.

- [ ] In `categorise_transactions_ai`, query `merchant_learning` first; only send unresolved rows to the LLM.
- [ ] Acceptance: after the user reviews/saves transactions once, similar merchants in the next upload should be auto-categorized without an AI call.

### P1.4 — PDF & image AI translation
Today only `extract_transactions_from_csv` accepts `translation_mode/api_key/ai_model`.

- [ ] Refactor `extract_transactions_from_pdf` and `extract_transactions_from_image` to accept the same params and run rows through `translate_batch_ai`.
- [ ] Acceptance: uploading a Japanese PDF statement should produce English-translated rows.

---

## P2 — Observability (after P1)

### P2.1 — Token / cost meter
- [ ] Capture `response.usage` (OpenAI) and `response.usage_metadata` (Gemini) inside `_openai_call` / `_gemini_call`. Aggregate into `st.session_state['ai_usage']`.
- [ ] Show a small sidebar block:  `AI calls: X · retries: Y · fallbacks: Z · in: 12.4k tok · out: 1.1k tok`.
- [ ] Estimated cost line based on the price table for the active model.

### P2.2 — Cache management UI
- [ ] "Clear translation cache" button.
- [ ] "Clear categorization cache" button.
- [ ] Show cache size (#rows in `translation_cache`).

---

## P3 — UX Polish (after P2)

### P3.1 — Persist provider/model preference
- [ ] Save `ai_provider` and `ai_model` selection to the `settings` table on change. Restore on app load.

### P3.2 — Always show provider dropdown
- [ ] Even when only one secret is configured, show the dropdown so the user can override (currently hidden — see line ~1880).

### P3.3 — Custom model option
- [ ] Add `"Custom..."` to the model selectbox; when chosen, show a `text_input` for the model name. Future-proofs for new releases.

### P3.4 — Remove stale labels
- [ ] Remove the legacy `"AI-Powered (GPT-4o-mini)"` mode string check in `translate_japanese_to_english` and `extract_transactions_from_csv`.

### P3.5 — Cost/quota help text
- [ ] Replace hard-coded `~$0.01/100 txns` strings with values pulled from the model registry (or remove and let the meter show actual usage).

---

## Test Automation

### Smoke-test agent
A single Python script that runs the full AI pipeline against a sample Japanese CSV, so neither you nor I have to manually drag-and-drop a file in the browser to verify a change.

- [x] **Sample CSV.** `tests/data/sample_japanese_statement.csv` — diverse Japanese merchants (restaurants, drugstores, transit, online services, abbreviated katakana).
- [x] **Smoke test script.** `tests/smoke_test_ai.py`:
  - Loads the sample CSV
  - Runs translation through both Gemini (default) and OpenAI (if key present)
  - Runs categorization on the translated rows
  - Reports pass/fail + a summary table (merchant → translation → category)
  - Returns non-zero exit code on failure
- [x] **How to invoke.**
  - From terminal: `python tests/smoke_test_ai.py --provider gemini`
  - Through me: just say "run the smoke test" and I'll execute it via a subagent.
- [ ] **Pytest integration (optional).** Mark it with `@pytest.mark.smoke` and add a `pytest -m smoke` recipe to the README so CI can opt in.

### Existing pytest suite
- [x] `tests/test_import_dedupe.py` — 12 tests
- [x] `tests/test_recurring.py` — 14 tests
- [ ] **Coverage gap:** no tests exist for `_ai_generate`, `_resolve_provider`, `_is_retryable_error`. Add unit tests with mocks (no real API calls).

---

## Definition of Done (each P1-P3 item)

1. Code change merged to working tree
2. `python -m py_compile transaction_web_app.py` passes
3. `pytest -v` passes (no regressions)
4. `python tests/smoke_test_ai.py` passes against at least one provider
5. Roadmap checkbox flipped to `[x]`
6. Commit message follows project convention (`feat:`, `fix:`, `chore:`, etc.)

---

## Future / Backlog (not yet prioritized)

- Receipt scanning with stronger OCR (PaddleOCR, surya)
- Budget envelope system
- Email notifications for unusual spending
- Mobile app (React Native)
- Supabase migration for cloud persistence + multi-device
- Native Anthropic Claude provider (similar pattern to Gemini integration)
- Streaming responses for very long files
