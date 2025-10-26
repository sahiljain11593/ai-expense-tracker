import re
from collections import defaultdict

import pandas as pd
import streamlit as st

from data_store import (
    init_db,
    compute_dedupe_hash,
    insert_transactions,
    create_import_record,
)


def render_duplicate_analysis():
    st.title("🔍 Duplicate Analysis & Selective Import (Isolated)")
    st.caption("This page runs independently from the main processing to avoid reruns and delays.")

    init_db()

    # Optional uploader on this page
    st.subheader("Upload CSV (optional)")
    up = st.file_uploader("Upload a CSV with transactions", type=["csv"], key="dup_page_uploader")
    if up is not None:
        try:
            df = pd.read_csv(up)
            st.session_state['csv_file_uploaded'] = True
            st.session_state['csv_file_name'] = getattr(up, 'name', 'uploaded.csv')
            st.session_state['csv_file_size'] = getattr(up, 'size', len(df))
            st.session_state['csv_dataframe'] = df.to_dict('records')
            st.success("CSV loaded for analysis.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    # Require data
    if not st.session_state.get('csv_file_uploaded') or not st.session_state.get('csv_dataframe'):
        st.info("Upload a CSV here or use the main page to upload, then return.")
        return

    # Build file key
    raw_file_name = st.session_state.get('csv_file_name', 'unknown')
    file_size = st.session_state.get('csv_file_size', 0)
    sanitized_file_name = re.sub(r'[^a-zA-Z0-9_-]', '_', str(raw_file_name))
    file_key = f"duplicate_analysis_{sanitized_file_name}_{file_size}"

    # Analyze once per file
    if file_key not in st.session_state or not st.session_state[file_key]:
        st.info("Analyzing CSV for potential duplicates...")
        try:
            csv_data = st.session_state.get('csv_dataframe', [])
            if not csv_data:
                st.error("No CSV data available for analysis")
                st.session_state[file_key] = True
                return

            try:
                df_analysis = pd.DataFrame(csv_data)
            except Exception as e:
                st.error(f"Error creating dataframe: {e}")
                st.session_state[file_key] = True
                return

            hash_groups = defaultdict(list)
            for idx, row in df_analysis.iterrows():
                try:
                    date = str(row.get('date', ''))
                    desc = str(row.get('description', ''))
                    amount = float(row.get('amount', 0))
                    dh = compute_dedupe_hash(date, desc, amount)
                    hash_groups[dh].append({
                        'row': idx + 1,
                        'date': date,
                        'description': desc,
                        'amount': amount,
                    })
                except Exception:
                    continue

            duplicates = {h: g for h, g in hash_groups.items() if len(g) > 1}
            st.session_state[f'{file_key}_groups'] = duplicates
            st.session_state[f'{file_key}_df'] = df_analysis
            st.session_state[file_key] = True

            if duplicates:
                st.warning(f"Found {len(duplicates)} potential duplicate groups.")
            else:
                st.success("No duplicate transactions found in your CSV file.")
        except Exception as e:
            st.error(f"Duplicate analysis error: {e}")
            st.session_state[file_key] = True
            st.session_state.pop(f'{file_key}_groups', None)
            st.session_state.pop(f'{file_key}_df', None)

    # Display and import
    duplicates = st.session_state.get(f'{file_key}_groups', {})
    df_analysis = st.session_state.get(f'{file_key}_df')
    if not duplicates or df_analysis is None:
        return

    with st.expander(f"View {len(duplicates)} Duplicate Groups", expanded=True):
        with st.form(f"dup_form_{file_key}"):
            select_all = st.checkbox("Select all", key=f"dup_select_all_{file_key}")

            for i, (dedupe_hash, group) in enumerate(duplicates.items(), 1):
                st.write(f"**Group {i}** ({len(group)} transactions):")
                for j, tx in enumerate(group):
                    c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
                    with c1:
                        st.write(f"  • Row {tx['row']}: {tx['date']} | {tx['description']} | ¥{tx['amount']:.2f}")
                    with c2:
                        tx_key = f"{file_key}_{i}_{j}_{tx['row']}"
                        st.checkbox("Select", key=f"dup_select_{tx_key}", value=bool(select_all))
                    with c3:
                        st.caption("Duplicate")
                    with c4:
                        st.caption(f"Group {i}")
                st.caption(f"   Hash: {dedupe_hash[:16]}...")
                st.divider()

            if st.form_submit_button("Import Selected", type="primary"):
                try:
                    selected_keys = []
                    for i, group in enumerate(duplicates.values(), 1):
                        for j, tx in enumerate(group):
                            tx_key = f"{file_key}_{i}_{j}_{tx['row']}"
                            if st.session_state.get(f"dup_select_{tx_key}", False):
                                selected_keys.append(tx_key)

                    if not selected_keys:
                        st.warning("No transactions selected.")
                        return

                    safe_file_name = st.session_state.get('csv_file_name', 'unknown_file')
                    import_batch_id = create_import_record(f"{safe_file_name}_duplicates", len(selected_keys))

                    transactions_to_import = []
                    for tx_key in selected_keys:
                        parts = tx_key.split('_')
                        row_num = int(parts[-1]) - 1
                        if 0 <= row_num < len(df_analysis):
                            row_data = df_analysis.iloc[row_num].to_dict()
                            try:
                                tx_data = {
                                    'date': str(row_data.get('date', '')),
                                    'description': str(row_data.get('description', '')),
                                    'original_description': str(row_data.get('original_description', '')),
                                    'amount': float(row_data.get('amount', 0)),
                                    'currency': str(row_data.get('currency', 'JPY')),
                                    'fx_rate': float(row_data.get('fx_rate', 1.0)),
                                    'amount_jpy': float(row_data.get('amount_jpy', row_data.get('amount', 0))),
                                    'category': str(row_data.get('category', '')),
                                    'subcategory': str(row_data.get('subcategory', '')),
                                    'transaction_type': str(row_data.get('transaction_type', 'Expense')),
                                }
                            except Exception:
                                continue
                            transactions_to_import.append(tx_data)

                    inserted, dupes, errors = insert_transactions(transactions_to_import, import_batch_id)
                    if inserted:
                        st.success(f"Imported {inserted} transactions (skipped {dupes} duplicates).")
                    else:
                        st.error("No transactions were imported.")
                except Exception as e:
                    st.error(f"Import failed: {e}")


if __name__ == "__main__":
    render_duplicate_analysis()
