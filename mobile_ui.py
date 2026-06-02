"""
Mobile-friendly layout helpers and global CSS for the Streamlit expense tracker.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import streamlit as st

COMPACT_LAYOUT_KEY = "compact_layout"


def use_compact_layout() -> bool:
    return bool(st.session_state.get(COMPACT_LAYOUT_KEY, False))


def set_compact_layout_default() -> None:
    if COMPACT_LAYOUT_KEY not in st.session_state:
        st.session_state[COMPACT_LAYOUT_KEY] = False


def render_sidebar_mobile_controls() -> None:
    set_compact_layout_default()
    st.sidebar.markdown("### 📱 Mobile")
    st.session_state[COMPACT_LAYOUT_KEY] = st.sidebar.toggle(
        "Compact layout",
        value=st.session_state[COMPACT_LAYOUT_KEY],
        help="Stacks columns, larger tap targets, and card-style categorization.",
    )
    st.sidebar.caption(
        "Narrow screens auto-stack columns. Bookmark your Streamlit URL on your phone."
    )


def inject_mobile_styles() -> None:
    compact_js = "true" if use_compact_layout() else "false"
    st.markdown(
        f"""
<style>
  .stButton > button {{ min-height: 2.75rem; padding: 0.5rem 1rem; }}
  .stSelectbox > div, .stMultiSelect > div, .stTextInput > div input {{ min-height: 2.75rem; }}
  div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {{
    overflow-x: auto; -webkit-overflow-scrolling: touch;
  }}
  body.compact-layout .stHorizontalBlock {{ flex-wrap: wrap !important; gap: 0.5rem; }}
  body.compact-layout [data-testid="column"] {{
    width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important;
  }}
  @media (max-width: 768px) {{
    .stHorizontalBlock {{ flex-wrap: wrap !important; gap: 0.5rem; }}
    [data-testid="column"] {{
      width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important;
    }}
    h1 {{ font-size: 1.5rem !important; }}
    .stTabs [data-baseweb="tab"] {{ min-height: 2.75rem; }}
  }}
  .mobile-tx-card {{
    border: 1px solid rgba(128,128,128,0.35);
    border-radius: 0.75rem; padding: 0.75rem 1rem; margin-bottom: 0.75rem;
  }}
</style>
<script>document.body.classList.toggle("compact-layout", {compact_js});</script>
        """,
        unsafe_allow_html=True,
    )


def layout_columns(count: int, *, weights: Optional[Sequence[float]] = None):
    if use_compact_layout():
        return [st.container() for _ in range(count)]
    if weights is not None:
        return st.columns(list(weights))
    return st.columns(count)


def default_description_column_width() -> str:
    return "small" if use_compact_layout() else "large"


def mobile_saved_transactions_display_columns(df_saved) -> List[str]:
    base = ["date", "description", "amount_jpy", "category", "transaction_type"]
    if not use_compact_layout() and "original_description" in df_saved.columns:
        base.insert(3, "original_description")
    return [c for c in base if c in df_saved.columns]


def render_transaction_filters(
    *,
    key_prefix: str,
    category_options: Sequence[str],
    type_options: Sequence[str],
    currency_options: Optional[Sequence[str]] = None,
    show_dates: bool = False,
) -> Tuple[Optional[list], Optional[list], Optional[list], Optional[str], Optional[object], Optional[object]]:
    categories = types = currencies = None
    search = None
    date_from = date_to = None
    expanded = not use_compact_layout()

    with st.expander("🔍 Filters", expanded=expanded):
        if use_compact_layout():
            categories = st.multiselect(
                "Categories", options=sorted(category_options), default=None, key=f"{key_prefix}_cat"
            )
            types = st.multiselect(
                "Type", options=list(type_options), default=None, key=f"{key_prefix}_type"
            )
            if currency_options is not None:
                currencies = st.multiselect(
                    "Currency", options=sorted(currency_options), default=None, key=f"{key_prefix}_cur"
                )
            if show_dates:
                date_from = st.date_input("From Date", value=None, key=f"{key_prefix}_from")
                date_to = st.date_input("To Date", value=None, key=f"{key_prefix}_to")
            search = st.text_input("Search Description", placeholder="Search...", key=f"{key_prefix}_search")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                categories = st.multiselect(
                    "Categories", options=sorted(category_options), default=None, key=f"{key_prefix}_cat"
                )
            with c2:
                types = st.multiselect(
                    "Type", options=list(type_options), default=None, key=f"{key_prefix}_type"
                )
            with c3:
                if show_dates:
                    date_from = st.date_input("From Date", value=None, key=f"{key_prefix}_from")
                    date_to = st.date_input("To Date", value=None, key=f"{key_prefix}_to")
                elif currency_options is not None:
                    currencies = st.multiselect(
                        "Currency", options=sorted(currency_options), default=None, key=f"{key_prefix}_cur"
                    )
            with c4:
                search = st.text_input("Search Description", placeholder="Search...", key=f"{key_prefix}_search")

    return categories, types, currencies, search, date_from, date_to
