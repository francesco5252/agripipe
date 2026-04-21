import tempfile
from pathlib import Path

import streamlit as st

from agripipe.loader import load_raw

st.set_page_config(page_title="Ingestion", page_icon="📥", layout="wide")
st.header("Step 1 — Ingestion")
st.markdown(
    "Carica un file Excel (`.xlsx`/`.xls`) o CSV. Lo schema minimo richiesto: `date`, `field_id`, `temp`, `humidity`, `ph`, `yield`."
)

uploaded_file = st.file_uploader(
    "File agricolo grezzo", type=["xlsx", "xls", "csv"], key="uploader"
)

if uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    try:
        df_raw = load_raw(tmp_path)
        st.session_state.df_raw = df_raw
        st.session_state.file_hash = df_raw.attrs.get("file_hash", "unknown")
        st.session_state.source_name = uploaded_file.name

        c1, c2, c3 = st.columns(3)
        c1.metric("Righe caricate", f"{len(df_raw):,}")
        c2.metric("Colonne", f"{df_raw.shape[1]}")
        c3.metric("SHA-256 (12)", st.session_state.file_hash[:12])
        st.caption(f"File: `{uploaded_file.name}` · Hash completo: `{st.session_state.file_hash}`")

        with st.expander("Anteprima dati grezzi (prime 20 righe)"):
            st.dataframe(df_raw.head(20), use_container_width=True)
    except (ValueError, FileNotFoundError) as e:
        st.error(f"Errore di caricamento: {e}")
        st.session_state.df_raw = None
