"""AgriPipe — UI Multipagina Streamlit.

Questa è la Landing Page dell'applicazione. Usa il menu laterale
per navigare nei tre step:
1. Ingestion
2. Refinery
3. Tensorizer
"""

import streamlit as st

# Setup pagina
st.set_page_config(
    page_title="AgriPipe — ML-Ops Pipeline",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 AgriPipe")
st.markdown(
    "Benvenuto in **AgriPipe**, la pipeline Data-to-Tensor per l'agricoltura di precisione."
)
st.caption("Excel agronomico grezzo → DataFrame pulito → Tensor PyTorch.")
st.markdown(
    "⬅️ **Naviga utilizzando il pannello laterale** per affrontare le 3 fasi del processing."
)

# Inizializzo chiavi di stato condivise globalmente fra le pagine
for k in ["df_raw", "df_clean", "cleaner", "file_hash", "source_name"]:
    if k not in st.session_state:
        st.session_state[k] = None
