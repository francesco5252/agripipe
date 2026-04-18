import streamlit as st
import pandas as pd
import yaml
from pathlib import Path
import io

from agripipe.loader import load_raw
from agripipe.cleaner import AgriCleaner, CleanerConfig
from agripipe.report import generate_report

st.set_page_config(page_title="AgriPipe UI - Pro Edition", page_icon="🇮🇹", layout="wide")

# --- CARICAMENTO CONOSCENZA ---
@st.cache_data
def load_agri_knowledge():
    with open("configs/agri_knowledge.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

knowledge = load_agri_knowledge()
presets = knowledge.get("regional_presets", {})

st.title("🌱 AgriPipe: Intelligenza Territoriale Italiana")
st.markdown("""
Questa applicazione pulisce e ottimizza i dati agronomici basandosi sulla **specificità del territorio italiano**.
Seleziona la tua coltura e zona per applicare le regole personalizzate.
""")

# --- STEP 1: SELEZIONE TERRITORIO ---
st.header("📍 1. Inquadramento Territoriale")
selected_key = st.selectbox(
    "Seleziona la combinazione Coltura + Zona:",
    options=list(presets.keys()),
    format_func=lambda x: f"{presets[x]['crop'].replace('_', ' ').title()} - {presets[x]['zona']}"
)

if selected_key:
    conf = presets[selected_key]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Tessitura Suolo:** {conf['suolo_tessitura']}")
    with col2:
        st.info(f"**Resa Max Attesa:** {conf['max_yield']} t/ha")
    with col3:
        st.info(f"**Range pH Ideale:** {conf['ideal_ph'][0]} - {conf['ideal_ph'][1]}")
    
    st.caption(f"ℹ️ **Note del territorio:** {conf['note']}")

# --- STEP 2: CARICAMENTO DATI ---
st.header("📁 2. Caricamento Dati Aziendali")
uploaded_file = st.file_uploader("Trascina qui il file Excel di X Farm", type=["xlsx", "csv"])

if uploaded_file is not None:
    st.success(f"✅ File per '{conf['zona']}' pronto per l'analisi.")
    
    if st.button("🚀 Avvia Ottimizzazione Personalizzata"):
        with st.spinner("Applicando regole territoriali specifiche..."):
            # Lettura
            if uploaded_file.name.endswith(".xlsx"):
                df_raw = pd.read_excel(uploaded_file)
            else:
                df_raw = pd.read_csv(uploaded_file)

            # Preparazione Cleaner Personalizzato
            # Creiamo una configurazione che forza i limiti della zona scelta
            custom_config = CleanerConfig(
                numeric_columns=df_raw.select_dtypes(include=['number']).columns.tolist(),
                physical_bounds={
                    "ph": tuple(conf['ideal_ph']),
                    "yield": (0, conf['max_yield']),
                    "temp": tuple(conf['temp_range'])
                },
                knowledge_path="configs/agri_knowledge.yaml"
            )
            
            cleaner = AgriCleaner(custom_config)
            df_clean = cleaner.clean(df_raw)
            
            # --- RISULTATI ---
            st.header("📊 3. Risultati dell'Ottimizzazione")
            
            # Metriche
            m1, m2, m3 = st.columns(3)
            m1.metric("Righe Pulite", len(df_clean))
            m2.metric("Anomalie Corrette", len(df_raw) - len(df_clean) + df_clean.isna().sum().sum())
            m3.metric("Affidabilità Dato", f"{min(100, (len(df_clean)/len(df_raw))*100):.1f}%")

            # Preparazione Esportazione
            st.write("---")
            
            # 1. Funzione per Excel
            def to_excel(df):
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Dati_Ottimizzati')
                return output.getvalue()

            # 2. Funzione per Tensor PyTorch (Novità!)
            from agripipe.dataset import AgriDataset
            import torch

            def to_tensors(df, conf):
                # Usiamo AgriDataset per la conversione rigida
                ds = AgriDataset(
                    df=df,
                    numeric_columns=df.select_dtypes(include=['number']).columns.tolist(),
                    target="yield" if "yield" in df.columns else None
                )
                output = io.BytesIO()
                # Salviamo un dizionario contenente tutto il necessario per l'IA
                torch.save({
                    "features": ds.features,
                    "target": ds.target,
                    "feature_names": ds.feature_names,
                    "metadata": {
                        "zona": conf['zona'],
                        "crop": conf['crop'],
                        "tessitura": conf['suolo_tessitura']
                    }
                }, output)
                return output.getvalue()

            excel_data = to_excel(df_clean)
            tensor_data = to_tensors(df_clean, conf)
            
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label=f"📥 Scarica Excel Pulito ({conf['zona']})",
                    data=excel_data,
                    file_name=f"agripipe_{selected_key}_clean.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.caption("Ideale per Agronomi e gestionali Excel.")
            
            with col_dl2:
                st.download_button(
                    label="💾 Scarica Tensor (PyTorch)",
                    data=tensor_data,
                    file_name=f"agripipe_{selected_key}_tensors.pt",
                    mime="application/octet-stream"
                )
                st.caption("Destinato al team AI (formato tensoriale rigido).")
            
            # --- ANALISI VISIVA (Novità!) ---
            st.write("---")
            st.header("📊 4. Analisi Visiva della Qualità")
            st.markdown("Confronto tra i dati **Grezzi** (input) e quelli **Ottimizzati** (output) secondo le regole territoriali.")

            import matplotlib.pyplot as plt
            import seaborn as sns

            numeric_cols = df_raw.select_dtypes(include=['number']).columns.tolist()
            # Mostriamo i grafici per le colonne più importanti
            for col in [c for c in ["yield", "temp", "ph", "humidity"] if c in numeric_cols]:
                with st.expander(f"Dettaglio Colonna: {col.upper()}", expanded=(col == "yield")):
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # 1. Boxplot per Outlier
                    data_cmp = pd.concat([
                        df_raw[[col]].assign(Stato="Grezzo"),
                        df_clean[[col]].assign(Stato="Ottimizzato")
                    ])
                    sns.boxplot(data=data_cmp, x="Stato", y=col, ax=ax1, palette=["#ff9999", "#66b3ff"])
                    ax1.set_title(f"Rimozione Anomalie in {col}")
                    
                    # 2. Distribuzione
                    sns.kdeplot(df_raw[col], fill=True, label="Grezzo", ax=ax2, color="#ff9999")
                    sns.kdeplot(df_clean[col], fill=True, label="Ottimizzato", ax=ax2, color="#66b3ff")
                    ax2.set_title(f"Distribuzione di {col}")
                    ax2.legend()
                    
                    st.pyplot(fig)
            
            st.subheader("Anteprima Dati Ottimizzati")
            st.dataframe(df_clean.head(20))
