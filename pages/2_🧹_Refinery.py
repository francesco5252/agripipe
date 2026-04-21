import pandas as pd
import streamlit as st

from agripipe.cleaner import AgriCleaner, CleanerConfig
from typing import cast
from agripipe.cleaner import ImputationStrategy, OutlierMethod

st.set_page_config(page_title="Refinery", page_icon="🧹", layout="wide")
st.header("Step 2 — Refinery")

if "df_raw" not in st.session_state or st.session_state.df_raw is None:
    st.info("Carica prima un file nello Step 1 (Ingestion).")
    st.stop()

df_raw = st.session_state.df_raw

st.markdown(
    "Scegli la strategia di imputazione e il metodo di rilevamento outlier, poi avvia la pulizia."
)

c_strat, c_outlier = st.columns(2)
with c_strat:
    missing_strategy = st.selectbox(
        "Strategia imputazione valori mancanti",
        options=["median", "mean", "ffill", "time", "drop"],
        help="median: sostituisce con la mediana · mean: con la media · ffill: propaga l'ultimo valore · time: interpolazione temporale · drop: rimuove le righe con NaN.",
    )
with c_outlier:
    outlier_method = st.selectbox(
        "Metodo rilevamento outlier",
        options=["iqr", "zscore", "none"],
    )

st.markdown("**Limiti fisici (opzionali)**")
c_ph, c_hum, c_temp = st.columns(3)
ph_lo = c_ph.number_input("pH min", value=0.0, step=0.1)
ph_hi = c_ph.number_input("pH max", value=14.0, step=0.1)
hum_lo = c_hum.number_input("Umidità % min", value=0.0, step=1.0)
hum_hi = c_hum.number_input("Umidità % max", value=100.0, step=1.0)
temp_lo = c_temp.number_input("Temp °C min", value=-30.0, step=1.0)
temp_hi = c_temp.number_input("Temp °C max", value=60.0, step=1.0)

if st.button("🧹 Avvia pulizia", type="primary"):
    numeric_cols = df_raw.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in ["field_id", "crop_type"] if c in df_raw.columns]
    date_cols = [c for c in ["date"] if c in df_raw.columns]

    missing_strat_cast = cast(ImputationStrategy, missing_strategy)
    outlier_strat_cast = cast(OutlierMethod, outlier_method)

    config = CleanerConfig(
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        date_columns=date_cols,
        dedup_keys=[c for c in ["field_id", "date"] if c in df_raw.columns],
        missing_strategy=missing_strat_cast,
        outlier_method=outlier_strat_cast,
        physical_bounds={
            "ph": (float(ph_lo), float(ph_hi)),
            "humidity": (float(hum_lo), float(hum_hi)),
            "temp": (float(temp_lo), float(temp_hi)),
        },
    )
    cleaner = AgriCleaner(config)
    with st.spinner("Pulizia in corso..."):
        df_clean = cleaner.clean(df_raw)

    st.session_state.cleaner = cleaner
    st.session_state.df_clean = df_clean

if st.session_state.get("df_clean") is not None:
    df_clean = st.session_state.df_clean
    diag = st.session_state.cleaner.diagnostics

    st.subheader("Integrità dopo pulizia")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Righe in → out", f"{diag.total_rows} → {len(df_clean)}")
    c2.metric("Fuori-range rimossi", f"{diag.out_of_bounds_removed}")
    c3.metric("Outlier rimossi", f"{diag.outliers_removed}")
    c4.metric("Valori imputati", f"{diag.values_imputed}")
    st.caption(
        f"Strategia imputazione applicata: **{diag.imputation_strategy_used}** · Duplicati rimossi: {diag.duplicates_removed}"
    )

    nan_before = df_raw.isna().sum()
    nan_after = df_clean.isna().sum().reindex(nan_before.index, fill_value=0)
    nan_df = pd.DataFrame({"grezzo": nan_before, "pulito": nan_after})
    with st.expander("NaN per colonna — prima vs. dopo"):
        st.dataframe(nan_df, use_container_width=True)

    with st.expander("Anteprima dati puliti (prime 20 righe)"):
        st.dataframe(df_clean.head(20), use_container_width=True)
