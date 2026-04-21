"""AgriPipe — UI monolitica Streamlit.

Tre step chiari, niente decorazioni superflue:

1. **Ingestion** — carica Excel/CSV, mostra anteprima e SHA-256.
2. **Refinery** — pulisce i dati con la strategia scelta e mostra metriche
   prima/dopo.
3. **Tensorizer** — produce il bundle ML (``.pt`` + ``metadata.json``) con
   split train/val/test configurabile; download diretto del ``.zip``.

Eseguire con::

    streamlit run app.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from agripipe.cleaner import AgriCleaner, CleanerConfig
from agripipe.export import export_ml_bundle
from agripipe.loader import load_raw

# ---------------------------------------------------------------------------
# Setup pagina
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AgriPipe — ML-Ops Pipeline",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 AgriPipe")
st.caption(
    "Excel agronomico grezzo → DataFrame pulito → tensor PyTorch. Pipeline ML-Ops in 3 step."
)

# Stato condiviso fra gli step (Streamlit ricarica tutto lo script a ogni interazione)
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "cleaner" not in st.session_state:
    st.session_state.cleaner = None
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None
if "source_name" not in st.session_state:
    st.session_state.source_name = None


# ---------------------------------------------------------------------------
# Step 1 — Ingestion
# ---------------------------------------------------------------------------

st.header("Step 1 — Ingestion")
st.markdown(
    "Carica un file Excel (`.xlsx`/`.xls`) o CSV. Lo schema minimo richiesto: "
    "`date`, `field_id`, `temp`, `humidity`, `ph`, `yield`."
)

uploaded_file = st.file_uploader(
    "File agricolo grezzo", type=["xlsx", "xls", "csv"], key="uploader"
)

if uploaded_file is not None:
    # Salva il file su disco temporaneo per poterlo far leggere a load_raw (che prende un path)
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


# ---------------------------------------------------------------------------
# Step 2 — Refinery
# ---------------------------------------------------------------------------

st.header("Step 2 — Refinery")

if st.session_state.df_raw is None:
    st.info("Carica prima un file nello Step 1.")
else:
    df_raw = st.session_state.df_raw

    st.markdown(
        "Scegli la strategia di imputazione e il metodo di rilevamento outlier, "
        "poi avvia la pulizia."
    )

    c_strat, c_outlier = st.columns(2)
    with c_strat:
        missing_strategy = st.selectbox(
            "Strategia imputazione valori mancanti",
            options=["median", "mean", "ffill", "time", "drop"],
            help=(
                "median: sostituisce con la mediana · mean: con la media · "
                "ffill: propaga l'ultimo valore · time: interpolazione temporale · "
                "drop: rimuove le righe con NaN."
            ),
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

        config = CleanerConfig(
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            date_columns=date_cols,
            dedup_keys=[c for c in ["field_id", "date"] if c in df_raw.columns],
            missing_strategy=missing_strategy,
            outlier_method=outlier_method,
            physical_bounds={
                "ph": (ph_lo, ph_hi),
                "humidity": (hum_lo, hum_hi),
                "temp": (temp_lo, temp_hi),
            },
        )
        cleaner = AgriCleaner(config)
        with st.spinner("Pulizia in corso..."):
            df_clean = cleaner.clean(df_raw)

        st.session_state.cleaner = cleaner
        st.session_state.df_clean = df_clean

    # Mostra i risultati se abbiamo un df pulito
    if st.session_state.df_clean is not None:
        df_clean = st.session_state.df_clean
        diag = st.session_state.cleaner.diagnostics

        st.subheader("Integrità dopo pulizia")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Righe in → out", f"{diag.total_rows} → {len(df_clean)}")
        c2.metric("Fuori-range rimossi", f"{diag.out_of_bounds_removed}")
        c3.metric("Outlier rimossi", f"{diag.outliers_removed}")
        c4.metric("Valori imputati", f"{diag.values_imputed}")
        st.caption(
            f"Strategia imputazione applicata: **{diag.imputation_strategy_used}** · "
            f"Duplicati rimossi: {diag.duplicates_removed}"
        )

        # NaN prima/dopo
        nan_before = df_raw.isna().sum()
        nan_after = df_clean.isna().sum().reindex(nan_before.index, fill_value=0)
        nan_df = pd.DataFrame({"grezzo": nan_before, "pulito": nan_after})
        with st.expander("NaN per colonna — prima vs. dopo"):
            st.dataframe(nan_df, use_container_width=True)

        with st.expander("Anteprima dati puliti (prime 20 righe)"):
            st.dataframe(df_clean.head(20), use_container_width=True)


# ---------------------------------------------------------------------------
# Step 3 — Tensorizer & Export
# ---------------------------------------------------------------------------

st.header("Step 3 — Tensorizer")

if st.session_state.df_clean is None:
    st.info("Completa prima la pulizia nello Step 2.")
else:
    df_clean = st.session_state.df_clean
    cleaner = st.session_state.cleaner

    st.markdown(
        "Configura scaler, encoding categorico e split del dataset. "
        "Il bundle `.zip` conterrà i tensor `.pt` (uno per split) e il manifest `metadata.json`."
    )

    c_target, c_scale, c_cat = st.columns(3)
    target_options = [c for c in df_clean.columns if c in ("yield", "resa")] or ["(nessuno)"]
    target_choice = c_target.selectbox(
        "Colonna target",
        options=(
            target_options + ["(nessuno)"] if "(nessuno)" not in target_options else target_options
        ),
    )
    target = None if target_choice == "(nessuno)" else target_choice

    scaling_strategy = c_scale.selectbox("Scaler", options=["standard", "robust"])
    categorical_strategy = c_cat.selectbox("Encoding categoriche", options=["label", "onehot"])

    st.markdown("**Split train / val / test** — la somma deve fare 100%.")
    c_tr, c_va, c_te = st.columns(3)
    pct_train = c_tr.slider("Train %", 50, 95, 70, step=5)
    pct_val = c_va.slider("Val %", 0, 40, 15, step=5)
    pct_test = c_te.slider("Test %", 0, 40, 15, step=5)
    total = pct_train + pct_val + pct_test

    if total != 100:
        st.warning(f"La somma è {total}% (deve essere 100%).")
    else:
        split_ratios: tuple[float, float, float] | None = (
            pct_train / 100,
            pct_val / 100,
            pct_test / 100,
        )
        if pct_val == 0 or pct_test == 0:
            # Con uno dei due a zero il Tensorizer esplode: meglio disabilitare lo split.
            st.info("Val o Test a 0%: lo split verrà disabilitato.")
            split_ratios = None

        if st.button("🚀 Genera bundle ML", type="primary"):
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                name = Path(st.session_state.source_name or "agripipe_export").stem
                try:
                    with st.spinner("Tensorizzazione ed export in corso..."):
                        paths = export_ml_bundle(
                            df_clean=df_clean,
                            cleaner=cleaner,
                            preset={},
                            output_dir=td_path,
                            name=name,
                            target=target,
                            split_ratios=split_ratios,
                            scaling_strategy=scaling_strategy,
                            categorical_strategy=categorical_strategy,
                        )
                    zip_bytes = paths["zip"].read_bytes()
                    json_text = paths["json"].read_text(encoding="utf-8")
                except ValueError as e:
                    st.error(f"Errore nella tensorizzazione: {e}")
                    zip_bytes = None
                    json_text = None

            if zip_bytes is not None:
                st.success("Bundle pronto.")
                st.download_button(
                    "💾 Scarica bundle ML (.zip)",
                    data=zip_bytes,
                    file_name=f"{name}_bundle.zip",
                    mime="application/zip",
                )
                with st.expander("Anteprima metadata.json"):
                    try:
                        st.json(json.loads(json_text))
                    except json.JSONDecodeError:
                        st.code(json_text)
