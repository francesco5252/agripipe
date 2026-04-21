import json
import tempfile
from pathlib import Path
from typing import cast, Literal

import streamlit as st

from agripipe.export import export_ml_bundle

st.set_page_config(page_title="Tensorizer", page_icon="📦", layout="wide")
st.header("Step 3 — Tensorizer")

if st.session_state.get("df_clean") is None:
    st.info("Completa prima la pulizia nello Step 2 (Refinery).")
    st.stop()

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
scaling_strat_cast = cast(Literal["standard", "robust"], scaling_strategy)
categorical_strategy = c_cat.selectbox("Encoding categoriche", options=["label", "onehot"])
categorical_strat_cast = cast(Literal["label", "onehot"], categorical_strategy)

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
                        scaling_strategy=scaling_strat_cast,
                        categorical_strategy=categorical_strat_cast,
                    )
                zip_bytes: bytes | None = paths["zip"].read_bytes()
                json_text: str | None = paths["json"].read_text(encoding="utf-8")
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
                    if json_text is not None:
                        st.json(json.loads(json_text))
                except json.JSONDecodeError:
                    st.code(json_text)
