"""Componenti Streamlit riusabili per AgriPipe.

Ogni funzione emette HTML nel runtime Streamlit usando le classi CSS definite
in ``ui/theme.py``. Principio: NESSUNA logica di business qui, solo rendering.
"""

from __future__ import annotations

from typing import Iterable

import streamlit as st


def render_hero(title: str = "🌱 AgriPipe",
                subtitle: str = "Da Excel sporco a dati ML-ready in 30 secondi") -> None:
    """Intestazione principale della pagina.
    
    Args:
        title: Titolo con emoji/icona.
        subtitle: Frase di valore mostrata sotto il titolo.
    """
    st.markdown(
        f"""
        <div class="agri-hero">
            <h1>{title}</h1>
            <div class="subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_step(number: int, icon: str, title: str) -> None:
    """Intestazione di uno step numerato (1..5) con icona.
    
    Args:
        number: Numero dello step (1..5).
        icon: Emoji agronomica (es. "📍", "🌾").
        title: Titolo dello step.
    """
    st.markdown(
        f"""
        <div class="agri-step">
            <span class="agri-step-number">Step {number}</span>
            <span>{icon} {title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_motivation(message: str) -> None:
    """Banner motivazionale Verde Salvia.
    
    Args:
        message: Frase unica che spiega il valore (es. risparmio tempo).
    """
    st.markdown(
        f'<div class="agri-motivation">💬 {message}</div>',
        unsafe_allow_html=True,
    )


def render_info_cards(cards: Iterable[tuple[str, str, str]]) -> None:
    """Riga di card informative equidistanziate.
    
    Args:
        cards: Iterable di tuple ``(icon, label, value)``. Tipicamente 3 card.
    """
    cards = list(cards)
    cols = st.columns(len(cards))
    for col, (icon, label, value) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="agri-info-card">
                    <div class="label">{icon} {label}</div>
                    <div class="value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_metrics(total_input: int, total_output: int, anomalies: int) -> None:
    """Tre metriche principali dopo la pulizia.
    
    Args:
        total_input: Righe del dataframe grezzo.
        total_output: Righe del dataframe pulito.
        anomalies: Numero di anomalie corrette (da diagnostics).
    """
    c1, c2, c3 = st.columns(3)
    reliability = (total_output / total_input * 100) if total_input else 100.0
    c1.metric("Righe pulite", f"{total_output:,}")
    c2.metric("Anomalie corrette", f"{anomalies:,}")
    c3.metric("Affidabilità dato", f"{reliability:.1f}%")


def render_download_row(
    excel_bytes: bytes,
    bundle_zip_bytes: bytes,
    name_prefix: str,
) -> None:
    """Due bottoni download affiancati: Excel + Bundle ML (.zip).
    
    Args:
        excel_bytes: Contenuto binario del file Excel pulito.
        bundle_zip_bytes: Contenuto binario del file .zip (pt + json).
        name_prefix: Prefisso per i nomi file (es. "ulivo_pugliese").
    """
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label="📥 Excel Pulito",
            data=excel_bytes,
            file_name=f"agripipe_{name_prefix}_clean.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.caption("Per agronomi e gestionali Excel.")
    with c2:
        st.download_button(
            label="💾 Bundle ML (.zip)",
            data=bundle_zip_bytes,
            file_name=f"agripipe_{name_prefix}_bundle.zip",
            mime="application/zip",
            use_container_width=True,
        )
        st.caption("Tensor PyTorch + metadata.json, pronti per training.")


def render_before_after_plots(df_raw, df_clean) -> None:
    """Boxplot + KDE prima/dopo per ogni colonna numerica chiave.
    
    Usa la palette AgriPipe (Marrone Terra per grezzo, Verde Salvia per pulito).
    
    Args:
        df_raw: DataFrame grezzo originale.
        df_clean: DataFrame pulito dopo AgriCleaner.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    from agripipe.ui.theme import PALETTE
    
    # Seleziona colonne numeriche comuni
    cols = [c for c in df_raw.select_dtypes(include=["number"]).columns 
            if c in df_clean.columns]
    
    if not cols: return

    for col in cols:
        fig, (ax_box, ax_kde) = plt.subplots(
            2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)},
            figsize=(8, 5)
        )
        
        # Data preparation
        raw_s = df_raw[col].dropna()
        clean_s = df_clean[col].dropna()
        
        # Boxplot
        sns.boxplot(x=raw_s, ax=ax_box, color=PALETTE["earth"], width=0.5)
        sns.boxplot(x=clean_s, ax=ax_box, color=PALETTE["sage"], width=0.3)
        ax_box.set(xlabel='')
        ax_box.set_axis_off()
        
        # KDE
        sns.kdeplot(raw_s, ax=ax_kde, fill=True, color=PALETTE["earth"], label="Grezzo")
        sns.kdeplot(clean_s, ax=ax_kde, fill=True, color=PALETTE["sage"], label="Pulito")
        
        ax_kde.set_title(f"Distribuzione: {col}", color=PALETTE["forest"], loc='left')
        ax_kde.legend()
        
        st.pyplot(fig)
        plt.close(fig)
