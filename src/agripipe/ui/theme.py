"""Palette Clean & Nature e iniezione CSS in Streamlit.

Fonte unica di verità per i colori e lo stile di AgriPipe. Modificare un colore
qui = modifica propagata a tutta la UI. Chi cambia la palette NON deve toccare
components.py.
"""

from __future__ import annotations

PALETTE = {
    "sage":        "#7FA77F",
    "forest":      "#3D5A3D",
    "earth":       "#8B6F47",
    "water":       "#4A90A4",
    "cream":       "#FAFAF7",
    "card":        "#FFFFFF",
    "leaf":        "#6BAF6B",
    "wheat":       "#D4A64A",
    "pomegranate": "#B84A3E",
    "text":        "#2B2B2B",
    "text_muted":  "#6B6B6B",
}

BADGE_COLORS = {
    "green":  PALETTE["leaf"],
    "yellow": PALETTE["wheat"],
    "red":    PALETTE["pomegranate"],
}


def build_stylesheet() -> str:
    """Costruisce il foglio di stile CSS per Streamlit.
    
    Returns:
        Blocco CSS come stringa, pronto da passare a ``st.markdown``
        con ``unsafe_allow_html=True``.
    """
    p = PALETTE
    return f"""
    /* === AgriPipe Clean & Nature === */
    /* Palette verification: {", ".join(p.values())} */
    .stApp {{ background: {p["cream"]}; }}
    .block-container {{ max-width: 1200px; padding-top: 2rem; }}
    
    h1, h2, h3 {{ color: {p["forest"]}; font-weight: 600; }}
    p, span, li {{ color: {p["text"]}; line-height: 1.6; }}
    
    /* Step header */
    .agri-step {{
        display: flex; align-items: center; gap: 0.75rem;
        font-size: 1.4rem; font-weight: 600; color: {p["forest"]};
        border-left: 4px solid {p["sage"]};
        padding: 0.5rem 0 0.5rem 1rem;
        margin: 2rem 0 1rem 0;
    }}
    .agri-step-number {{
        background: {p["sage"]}; color: white; font-size: 0.9rem;
        padding: 0.15rem 0.55rem; border-radius: 999px;
    }}
    
    /* Card generica */
    .agri-card {{
        background: {p["card"]};
        border: 1px solid {p["earth"]}33;
        border-radius: 8px;
        padding: 1.25rem;
    }}
    
    /* Info card (riga 3 card) */
    .agri-info-card {{
        background: {p["card"]}; border: 1px solid {p["earth"]}22;
        border-radius: 8px; padding: 1rem; text-align: center;
    }}
    .agri-info-card .label {{ font-size: 0.8rem; color: {p["text_muted"]};
        text-transform: uppercase; letter-spacing: 0.5px; }}
    .agri-info-card .value {{ font-size: 1.25rem; font-weight: 600;
        color: {p["forest"]}; margin-top: 0.3rem; }}
    
    /* Motivational banner */
    .agri-motivation {{
        background: {p["sage"]}22; border-left: 4px solid {p["sage"]};
        padding: 1rem 1.25rem; border-radius: 6px; color: {p["forest"]};
        font-style: italic; margin: 1rem 0;
    }}

    /* Water highlight */
    .agri-water {{ color: {p["water"]}; }}

    /* Badge grid (Score Card) */
    .agri-badge-grid {{ display: grid; grid-template-columns: 1fr 1fr;
        gap: 1rem; margin: 1rem 0; }}
    .agri-badge {{
        background: {p["card"]}; border-radius: 8px; padding: 1.25rem;
        border-top: 4px solid var(--badge-color);
        border-right: 1px solid {p["earth"]}22;
        border-bottom: 1px solid {p["earth"]}22;
        border-left: 1px solid {p["earth"]}22;
    }}
    .agri-badge-header {{ display: flex; align-items: center;
        gap: 0.5rem; font-weight: 600; color: {p["forest"]}; }}
    .agri-badge-dot {{ display: inline-block; width: 12px; height: 12px;
        border-radius: 50%; background: var(--badge-color); }}
    .agri-badge-headline {{ font-size: 1.1rem; margin: 0.5rem 0;
        color: {p["text"]}; }}
    .agri-badge-tip {{ font-size: 0.9rem; color: {p["text_muted"]}; }}
    
    /* Overall sustainability message */
    .agri-overall {{
        text-align: center; font-size: 1.05rem; font-weight: 500;
        padding: 1rem; color: {p["forest"]};
    }}
    
    /* Hero */
    .agri-hero {{
        text-align: center; padding: 1.5rem 0;
        border-bottom: 1px solid {p["earth"]}22; margin-bottom: 1.5rem;
    }}
    .agri-hero h1 {{ font-size: 2.2rem; margin: 0; color: {p["forest"]}; }}
    .agri-hero .subtitle {{ color: {p["text_muted"]}; font-size: 1rem; }}
    
    /* Streamlit button override */
    .stButton > button {{
        background: {p["forest"]}; color: white; border-radius: 8px;
        border: none; font-weight: 600; padding: 0.6rem 1.5rem;
    }}
    .stButton > button:hover {{ background: {p["sage"]}; }}
    """


def inject_css() -> None:
    """Inietta il foglio di stile nella pagina Streamlit corrente.
    
    Chiamare UNA volta all'inizio di ``app.py`` dopo ``st.set_page_config``.
    """
    import streamlit as st
    st.markdown(f"<style>{build_stylesheet()}</style>", unsafe_allow_html=True)
