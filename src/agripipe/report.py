import base64
import io
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from agripipe.utils.logging_setup import get_logger

logger = get_logger(__name__)


def generate_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    output_path: str | Path,
    title: str = "AgriPipe — Data Quality Report",
) -> Path:
    """Genera un report HTML standalone con tabelle e grafici prima/dopo."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Calcola statistiche testuali
    stats = _compute_stats(df_before, df_after)
    
    # 2. Genera i grafici
    plots_html = _generate_plots(df_before, df_after)
    
    # 3. Assembla l'HTML finale
    html = _render_html(stats, plots_html, title)
    output_path.write_text(html, encoding="utf-8")
    
    logger.info("Report salvato con grafici: %s", output_path)
    return output_path


def _generate_plots(df_before: pd.DataFrame, df_after: pd.DataFrame) -> str:
    """Crea grafici prima/dopo per ogni colonna numerica."""
    numeric_cols = df_before.select_dtypes(include=["number"]).columns
    html_snippets = []

    # Imposta lo stile di Seaborn
    sns.set_theme(style="whitegrid", palette="muted")

    for col in numeric_cols:
        if col not in df_after.columns:
            continue
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Grafico 1: Boxplot (per vedere gli outlier)
        data_cmp = pd.concat([
            df_before[[col]].assign(Stato="Grezzo (Prima)"),
            df_after[[col]].assign(Stato="Pulito (Dopo)")
        ])
        sns.boxplot(data=data_cmp, x="Stato", y=col, ax=ax1, width=0.5)
        ax1.set_title(f"Outlier in '{col}'")
        
        # Grafico 2: Istogramma (per vedere la distribuzione)
        sns.kdeplot(df_before[col], fill=True, label="Prima", ax=ax2, color="orange")
        sns.kdeplot(df_after[col], fill=True, label="Dopo", ax=ax2, color="green")
        ax2.set_title(f"Distribuzione di '{col}'")
        ax2.legend()

        plt.tight_layout()
        
        # Salva in base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        
        html_snippets.append(f"""
        <div class="plot-container">
            <h3>Analisi Colonna: {col}</h3>
            <img src="data:image/png;base64,{data}" alt="Grafico {col}">
        </div>
        """)

    return "\n".join(html_snippets)


def _compute_stats(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows_before": len(df_before),
        "rows_after": len(df_after),
        "rows_removed": len(df_before) - len(df_after),
        "cols_before": df_before.shape[1],
        "cols_after": df_after.shape[1],
        "nan_before": _nan_table(df_before),
        "nan_after": _nan_table(df_after),
        "describe_after": df_after.describe(include="all").round(3).fillna("-"),
        "dtypes_after": df_after.dtypes.astype(str).to_frame("dtype"),
    }


def _nan_table(df: pd.DataFrame) -> pd.DataFrame:
    nan_count = df.isna().sum()
    nan_pct = (df.isna().mean() * 100).round(2)
    return pd.DataFrame({"nan_count": nan_count, "nan_pct": nan_pct})


def _render_html(s: dict, plots_html: str, title: str) -> str:
    css = """
    body { font-family: -apple-system, Segoe UI, sans-serif; margin: 2rem; color: #222; max-width: 1100px; }
    h1 { color: #2d6a4f; border-bottom: 3px solid #2d6a4f; padding-bottom: .4rem; }
    h2 { color: #1b4332; margin-top: 2rem; border-left: 5px solid #2d6a4f; padding-left: 1rem; }
    .cards { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
    .card { flex: 1; min-width: 180px; padding: 1rem; background: #f1f8f4; border-radius: 4px; }
    .card .value { font-size: 1.8rem; font-weight: bold; color: #2d6a4f; }
    .card .label { font-size: .85rem; color: #555; text-transform: uppercase; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
    table { border-collapse: collapse; width: 100%; margin: .5rem 0; font-size: .9rem; }
    th, td { border: 1px solid #ddd; padding: .4rem .6rem; text-align: left; }
    th { background: #e9f5ec; }
    tr:nth-child(even) { background: #fafafa; }
    .plot-container { margin: 2rem 0; background: #fff; padding: 1rem; border: 1px solid #eee; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .plot-container img { max-width: 100%; height: auto; }
    footer { margin-top: 3rem; color: #888; font-size: .8rem; text-align: center; }
    """

    removed_pct = (s["rows_removed"] / s["rows_before"] * 100) if s["rows_before"] else 0

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>{css}</style>
</head>
<body>
    <h1>🌱 {title}</h1>
    <p><em>Generato il {s["timestamp"]}</em></p>

    <h2>1. Riepilogo Pipeline</h2>
    <div class="cards">
        <div class="card"><div class="label">Righe input</div><div class="value">{s["rows_before"]:,}</div></div>
        <div class="card"><div class="label">Righe output</div><div class="value">{s["rows_after"]:,}</div></div>
        <div class="card"><div class="label">Rimosse</div><div class="value">{s["rows_removed"]:,}</div></div>
        <div class="card"><div class="label">% Rimossa</div><div class="value">{removed_pct:.1f}%</div></div>
        <div class="card"><div class="label">Colonne</div><div class="value">{s["cols_before"]} → {s["cols_after"]}</div></div>
    </div>

    <h2>2. Visualizzazione Automatica (Prima vs Dopo)</h2>
    <p>Questi grafici mostrano come AgriPipe ha pulito i dati. A sinistra vedi se c'erano valori fuori scala (pallini neri), a destra vedi come è cambiata la distribuzione.</p>
    {plots_html}

    <h2>3. Analisi Dati Mancanti (NaN)</h2>
    <div class="grid">
        <div><h3>Prima della pulizia</h3>{s["nan_before"].to_html()}</div>
        <div><h3>Dopo la pulizia</h3>{s["nan_after"].to_html()}</div>
    </div>

    <h2>4. Statistiche Finali</h2>
    {s["describe_after"].to_html()}

    <footer>Generato da <strong>AgriPipe</strong> · Strumento per Agricoltura Sostenibile</footer>
</body>
</html>
"""
