"""Report HTML di qualità dei dati prima/dopo la pulizia."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from agripipe.utils.logging_setup import get_logger

logger = get_logger(__name__)


def generate_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    output_path: str | Path,
    title: str = "AgriPipe — Data Quality Report",
) -> Path:
    """Genera un report HTML standalone confrontando dataset grezzo e pulito."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = _compute_stats(df_before, df_after)
    html = _render_html(stats, title)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Report salvato: %s", output_path)
    return output_path


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


def _render_html(s: dict, title: str) -> str:
    css = """
    body { font-family: -apple-system, Segoe UI, sans-serif; margin: 2rem; color: #222; max-width: 1100px; }
    h1 { color: #2d6a4f; border-bottom: 3px solid #2d6a4f; padding-bottom: .4rem; }
    h2 { color: #1b4332; margin-top: 2rem; }
    .cards { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
    .card { flex: 1; min-width: 180px; padding: 1rem; background: #f1f8f4; border-left: 4px solid #2d6a4f; border-radius: 4px; }
    .card .value { font-size: 1.8rem; font-weight: bold; color: #2d6a4f; }
    .card .label { font-size: .85rem; color: #555; text-transform: uppercase; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
    table { border-collapse: collapse; width: 100%; margin: .5rem 0; font-size: .9rem; }
    th, td { border: 1px solid #ddd; padding: .4rem .6rem; text-align: left; }
    th { background: #e9f5ec; }
    tr:nth-child(even) { background: #fafafa; }
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

    <h2>Riepilogo</h2>
    <div class="cards">
        <div class="card"><div class="label">Righe input</div><div class="value">{s["rows_before"]:,}</div></div>
        <div class="card"><div class="label">Righe output</div><div class="value">{s["rows_after"]:,}</div></div>
        <div class="card"><div class="label">Rimosse</div><div class="value">{s["rows_removed"]:,}</div></div>
        <div class="card"><div class="label">% Rimossa</div><div class="value">{removed_pct:.1f}%</div></div>
        <div class="card"><div class="label">Colonne</div><div class="value">{s["cols_before"]} → {s["cols_after"]}</div></div>
    </div>

    <h2>NaN per colonna</h2>
    <div class="grid">
        <div><h3>Prima</h3>{s["nan_before"].to_html()}</div>
        <div><h3>Dopo</h3>{s["nan_after"].to_html()}</div>
    </div>

    <h2>Statistiche descrittive (dopo pulizia)</h2>
    {s["describe_after"].to_html()}

    <h2>Tipi di dato finali</h2>
    {s["dtypes_after"].to_html()}

    <footer>Generato da <strong>AgriPipe</strong> · pipeline automatica per dati agronomici</footer>
</body>
</html>
"""
