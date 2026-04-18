"""Costruzione e salvataggio del file metadata.json che accompagna il tensor .pt.

Serve da "manuale d'uso" del dataset per il team Data Science di X Farm:
spiega ogni colonna, il contesto agronomico, e mostra un esempio PyTorch.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agripipe.dataset import AgriDataset

SCHEMA_VERSION = 1

_COLUMN_DESCRIPTIONS = {
    "temp":        ("°C",       "Temperatura media giornaliera"),
    "temperatura": ("°C",       "Temperatura media giornaliera"),
    "humidity":    ("%",        "Umidità relativa aria"),
    "umidità":     ("%",        "Umidità relativa aria"),
    "rainfall":    ("mm",       "Precipitazione giornaliera"),
    "pioggia":     ("mm",       "Precipitazione giornaliera"),
    "ph":          ("pH",       "Acidità del suolo"),
    "yield":       ("t/ha",     "Resa colturale"),
    "resa":        ("t/ha",     "Resa colturale"),
    "n":           ("kg/ha",    "Concimazione azotata"),
    "azoto":       ("kg/ha",    "Concimazione azotata"),
    "soil_moisture": ("%",      "Umidità del suolo"),
    "irrigation":  ("mm",       "Irrigazione applicata"),
    "organic_matter": ("%",     "Sostanza organica del suolo"),
    "gdd_daily":       ("°C·d", "Gradi Giorno giornalieri"),
    "gdd_accumulated": ("°C·d", "Gradi Giorno cumulati"),
    "huglin_index":    ("idx",  "Indice di Huglin (qualità vitivinicola)"),
    "huglin_daily":    ("idx",  "Contributo Huglin giornaliero"),
    "daily_wb":        ("mm",   "Bilancio idrico giornaliero"),
    "drought_7d_score":("mm",   "Indice di siccità cumulata a 7 giorni"),
    "n_efficiency":    ("t/kg", "Efficienza dell'azoto (NUE)"),
}


def _describe_column(name: str, index: int) -> dict:
    unit, description = _COLUMN_DESCRIPTIONS.get(
        name.lower(), ("",  f"Colonna {name}")
    )
    return {
        "name": name,
        "index": index,
        "unit": unit,
        "description": description,
        "normalized": True,  # Tensorizer applica StandardScaler di default
    }


def build_metadata(
    dataset: AgriDataset,
    preset: dict,
    cleaner_diagnostics: dict,
    target: str | None = None,
    name: str = "agripipe_export",
) -> dict:
    """Costruisce il dizionario metadata dal dataset e dal contesto agronomico.
    
    Args:
        dataset: ``AgriDataset`` già fit (features + feature_names).
        preset: Entry di ``regional_presets`` del YAML (region, crop, zona, ...).
        cleaner_diagnostics: ``dataclasses.asdict(cleaner.diagnostics)``.
        target: Nome della colonna target (es. "yield").
        name: Identificatore del dataset (diventa ``dataset_info.name``).
    
    Returns:
        Dizionario pronto per JSON con: schema_version, generated_at,
        dataset_info, columns, agronomic_context, cleaning_stats,
        pytorch_usage.
    """
    n_rows = dataset.features.shape[0]
    n_features = dataset.features.shape[1]
    
    columns = [_describe_column(n, i) for i, n in enumerate(dataset.feature_names)]
    
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_info": {
            "name": name,
            "rows": int(n_rows),
            "features": int(n_features),
            "target": target,
            "target_unit": _COLUMN_DESCRIPTIONS.get((target or "").lower(), ("", ""))[0],
            "task": "regression" if target else "unsupervised",
        },
        "columns": columns,
        "agronomic_context": {
            "crop": preset.get("crop", "unknown"),
            "crop_display": preset.get("crop_display", ""),
            "region": preset.get("region", "unknown"),
            "zona": preset.get("zona", ""),
            "cleaning_rules": [
                "Direttiva Nitrati (soglia azoto)",
                "Regola Tre 10 (peronospora vite)",
                "Coerenza pioggia/umidità",
                "Imputazione time-series",
            ],
        },
        "cleaning_stats": cleaner_diagnostics,
        "pytorch_usage": {
            "example_code": (
                "import torch\n"
                "from torch.utils.data import TensorDataset, DataLoader\n\n"
                "bundle = torch.load('agripipe_export.pt')\n"
                "features, target = bundle['features'], bundle['target']\n"
                "loader = DataLoader(TensorDataset(features, target), batch_size=32, shuffle=True)\n"
                "# features è già normalizzato (StandardScaler): passa direttamente alla rete."
            ),
        },
    }


def save_metadata_json(metadata: dict, path: str | Path) -> Path:
    """Scrive il dict metadata su disco come JSON UTF-8 indentato.
    
    Args:
        metadata: Output di ``build_metadata``.
        path: Destinazione (cartelle create se mancanti).
    
    Returns:
        Path al file scritto.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
