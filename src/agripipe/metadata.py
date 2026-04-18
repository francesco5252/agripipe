"""Costruzione e salvataggio del file metadata.json che accompagna il tensor .pt.

Serve da "manuale d'uso" del dataset per il team Data Science di X Farm:
spiega ogni colonna, il contesto agronomico, e mostra un esempio PyTorch.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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
    """Costruisce il dizionario metadata con profili statistici avanzati.
    
    Include: info dataset, descrizioni colonne, contesto pipeline,
    statistiche descrittive (mean, std, min, max) e matrice di correlazione.
    """
    n_rows = dataset.features.shape[0]
    n_features = dataset.features.shape[1]
    
    # 1. Statistiche Descrittive (usando il DataFrame originale del dataset se disponibile o calcolando dai tensor)
    # Per semplicità e precisione, le calcoliamo qui come dizionario
    columns_stats = []
    # Trasformiamo il tensor in array per calcoli veloci
    X = dataset.features.numpy()
    
    for i, col_name in enumerate(dataset.feature_names):
        col_data = X[:, i]
        desc = _describe_column(col_name, i)
        desc["stats"] = {
            "mean": float(np.mean(col_data)),
            "std": float(np.std(col_data)),
            "min": float(np.min(col_data)),
            "max": float(np.max(col_data)),
        }
        columns_stats.append(desc)

    # 2. Matrice di Correlazione (solo se abbiamo più di una colonna)
    correlation_map = {}
    if n_features > 1:
        corr_matrix = np.corrcoef(X, rowvar=False)
        for i, col_i in enumerate(dataset.feature_names):
            correlation_map[col_i] = {
                col_j: float(corr_matrix[i, j]) 
                for j, col_j in enumerate(dataset.feature_names)
            }

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
            "file_fingerprint_sha256": dataset.df.attrs.get("file_hash", "unknown"),
            "schema_lock_hash": dataset.metadata.get("schema_lock_hash", "unknown"),
            "precision": dataset.metadata.get("precision", "float32"),
        },
        "columns": columns_stats,
        "correlations": correlation_map,
        "data_integrity_report": dataset.df.attrs.get("integrity_report", {}),
        "split_info": {
            "is_split": dataset.train_indices is not None,
            "counts": {
                "train": len(dataset.train_indices) if dataset.train_indices else 0,
                "val": len(dataset.val_indices) if dataset.val_indices else 0,
                "test": len(dataset.test_indices) if dataset.test_indices else 0,
                "total": len(dataset)
            },
            "ratios": dataset.metadata.get("split_ratios", "none")
        },
        "pipeline_context": {
            "preset_applied": preset.get("crop_display", "custom"),
            "region": preset.get("region", "unknown"),
            "categorical_mappings": dataset.tensorizer.get_categorical_mappings(),
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
