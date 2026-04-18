"""Orchestrazione dell'esportazione ML-ready: .pt + metadata.json + .zip.

Questo modulo è la "uscita" di AgriPipe verso il team Data Science di X Farm.
Produce un bundle completo: features normalizzate, target, nomi colonne,
parametri dello scaler, più un manuale d'uso in JSON.
"""

from __future__ import annotations

import zipfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from agripipe.cleaner import AgriCleaner
from agripipe.dataset import AgriDataset
from agripipe.metadata import build_metadata, save_metadata_json


def export_ml_bundle(
    df_clean: pd.DataFrame,
    cleaner: AgriCleaner,
    preset: dict,
    output_dir: str | Path,
    name: str = "agripipe_export",
    target: str = "yield",
) -> dict[str, Path]:
    """Esporta un bundle completo per training PyTorch.
    
    Crea nella ``output_dir``:
        - ``{name}.pt``   : bundle tensoriale (features, target, feature_names, scaler).
        - ``{name}.json`` : metadata auto-documentato (build_metadata output).
        - ``{name}.zip``  : zip di entrambi, per download singolo dall'UI.
    
    Args:
        df_clean: DataFrame già pulito da ``AgriCleaner.clean``.
        cleaner: Istanza usata per la pulizia (per accedere a diagnostics).
        preset: Entry di ``regional_presets`` selezionata dall'utente.
        output_dir: Cartella destinazione (creata se mancante).
        name: Prefisso per i file generati.
        target: Colonna target (passa ``None`` per bundle unsupervised).
    
    Returns:
        Dict con i Path dei 3 file: ``{"pt", "json", "zip"}``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    numeric_cols = [c for c in df_clean.select_dtypes(include=["number"]).columns
                    if c != target]
    target_col = target if target and target in df_clean.columns else None
    
    ds = AgriDataset(
        df=df_clean,
        numeric_columns=numeric_cols,
        target=target_col,
    )
    
    pt_path = output_dir / f"{name}.pt"
    json_path = output_dir / f"{name}.json"
    zip_path = output_dir / f"{name}.zip"
    
    bundle = {
        "features": ds.features,
        "target": ds.target,
        "feature_names": ds.feature_names,
        "scaler_mean": torch.tensor(ds.tensorizer.scaler.mean_, dtype=torch.float32),
        "scaler_scale": torch.tensor(ds.tensorizer.scaler.scale_, dtype=torch.float32),
    }
    torch.save(bundle, pt_path)
    
    metadata = build_metadata(
        dataset=ds,
        preset=preset,
        cleaner_diagnostics=asdict(cleaner.diagnostics),
        target=target_col,
        name=name,
    )
    save_metadata_json(metadata, json_path)
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pt_path, arcname=pt_path.name)
        zf.write(json_path, arcname=json_path.name)
    
    return {"pt": pt_path, "json": json_path, "zip": zip_path}
