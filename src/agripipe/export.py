"""Orchestrazione Industrial-Standard: Multi-Set, Parquet e Precision.

Modulo di uscita finale: supporta split, precisione float16 e l'esportazione 
in formato Parquet per massima interoperabilità Big Data.
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
    split_ratios: tuple[float, float, float] | None = None,
    scaling_strategy: str = "standard",
    categorical_strategy: str = "label",
    precision: str = "float32",
) -> dict[str, Path]:
    """Esporta un bundle industriale completo (PT + JSON + Parquet + ZIP)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    numeric_cols = [c for c in df_clean.select_dtypes(include=["number"]).columns if c != target]
    categorical_cols = [c for c in ["crop", "field_id", "campo"] if c in df_clean.columns]
    
    # 1. Creazione Dataset con Precisione e Split
    ds = AgriDataset(
        df=df_clean,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        target=target if target in df_clean.columns else None,
        categorical_strategy=categorical_strategy,
        scaling_strategy=scaling_strategy,
        split_ratios=split_ratios,
        precision=precision
    )

    paths = {}
    zip_path = output_dir / f"{name}.zip"
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        
        # 2. Esportazione Parquet (Interoperabilità)
        parquet_path = output_dir / f"{name}_clean.parquet"
        df_clean.to_parquet(parquet_path, index=False)
        zf.write(parquet_path, arcname=parquet_path.name)
        paths["parquet"] = parquet_path
        
        # 3. Salvataggio Tensor (Performance)
        if ds.train_indices is not None:
            for s_name, indices in [("train", ds.train_indices), 
                                   ("val", ds.val_indices), 
                                   ("test", ds.test_indices)]:
                s_path = output_dir / f"{name}_{s_name}.pt"
                bundle = {
                    "features": ds.features[indices],
                    "target": ds.target[indices] if ds.target is not None else None,
                    "feature_names": ds.feature_names,
                    "metadata": ds.metadata
                }
                torch.save(bundle, s_path)
                zf.write(s_path, arcname=s_path.name)
                paths[s_name] = s_path
        else:
            pt_path = output_dir / f"{name}.pt"
            bundle = {
                "features": ds.features,
                "target": ds.target,
                "feature_names": ds.feature_names,
                "metadata": ds.metadata
            }
            torch.save(bundle, pt_path)
            zf.write(pt_path, arcname=pt_path.name)
            paths["pt"] = pt_path

        # 4. Metadata
        json_path = output_dir / f"{name}.json"
        metadata = build_metadata(
            dataset=ds,
            preset=preset,
            cleaner_diagnostics=asdict(cleaner.diagnostics),
            target=target,
            name=name,
        )
        save_metadata_json(metadata, json_path)
        zf.write(json_path, arcname=json_path.name)
        paths["json"] = json_path

    paths["zip"] = zip_path
    return paths
