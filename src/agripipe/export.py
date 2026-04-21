"""Export del bundle ML: tensor ``.pt`` + ``metadata.json`` + ``.zip``.

Nessun export Parquet o CSV: il formato di scambio con il team ML di X Farm è
esclusivamente PyTorch ``.pt`` + JSON per il manifest.
"""

from __future__ import annotations

import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from agripipe.cleaner import AgriCleaner
from agripipe.dataset import AgriDataset
from agripipe.metadata import build_metadata, save_metadata_json


from typing import Literal


def export_ml_bundle(
    df_clean: pd.DataFrame,
    cleaner: AgriCleaner,
    preset: dict,
    output_dir: str | Path,
    name: str = "agripipe_export",
    target: str | None = "yield",
    split_ratios: tuple[float, float, float] | None = None,
    scaling_strategy: Literal["standard", "robust"] = "standard",
    categorical_strategy: Literal["label", "onehot"] = "label",
) -> dict[str, Path]:
    """Esporta tensor + metadata + zip in ``output_dir``.

    Args:
        df_clean: DataFrame già pulito (``AgriCleaner.clean``).
        cleaner: Istanza usata per la pulizia (fornisce i diagnostics).
        preset: Dict preset applicato (informativo, finisce nei metadata).
        output_dir: Cartella di destinazione (creata se non esiste).
        name: Prefisso dei file generati.
        target: Colonna target. ``None`` = esporto solo le features.
        split_ratios: ``(train, val, test)``. Se presenti, i tensor vengono divisi
            in tre file ``.pt`` separati.
        scaling_strategy: ``"standard"`` | ``"robust"``.
        categorical_strategy: ``"label"`` | ``"onehot"``.

    Returns:
        Dict con i path di ogni artefatto prodotto. Chiavi:
        - ``"pt"`` (solo se nessuno split) o ``"train"``/``"val"``/``"test"``.
        - ``"json"`` — manifest del dataset.
        - ``"zip"`` — archivio che racchiude tutti gli artefatti.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Feature columns: numeriche meno il target
    numeric_cols = [c for c in df_clean.select_dtypes(include=["number"]).columns if c != target]
    categorical_cols = [c for c in cleaner.config.categorical_columns if c in df_clean.columns]

    ds = AgriDataset(
        df=df_clean,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        target=target if target in df_clean.columns else None,
        categorical_strategy=categorical_strategy,
        scaling_strategy=scaling_strategy,
        split_ratios=split_ratios,
    )

    paths: dict[str, Path] = {}
    zip_path = output_dir / f"{name}.zip"

    scaler_mean, scaler_scale = _extract_scaler_arrays(ds)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Tensor .pt (uno o tre file a seconda dello split)
        if ds.train_indices is not None:
            for split_name, indices in [
                ("train", ds.train_indices),
                ("val", ds.val_indices),
                ("test", ds.test_indices),
            ]:
                s_path = output_dir / f"{name}_{split_name}.pt"
                bundle = _build_bundle(ds, scaler_mean, scaler_scale, indices=indices)
                torch.save(bundle, s_path)
                zf.write(s_path, arcname=s_path.name)
                paths[split_name] = s_path
        else:
            pt_path = output_dir / f"{name}.pt"
            bundle = _build_bundle(ds, scaler_mean, scaler_scale, indices=None)
            torch.save(bundle, pt_path)
            zf.write(pt_path, arcname=pt_path.name)
            paths["pt"] = pt_path

        # 2. Metadata JSON
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

    # Integrazione MLOps (Step 9 e 10)
    baseline_metrics = _compute_baseline(ds)
    try:
        from agripipe.tracking import log_export_run

        log_export_run(
            config_dict=cleaner.config.model_dump(),
            diag_dict=asdict(cleaner.diagnostics),
            file_name=name,
            target=target,
            split_ratios=split_ratios,
            baseline_metrics=baseline_metrics,
        )
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning("MLflow tracking fallito o non configurato: %s", e)

    return paths


def _build_bundle(
    ds: AgriDataset,
    scaler_mean: list[float],
    scaler_scale: list[float],
    indices: list[int] | None,
) -> dict:
    """Costruisce il dict da salvare in ``torch.save``."""
    if indices is None:
        features = ds.features
        target = ds.target
    else:
        features = ds.features[indices]
        target = ds.target[indices] if ds.target is not None else None
    return {
        "features": features,
        "target": target,
        "feature_names": ds.feature_names,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "metadata": ds.metadata,
    }


def _extract_scaler_arrays(ds: AgriDataset) -> tuple[list[float], list[float]]:
    """Prende mean/scale (o center/scale) dallo scaler del Tensorizer."""
    scaler = ds.tensorizer.scaler
    if hasattr(scaler, "mean_"):
        mean = np.asarray(scaler.mean_, dtype=float).tolist()
    elif hasattr(scaler, "center_"):
        mean = np.asarray(scaler.center_, dtype=float).tolist()
    else:
        mean = []
    scale = np.asarray(scaler.scale_, dtype=float).tolist() if hasattr(scaler, "scale_") else []
    return mean, scale


def _compute_baseline(ds: AgriDataset) -> dict[str, float]:
    """Allena un modello dummy per validare la non assurdità del dataset (Step 10)."""
    X = ds.features.numpy()
    if ds.target is None:
        return {}
    y = ds.target.numpy()

    if ds.train_indices is not None and ds.val_indices is not None:
        X_train, y_train = X[ds.train_indices], y[ds.train_indices]
        X_val, y_val = X[ds.val_indices], y[ds.val_indices]
    else:
        X_train, y_train = X, y
        X_val, y_val = X, y

    if len(X_train) < 5:
        return {}

    model = Ridge()
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mse = float(mean_squared_error(y_val, preds))
    return {"baseline_mse": mse}
