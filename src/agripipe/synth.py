"""Generatore di Excel agronomici 'sporchi' sintetici per testing e demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from agripipe.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class SynthConfig:
    """Parametri di generazione. Tutti i 'dirty rate' sono frazioni [0, 1]."""

    n_rows: int = 500
    n_fields: int = 5
    start_date: str = "2024-01-01"
    seed: int = 42

    # Tassi di "sporcizia"
    nan_rate: float = 0.08
    outlier_rate: float = 0.03
    physical_violation_rate: float = 0.02
    duplicate_rate: float = 0.05
    wrong_type_rate: float = 0.02  # numeri scritti come stringhe ("12,5" invece di 12.5)


def generate_dirty_excel(
    output_path: str | Path,
    config: SynthConfig | None = None,
) -> Path:
    """Crea un .xlsx con dati agronomici realistici e anomalie iniettate.

    Colonne generate: date, field_id, crop_type, temp, humidity, ph, rainfall, yield.
    """
    cfg = config or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = _base_dataset(cfg, rng)
    df = _inject_nan(df, cfg, rng)
    df = _inject_outliers(df, cfg, rng)
    df = _inject_physical_violations(df, cfg, rng)
    df = _inject_wrong_types(df, cfg, rng)
    df = _inject_duplicates(df, cfg, rng)

    df.to_excel(output_path, index=False, engine="openpyxl")
    logger.info("Generato Excel sporco: %s (%d righe)", output_path, len(df))
    return output_path


# ---------- step privati ----------

def _base_dataset(cfg: SynthConfig, rng: np.random.Generator) -> pd.DataFrame:
    fields = [f"F{i+1}" for i in range(cfg.n_fields)]
    crops = ["wheat", "corn", "soy", "barley"]
    dates = pd.date_range(cfg.start_date, periods=cfg.n_rows // cfg.n_fields + 1, freq="D")

    rows = []
    for field_id in fields:
        crop = rng.choice(crops)
        for d in dates[: cfg.n_rows // cfg.n_fields]:
            rows.append(
                {
                    "date": d,
                    "field_id": field_id,
                    "crop_type": crop,
                    "temp": float(rng.normal(20, 5)),
                    "humidity": float(rng.uniform(40, 85)),
                    "ph": float(rng.uniform(5.8, 7.4)),
                    "rainfall": float(max(0, rng.normal(30, 20))),
                    "yield": float(rng.normal(5.0, 1.2)),
                }
            )
    return pd.DataFrame(rows)


def _inject_nan(df: pd.DataFrame, cfg: SynthConfig, rng: np.random.Generator) -> pd.DataFrame:
    numeric_cols = ["temp", "humidity", "ph", "rainfall", "yield"]
    for col in numeric_cols:
        mask = rng.random(len(df)) < cfg.nan_rate
        df.loc[mask, col] = np.nan
    return df


def _inject_outliers(df: pd.DataFrame, cfg: SynthConfig, rng: np.random.Generator) -> pd.DataFrame:
    for col, spike in [("temp", 200.0), ("yield", 50.0), ("rainfall", 5000.0)]:
        mask = rng.random(len(df)) < cfg.outlier_rate
        df.loc[mask, col] = spike * rng.uniform(0.5, 1.5, mask.sum())
    return df


def _inject_physical_violations(
    df: pd.DataFrame, cfg: SynthConfig, rng: np.random.Generator
) -> pd.DataFrame:
    # pH impossibile (negativo o > 14)
    mask = rng.random(len(df)) < cfg.physical_violation_rate
    df.loc[mask, "ph"] = rng.choice([-2.0, 17.5, 99.0], mask.sum())
    # humidity > 100%
    mask = rng.random(len(df)) < cfg.physical_violation_rate
    df.loc[mask, "humidity"] = rng.uniform(101, 150, mask.sum())
    return df


def _inject_wrong_types(
    df: pd.DataFrame, cfg: SynthConfig, rng: np.random.Generator
) -> pd.DataFrame:
    """Alcune celle numeriche scritte come stringa con virgola decimale (stile IT)."""
    mask = rng.random(len(df)) < cfg.wrong_type_rate
    idx = df.index[mask]
    df["temp"] = df["temp"].astype(object)
    for i in idx:
        val = df.at[i, "temp"]
        if pd.notna(val):
            df.at[i, "temp"] = str(val).replace(".", ",")
    return df


def _inject_duplicates(
    df: pd.DataFrame, cfg: SynthConfig, rng: np.random.Generator
) -> pd.DataFrame:
    n_dup = int(len(df) * cfg.duplicate_rate)
    if n_dup == 0:
        return df
    dup_idx = rng.choice(df.index, size=n_dup, replace=False)
    return pd.concat([df, df.loc[dup_idx]], ignore_index=True)
