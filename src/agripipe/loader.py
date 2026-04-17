"""Caricamento robusto di Excel agronomici grezzi."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from agripipe.utils.logging_setup import get_logger

logger = get_logger(__name__)


class RawSchema(BaseModel):
    """Schema atteso per il file di input. Adatta ai tuoi campi reali."""

    required_columns: list[str] = Field(
        default_factory=lambda: ["date", "field_id", "temp", "humidity", "ph", "yield"]
    )


def load_raw(
    path: str | Path,
    sheet_name: str | int | None = 0,
    schema: RawSchema | None = None,
) -> pd.DataFrame:
    """Legge un Excel grezzo, valida lo schema minimo e logga anomalie.

    Args:
        path: percorso al file .xlsx / .xls.
        sheet_name: foglio da leggere (default: primo).
        schema: schema di validazione; se None usa quello di default.

    Returns:
        DataFrame grezzo (non ancora pulito).

    Raises:
        FileNotFoundError: se il path non esiste.
        ValueError: se mancano colonne obbligatorie.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    schema = schema or RawSchema()
    logger.info("Carico %s (sheet=%s)", path, sheet_name)

    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")

    missing = set(schema.required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nello schema: {missing}")

    logger.info("Caricate %d righe, %d colonne", len(df), df.shape[1])
    _log_quality_report(df)
    return df


def _log_quality_report(df: pd.DataFrame) -> None:
    """Log sintetico di NaN, duplicati, tipi."""
    nan_pct = (df.isna().sum() / len(df) * 100).round(2)
    dup = df.duplicated().sum()
    logger.info("NaN %% per colonna:\n%s", nan_pct.to_string())
    logger.info("Righe duplicate: %d", dup)
