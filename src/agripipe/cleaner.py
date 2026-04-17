"""Pulizia automatica: missing, outlier, deduplica, type coercion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml

from agripipe.utils.logging_setup import get_logger

logger = get_logger(__name__)

ImputationStrategy = Literal["mean", "median", "ffill", "drop"]


@dataclass
class CleanerConfig:
    """Configurazione della pulizia. Serializzabile da YAML."""

    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    date_columns: list[str] = field(default_factory=list)
    dedup_keys: list[str] = field(default_factory=list)
    missing_strategy: ImputationStrategy = "median"
    missing_drop_threshold: float = 0.5  # drop colonna se >50% NaN
    outlier_method: Literal["iqr", "zscore", "none"] = "iqr"
    outlier_iqr_multiplier: float = 1.5
    physical_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)


class AgriCleaner:
    """Pipeline di pulizia configurabile per dati agronomici."""

    def __init__(self, config: CleanerConfig):
        self.config = config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgriCleaner":
        """Istanzia da file YAML."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        # Converti liste in tuple per physical_bounds
        if "physical_bounds" in raw:
            raw["physical_bounds"] = {k: tuple(v) for k, v in raw["physical_bounds"].items()}
        return cls(CleanerConfig(**raw))

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica l'intera pipeline di pulizia."""
        logger.info("Avvio pulizia su %d righe", len(df))
        df = df.copy()
        df = self._coerce_types(df)
        df = self._drop_sparse_columns(df)
        df = self._apply_physical_bounds(df)
        df = self._handle_outliers(df)
        df = self._impute_missing(df)
        df = self._deduplicate(df)
        logger.info("Pulizia completata: %d righe finali", len(df))
        return df

    # ---------- step privati ----------

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.config.date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        for col in self.config.numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _drop_sparse_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        nan_ratio = df.isna().mean()
        to_drop = nan_ratio[nan_ratio > self.config.missing_drop_threshold].index.tolist()
        if to_drop:
            logger.warning("Drop colonne troppo sparse: %s", to_drop)
            df = df.drop(columns=to_drop)
        return df

    def _apply_physical_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imposta a NaN valori fuori dai limiti fisici (es. pH ∉ [0,14])."""
        for col, (lo, hi) in self.config.physical_bounds.items():
            if col in df.columns:
                mask = (df[col] < lo) | (df[col] > hi)
                n = int(mask.sum())
                if n:
                    logger.warning("%s: %d valori fuori range [%s, %s] → NaN", col, n, lo, hi)
                    df.loc[mask, col] = np.nan
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.outlier_method == "none":
            return df
        for col in self.config.numeric_columns:
            if col not in df.columns:
                continue
            series = df[col]
            if self.config.outlier_method == "iqr":
                q1, q3 = series.quantile([0.25, 0.75])
                iqr = q3 - q1
                lo = q1 - self.config.outlier_iqr_multiplier * iqr
                hi = q3 + self.config.outlier_iqr_multiplier * iqr
            else:  # zscore
                mu, sigma = series.mean(), series.std()
                lo, hi = mu - 3 * sigma, mu + 3 * sigma
            mask = (series < lo) | (series > hi)
            if mask.any():
                logger.info("%s: %d outlier → NaN", col, int(mask.sum()))
                df.loc[mask, col] = np.nan
        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        strategy = self.config.missing_strategy
        if strategy == "drop":
            return df.dropna()
        for col in self.config.numeric_columns:
            if col not in df.columns:
                continue
            if strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "ffill":
                df[col] = df[col].ffill().bfill()
        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.dedup_keys:
            return df.drop_duplicates()
        before = len(df)
        df = df.drop_duplicates(subset=self.config.dedup_keys, keep="last")
        logger.info("Deduplica: %d → %d righe", before, len(df))
        return df
