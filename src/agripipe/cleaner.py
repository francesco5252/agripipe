"""Pulizia automatica: missing, outlier, deduplica, type coercion.

Fase 2 della pipeline: trasforma dati grezzi sporchi in dati puliti e coerenti
pronti per la tensorizzazione ML. Focus su integrità statistica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml

from agripipe.utils.logging_setup import get_logger

logger = get_logger(__name__)

ImputationStrategy = Literal["mean", "median", "ffill", "drop", "time"]


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
    knowledge_path: str = "configs/agri_knowledge.yaml"


@dataclass
class CleanerDiagnostics:
    """Metriche di qualità del dato raccolte durante la pulizia."""

    total_rows: int = 0
    current_preset_name: str | None = None
    imputation_strategy_used: str = ""
    values_imputed: int = 0
    outliers_removed: int = 0
    out_of_bounds_removed: int = 0
    duplicates_removed: int = 0


class AgriCleaner:
    """Pipeline di pulizia statistica per dati agronomici."""

    def __init__(self, config: CleanerConfig):
        self.config = config
        self.knowledge = self._load_knowledge()
        self.diagnostics = CleanerDiagnostics()

    def _load_knowledge(self) -> dict:
        path = Path(self.config.knowledge_path)
        if not path.exists():
            logger.warning("Configurazione conoscenza non trovata in %s", path)
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgriCleaner":
        """Istanzia da file YAML."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        if "physical_bounds" in raw:
            raw["physical_bounds"] = {k: tuple(v) for k, v in raw["physical_bounds"].items()}
        return cls(CleanerConfig(**raw))

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        knowledge_path: str = "configs/agri_knowledge.yaml",
    ) -> "AgriCleaner":
        """Istanzia caricando un preset regionale."""
        path = Path(knowledge_path)
        with open(path, "r", encoding="utf-8") as f:
            knowledge = yaml.safe_load(f)
        
        presets = knowledge.get("regional_presets", {})
        if preset_name not in presets:
            raise ValueError(f"Preset '{preset_name}' non trovato.")
        
        preset_data = presets[preset_name]
        
        config = CleanerConfig(
            knowledge_path=knowledge_path,
            numeric_columns=[], 
            date_columns=["date", "data"],
            dedup_keys=["field_id", "date", "campo", "data"],
        )
        
        if "temp_range" in preset_data:
            config.physical_bounds["temp"] = tuple(preset_data["temp_range"])
            config.physical_bounds["temperatura"] = tuple(preset_data["temp_range"])
        if "ideal_ph" in preset_data:
            config.physical_bounds["ph"] = tuple(preset_data["ideal_ph"])
            
        cleaner = cls(config)
        cleaner.diagnostics.current_preset_name = preset_name
        return cleaner

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica la pipeline di pulizia statistica."""
        self.diagnostics = CleanerDiagnostics(total_rows=len(df))
        self.diagnostics.current_preset_name = getattr(self.diagnostics, "current_preset_name", None)
        
        logger.info("Avvio pulizia su %d righe", len(df))
        df = df.copy()

        if not self.config.numeric_columns:
            self.config.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        df = self._coerce_types(df)
        df = self._drop_sparse_columns(df)
        df = self._apply_physical_bounds(df)
        df = self._handle_outliers(df)
        df = self._impute_missing(df)
        df = self._deduplicate(df)

        logger.info("Pulizia completata: %d righe finali", len(df))
        return df

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
        for col, (lo, hi) in self.config.physical_bounds.items():
            if col in df.columns:
                mask = (df[col] < lo) | (df[col] > hi)
                if mask.any():
                    count = int(mask.sum())
                    logger.warning("%s: %d fuori range [%s, %s] -> NaN", col, count, lo, hi)
                    df.loc[mask, col] = np.nan
                    self.diagnostics.out_of_bounds_removed += count
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.outlier_method == "none":
            return df
        for col in self.config.numeric_columns:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if len(s) < 3: continue
            
            if self.config.outlier_method == "iqr":
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                lo, hi = (q1 - self.config.outlier_iqr_multiplier * iqr,
                          q3 + self.config.outlier_iqr_multiplier * iqr)
            else:
                mu, sigma = s.mean(), s.std()
                lo, hi = mu - 3 * sigma, mu + 3 * sigma
            
            mask = (df[col] < lo) | (df[col] > hi)
            if mask.any():
                count = int(mask.sum())
                logger.info("%s: %d outlier rilevati -> NaN", col, count)
                df.loc[mask, col] = np.nan
                self.diagnostics.outliers_removed += count
        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        strat = self.config.missing_strategy

        if strat == "time":
            date_col = next((c for c in ["date", "data"] if c in df.columns), None)
            if not date_col or len(df) < 3:
                strat = "median"
            else:
                self.diagnostics.imputation_strategy_used = "time"
                return self._impute_time(df, date_col)

        self.diagnostics.imputation_strategy_used = strat

        if strat == "drop":
            return df.dropna()

        before_na = int(df[self.config.numeric_columns].isna().sum().sum()) if self.config.numeric_columns else 0
        for col in self.config.numeric_columns:
            if col not in df.columns: continue
            fill_val = df[col].mean() if strat == "mean" else df[col].median()
            df[col] = df[col].fillna(fill_val)
            if strat == "ffill":
                df[col] = df[col].ffill().bfill()
        
        after_na = int(df[self.config.numeric_columns].isna().sum().sum()) if self.config.numeric_columns else 0
        self.diagnostics.values_imputed += (before_na - after_na)
        return df

    def _impute_time(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        field_col = next((c for c in ["field_id", "campo"] if c in df.columns), None)
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        sort_cols = [field_col, date_col] if field_col else [date_col]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        before_na = int(df[self.config.numeric_columns].isna().sum().sum())
        df_indexed = df.set_index(date_col)

        for col in self.config.numeric_columns:
            if col not in df_indexed.columns: continue
            if field_col:
                df_indexed[col] = df_indexed.groupby(field_col)[col].transform(
                    lambda s: s.interpolate(method="time", limit=3).ffill().bfill()
                )
            else:
                df_indexed[col] = df_indexed[col].interpolate(method="time", limit=3).ffill().bfill()

        df = df_indexed.reset_index()
        after_na = int(df[self.config.numeric_columns].isna().sum().sum())
        self.diagnostics.values_imputed += (before_na - after_na)
        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        keys = [k for k in self.config.dedup_keys if k in df.columns]
        before = len(df)
        df = df.drop_duplicates(subset=keys if keys else None, keep="last")
        self.diagnostics.duplicates_removed = before - len(df)
        return df
