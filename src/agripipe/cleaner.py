"""Pulizia Master-Level: Iterative Imputation (MICE), Delta-Checks e ML-Ops.

Fase 2 definitiva: garantisce la preservazione assoluta dell'ordine originale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, IterativeImputer 

from agripipe.utils.logging_setup import get_logger

logger = get_logger(__name__)

ImputationStrategy = Literal["mean", "median", "ffill", "drop", "time", "knn", "mice"]


@dataclass
class CleanerConfig:
    """Configurazione della pulizia Master."""
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    date_columns: list[str] = field(default_factory=list)
    dedup_keys: list[str] = field(default_factory=list)
    missing_strategy: ImputationStrategy = "mice"
    missing_drop_threshold: float = 0.5
    outlier_method: Literal["iqr", "zscore", "ml", "none"] = "iqr"
    outlier_iqr_multiplier: float = 1.5
    auto_log_transform: bool = True
    enable_seasonal_checks: bool = True
    enable_peer_validation: bool = True
    max_deltas: dict[str, float] = field(default_factory=lambda: {"temp": 15.0, "ph": 1.0})
    physical_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    knowledge_path: str = "configs/agri_knowledge.yaml"


@dataclass
class CleanerDiagnostics:
    """Metriche di qualità Master."""
    total_rows: int = 0
    current_preset_name: str | None = None
    imputation_strategy_used: str = ""
    values_imputed: int = 0
    outliers_removed: int = 0
    seasonal_outliers: int = 0
    peer_anomalies: int = 0
    out_of_bounds_removed: int = 0
    duplicates_removed: int = 0
    inconsistent_rows: int = 0
    delta_anomalies: int = 0
    target_bias_detected: bool = False
    skewed_columns: list[str] = field(default_factory=list)
    log_transformed_columns: list[str] = field(default_factory=list)


class AgriCleaner:
    """Pipeline di pulizia definitiva Master per dati agronomici."""

    def __init__(self, config: CleanerConfig):
        self.config = config
        self.knowledge = self._load_knowledge()
        self.diagnostics = CleanerDiagnostics()

    def _load_knowledge(self) -> dict:
        path = Path(self.config.knowledge_path)
        return yaml.safe_load(open(path, "r", encoding="utf-8")) if path.exists() else {}

    @classmethod
    def from_preset(cls, preset_name: str, knowledge_path: str = "configs/agri_knowledge.yaml") -> "AgriCleaner":
        path = Path(knowledge_path)
        knowledge = yaml.safe_load(open(path, "r", encoding="utf-8"))
        preset_data = knowledge.get("regional_presets", {}).get(preset_name)
        if not preset_data: raise ValueError(f"Preset '{preset_name}' non trovato.")
        
        config = CleanerConfig(
            knowledge_path=knowledge_path,
            numeric_columns=[], 
            date_columns=["date", "data"],
            dedup_keys=["field_id", "date", "campo", "data"],
            categorical_columns=["crop", "crop_type", "field_id", "campo"]
        )
        if "temp_range" in preset_data:
            config.physical_bounds["temp"] = tuple(preset_data["temp_range"])
        if "ideal_ph" in preset_data:
            config.physical_bounds["ph"] = tuple(preset_data["ideal_ph"])
        return cls(config)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline Master che preserva l'indice originale."""
        self.diagnostics = CleanerDiagnostics(total_rows=len(df))
        df = df.copy()
        
        # Salvataggio ordine originale tramite colonna nascosta
        df["_order_id"] = range(len(df))

        if not self.config.numeric_columns:
            self.config.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if "_order_id" in self.config.numeric_columns:
                self.config.numeric_columns.remove("_order_id")

        df = self._coerce_types(df)
        self._analyze_distributions(df)
        if self.config.auto_log_transform:
            df = self._apply_log_transforms(df)
            
        df = self._impute_categorical(df)
        if self.config.enable_peer_validation:
            df = self._apply_peer_validation(df)
        
        df = self._check_delta_rates(df)
        if self.config.enable_seasonal_checks:
            df = self._handle_seasonal_outliers(df)

        df = self._drop_sparse_columns(df)
        df = self._check_logical_consistency(df)
        df = self._apply_physical_bounds(df)
        df = self._handle_outliers(df)
        df = self._impute_missing(df)
        df = self._deduplicate(df)
        self._check_target_bias(df)
        
        # Ritorno all'ordine iniziale rigoroso
        return df.sort_values("_order_id").drop(columns=["_order_id"])

    def _analyze_distributions(self, df: pd.DataFrame) -> None:
        for col in self.config.numeric_columns:
            if col in df.columns:
                s = df[col].dropna()
                if len(s) > 10:
                    skew = s.skew()
                    if abs(skew) > 1.5:
                        self.diagnostics.skewed_columns.append(col)

    def _apply_log_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.diagnostics.skewed_columns:
            if col in df.columns:
                s = df[col].dropna()
                if (s >= 0).all() and s.skew() > 1.5:
                    df[col] = np.log1p(df[col])
                    self.diagnostics.log_transformed_columns.append(col)
        return df

    def _apply_peer_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        date_col = next((c for c in ["date", "data"] if c in df.columns), None)
        field_col = next((c for c in ["field_id", "campo"] if c in df.columns), None)
        if not date_col or not field_col or df[field_col].nunique() < 3: return df
        for col in ["temp", "humidity", "ph"]:
            if col in df.columns:
                peer_median = df.groupby(date_col)[col].transform("median")
                deviation = (df[col] - peer_median).abs()
                threshold = 5.0 if col == "temp" else 20.0
                mask = deviation > threshold
                if mask.any():
                    df.loc[mask, col] = np.nan
                    self.diagnostics.peer_anomalies += int(mask.sum())
        return df

    def _check_target_bias(self, df: pd.DataFrame) -> None:
        target_col = next((c for c in ["yield", "resa"] if c in df.columns), None)
        if not target_col: return
        s = df[target_col].dropna()
        if len(s) < 10: return
        cv = s.std() / s.mean() if s.mean() != 0 else 0
        if cv < 0.05 or abs(s.skew()) > 2.0:
            self.diagnostics.target_bias_detected = True

    def _handle_seasonal_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        date_col = next((c for c in ["date", "data"] if c in df.columns), None)
        if not date_col or len(df) < 50: return df
        temp_df = df.copy()
        temp_df["_month"] = pd.to_datetime(temp_df[date_col]).dt.month
        for col in self.config.numeric_columns:
            if col not in temp_df.columns or col == "_month": continue
            stats = temp_df.groupby("_month")[col].transform(lambda x: x.median())
            stds = temp_df.groupby("_month")[col].transform(lambda x: x.std())
            mask = (temp_df[col] - stats).abs() > (4 * stds)
            if mask.any():
                df.loc[mask, col] = np.nan
                self.diagnostics.seasonal_outliers += int(mask.sum())
        return df

    def _check_delta_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        date_col = next((c for c in ["date", "data"] if c in df.columns), None)
        field_col = next((c for c in ["field_id", "campo"] if c in df.columns), None)
        if not date_col or not field_col: return df
        df_sorted = df.sort_values([field_col, date_col])
        for col, max_delta in self.config.max_deltas.items():
            if col in df_sorted.columns:
                deltas = df_sorted.groupby(field_col)[col].diff().abs()
                mask = deltas > max_delta
                if mask.any():
                    bad_indices = df_sorted.loc[mask].index
                    df.loc[bad_indices, col] = np.nan
                    self.diagnostics.delta_anomalies += len(bad_indices)
        return df

    def _impute_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        date_col = next((c for c in ["date", "data"] if c in df.columns), None)
        if date_col:
            # Ordiniamo in una copia separata per fare ffill/bfill
            df_sorted = df.sort_values(date_col).copy()
            for col in self.config.categorical_columns:
                if col in df_sorted.columns:
                    df_sorted[col] = df_sorted[col].ffill().bfill()
            # Ri-iniettiamo i valori nel df originale usando l'indice (Pandas fa l'allineamento automatico)
            for col in self.config.categorical_columns:
                if col in df.columns:
                    df[col] = df_sorted[col]
        return df

    def _check_logical_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        inconsistencies = pd.Series(False, index=df.index)
        if "rainfall" in df.columns and "soil_moisture" in df.columns:
            mask = (df["rainfall"] > 20) & (df["soil_moisture"] < 15)
            inconsistencies |= mask
        if "temp" in df.columns and "humidity" in df.columns:
            mask = (df["temp"] > 35) & (df["humidity"] > 98)
            inconsistencies |= mask
        self.diagnostics.inconsistent_rows = int(inconsistencies.sum())
        cols_to_null = [c for c in ["temp", "humidity", "rainfall", "soil_moisture"] if c in df.columns]
        df.loc[inconsistencies, cols_to_null] = np.nan
        return df

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.config.date_columns:
            if col in df.columns: df[col] = pd.to_datetime(df[col], errors="coerce")
        for col in self.config.numeric_columns:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _drop_sparse_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        nan_ratio = df.isna().mean()
        to_drop = nan_ratio[nan_ratio > self.config.missing_drop_threshold].index.tolist()
        to_drop = [c for c in to_drop if c != "_order_id"]
        return df.drop(columns=to_drop) if to_drop else df

    def _apply_physical_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, (lo, hi) in self.config.physical_bounds.items():
            if col in df.columns:
                mask = (df[col] < lo) | (df[col] > hi)
                if mask.any():
                    df.loc[mask, col] = np.nan
                    self.diagnostics.out_of_bounds_removed += int(mask.sum())
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.outlier_method == "ml": return self._handle_outliers_ml(df)
        if self.config.outlier_method == "none": return df
        for col in self.config.numeric_columns:
            if col not in df.columns: continue
            s = df[col].dropna()
            if len(s) < 10: continue
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
            mask = (df[col] < lo) | (df[col] > hi)
            if mask.any():
                df.loc[mask, col] = np.nan
                self.diagnostics.outliers_removed += int(mask.sum())
        return df

    def _handle_outliers_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.config.numeric_columns if c in df.columns]
        if len(df) < 15: return df
        temp_df = df[cols].fillna(df[cols].median())
        iso = IsolationForest(contamination=0.05, random_state=42)
        mask = (iso.fit_predict(temp_df) == -1)
        if mask.any():
            df.loc[mask, cols] = np.nan
            self.diagnostics.outliers_removed += int(mask.sum())
        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        strat = self.config.missing_strategy
        self.diagnostics.imputation_strategy_used = strat
        if strat == "mice": return self._impute_mice(df)
        if strat == "knn": return self._impute_knn(df)
        if strat == "time":
            date_col = next((c for c in ["date", "data"] if c in df.columns), None)
            if date_col and len(df) >= 3: return self._impute_time(df, date_col)
        before_na = int(df[self.config.numeric_columns].isna().sum().sum())
        for col in self.config.numeric_columns:
            if col not in df.columns: continue
            df[col] = df[col].fillna(df[col].median())
        self.diagnostics.values_imputed += (before_na - int(df[self.config.numeric_columns].isna().sum().sum()))
        return df

    def _impute_mice(self, df: pd.DataFrame) -> pd.DataFrame:
        """L'Imputazione Scientifica Definitiva: MICE."""
        cols = [c for c in self.config.numeric_columns if c in df.columns]
        if len(df) < 10 or not cols: return df
        before_na = int(df[cols].isna().sum().sum())
        imputer = IterativeImputer(max_iter=10, random_state=42)
        df[cols] = imputer.fit_transform(df[cols])
        self.diagnostics.values_imputed += (before_na - int(df[cols].isna().sum().sum()))
        return df

    def _impute_knn(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.config.numeric_columns if c in df.columns]
        if len(df) < 5: return df
        before_na = int(df[cols].isna().sum().sum())
        imputer = KNNImputer(n_neighbors=3)
        df[cols] = imputer.fit_transform(df[cols])
        self.diagnostics.values_imputed += (before_na - int(df[cols].isna().sum().sum()))
        return df

    def _impute_time(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        field_col = next((c for c in ["field_id", "campo"] if c in df.columns), None)
        df_indexed = df.set_index(date_col)
        before_na = int(df[self.config.numeric_columns].isna().sum().sum())
        for col in self.config.numeric_columns:
            if col not in df_indexed.columns: continue
            if field_col:
                df_indexed[col] = df_indexed.groupby(field_col)[col].transform(lambda s: s.interpolate(method="time", limit=3).ffill().bfill())
            else:
                df_indexed[col] = df_indexed[col].interpolate(method="time", limit=3).ffill().bfill()
        df_res = df_indexed.reset_index()
        self.diagnostics.values_imputed += (before_na - int(df_res[self.config.numeric_columns].isna().sum().sum()))
        return df_res

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        keys = [k for k in self.config.dedup_keys if k in df.columns]
        before = len(df)
        df = df.drop_duplicates(subset=keys if keys else None, keep="last")
        self.diagnostics.duplicates_removed = (before - len(df))
        return df
