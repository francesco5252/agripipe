"""Moduli pipeline (Transformers) estratti dalla vecchia logica monolitica."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from agripipe.base import AgriTransformer
from agripipe.utils.logging_setup import get_logger

logger = get_logger(__name__)


class ConfidenceInitializer(AgriTransformer):
    def __init__(self, soft_cleaning: bool):
        self.soft_cleaning = soft_cleaning

    def fit(self, df: pd.DataFrame, y: Any = None) -> ConfidenceInitializer:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.soft_cleaning:
            df["confidence"] = 1.0
        return df


class UnitConverter(AgriTransformer):
    def __init__(self, active: bool, use_heuristic: bool, diagnostics: Any):
        self.active = active
        self.use_heuristic = use_heuristic
        self.diagnostics = diagnostics

    def fit(self, df: pd.DataFrame, y: Any = None) -> UnitConverter:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.active:
            return df
        from agripipe.units import detect_and_convert_units

        df, report = detect_and_convert_units(df, use_range_heuristic=self.use_heuristic)
        self.diagnostics.unit_conversions = report
        return df


class TypeCoercer(AgriTransformer):
    def __init__(self, date_columns: list[str], numeric_columns: list[str]):
        self.date_columns = date_columns
        self.numeric_columns = numeric_columns

    def fit(self, df: pd.DataFrame, y: Any = None) -> TypeCoercer:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        for col in self.numeric_columns:
            if col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].apply(
                        lambda v: str(v).replace(",", ".") if isinstance(v, str) else v
                    )
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df


class SparseColumnDropper(AgriTransformer):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def fit(self, df: pd.DataFrame, y: Any = None) -> SparseColumnDropper:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        nan_ratio = df.isna().mean()
        to_drop = nan_ratio[nan_ratio > self.threshold].index.tolist()
        return df.drop(columns=to_drop) if to_drop else df


class PhysicalBoundsFilter(AgriTransformer):
    def __init__(self, bounds: dict[str, tuple[float, float]], diagnostics: Any):
        self.bounds = bounds
        self.diagnostics = diagnostics

    def fit(self, df: pd.DataFrame, y: Any = None) -> PhysicalBoundsFilter:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, (lo, hi) in self.bounds.items():
            if col in df.columns:
                mask = (df[col] < lo) | (df[col] > hi)
                n_bad = int(mask.sum())
                if n_bad:
                    df.loc[mask, col] = np.nan
                    self.diagnostics.out_of_bounds_removed += n_bad
        return df


class AgronomicRulesFilter(AgriTransformer):
    def __init__(
        self,
        max_yield: float | None,
        harvest_months: list[int],
        soft_cleaning: bool,
        diagnostics: Any,
    ):
        self.max_yield = max_yield
        self.harvest_months = harvest_months
        self.soft_cleaning = soft_cleaning
        self.diagnostics = diagnostics

    def fit(self, df: pd.DataFrame, y: Any = None) -> AgronomicRulesFilter:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        soft = self.soft_cleaning
        if self.max_yield is not None and "yield" in df.columns:
            mask = df["yield"] > self.max_yield
            n_bad = int(mask.sum())
            if n_bad:
                if soft:
                    df.loc[mask, "confidence"] *= 0.3
                    logger.info("Agronomic soft-filter: penalized %d yield values", n_bad)
                else:
                    df.loc[mask, "yield"] = np.nan
                    self.diagnostics.agronomic_outliers_removed += n_bad
                    logger.info("Agronomic filter: removed %d yield values", n_bad)

        if self.harvest_months and "yield" in df.columns:
            date_col = next((c for c in ["date", "data"] if c in df.columns), None)
            if date_col:
                temp_dates = pd.to_datetime(df[date_col], errors="coerce")
                mask = (df["yield"] > 0) & (~temp_dates.dt.month.isin(self.harvest_months))
                n_bad = int(mask.sum())
                if n_bad:
                    if soft:
                        df.loc[mask, "confidence"] *= 0.1
                        logger.warning("Temporal soft-filter: penalized %d yield values", n_bad)
                    else:
                        df.loc[mask, "yield"] = np.nan
                        self.diagnostics.agronomic_outliers_removed += n_bad
                        logger.warning("Temporal filter: removed %d yield values", n_bad)
        return df


class OutlierHandler(AgriTransformer):
    def __init__(
        self,
        method: str,
        iqr_multiplier: float,
        numeric_columns: list[str],
        soft_cleaning: bool,
        diagnostics: Any,
    ):
        self.method = method
        self.iqr_multiplier = iqr_multiplier
        self.numeric_columns = numeric_columns
        self.soft_cleaning = soft_cleaning
        self.diagnostics = diagnostics

    def fit(self, df: pd.DataFrame, y: Any = None) -> OutlierHandler:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.method == "none":
            return df
        soft = self.soft_cleaning

        for col in self.numeric_columns:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if len(s) < 10:
                continue

            if self.method == "zscore":
                mean, std = s.mean(), s.std()
                if std == 0:
                    continue
                mask = ((df[col] - mean).abs() / std) > 3.0
            else:  # iqr
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                k = self.iqr_multiplier
                lo, hi = q1 - k * iqr, q3 + k * iqr
                mask = (df[col] < lo) | (df[col] > hi)

            n_bad = int(mask.sum())
            if n_bad:
                if soft:
                    df.loc[mask, "confidence"] *= 0.5
                else:
                    df.loc[mask, col] = np.nan
                    self.diagnostics.outliers_removed += n_bad
        return df


class CategoricalImputer(AgriTransformer):
    def __init__(self, categorical_columns: list[str]):
        self.categorical_columns = categorical_columns

    def fit(self, df: pd.DataFrame, y: Any = None) -> CategoricalImputer:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        date_col = next((c for c in ["date", "data"] if c in df.columns), None)
        if not date_col:
            return df
        sort_idx = df[date_col].argsort(kind="mergesort")
        for col in self.categorical_columns:
            if col in df.columns:
                filled = df[col].iloc[sort_idx].ffill().bfill()
                df[col] = filled.reindex(df.index)
        return df


class MissingValueImputer(AgriTransformer):
    def __init__(self, strategy: str, numeric_columns: list[str], diagnostics: Any):
        self.strategy = strategy
        self.numeric_columns = numeric_columns
        self.diagnostics = diagnostics

    def fit(self, df: pd.DataFrame, y: Any = None) -> MissingValueImputer:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.numeric_columns if c in df.columns]
        if not cols:
            self.diagnostics.imputation_strategy_used = self.strategy
            return df

        strat = self.strategy
        date_col = next((c for c in ["date", "data"] if c in df.columns), None)
        if strat == "time" and not date_col:
            logger.info("Strategia 'time' richiesta ma nessuna colonna data: fallback a 'median'.")
            strat = "median"

        before_na = int(df[cols].isna().sum().sum())

        if strat == "drop":
            df = df.dropna(subset=cols).reset_index(drop=True)
        elif strat == "time":
            df = self._impute_time(df, cols, date_col)  # type: ignore
        else:
            for col in cols:
                if strat == "mean":
                    fill = df[col].mean()
                elif strat == "ffill":
                    df[col] = df[col].ffill().bfill()
                    continue
                else:
                    fill = df[col].median()
                df[col] = df[col].fillna(fill)

        after_na = int(df[cols].isna().sum().sum()) if strat != "drop" else 0
        self.diagnostics.values_imputed += max(0, before_na - after_na)
        self.diagnostics.imputation_strategy_used = strat
        return df

    def _impute_time(self, df: pd.DataFrame, cols: list[str], date_col: str) -> pd.DataFrame:
        field_col = next((c for c in ["field_id", "campo"] if c in df.columns), None)
        df = df.sort_values(date_col).copy()
        df_indexed = df.set_index(date_col)
        for col in cols:
            if col not in df_indexed.columns:
                continue
            if field_col:
                df_indexed[col] = df_indexed.groupby(field_col)[col].transform(
                    lambda s: s.interpolate(method="time", limit=3).ffill().bfill()
                )
            else:
                df_indexed[col] = (
                    df_indexed[col].interpolate(method="time", limit=3).ffill().bfill()
                )
        for col in cols:
            if col in df_indexed.columns and df_indexed[col].isna().any():
                df_indexed[col] = df_indexed[col].fillna(df_indexed[col].median())
        return df_indexed.reset_index()


class Deduplicator(AgriTransformer):
    def __init__(self, keys: list[str], diagnostics: Any):
        self.keys = keys
        self.diagnostics = diagnostics

    def fit(self, df: pd.DataFrame, y: Any = None) -> Deduplicator:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        keys = [k for k in self.keys if k in df.columns]
        before = len(df)
        df = df.drop_duplicates(subset=keys if keys else None, keep="last").reset_index(drop=True)
        self.diagnostics.duplicates_removed = before - len(df)
        return df


class GDDCalculator(AgriTransformer):
    def __init__(self, t_base: float | None):
        self.t_base = t_base

    def fit(self, df: pd.DataFrame, y: Any = None) -> GDDCalculator:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.t_base is None or "temp" not in df.columns:
            return df

        date_col = next((c for c in ["date", "data"] if c in df.columns), None)
        field_col = next((c for c in ["field_id", "campo"] if c in df.columns), None)
        if not date_col:
            return df

        df = df.sort_values([field_col, date_col]) if field_col else df.sort_values(date_col)

        def gdd_daily(t: float) -> float:
            # Asserzione necessaria per mypy
            return max(0, t - float(self.t_base))  # type: ignore

        df["gdd_daily"] = df["temp"].apply(gdd_daily)

        if field_col:
            df["gdd_accumulated"] = df.groupby(field_col)["gdd_daily"].cumsum()
        else:
            df["gdd_accumulated"] = df["gdd_daily"].cumsum()

        logger.info("Calculated GDD with t_base=%.1f", self.t_base)
        return df
