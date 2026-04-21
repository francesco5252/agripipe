"""Step 2 — Cleaner: pulizia statistica dei dati agronomici.

Nessuna regola agronomica interpretativa: il Cleaner è un filtro oggettivo.

Pipeline (``AgriCleaner.clean``):

1. **Coercizione tipi** — stringhe numeriche "stile IT" (``"12,5"``) → ``float``.
2. **Limiti fisici** — valori fuori range definito dall'utente → ``NaN``
   (es. ``pH < 0`` o ``pH > 14`` è impossibile).
3. **Outlier IQR / Z-Score** — anomalie statistiche marcate ``NaN``.
4. **Imputazione** — ``mean`` / ``median`` / ``ffill`` / ``drop`` / ``time``
   (interpolazione temporale con fallback automatico alla mediana se manca
   la colonna data).
5. **Imputazione categorica** — ``ffill`` + ``bfill`` ordinato per data.
6. **Deduplicazione** — rimozione righe duplicate sulle chiavi richieste.
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
OutlierMethod = Literal["iqr", "zscore", "none"]


@dataclass
class CleanerConfig:
    """Configurazione del Cleaner. Tutti i campi sono opzionali.

    ``numeric_columns`` vuota ⇒ auto-detect delle colonne numeriche al runtime.
    """

    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    date_columns: list[str] = field(default_factory=list)
    dedup_keys: list[str] = field(default_factory=list)
    missing_strategy: ImputationStrategy = "median"
    missing_drop_threshold: float = 0.5
    outlier_method: OutlierMethod = "iqr"
    outlier_iqr_multiplier: float = 1.5
    physical_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    auto_unit_conversion: bool = False
    unit_range_heuristic: bool = False
    knowledge_path: str = "configs/agri_knowledge.yaml"
    max_yield: float | None = None
    salinity_tolerance: float | None = None
    harvest_months: list[int] = field(default_factory=list)
    soil_texture: str | None = None
    soft_cleaning: bool = False
    calculate_gdd: bool = False
    t_base: float | None = None


@dataclass
class CleanerDiagnostics:
    """Contatori d'integrità calcolati durante ``clean``."""

    total_rows: int = 0
    current_preset_name: str | None = None
    imputation_strategy_used: str = ""
    values_imputed: int = 0
    outliers_removed: int = 0
    out_of_bounds_removed: int = 0
    agronomic_outliers_removed: int = 0
    duplicates_removed: int = 0
    unit_conversions: dict[str, dict[str, str]] = field(default_factory=dict)


class AgriCleaner:
    """Pipeline di pulizia statistica per dati agronomici.

    Esempio:
        >>> from agripipe.cleaner import AgriCleaner, CleanerConfig
        >>> cleaner = AgriCleaner(CleanerConfig(numeric_columns=["temp", "ph"]))
        >>> df_clean = cleaner.clean(df_raw)
    """

    def __init__(self, config: CleanerConfig):
        self.config = config
        self.knowledge = self._load_knowledge()
        self.diagnostics = CleanerDiagnostics()

    # ---- factory helpers ---------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgriCleaner":
        """Carica la configurazione del Cleaner da un file YAML."""
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        # physical_bounds nel YAML arriva come dict[str, list[2]] — convertiamo a tupla.
        bounds = data.get("physical_bounds") or {}
        data["physical_bounds"] = {k: tuple(v) for k, v in bounds.items()}
        return cls(CleanerConfig(**data))

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        knowledge_path: str = "configs/agri_knowledge.yaml",
    ) -> "AgriCleaner":
        """Carica bounds fisici dal preset regionale nell'``agri_knowledge.yaml``."""
        path = Path(knowledge_path)
        knowledge = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        preset_data = knowledge.get("regional_presets", {}).get(preset_name)
        if not preset_data:
            raise ValueError(f"Preset '{preset_name}' non trovato in {path}.")

        config = CleanerConfig(
            knowledge_path=str(path),
            date_columns=["date"],
            dedup_keys=["field_id", "date"],
            categorical_columns=["crop_type", "field_id"],
        )
        if "temp_range" in preset_data:
            lo, hi = preset_data["temp_range"]
            config.physical_bounds["temp"] = (float(lo), float(hi))
        if "ideal_ph" in preset_data:
            lo, hi = preset_data["ideal_ph"]
            config.physical_bounds["ph"] = (float(lo), float(hi))
        if "max_yield" in preset_data:
            config.max_yield = float(preset_data["max_yield"])
        if "salinity_tolerance" in preset_data:
            config.salinity_tolerance = float(preset_data["salinity_tolerance"])
        if "harvest_months" in preset_data:
            config.harvest_months = [int(m) for m in preset_data["harvest_months"]]
        if "suolo_tessitura" in preset_data:
            config.soil_texture = str(preset_data["suolo_tessitura"])

        # Carichiamo la t_base specifica della coltura se disponibile
        crop_type = preset_data.get("crop")
        if crop_type and "crops" in knowledge and crop_type in knowledge["crops"]:
            config.t_base = knowledge["crops"][crop_type].get("t_base")

        inst = cls(config)
        inst.diagnostics.current_preset_name = preset_name
        return inst

    def _load_knowledge(self) -> dict:
        path = Path(self.config.knowledge_path)
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # ---- main API ----------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Esegue la pipeline di pulizia e ritorna un DataFrame pulito.

        ``self.diagnostics`` viene popolato con i contatori di ogni fase.

        Args:
            df: DataFrame grezzo (già caricato via ``load_raw``).

        Returns:
            DataFrame pulito con punteggi di fiducia e (opzionalmente) GDD.
        """
        preset_name = self.diagnostics.current_preset_name  # preservato tra run
        self.diagnostics = CleanerDiagnostics(total_rows=len(df), current_preset_name=preset_name)
        df = df.copy()

        # Step 0: Inizializzazione Confidence Score
        if self.config.soft_cleaning:
            df["confidence"] = 1.0

        # Step 1: Conversione unità (se abilitata)
        df = self._convert_units(df)

        # Auto-detect colonne numeriche se non specificate
        if not self.config.numeric_columns:
            self.config.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        df = self._coerce_types(df)
        df = self._impute_categorical(df)
        df = self._drop_sparse_columns(df)
        df = self._apply_physical_bounds(df)
        df = self._apply_agronomic_rules(df)
        df = self._handle_outliers(df)
        df = self._impute_missing(df)
        df = self._deduplicate(df)

        # Step Finale: Calcolo GDD (Growing Degree Days)
        if self.config.calculate_gdd:
            df = self._calculate_gdd(df)

        return df

    # ---- private stages ----------------------------------------------------

    def _calculate_gdd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcola i Gradi Giorno accumulati (GDD) basati sulla t_base."""
        if self.config.t_base is None or "temp" not in df.columns:
            return df

        date_col = next((c for c in ["date", "data"] if c in df.columns), None)
        field_col = next((c for c in ["field_id", "campo"] if c in df.columns), None)

        if not date_col:
            return df

        # Calcolo GDD giornaliero: max(0, T_media - T_base)
        # Qui usiamo 'temp' come proxy della media giornaliera se il dato è giornaliero
        df = df.sort_values([field_col, date_col]) if field_col else df.sort_values(date_col)

        def gdd_daily(t):
            return max(0, t - self.config.t_base)

        df["gdd_daily"] = df["temp"].apply(gdd_daily)

        # Accumulo per campo (cumsum)
        if field_col:
            df["gdd_accumulated"] = df.groupby(field_col)["gdd_daily"].cumsum()
        else:
            df["gdd_accumulated"] = df["gdd_daily"].cumsum()

        logger.info("Calculated GDD with t_base=%.1f", self.config.t_base)
        return df

    def _apply_agronomic_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica filtri basati su limiti agronomici del preset."""
        soft = self.config.soft_cleaning

        # 1. Resa massima (yield)
        if self.config.max_yield is not None and "yield" in df.columns:
            mask = df["yield"] > self.config.max_yield
            n_bad = int(mask.sum())
            if n_bad:
                if soft:
                    df.loc[mask, "confidence"] *= 0.3  # Penalità forte
                    logger.info(
                        "Agronomic soft-filter: penalized %d yield values > %.1f",
                        n_bad,
                        self.config.max_yield,
                    )
                else:
                    df.loc[mask, "yield"] = np.nan
                    self.diagnostics.agronomic_outliers_removed += n_bad
                    logger.info(
                        "Agronomic filter: removed %d yield values > %.1f",
                        n_bad,
                        self.config.max_yield,
                    )

        # 2. Calendario di raccolta (harvest_months)
        if self.config.harvest_months and "yield" in df.columns:
            date_col = next((c for c in ["date", "data"] if c in df.columns), None)
            if date_col:
                temp_dates = pd.to_datetime(df[date_col], errors="coerce")
                mask = (df["yield"] > 0) & (~temp_dates.dt.month.isin(self.config.harvest_months))
                n_bad = int(mask.sum())
                if n_bad:
                    if soft:
                        df.loc[mask, "confidence"] *= 0.1  # Dato quasi certamente errato
                        logger.warning(
                            "Temporal soft-filter: penalized %d yield values outside window", n_bad
                        )
                    else:
                        df.loc[mask, "yield"] = np.nan
                        self.diagnostics.agronomic_outliers_removed += n_bad
                        logger.warning(
                            "Temporal filter: removed %d yield values outside harvest window", n_bad
                        )

        return df

    def _convert_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rileva e converte unità non-SI se abilitato in config."""
        if not self.config.auto_unit_conversion:
            return df

        from agripipe.units import detect_and_convert_units

        df, report = detect_and_convert_units(
            df, use_range_heuristic=self.config.unit_range_heuristic
        )
        self.diagnostics.unit_conversions = report
        return df

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Date → ``datetime``; numerici "12,5"/"12.5" → ``float``."""
        for col in self.config.date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        for col in self.config.numeric_columns:
            if col in df.columns:
                # Supporta numeri in formato italiano ("12,5") convertendo la virgola
                if df[col].dtype == object:
                    df[col] = df[col].apply(
                        lambda v: str(v).replace(",", ".") if isinstance(v, str) else v
                    )
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _drop_sparse_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rimuove colonne con % NaN oltre ``missing_drop_threshold``."""
        nan_ratio = df.isna().mean()
        to_drop = nan_ratio[nan_ratio > self.config.missing_drop_threshold].index.tolist()
        return df.drop(columns=to_drop) if to_drop else df

    def _apply_physical_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valori fuori dai limiti fisici → NaN (sempre, anche in soft_cleaning)."""
        for col, (lo, hi) in self.config.physical_bounds.items():
            if col in df.columns:
                mask = (df[col] < lo) | (df[col] > hi)
                n_bad = int(mask.sum())
                if n_bad:
                    df.loc[mask, col] = np.nan
                    self.diagnostics.out_of_bounds_removed += n_bad
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Outlier statistici: NaN in hard mode, penalità confidence in soft mode."""
        method = self.config.outlier_method
        soft = self.config.soft_cleaning
        if method == "none":
            return df

        for col in self.config.numeric_columns:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if len(s) < 10:
                continue

            if method == "zscore":
                mean, std = s.mean(), s.std()
                if std == 0:
                    continue
                mask = ((df[col] - mean).abs() / std) > 3.0
            else:  # iqr
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                k = self.config.outlier_iqr_multiplier
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

    def _impute_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Colonne categoriche: ``ffill`` + ``bfill`` ordinato per data."""
        date_col = next((c for c in ["date", "data"] if c in df.columns), None)
        if not date_col:
            return df
        sort_idx = df[date_col].argsort(kind="mergesort")
        for col in self.config.categorical_columns:
            if col in df.columns:
                filled = df[col].iloc[sort_idx].ffill().bfill()
                df[col] = filled.reindex(df.index)
        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputazione numerica. ``time`` con fallback a mediana se manca data."""
        cols = [c for c in self.config.numeric_columns if c in df.columns]
        if not cols:
            self.diagnostics.imputation_strategy_used = self.config.missing_strategy
            return df

        strat: str = self.config.missing_strategy
        date_col = next((c for c in ["date", "data"] if c in df.columns), None)
        if strat == "time" and not date_col:
            logger.info("Strategia 'time' richiesta ma nessuna colonna data: fallback a 'median'.")
            strat = "median"

        before_na = int(df[cols].isna().sum().sum())

        if strat == "drop":
            df = df.dropna(subset=cols).reset_index(drop=True)
        elif strat == "time":
            df = self._impute_time(df, cols, date_col)  # type: ignore[arg-type]
        else:
            for col in cols:
                if strat == "mean":
                    fill = df[col].mean()
                elif strat == "ffill":
                    df[col] = df[col].ffill().bfill()
                    continue
                else:  # median (default)
                    fill = df[col].median()
                df[col] = df[col].fillna(fill)

        after_na = int(df[cols].isna().sum().sum()) if strat != "drop" else 0
        self.diagnostics.values_imputed += max(0, before_na - after_na)
        self.diagnostics.imputation_strategy_used = strat
        return df

    def _impute_time(self, df: pd.DataFrame, cols: list[str], date_col: str) -> pd.DataFrame:
        """Interpolazione temporale per campo (resta dentro il field_id)."""
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
        # Fallback finale: mediana per eventuali residui
        for col in cols:
            if col in df_indexed.columns and df_indexed[col].isna().any():
                df_indexed[col] = df_indexed[col].fillna(df_indexed[col].median())
        return df_indexed.reset_index()

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rimuove duplicati sulle ``dedup_keys``."""
        keys = [k for k in self.config.dedup_keys if k in df.columns]
        before = len(df)
        df = df.drop_duplicates(subset=keys if keys else None, keep="last").reset_index(drop=True)
        self.diagnostics.duplicates_removed = before - len(df)
        return df
