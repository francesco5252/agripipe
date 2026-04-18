"""Pulizia automatica: missing, outlier, deduplica, type coercion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml

from agripipe.indices import compute_agronomic_indices
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
    """Conteggi raccolti durante la pulizia per la Sustainability Score Card.

    Popolato da ``AgriCleaner.clean()`` e consumato da
    ``sustainability.compute_scorecard`` e ``metadata.build_metadata``.
    Non influenza il comportamento del cleaner: è solo esposizione dati.
    """

    total_rows: int = 0
    imputation_strategy_used: str = ""
    current_preset_name: str | None = None
    values_imputed: int = 0
    outliers_removed: int = 0
    out_of_bounds_removed: int = 0
    nitrogen_violations: int = 0
    peronospora_events: int = 0
    irrigation_inefficient: int = 0
    soil_organic_low: int = 0
    heat_stress_flowering: int = 0
    late_frost_events: int = 0


class AgriCleaner:
    """Pipeline di pulizia configurabile per dati agronomici."""

    def __init__(self, config: CleanerConfig):
        self.config = config
        self.knowledge = self._load_knowledge()
        self.diagnostics = CleanerDiagnostics()

    def _load_knowledge(self) -> dict:
        path = Path(self.config.knowledge_path)
        if not path.exists():
            logger.warning("Cervello agronomico non trovato in %s", path)
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
        """Istanzia caricando un preset regionale dal file di conoscenza.
        
        Args:
            preset_name: Nome della chiave in ``regional_presets`` (es. "ulivo_ligure").
            knowledge_path: Path al file YAML dei preset.
            
        Raises:
            ValueError: Se il preset non esiste.
        """
        path = Path(knowledge_path)
        with open(path, "r", encoding="utf-8") as f:
            knowledge = yaml.safe_load(f)
        
        presets = knowledge.get("regional_presets", {})
        if preset_name not in presets:
            raise ValueError(f"Preset '{preset_name}' non trovato in {knowledge_path}")
        
        preset_data = presets[preset_name]
        
        # Configurazione minima: le colonne verranno scoperte o passate
        # Per ora lasciamo che AgriCleaner scopra le numeriche se non forzate
        config = CleanerConfig(
            knowledge_path=knowledge_path,
            # Non forziamo colonne che potrebbero mancare nel dataset sintetico
            # ma impostiamo quelle che ci aspettiamo di avere se presenti
            numeric_columns=[], 
            date_columns=["date", "data"],
            dedup_keys=["field_id", "date"],
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
        """Applica l'intera pipeline di pulizia e calcola gli indici agronomici."""
        self.diagnostics = CleanerDiagnostics(total_rows=len(df))
        logger.info("Avvio pulizia intelligente su %d righe", len(df))
        df = df.copy()

        # Se numeric_columns è vuoto, lo popoliamo con tutte le colonne numeriche presenti
        if not self.config.numeric_columns:
            self.config.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            logger.info("Colonne numeriche scoperte: %s", self.config.numeric_columns)

        df = self._coerce_types(df)
        df = self._drop_sparse_columns(df)
        df = self._apply_agronomic_rules(df)
        df = self._apply_physical_bounds(df)
        df = self._handle_outliers(df)
        df = self._impute_missing(df)
        df = self._deduplicate(df)

        # Novità: Calcolo degli indici agronomici (GDD, Bilancio Idrico, etc.)
        df = compute_agronomic_indices(df, self.knowledge)
        logger.info("Indici agronomici calcolati con successo.")

        logger.info("Pulizia completata: %d righe finali", len(df))
        return df

    def _apply_agronomic_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida i dati usando regole biologiche, stagionali e di sostenibilità."""
        if not self.knowledge or "crops" not in self.knowledge:
            return df

        crop_col = next((c for c in ["crop_type", "crop", "coltura"] if c in df.columns), None)
        date_col = next((c for c in ["date", "data"] if c in df.columns), None)
        rain_col = next((c for c in ["rainfall", "pioggia"] if c in df.columns), None)
        hum_col = next((c for c in ["humidity", "umidità"] if c in df.columns), None)
        next((c for c in ["salinity", "salinità", "ec"] if c in df.columns), None)
        soil_moist_col = next(
            (c for c in ["soil_moisture", "umidità_suolo"] if c in df.columns), None
        )
        temp_col = next((c for c in ["temp", "temperatura"] if c in df.columns), None)
        irrig_col = next((c for c in ["irrigation", "irrigazione"] if c in df.columns), None)
        n_col = next((c for c in ["n", "azoto"] if c in df.columns), None)
        som_col = next(
            (c for c in ["organic_matter", "sostanza_organica", "som"] if c in df.columns), None
        )

        if not crop_col:
            return df

        # --- A. COERENZA AMBIENTALE ---
        if rain_col and hum_col:
            mask = (df[rain_col] > 10) & (df[hum_col] < 40)
            if mask.any():
                logger.warning(
                    "Coerenza: %d sensori umidità segnalati come guasti (piove ma <40%%).",
                    int(mask.sum()),
                )
                df.loc[mask, hum_col] = np.nan

        # --- B. STRESS IDRICO ---
        if soil_moist_col and temp_col:
            mask = (df[temp_col] > 38) & (df[soil_moist_col] < 10)
            if mask.any():
                logger.warning(
                    "Stress: %d eventi di calore estremo su suolo arido rilevati.", int(mask.sum())
                )

        # --- C. CONCIMAZIONE SOSTENIBILE ---
        if n_col and (rain_col or soil_moist_col):
            dry_soil = df[soil_moist_col] < 15 if soil_moist_col else True
            no_rain = df[rain_col] < 2 if rain_col else True
            mask = (df[n_col] > 10) & dry_soil & no_rain
            if mask.any():
                count = int(mask.sum())
                logger.warning("Sostenibilità: %d concimazioni su terreno troppo secco.", count)
                self.diagnostics.nitrogen_violations += count

        # --- D. RISTAGNO IDRICO ---
        if rain_col and soil_moist_col:
            mask = (df[rain_col] > 150) & (df[soil_moist_col] > 90)
            if mask.any():
                logger.warning("Ristagno: %d zone a rischio soffocamento radici.", int(mask.sum()))

        # --- E. EFFICIENZA IDRICA ---
        if irrig_col and soil_moist_col:
            mask = (df[irrig_col] > 5) & (df[soil_moist_col] > 85)
            if mask.any():
                count = int(mask.sum())
                logger.warning("Efficienza: %d irrigazioni inutili su suolo saturo.", count)
                self.diagnostics.irrigation_inefficient += count

        # --- F. SALUTE SUOLO ---
        if som_col:
            min_som = self.knowledge.get("general", {}).get("min_organic_matter", 1.5)
            mask = df[som_col] < min_som
            if mask.any():
                count = int(mask.sum())
                logger.warning(
                    "Salute: %d lotti con sostanza organica degradata (<%s%%).", count, min_som
                )
                self.diagnostics.soil_organic_low += count

        # --- GELATE E MALATTIE (Specifiche per coltura) ---
        for crop_name, rules in self.knowledge["crops"].items():
            mask = df[crop_col].str.lower() == crop_name.lower()
            if not mask.any():
                continue

            # Regola dei Tre 10 (Peronospora Vite)
            if crop_name == "wine_grape_docg" and temp_col and rain_col:
                inf_mask = (
                    mask
                    & (df[temp_col] > rules.get("rule_10_temp", 10))
                    & (df[rain_col] > rules.get("rule_10_rain", 10))
                )
                if inf_mask.any():
                    count = int(inf_mask.sum())
                    logger.warning("Malattia [It-Vite]: %d eventi a rischio Peronospora.", count)
                    self.diagnostics.peronospora_events += count

            # Resa
            if "yield" in df.columns and "max_yield" in rules:
                y_mask = mask & (df["yield"] > rules["max_yield"])
                if y_mask.any():
                    logger.info(
                        "Regola [It-%s]: %d rese impossibili (>%s t/ha)",
                        crop_name,
                        int(y_mask.sum()),
                        rules["max_yield"],
                    )
                    df.loc[y_mask, "yield"] = np.nan

            # Colpo di calore in fioritura
            if date_col and temp_col and "flowering_months" in rules:
                m = pd.to_datetime(df[date_col]).dt.month
                ct = rules.get("critical_temp_flowering", 35)
                h_mask = mask & m.isin(rules["flowering_months"]) & (df[temp_col] > ct)
                if h_mask.any():
                    count = int(h_mask.sum())
                    logger.warning(
                        "Stress [It-%s]: %d colpi di calore in fioritura.", crop_name, count
                    )
                    self.diagnostics.heat_stress_flowering += count

            # Gelo tardivo
            if date_col and temp_col and "frost_danger_months" in rules:
                m = pd.to_datetime(df[date_col]).dt.month
                f_mask = mask & m.isin(rules["frost_danger_months"]) & (df[temp_col] < 0)
                if f_mask.any():
                    count = int(f_mask.sum())
                    logger.warning("Gelo [It-%s]: %d gelate tardive rilevate.", crop_name, count)
                    self.diagnostics.late_frost_events += count

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
            logger.warning("Drop colonne sparse: %s", to_drop)
            df = df.drop(columns=to_drop)
        return df

    def _apply_physical_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, (lo, hi) in self.config.physical_bounds.items():
            if col in df.columns:
                mask = (df[col] < lo) | (df[col] > hi)
                if mask.any():
                    count = int(mask.sum())
                    logger.warning("%s: %d fuori range fisico [%s, %s] → NaN", col, count, lo, hi)
                    df.loc[mask, col] = np.nan
                    self.diagnostics.out_of_bounds_removed += count
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.outlier_method == "none":
            return df
        for col in self.config.numeric_columns:
            if col not in df.columns:
                continue
            s = df[col]
            if self.config.outlier_method == "iqr":
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                lo, hi = (
                    q1 - self.config.outlier_iqr_multiplier * iqr,
                    q3 + self.config.outlier_iqr_multiplier * iqr,
                )
            else:
                mu, sigma = s.mean(), s.std()
                lo, hi = mu - 3 * sigma, mu + 3 * sigma
            mask = (s < lo) | (s > hi)
            if mask.any():
                count = int(mask.sum())
                logger.info("%s: %d outlier → NaN", col, count)
                df.loc[mask, col] = np.nan
                self.diagnostics.outliers_removed += count
        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputa i valori mancanti secondo la strategia configurata.

        Per ``strategy="time"`` richiede una colonna ``date``. Se manca, effettua
        fallback a ``median`` con log di warning (scelta di design D1: robustezza
        sopra la rigorosità).
        """
        strat = self.config.missing_strategy

        if strat == "time":
            date_col = next((c for c in ["date", "data"] if c in df.columns), None)
            if not date_col or len(df) < 3:
                logger.warning(
                    "Strategia 'time' richiede colonna date e >=3 righe; "
                    "fallback automatico a 'median'."
                )
                strat = "median"
                self.diagnostics.imputation_strategy_used = "median"
            else:
                return self._impute_time(df, date_col)

        self.diagnostics.imputation_strategy_used = strat

        if strat == "drop":
            return df.dropna()

        before_na = (
            int(df[self.config.numeric_columns].isna().sum().sum())
            if self.config.numeric_columns
            else 0
        )
        for col in self.config.numeric_columns:
            if col not in df.columns:
                continue
            if strat == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strat == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strat == "ffill":
                df[col] = df[col].ffill().bfill()
        after_na = (
            int(df[self.config.numeric_columns].isna().sum().sum())
            if self.config.numeric_columns
            else 0
        )
        self.diagnostics.values_imputed += before_na - after_na
        return df

    def _impute_time(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Interpolazione temporale per-campo.

        Ordina per (field, date), interpola ``method="time"`` con ``limit=3``,
        chiude i bordi con ``ffill().bfill()``. Raggruppa per ``field_id`` per
        non mischiare campi diversi.
        """
        field_col = next((c for c in ["field_id", "campo", "lotto"] if c in df.columns), None)

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        sort_cols = [field_col, date_col] if field_col else [date_col]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        before_na = int(df[self.config.numeric_columns].isna().sum().sum())

        original_index = df.index
        df_indexed = df.set_index(date_col)

        for col in self.config.numeric_columns:
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

        df = df_indexed.reset_index()
        df.index = original_index

        after_na = int(df[self.config.numeric_columns].isna().sum().sum())
        self.diagnostics.values_imputed += before_na - after_na
        self.diagnostics.imputation_strategy_used = "time"

        if not field_col:
            logger.warning("Imputazione 'time' senza field_id: interpolazione globale.")

        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        keys = [k for k in self.config.dedup_keys if k in df.columns]
        if not keys:
            return df.drop_duplicates()
        before = len(df)
        df = df.drop_duplicates(subset=keys, keep="last")
        logger.info("Deduplica: %d → %d righe", before, len(df))
        return df
