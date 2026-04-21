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
from pydantic import BaseModel, Field, model_validator

from agripipe.utils.logging_setup import get_logger

logger = get_logger(__name__)

ImputationStrategy = Literal["mean", "median", "ffill", "drop", "time"]
OutlierMethod = Literal["iqr", "zscore", "none"]


class CleanerConfig(BaseModel):
    """Configurazione del Cleaner pilotata da Pydantic. Tutti i campi sono opzionali.

    ``numeric_columns`` vuota ⇒ auto-detect delle colonne numeriche al runtime.
    """

    model_config = {"protected_namespaces": ()}

    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    date_columns: list[str] = Field(default_factory=list)
    dedup_keys: list[str] = Field(default_factory=list)
    missing_strategy: ImputationStrategy = "median"
    missing_drop_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    outlier_method: OutlierMethod = "iqr"
    outlier_iqr_multiplier: float = Field(default=1.5, gt=0.0)
    physical_bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)
    auto_unit_conversion: bool = False
    unit_range_heuristic: bool = False
    knowledge_path: str = "configs/agri_knowledge.yaml"
    max_yield: float | None = Field(default=None, ge=0.0)
    salinity_tolerance: float | None = Field(default=None, ge=0.0)
    harvest_months: list[int] = Field(default_factory=list)
    soil_texture: str | None = None
    soft_cleaning: bool = False
    calculate_gdd: bool = False
    t_base: float | None = None

    @model_validator(mode="after")
    def check_physical_bounds(self) -> "CleanerConfig":
        for col, (lo, hi) in self.physical_bounds.items():
            if lo > hi:
                raise ValueError(f"Livello minimo {lo} per '{col}' maggiore del massimo {hi}.")
        return self


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

        # Auto-detect colonne numeriche se non specificate
        if not self.config.numeric_columns:
            self.config.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        from agripipe.pipeline import Pipeline
        from agripipe.transformers import (
            AgronomicRulesFilter,
            CategoricalImputer,
            ConfidenceInitializer,
            Deduplicator,
            GDDCalculator,
            MissingValueImputer,
            OutlierHandler,
            PhysicalBoundsFilter,
            SparseColumnDropper,
            TypeCoercer,
            UnitConverter,
        )

        steps = [
            ("conf", ConfidenceInitializer(self.config.soft_cleaning)),
            (
                "units",
                UnitConverter(
                    self.config.auto_unit_conversion,
                    self.config.unit_range_heuristic,
                    self.diagnostics,
                ),
            ),
            ("coerce", TypeCoercer(self.config.date_columns, self.config.numeric_columns)),
            ("cat_impute", CategoricalImputer(self.config.categorical_columns)),
            ("sparse", SparseColumnDropper(self.config.missing_drop_threshold)),
            ("bounds", PhysicalBoundsFilter(self.config.physical_bounds, self.diagnostics)),
            (
                "agro",
                AgronomicRulesFilter(
                    self.config.max_yield,
                    self.config.harvest_months,
                    self.config.soft_cleaning,
                    self.diagnostics,
                ),
            ),
            (
                "outlier",
                OutlierHandler(
                    self.config.outlier_method,
                    self.config.outlier_iqr_multiplier,
                    self.config.numeric_columns,
                    self.config.soft_cleaning,
                    self.diagnostics,
                ),
            ),
            (
                "missing",
                MissingValueImputer(
                    self.config.missing_strategy, self.config.numeric_columns, self.diagnostics
                ),
            ),
            ("dedup", Deduplicator(self.config.dedup_keys, self.diagnostics)),
        ]

        if self.config.calculate_gdd:
            steps.append(("gdd", GDDCalculator(self.config.t_base)))

        pipeline = Pipeline(steps)  # type: ignore[arg-type]
        return pipeline.fit_transform(df.copy())
