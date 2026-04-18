"""Unified AgriPipe Wrapper: L'Infrastruttura Integrale.

Il cuore del prodotto: incapsula Loader, Cleaner e Tensorizer in un unico 
oggetto persistibile per produzione e ricerca.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import torch

from agripipe.loader import load_raw, load_from_dir
from agripipe.cleaner import AgriCleaner, CleanerConfig
from agripipe.tensorizer import Tensorizer, TensorBundle


class AgriPipe:
    """Il 'Cervello' unificato di AgriPipe (Production Wrapper)."""

    def __init__(self, cleaner_config: CleanerConfig, tensorizer_args: dict):
        self.cleaner = AgriCleaner(cleaner_config)
        self.tensorizer_args = tensorizer_args
        self.tensorizer: Tensorizer | None = None
        self._fitted = False

    def process_file(
        self, 
        path: str | Path, 
        split_ratios: tuple[float, float, float] | None = None
    ) -> TensorBundle:
        """Esegue l'intero workflow su un singolo file."""
        df_raw = load_raw(path)
        return self._process_df(df_raw, split_ratios)

    def process_directory(
        self, 
        dir_path: str | Path, 
        split_ratios: tuple[float, float, float] | None = None
    ) -> TensorBundle:
        """Esegue l'intero workflow su una cartella di file."""
        df_raw = load_from_dir(dir_path)
        return self._process_df(df_raw, split_ratios)

    def _process_df(self, df: pd.DataFrame, split_ratios: tuple[float, float, float] | None) -> TensorBundle:
        # 1. Pulizia Master
        df_clean = self.cleaner.clean(df)
        
        # 2. Tensorizzazione Master
        if not self.tensorizer:
            # Assicuriamoci che il target non sia nelle numeric_columns delle feature
            target = self.tensorizer_args.get("target")
            if target and target in self.tensorizer_args.get("numeric_columns", []):
                self.tensorizer_args["numeric_columns"].remove(target)
                
            self.tensorizer = Tensorizer(**self.tensorizer_args)
            
        bundle = self.tensorizer.fit_transform(df_clean, split_ratios=split_ratios)
        self._fitted = True
        return bundle

    def inverse_predict(self, y_scaled: torch.Tensor | np.ndarray) -> np.ndarray:
        """Riconverte le predizioni dell'IA in unità reali (es: da log a t/ha)."""
        if not self.tensorizer:
            raise RuntimeError("Pipeline non ancora addestrata.")
        return self.tensorizer.inverse_transform_target(y_scaled)

    def save(self, path: str | Path) -> None:
        """Salva l'intera pipeline (Cervello + Scale)."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "AgriPipe":
        """Carica una pipeline salvata."""
        return joblib.load(path)
