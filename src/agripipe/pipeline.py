"""Gestore della sequenzializzazione stile Pipeline scikit-learn."""

from __future__ import annotations

from typing import Any

import pandas as pd

from agripipe.base import AgriTransformer


class Pipeline(AgriTransformer):
    """Esegue sequenzialmente una lista di AgriTransformer."""

    def __init__(self, steps: list[tuple[str, AgriTransformer]]):
        """Costruisce una pipeline.

        Args:
            steps: Lista di tuple `(name, transformer)`.
        """
        self.steps = steps

    def fit(self, df: pd.DataFrame, y: Any = None) -> Pipeline:
        """Esegue `fit_transform` sequenzialmente fino all'ultimo, su cui fa solo `fit`."""
        X = df
        for name, step in self.steps[:-1]:
            X = step.fit_transform(X, y)
        self.steps[-1][1].fit(X, y)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Esegue `transform` sequenzialmente su tutti i transformer."""
        X = df
        for name, step in self.steps:
            X = step.transform(X)
        return X

    def fit_transform(self, df: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Esegue fit e transform simultaneamente attraverso l'intera pipeline."""
        X = df
        for name, step in self.steps:
            X = step.fit_transform(X, y)
        return X
