"""Classi base e interfacce per l'architettura a pipeline di AgriPipe."""

from __future__ import annotations

import abc
from typing import Any

import pandas as pd


class AgriTransformer(abc.ABC):
    """Classe base astratta per tutte le trasformazioni della pipeline AgriPipe.

    Ispirata all'API di scikit-learn (BaseEstimator, TransformerMixin),
    fornisce un contratto comune per l'orchestrazione modulare.
    """

    @abc.abstractmethod
    def fit(self, df: pd.DataFrame, y: Any = None) -> AgriTransformer:
        """Apprende i parametri statistici dal dato.

        Per trasformazioni stateless (es. coercizione tipi, bound fisici),
        deve semplicemente restituire `self`.

        Args:
            df: DataFrame di input.
            y: Target opzionale (nella maggior parte delle pulizie agronomiche è None).

        Returns:
            L'istanza corrente (self).
        """
        pass

    @abc.abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica la trasformazione agronomica/statistica.

        Args:
            df: DataFrame grezzo o parzialmente pulito.

        Returns:
            Nuovo DataFrame modificato. Non deve mutare l'input.
        """
        pass

    def fit_transform(self, df: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Shorthand per eseguire fit() seguito da transform()."""
        return self.fit(df, y).transform(df)
