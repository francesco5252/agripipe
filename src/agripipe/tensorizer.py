"""Conversione DataFrame → tensor PyTorch con encoder/scaler persistibili."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

from agripipe.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class TensorBundle:
    """Contenitore dei tensor in output."""

    features: torch.Tensor  # float32 [N, D]
    target: torch.Tensor | None  # float32 o long [N]
    feature_names: list[str]


class Tensorizer:
    """Converte un DataFrame pulito in tensor PyTorch normalizzati.

    Normalizza le colonne numeriche con ``StandardScaler`` (media 0, dev std 1)
    e codifica le colonne categoriche con ``LabelEncoder``. Serializzabile per
    inferenza futura.

    Attributes:
        numeric_columns: Colonne continue da normalizzare.
        categorical_columns: Colonne categoriche da codificare.
        target: Colonna target (opzionale, esclusa dalle features).
        target_dtype: Tipo del tensor target (``"float32"`` | ``"long"``).
        scaler: ``StandardScaler`` fit sui numeric_columns.
        encoders: Dict ``{col_name: LabelEncoder}`` per le categoriche.
    """

    def __init__(
        self,
        numeric_columns: list[str],
        categorical_columns: list[str],
        target: str | None = None,
        target_dtype: str = "float32",
    ):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.target = target
        self.target_dtype = target_dtype
        self.scaler = StandardScaler()
        self.encoders: dict[str, LabelEncoder] = {}
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> TensorBundle:
        """Fit di scaler/encoder sul DataFrame e successiva trasformazione.

        Equivalente a ``_fit(df)`` seguito da ``transform(df)``. Usare questo
        metodo la prima volta sui dati di training; per dati successivi usare
        direttamente ``transform``.

        Args:
            df: DataFrame pulito contenente tutte le colonne numeriche,
                categoriche e (opzionalmente) il target.

        Returns:
            ``TensorBundle`` con ``features`` normalizzate, ``target`` (o None)
            e ``feature_names`` nell'ordine delle colonne.

        Raises:
            ValueError: Se il tensor risultante contiene NaN/Inf.
        """
        self._fit(df)
        return self.transform(df)

    def _fit(self, df: pd.DataFrame) -> None:
        if self.numeric_columns:
            self.scaler.fit(df[self.numeric_columns].values)
        for col in self.categorical_columns:
            le = LabelEncoder()
            le.fit(df[col].astype(str).values)
            self.encoders[col] = le
        self._fitted = True
        logger.info(
            "Tensorizer fit: %d numerici, %d categoriche",
            len(self.numeric_columns),
            len(self.categorical_columns),
        )

    def transform(self, df: pd.DataFrame) -> TensorBundle:
        """Applica la trasformazione già fit a un nuovo DataFrame.

        Args:
            df: DataFrame con le stesse colonne usate in ``fit_transform``.

        Returns:
            ``TensorBundle`` con ``features``, ``target`` (o None) e
            ``feature_names``.

        Raises:
            RuntimeError: Se ``_fitted`` non è ancora stato chiamato.
            ValueError: Se il tensor risultante contiene NaN/Inf.
        """

        parts: list[np.ndarray] = []
        names: list[str] = []

        if self.numeric_columns:
            num = self.scaler.transform(df[self.numeric_columns].values)
            parts.append(num)
            names.extend(self.numeric_columns)

        for col in self.categorical_columns:
            enc = self.encoders[col].transform(df[col].astype(str).values)
            parts.append(enc.reshape(-1, 1).astype(np.float32))
            names.append(col)

        X = np.concatenate(parts, axis=1).astype(np.float32)
        self._assert_finite(X)
        features = torch.from_numpy(X)

        target_tensor: torch.Tensor | None = None
        if self.target and self.target in df.columns:
            y = df[self.target].values
            if self.target_dtype == "long":
                target_tensor = torch.tensor(y, dtype=torch.long)
            else:
                target_tensor = torch.tensor(y.astype(np.float32), dtype=torch.float32)

        logger.info("Output shape: features=%s", tuple(features.shape))
        return TensorBundle(features=features, target=target_tensor, feature_names=names)

    @staticmethod
    def _assert_finite(X: np.ndarray) -> None:
        if not np.isfinite(X).all():
            raise ValueError("Tensor contiene NaN o Inf: pulire i dati prima della conversione.")

    def save(self, path: str | Path) -> None:
        """Serializza scaler + encoder su disco per inferenza futura."""
        joblib.dump(
            {"scaler": self.scaler, "encoders": self.encoders, "config": self.__dict__}, path
        )
        logger.info("Tensorizer salvato in %s", path)
