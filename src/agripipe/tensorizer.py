"""Step 3 — Tensorizer: dati puliti → ``torch.Tensor`` pronti per PyTorch.

Prende un DataFrame già pulito e produce un ``TensorBundle`` con:

* Features numeriche scalate (StandardScaler o RobustScaler).
* Categoriche codificate (LabelEncoder o OneHotEncoder).
* Target opzionale.
* Split opzionale Train/Val/Test (indici).
* Metadata tecnico: ``file_hash``, ``schema_lock_hash``, parametri scaler, ecc.

Solleva ``ValueError`` se incontra ``NaN``/``Inf`` in ingresso: è compito del
``AgriCleaner`` consegnare dati puliti.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, StandardScaler

from agripipe.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class TensorBundle:
    """Contenitore leggero per features, target e metadata pronti per PyTorch."""

    features: torch.Tensor
    target: torch.Tensor | None
    feature_names: list[str]
    metadata: dict
    train_indices: list[int] | None = None
    val_indices: list[int] | None = None
    test_indices: list[int] | None = None


class Tensorizer:
    """Converte un DataFrame pulito in tensor ``float32`` per PyTorch."""

    def __init__(
        self,
        numeric_columns: list[str],
        categorical_columns: list[str],
        target: str | None = None,
        scaling_strategy: Literal["standard", "robust"] = "standard",
        categorical_strategy: Literal["label", "onehot"] = "label",
    ):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.target = target
        self.scaling_strategy = scaling_strategy
        self.categorical_strategy = categorical_strategy

        self.scaler = StandardScaler() if scaling_strategy == "standard" else RobustScaler()
        self.encoders: dict[str, LabelEncoder | OneHotEncoder] = {}
        self.feature_names: list[str] = []
        self.schema_hash: str = ""
        self._fitted = False

    # ---- main API ----------------------------------------------------------

    def fit_transform(
        self,
        df: pd.DataFrame,
        split_ratios: tuple[float, float, float] | None = None,
    ) -> TensorBundle:
        """Esegue fit + transform in un colpo solo.

        Args:
            df: DataFrame già pulito (nessun NaN nelle colonne numeriche).
            split_ratios: ``(train, val, test)`` con somma = 1.0. Se ``None``,
                nessuno split.

        Returns:
            TensorBundle con tensor float32 pronti all'uso.

        Raises:
            ValueError: Se ``df`` contiene ``NaN`` o ``Inf`` nelle colonne usate.
        """
        df = df.copy()
        self.feature_names = []
        self.schema_hash = _compute_schema_hash(df.columns.tolist())

        # 1. Validazione: niente NaN/Inf nelle colonne numeriche o nel target
        active_num = [c for c in self.numeric_columns if c in df.columns]
        active_cat = [c for c in self.categorical_columns if c in df.columns]
        cols_to_check = active_num + ([self.target] if self.target in df.columns else [])
        if cols_to_check:
            arr = df[cols_to_check].to_numpy(dtype=float, na_value=np.nan)
            if not np.isfinite(arr).all():
                raise ValueError(
                    "Il DataFrame contiene NaN o Inf nelle colonne usate dal Tensorizer: "
                    f"{cols_to_check}. Eseguire prima AgriCleaner.clean()."
                )

        # 2. Features numeriche
        num_data = np.empty((len(df), 0), dtype=np.float32)
        if active_num:
            num_data = self.scaler.fit_transform(df[active_num].to_numpy(dtype=float))
            self.feature_names.extend(active_num)

        # 3. Features categoriche
        cat_data = np.empty((len(df), 0), dtype=np.float32)
        for col in active_cat:
            if self.categorical_strategy == "label":
                le = LabelEncoder()
                encoded = le.fit_transform(df[col].astype(str)).reshape(-1, 1)
                self.encoders[col] = le
                cat_data = np.hstack([cat_data, encoded])
                self.feature_names.append(col)
            else:  # onehot
                ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoded = ohe.fit_transform(df[[col]].astype(str))
                self.encoders[col] = ohe
                cat_data = np.hstack([cat_data, encoded])
                self.feature_names.extend(ohe.get_feature_names_out([col]).tolist())

        # 4. Combine → float32
        X = np.hstack([num_data, cat_data]).astype(np.float32)

        # 5. Target
        y_tensor: torch.Tensor | None = None
        if self.target and self.target in df.columns:
            y_tensor = torch.tensor(df[self.target].to_numpy(dtype=np.float32))

        # 6. Split indici
        train_idx = val_idx = test_idx = None
        if split_ratios is not None:
            r_train, r_val, r_test = split_ratios
            if not np.isclose(r_train + r_val + r_test, 1.0):
                raise ValueError(f"split_ratios devono sommare a 1.0, ricevuto: {split_ratios}")
            indices = np.arange(len(df))
            train_idx, tmp_idx = train_test_split(indices, train_size=r_train, random_state=42)
            val_rel = r_val / (r_val + r_test)
            val_idx, test_idx = train_test_split(tmp_idx, train_size=val_rel, random_state=42)

        self._fitted = True

        return TensorBundle(
            features=torch.tensor(X),
            target=y_tensor,
            feature_names=list(self.feature_names),
            train_indices=train_idx.tolist() if train_idx is not None else None,
            val_indices=val_idx.tolist() if val_idx is not None else None,
            test_indices=test_idx.tolist() if test_idx is not None else None,
            metadata={
                "file_hash": df.attrs.get("file_hash", "unknown"),
                "schema_lock_hash": self.schema_hash,
                "scaling": self.scaling_strategy,
                "categorical": self.categorical_strategy,
                "scaler_params": _scaler_params(self.scaler),
                "split_ratios": split_ratios,
            },
        )

    # ---- helpers -----------------------------------------------------------

    def get_categorical_mappings(self) -> dict:
        """Ritorna i valori originali di ogni colonna categorica codificata."""
        mappings: dict = {}
        for col, enc in self.encoders.items():
            if isinstance(enc, LabelEncoder):
                mappings[col] = {int(i): str(v) for i, v in enumerate(enc.classes_)}
            elif isinstance(enc, OneHotEncoder):
                mappings[col] = {
                    "type": "onehot",
                    "features": enc.get_feature_names_out([col]).tolist(),
                }
        return mappings


# ---- module helpers --------------------------------------------------------


def _compute_schema_hash(columns: list[str]) -> str:
    """Hash SHA-256 (16 char) stabile rispetto all'ordine delle colonne."""
    schema_str = ",".join(sorted(str(c) for c in columns))
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


def _scaler_params(scaler: StandardScaler | RobustScaler) -> dict:
    """Estrae i parametri serializzabili dello scaler (mean/scale o center/scale)."""
    if isinstance(scaler, StandardScaler):
        return {
            "type": "standard",
            "mean": scaler.mean_.tolist() if hasattr(scaler, "mean_") else [],
            "scale": scaler.scale_.tolist() if hasattr(scaler, "scale_") else [],
        }
    return {
        "type": "robust",
        "center": scaler.center_.tolist() if hasattr(scaler, "center_") else [],
        "scale": scaler.scale_.tolist() if hasattr(scaler, "scale_") else [],
    }
