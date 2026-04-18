"""Tensorizzazione Master: Schema Lock, Precision e Target Transformation.

Fase 3 definitiva: implementa il supporto per la trasformazione logaritmica 
del target e l'inverse mapping per rendere l'IA reversibile.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from agripipe.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class TensorBundle:
    """Contenitore per dati pronti per PyTorch, ottimizzato per l'industria."""
    features: torch.Tensor
    target: torch.Tensor | None
    feature_names: list[str]
    metadata: dict
    train_indices: list[int] | None = None
    val_indices: list[int] | None = None
    test_indices: list[int] | None = None


class Tensorizer:
    """Motore Master con Target Transformer e Feature Selection."""

    def __init__(
        self,
        numeric_columns: list[str],
        categorical_columns: list[str],
        target: str | None = None,
        scaling_strategy: Literal["standard", "robust"] = "standard",
        categorical_strategy: Literal["label", "onehot"] = "label",
        precision: Literal["float32", "float16"] = "float32",
        transform_target: bool = True, # Novità Master: Log del target
        drop_redundant: bool = True,
    ):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.target = target
        self.scaling_strategy = scaling_strategy
        self.categorical_strategy = categorical_strategy
        self.precision = precision
        self.transform_target = transform_target
        self.drop_redundant = drop_redundant
        
        self.scaler = StandardScaler() if scaling_strategy == "standard" else RobustScaler()
        self.encoders = {}
        self._fitted = False
        self.feature_names = []
        self.dropped_columns = []
        self.schema_hash = ""

    def fit_transform(self, df: pd.DataFrame, split_ratios: tuple[float, float, float] | None = None) -> TensorBundle:
        """Trasforma il DataFrame ottimizzando feature e target."""
        df = df.copy()
        
        # 1. Feature Selection
        if self.drop_redundant:
            df = self._remove_redundant_columns(df)
        
        self.schema_hash = self._generate_schema_hash(df.columns.tolist())
        df = self._apply_cyclic_encoding(df)
        
        # Features
        num_data = np.empty((len(df), 0))
        active_numerics = [c for c in self.numeric_columns if c in df.columns]
        if active_numerics:
            num_data = self.scaler.fit_transform(df[active_numerics].values)
            self.feature_names = list(active_numerics)

        # Categories
        cat_data = np.empty((len(df), 0))
        active_categoricals = [c for c in self.categorical_columns if c in df.columns]
        for col in active_categoricals:
            if self.categorical_strategy == "label":
                le = LabelEncoder()
                encoded = le.fit_transform(df[col].astype(str)).reshape(-1, 1)
                self.encoders[col] = le
                cat_data = np.hstack([cat_data, encoded])
                self.feature_names.append(col)
            else:
                ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoded = ohe.fit_transform(df[[col]].astype(str))
                self.encoders[col] = ohe
                cat_data = np.hstack([cat_data, encoded])
                self.feature_names.extend(ohe.get_feature_names_out([col]).tolist())

        # Combine
        dtype = np.float32 if self.precision == "float32" else np.float16
        X = np.hstack([num_data, cat_data]).astype(dtype)
        
        # 4. Target Transformation
        y = None
        if self.target and self.target in df.columns:
            y_raw = df[self.target].values.astype(dtype)
            if self.transform_target:
                logger.info("Master Transformer: applico log(1+x) al target %s", self.target)
                y = np.log1p(y_raw)
            else:
                y = y_raw

        # 5. Splitting
        indices = np.arange(len(df))
        train_idx, val_idx, test_idx = None, None, None
        if split_ratios:
            r_train, r_val, r_test = split_ratios
            train_idx, temp_idx = train_test_split(indices, train_size=r_train, random_state=42)
            val_relative_size = r_val / (r_val + r_test)
            val_idx, test_idx = train_test_split(temp_idx, train_size=val_relative_size, random_state=42)

        self._fitted = True
        
        return TensorBundle(
            features=torch.tensor(X),
            target=torch.tensor(y) if y is not None else None,
            feature_names=self.feature_names,
            train_indices=train_idx.tolist() if train_idx is not None else None,
            val_indices=val_idx.tolist() if val_idx is not None else None,
            test_indices=test_idx.tolist() if test_idx is not None else None,
            metadata={
                "file_hash": df.attrs.get("file_hash", "unknown"),
                "schema_lock_hash": self.schema_hash,
                "precision": self.precision,
                "target_transformed": self.transform_target,
                "dropped_redundant": self.dropped_columns,
                "scaling": self.scaling_strategy,
                "categorical": self.categorical_strategy,
                "split_ratios": split_ratios
            }
        )

    def inverse_transform_target(self, y_scaled: np.ndarray | torch.Tensor) -> np.ndarray:
        """Torna dal valore predetto dall'IA al valore reale (t/ha)."""
        if isinstance(y_scaled, torch.Tensor):
            y_scaled = y_scaled.detach().cpu().numpy()
        
        if self.transform_target:
            return np.expm1(y_scaled)
        return y_scaled

    def _remove_redundant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scherma il target dalla rimozione ridondanze e previene il riordino delle righe."""
        # Mai droppare o alterare la riga se è il target
        to_drop = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != self.target and df[col].nunique() <= 1:
                to_drop.append(col)
                
        # Deduplica colonne (escluso target)
        numeric_only = df.select_dtypes(include=[np.number]).columns
        cols_to_check = [c for c in numeric_only if c != self.target]
        if len(cols_to_check) > 1:
            df_check = df[cols_to_check].T
            is_dupe = df_check.duplicated()
            dupes = df_check.index[is_dupe].tolist()
            to_drop.extend(dupes)
            
        final_to_drop = list(set(to_drop))
        self.dropped_columns.extend(final_to_drop)
        return df.drop(columns=final_to_drop)

    def _generate_schema_hash(self, columns: list[str]) -> str:
        schema_str = ",".join(sorted(columns))
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]

    def _apply_cyclic_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        date_col = next((c for c in ["date", "data"] if c in df.columns), None)
        if not date_col: return df
        day_of_year = pd.to_datetime(df[date_col]).dt.dayofyear
        df["date_sin"] = np.sin(2 * np.pi * day_of_year / 366)
        df["date_cos"] = np.cos(2 * np.pi * day_of_year / 366)
        return df

    def get_categorical_mappings(self) -> dict:
        mappings = {}
        for col, enc in self.encoders.items():
            if isinstance(enc, LabelEncoder):
                mappings[col] = {int(i): str(l) for i, l in enumerate(enc.classes_)}
            elif isinstance(enc, OneHotEncoder):
                mappings[col] = {"type": "onehot", "features": enc.get_feature_names_out([col]).tolist()}
        return mappings

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)
