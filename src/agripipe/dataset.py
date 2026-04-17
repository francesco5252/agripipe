"""`torch.utils.data.Dataset` per integrazione diretta con DataLoader."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from agripipe.tensorizer import Tensorizer


class AgriDataset(Dataset):
    """Dataset PyTorch costruito da un DataFrame già pulito."""

    def __init__(
        self,
        df: pd.DataFrame,
        numeric_columns: list[str],
        categorical_columns: list[str] | None = None,
        target: str | None = None,
        target_dtype: str = "float32",
    ):
        self.tensorizer = Tensorizer(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns or [],
            target=target,
            target_dtype=target_dtype,
        )
        bundle = self.tensorizer.fit_transform(df)
        self.features = bundle.features
        self.target = bundle.target
        self.feature_names = bundle.feature_names

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.target is None:
            return self.features[idx]
        return self.features[idx], self.target[idx]
