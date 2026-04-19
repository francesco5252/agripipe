"""``torch.utils.data.Dataset`` costruito sopra un DataFrame pulito.

Wrapper leggero attorno a ``Tensorizer``: espone feature/target come tensor
e fornisce ``__getitem__`` standard, compatibile con ``DataLoader``.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from agripipe.tensorizer import Tensorizer


class AgriDataset(Dataset):
    """PyTorch ``Dataset`` per dati agronomici puliti."""

    def __init__(
        self,
        df: pd.DataFrame,
        numeric_columns: list[str],
        categorical_columns: list[str] | None = None,
        target: str | None = None,
        categorical_strategy: str = "label",
        scaling_strategy: str = "standard",
        split_ratios: tuple[float, float, float] | None = None,
    ):
        self.tensorizer = Tensorizer(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns or [],
            target=target,
            scaling_strategy=scaling_strategy,
            categorical_strategy=categorical_strategy,
        )
        self.df = df
        self.target_col = target

        bundle = self.tensorizer.fit_transform(df, split_ratios=split_ratios)
        self.features = bundle.features
        self.target = bundle.target
        self.feature_names = bundle.feature_names
        self.metadata = bundle.metadata
        self.train_indices = bundle.train_indices
        self.val_indices = bundle.val_indices
        self.test_indices = bundle.test_indices

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.target is None:
            return self.features[idx]
        return self.features[idx], self.target[idx]
