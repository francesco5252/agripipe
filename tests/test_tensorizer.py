import pandas as pd
import pytest
import torch

from agripipe.cleaner import AgriCleaner
from agripipe.dataset import AgriDataset
from agripipe.tensorizer import Tensorizer


def test_tensorizer_output_shape_and_dtype(dirty_df, cleaner_config):
    df = AgriCleaner(cleaner_config).clean(dirty_df)
    t = Tensorizer(
        numeric_columns=cleaner_config.numeric_columns,
        categorical_columns=cleaner_config.categorical_columns,
        target="yield",
    )
    bundle = t.fit_transform(df)
    assert bundle.features.dtype == torch.float32
    assert bundle.features.shape[0] == len(df)
    assert bundle.target is not None


def test_tensorizer_raises_on_nan(cleaner_config):
    import numpy as np

    df = pd.DataFrame({"temp": [1.0, np.nan], "humidity": [1.0, 2.0]})
    t = Tensorizer(numeric_columns=["temp", "humidity"], categorical_columns=[])
    with pytest.raises(ValueError, match="NaN o Inf"):
        t.fit_transform(df)


def test_dataset_getitem(dirty_df, cleaner_config):
    df = AgriCleaner(cleaner_config).clean(dirty_df)
    ds = AgriDataset(
        df,
        numeric_columns=cleaner_config.numeric_columns,
        categorical_columns=cleaner_config.categorical_columns,
        target="yield",
    )
    x, y = ds[0]
    assert x.ndim == 1
    assert y.ndim == 0
