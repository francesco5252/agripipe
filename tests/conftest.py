"""Fixtures condivise per i test."""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

from agripipe.cleaner import CleanerConfig


@pytest.fixture
def dirty_df() -> pd.DataFrame:
    """DataFrame con errori realistici: NaN, outlier, duplicati, pH impossibili."""
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "field_id": ["F1"] * 25 + ["F2"] * 25,
            "temp": rng.normal(20, 3, n),
            "humidity": rng.uniform(40, 80, n),
            "ph": rng.uniform(5.5, 7.5, n),
            "yield": rng.normal(5.0, 1.0, n),
            "crop_type": ["wheat"] * 30 + ["corn"] * 20,
        }
    )
    # Inietta anomalie
    df.loc[0, "ph"] = 99.0  # fuori range fisico
    df.loc[1, "temp"] = 500.0  # outlier estremo
    df.loc[2, "humidity"] = np.nan
    df.loc[3, "yield"] = np.nan
    # Duplicato
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)
    return df


@pytest.fixture
def cleaner_config() -> CleanerConfig:
    return CleanerConfig(
        numeric_columns=["temp", "humidity", "ph", "yield"],
        categorical_columns=["field_id", "crop_type"],
        date_columns=["date"],
        dedup_keys=["field_id", "date"],
        missing_strategy="median",
        outlier_method="iqr",
        physical_bounds={"ph": (0.0, 14.0), "humidity": (0.0, 100.0)},
    )
