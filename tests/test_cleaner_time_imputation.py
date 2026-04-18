"""Test per la strategia di imputazione time-series."""

import numpy as np
import pandas as pd
import pytest

from agripipe.cleaner import AgriCleaner, CleanerConfig


def _ts_df_with_gap() -> pd.DataFrame:
    """Campo F1: temperatura 10 al giorno 1, NaN al giorno 2, 20 al giorno 3."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-03-01", periods=5, freq="D"),
            "field_id": ["F1"] * 5,
            "temp": [10.0, np.nan, 20.0, 22.0, 24.0],
            "humidity": [60.0, 65.0, np.nan, 70.0, 72.0],
        }
    )


def test_time_imputation_interpolates_inside_field():
    config = CleanerConfig(
        numeric_columns=["temp", "humidity"],
        date_columns=["date"],
        missing_strategy="time",
        outlier_method="none",
    )
    cleaner = AgriCleaner(config)
    df_clean = cleaner.clean(_ts_df_with_gap())
    # Il NaN del giorno 2 (temp) dovrebbe essere ~15 (interpolazione lineare temporale)
    assert df_clean.loc[df_clean["date"] == "2024-03-02", "temp"].iloc[0] == pytest.approx(
        15.0, abs=0.5
    )
    assert cleaner.diagnostics.imputation_strategy_used == "time"


def test_time_imputation_falls_back_to_median_when_no_date():
    df = pd.DataFrame(
        {
            "field_id": ["F1"] * 5,
            "temp": [10.0, np.nan, 20.0, 22.0, 24.0],
        }
    )
    config = CleanerConfig(
        numeric_columns=["temp"],
        date_columns=[],
        missing_strategy="time",
        outlier_method="none",
    )
    cleaner = AgriCleaner(config)
    df_clean = cleaner.clean(df)
    # Fallback: nessun NaN residuo
    assert df_clean["temp"].isna().sum() == 0
    assert cleaner.diagnostics.imputation_strategy_used == "median"


def test_time_imputation_respects_field_boundaries():
    """Il NaN di F1 non deve essere riempito con valori di F2."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-03-01",
                    "2024-03-02",
                    "2024-03-03",  # F1
                    "2024-03-01",
                    "2024-03-02",
                    "2024-03-03",  # F2
                ]
            ),
            "field_id": ["F1", "F1", "F1", "F2", "F2", "F2"],
            "temp": [10.0, np.nan, 20.0, 100.0, 105.0, 110.0],
        }
    )
    config = CleanerConfig(
        numeric_columns=["temp"],
        date_columns=["date"],
        missing_strategy="time",
        outlier_method="none",
    )
    cleaner = AgriCleaner(config)
    df_clean = cleaner.clean(df)
    f1_middle = df_clean[(df_clean["field_id"] == "F1") & (df_clean["date"] == "2024-03-02")][
        "temp"
    ].iloc[0]
    # Deve essere ~15 (media di 10 e 20), non ~100 (valori di F2)
    assert f1_middle == pytest.approx(15.0, abs=1.0)
