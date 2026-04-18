"""Test che CleanerDiagnostics sia istanziabile e collegato a AgriCleaner."""

import pandas as pd
import numpy as np

from agripipe.cleaner import AgriCleaner, CleanerConfig, CleanerDiagnostics


def test_cleaner_has_diagnostics_attribute():
    config = CleanerConfig(numeric_columns=["temp"])
    cleaner = AgriCleaner(config)
    assert isinstance(cleaner.diagnostics, CleanerDiagnostics)
    assert cleaner.diagnostics.total_rows == 0


def test_diagnostics_has_all_expected_fields():
    d = CleanerDiagnostics()
    expected_fields = {
        "total_rows", "imputation_strategy_used", "values_imputed",
        "outliers_removed", "out_of_bounds_removed",
        "nitrogen_violations", "peronospora_events",
        "irrigation_inefficient", "soil_organic_low",
        "heat_stress_flowering", "late_frost_events",
    }
    for f in expected_fields:
        assert hasattr(d, f), f"Missing field: {f}"


def _synth_violations_df() -> pd.DataFrame:
    """DataFrame costruito ad hoc per generare violazioni note."""
    return pd.DataFrame({
        "date": pd.date_range("2024-05-01", periods=5, freq="D"),
        "field_id": ["F1"] * 5,
        "crop_type": ["wine_grape_docg"] * 5,
        "temp": [11.0, 35.0, 20.0, 20.0, 20.0],   # riga 1 = Regola Tre 10 se pioggia>10
        "rainfall": [12.0, 0.0, 0.0, 0.0, 0.0],    # riga 0 = Peronospora
        "humidity": [30.0, 50.0, 50.0, 50.0, 50.0],
        "soil_moisture": [12.0, 50.0, 90.0, 50.0, 50.0],
        "irrigation": [0.0, 0.0, 10.0, 0.0, 0.0],  # riga 2 = irrigazione su suolo saturo
        "n": [15.0, 0.0, 0.0, 0.0, 0.0],           # riga 0 = azoto su suolo secco
        "organic_matter": [1.0, 2.0, 2.0, 2.0, 2.0],  # riga 0 = suolo povero
        "ph": [7.0] * 5,
        "yield": [5.0] * 5,
    })


def test_diagnostics_counts_peronospora():
    from agripipe.cleaner import AgriCleaner, CleanerConfig
    config = CleanerConfig(
        numeric_columns=["temp", "rainfall", "humidity", "soil_moisture",
                         "irrigation", "n", "organic_matter", "ph", "yield"],
        date_columns=["date"],
    )
    cleaner = AgriCleaner(config)
    cleaner.clean(_synth_violations_df())
    assert cleaner.diagnostics.peronospora_events >= 1
    assert cleaner.diagnostics.nitrogen_violations >= 1
    assert cleaner.diagnostics.irrigation_inefficient >= 1
    assert cleaner.diagnostics.soil_organic_low >= 1
