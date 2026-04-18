"""Test che CleanerDiagnostics sia istanziabile e collegato a AgriCleaner."""

import pandas as pd

from agripipe.cleaner import AgriCleaner, CleanerConfig, CleanerDiagnostics


def test_cleaner_has_diagnostics_attribute():
    config = CleanerConfig(numeric_columns=["temp"])
    cleaner = AgriCleaner(config)
    assert isinstance(cleaner.diagnostics, CleanerDiagnostics)
    assert cleaner.diagnostics.total_rows == 0


def test_diagnostics_has_all_expected_fields():
    d = CleanerDiagnostics()
    expected_fields = {
        "total_rows",
        "imputation_strategy_used",
        "values_imputed",
        "outliers_removed",
        "out_of_bounds_removed",
        "nitrogen_violations",
        "peronospora_events",
        "irrigation_inefficient",
        "soil_organic_low",
        "heat_stress_flowering",
        "late_frost_events",
    }
    for f in expected_fields:
        assert hasattr(d, f), f"Missing field: {f}"


def _synth_violations_df() -> pd.DataFrame:
    """DataFrame costruito ad hoc per generare violazioni note.

    Distribuzione delle violazioni per riga (ogni regola su righe separate
    per evitare interazioni tra condizioni mutuamente esclusive):
    - riga 0: Peronospora (temp>10 + rain>10) + sostanza organica povera (<1.5)
    - riga 1: concimazione azotata su suolo secco senza pioggia
    - riga 2: irrigazione inefficiente su suolo saturo
    - righe 3-4: benigne (baseline, nessuna violazione)
    """
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-05-01", periods=5, freq="D"),
            "field_id": ["F1"] * 5,
            "crop_type": ["wine_grape_docg"] * 5,
            "temp": [11.0, 20.0, 20.0, 20.0, 20.0],  # riga 0 = Peronospora (>10)
            "rainfall": [
                12.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],  # riga 0 = Peronospora (>10); riga 1 = no pioggia
            "humidity": [50.0, 50.0, 50.0, 50.0, 50.0],  # no sensori "guasti"
            "soil_moisture": [50.0, 12.0, 90.0, 50.0, 50.0],  # r1 secco, r2 saturo
            "irrigation": [0.0, 0.0, 10.0, 0.0, 0.0],  # riga 2 = irrigazione inutile
            "n": [0.0, 15.0, 0.0, 0.0, 0.0],  # riga 1 = concimazione eccessiva
            "organic_matter": [1.0, 2.0, 2.0, 2.0, 2.0],  # riga 0 = suolo povero
            "ph": [7.0] * 5,
            "yield": [5.0] * 5,
        }
    )


def test_diagnostics_counts_peronospora():
    config = CleanerConfig(
        numeric_columns=[
            "temp",
            "rainfall",
            "humidity",
            "soil_moisture",
            "irrigation",
            "n",
            "organic_matter",
            "ph",
            "yield",
        ],
        date_columns=["date"],
    )
    cleaner = AgriCleaner(config)
    cleaner.clean(_synth_violations_df())
    assert cleaner.diagnostics.peronospora_events >= 1
    assert cleaner.diagnostics.nitrogen_violations >= 1
    assert cleaner.diagnostics.irrigation_inefficient >= 1
    assert cleaner.diagnostics.soil_organic_low >= 1
