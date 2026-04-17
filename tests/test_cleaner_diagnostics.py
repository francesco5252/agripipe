"""Test che CleanerDiagnostics sia istanziabile e collegato a AgriCleaner."""

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
