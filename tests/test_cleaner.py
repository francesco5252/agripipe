import pandas as pd

from agripipe.cleaner import AgriCleaner, CleanerConfig


def test_clean_removes_physical_bounds_violations(dirty_df: pd.DataFrame, cleaner_config):
    cleaner = AgriCleaner(cleaner_config)
    out = cleaner.clean(dirty_df)
    # pH non deve contenere il 99.0 iniettato
    assert out["ph"].max() <= 14.0


def test_clean_handles_outliers(dirty_df: pd.DataFrame, cleaner_config):
    cleaner = AgriCleaner(cleaner_config)
    out = cleaner.clean(dirty_df)
    # temp 500.0 deve essere stato rimosso come outlier → imputato con mediana
    assert out["temp"].max() < 100.0


def test_clean_no_nan_after_imputation(dirty_df: pd.DataFrame, cleaner_config):
    cleaner = AgriCleaner(cleaner_config)
    out = cleaner.clean(dirty_df)
    for col in cleaner_config.numeric_columns:
        assert out[col].isna().sum() == 0, f"Colonna {col} ha ancora NaN"


def test_clean_deduplicates(dirty_df: pd.DataFrame, cleaner_config):
    cleaner = AgriCleaner(cleaner_config)
    out = cleaner.clean(dirty_df)
    assert not out.duplicated(subset=cleaner_config.dedup_keys).any()


def test_from_yaml(tmp_path):
    import yaml

    cfg = {
        "numeric_columns": ["temp"],
        "categorical_columns": [],
        "date_columns": [],
        "dedup_keys": [],
        "missing_strategy": "mean",
        "outlier_method": "none",
        "physical_bounds": {},
    }
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    cleaner = AgriCleaner.from_yaml(p)
    assert isinstance(cleaner.config, CleanerConfig)


def test_from_preset_loads_correct_bounds():
    # Carichiamo il preset Barolo (vite_nebbiolo_barolo)
    cleaner = AgriCleaner.from_preset("vite_nebbiolo_barolo")
    # Temp range nel YAML è [-15, 38]
    assert cleaner.config.physical_bounds["temp"] == (-15.0, 38.0)
    # pH range nel YAML è [7.2, 8.2]
    assert cleaner.config.physical_bounds["ph"] == (7.2, 8.2)
    assert cleaner.diagnostics.current_preset_name == "vite_nebbiolo_barolo"


def test_from_preset_raises_on_missing():
    import pytest

    with pytest.raises(ValueError, match="non trovato"):
        AgriCleaner.from_preset("preset_inesistente_123")


def test_clean_with_preset_discovers_columns():
    df = pd.DataFrame(
        {
            "temp": [25.0, 100.0, 26.0],  # 100.0 è fuori range per ulivo_taggiasco_dop
            "ph": [7.0, 7.2, 7.5],
            "crop_type": ["olive", "olive", "olive"],
        }
    )
    # ulivo_taggiasco_dop temp_range: [-5, 38]
    cleaner = AgriCleaner.from_preset("ulivo_taggiasco_dop")
    out = cleaner.clean(df)

    # Il 100.0 deve essere stato rimosso (NaN) e poi imputato (mediana ~25.5)
    assert out["temp"].max() <= 38.0
    assert "temp" in cleaner.config.numeric_columns
    assert "ph" in cleaner.config.numeric_columns


def test_preset_pomodoro_pachino():
    """Issue #8 — Verifica che il preset pomodoro_pachino_igp si carichi e applichi correttamente."""
    cleaner = AgriCleaner.from_preset("pomodoro_pachino_igp")

    # Bounds fisici devono riflettere la letteratura agronomica
    assert cleaner.config.physical_bounds["temp"] == (10.0, 45.0)
    assert cleaner.config.physical_bounds["ph"] == (6.5, 8.0)
    assert cleaner.diagnostics.current_preset_name == "pomodoro_pachino_igp"

    # Applicazione: un valore fuori range deve diventare NaN e poi essere imputato
    df = pd.DataFrame(
        {
            "temp": [25.0, 50.0, 28.0],  # 50.0 è fuori range [10, 45]
            "ph": [6.5, 7.0, 9.0],  # 9.0 è fuori range [6.5, 8.0]
        }
    )
    out = cleaner.clean(df)

    assert out["temp"].max() <= 45.0, "temp fuori range non rimossa"
    assert out["ph"].max() <= 8.0, "pH fuori range non rimossa"


def test_agronomic_rules_ulivo_coratina():
    """Verifica che max_yield sia caricato e applicato."""
    cleaner = AgriCleaner.from_preset("ulivo_coratina_bari")

    # Verifica caricamento config (Coratina ha max_yield=10.0)
    assert cleaner.config.max_yield == 10.0

    # Input con valori fuori dai limiti agronomici
    df = pd.DataFrame(
        {
            "yield": [8.0, 50.0, 9.0],  # 50.0 è assurdo (max 10)
            "temp": [25.0, 26.0, 27.0],
            "ph": [7.5, 7.6, 7.7],
            "field_id": ["A", "A", "A"],
            "date": ["2025-11-01", "2025-11-02", "2025-11-03"],
        }
    )

    out = cleaner.clean(df)

    # Il valore fuori limite (50.0) deve essere stato rimosso e imputato
    assert out["yield"].max() <= 10.0
    assert cleaner.diagnostics.agronomic_outliers_removed >= 1


def test_clean_with_auto_unit_conversion():
    """Verifica che il Cleaner chiami correttamente detect_and_convert_units."""
    import pytest

    config = CleanerConfig(auto_unit_conversion=True, numeric_columns=["temp"])
    cleaner = AgriCleaner(config)

    # Input con colonna fahrenheit
    df = pd.DataFrame(
        {"temp_f": [32.0, 68.0], "field_id": ["A", "A"], "date": ["2025-01-01", "2025-01-02"]}
    )
    df_clean = cleaner.clean(df)

    # temp_f deve essere sparito, temp deve esistere ed essere in Celsius
    assert "temp" in df_clean.columns
    assert "temp_f" not in df_clean.columns
    assert df_clean["temp"].iloc[0] == pytest.approx(0.0)
    assert df_clean["temp"].iloc[1] == pytest.approx(20.0, abs=0.1)
    assert "temp_f" in cleaner.diagnostics.unit_conversions
