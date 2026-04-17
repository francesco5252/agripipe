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
