"""Test del bundling ML-ready: .pt + metadata.json + .zip."""

import json
import zipfile
from pathlib import Path

import pandas as pd
import torch

from agripipe.cleaner import AgriCleaner, CleanerConfig
from agripipe.export import export_ml_bundle


def _clean_df_and_cleaner():
    df = pd.DataFrame(
        {
            "temp": [20.0, 22.0, 24.0, 26.0, 28.0],
            "humidity": [60.0, 65.0, 70.0, 75.0, 80.0],
            "yield": [5.0, 6.0, 7.0, 8.0, 9.0],
        }
    )
    config = CleanerConfig(
        numeric_columns=["temp", "humidity", "yield"],
        outlier_method="none",
        missing_strategy="median",
    )
    cleaner = AgriCleaner(config)
    df_clean = cleaner.clean(df)
    return df_clean, cleaner


def test_export_creates_three_files(tmp_path: Path):
    df, cleaner = _clean_df_and_cleaner()
    preset = {"region": "Puglia", "crop": "olive", "zona": "Salento"}
    paths = export_ml_bundle(df, cleaner, preset, tmp_path, name="test")
    assert paths["pt"].exists()
    assert paths["json"].exists()
    assert paths["zip"].exists()


def test_exported_pt_contains_all_bundle_keys(tmp_path: Path):
    df, cleaner = _clean_df_and_cleaner()
    paths = export_ml_bundle(df, cleaner, {}, tmp_path, name="test")
    bundle = torch.load(paths["pt"], weights_only=False)
    assert "features" in bundle
    assert "feature_names" in bundle
    assert "scaler_mean" in bundle
    assert "scaler_scale" in bundle
    assert isinstance(bundle["features"], torch.Tensor)


def test_exported_zip_contains_pt_and_json(tmp_path: Path):
    df, cleaner = _clean_df_and_cleaner()
    paths = export_ml_bundle(df, cleaner, {}, tmp_path, name="test")
    with zipfile.ZipFile(paths["zip"]) as z:
        names = z.namelist()
    assert any(n.endswith(".pt") for n in names)
    assert any(n.endswith(".json") for n in names)


def test_exported_metadata_is_valid_json(tmp_path: Path):
    df, cleaner = _clean_df_and_cleaner()
    paths = export_ml_bundle(df, cleaner, {"region": "X", "crop": "y"}, tmp_path, name="t")
    meta = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert meta["schema_version"] == 1
    assert meta["dataset_info"]["rows"] == 5
