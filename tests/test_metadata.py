"""Test della costruzione e serializzazione del metadata.json."""

import json
from pathlib import Path

import pandas as pd
import torch

from agripipe.cleaner import AgriCleaner, CleanerConfig
from agripipe.dataset import AgriDataset
from agripipe.metadata import build_metadata, save_metadata_json


def _prepare_dataset():
    df = pd.DataFrame({
        "temp": [20.0, 22.0, 24.0],
        "humidity": [60.0, 65.0, 70.0],
        "yield": [5.0, 6.0, 7.0],
    })
    ds = AgriDataset(df=df, numeric_columns=["temp", "humidity"], target="yield")
    return df, ds


def test_build_metadata_has_schema_version_and_timestamp():
    df, ds = _prepare_dataset()
    preset = {"region": "Puglia", "crop": "olive", "zona": "Salento"}
    cleaner_diag_dict = {"values_imputed": 0, "outliers_removed": 0}
    meta = build_metadata(ds, preset, cleaner_diag_dict, target="yield")
    assert meta["schema_version"] == 1
    assert "generated_at" in meta
    assert meta["dataset_info"]["target"] == "yield"


def test_build_metadata_describes_all_columns():
    df, ds = _prepare_dataset()
    preset = {"region": "Puglia", "crop": "olive", "zona": "Salento"}
    meta = build_metadata(ds, preset, {}, target="yield")
    col_names = [c["name"] for c in meta["columns"]]
    assert "temp" in col_names
    assert "humidity" in col_names
    for col in meta["columns"]:
        assert "index" in col
        assert "description" in col
        assert "normalized" in col


def test_save_metadata_json_writes_valid_json(tmp_path: Path):
    df, ds = _prepare_dataset()
    preset = {"region": "Puglia", "crop": "olive", "zona": "Salento"}
    meta = build_metadata(ds, preset, {}, target="yield")
    out = tmp_path / "metadata.json"
    save_metadata_json(meta, out)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 1


def test_metadata_includes_pipeline_context():
    df, ds = _prepare_dataset()
    preset = {"region": "Puglia", "crop": "olive", "crop_display": "Olivo DOP"}
    meta = build_metadata(ds, preset, {}, target="yield")
    assert meta["pipeline_context"]["preset_applied"] == "Olivo DOP"
    assert meta["pipeline_context"]["region"] == "Puglia"


def test_metadata_has_pytorch_example_code():
    df, ds = _prepare_dataset()
    meta = build_metadata(ds, {}, {}, target="yield")
    assert "example_code" in meta["pytorch_usage"]
    assert "torch.load" in meta["pytorch_usage"]["example_code"]
