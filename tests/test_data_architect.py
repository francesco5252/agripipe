"""Test del livello Data Architect: split train/val/test, robust scaling, schema hash."""

import numpy as np
import pandas as pd

from agripipe.cleaner import AgriCleaner, CleanerConfig
from agripipe.export import export_ml_bundle
from agripipe.tensorizer import Tensorizer


def test_stratified_split_ratios():
    df = pd.DataFrame(
        {
            "temp": np.random.normal(20, 5, 100),
            "yield": np.random.normal(50, 10, 100),
            "date": pd.date_range("2024-01-01", periods=100),
        }
    )

    t = Tensorizer(numeric_columns=["temp"], categorical_columns=[], target="yield")
    bundle = t.fit_transform(df, split_ratios=(0.7, 0.2, 0.1))

    assert len(bundle.train_indices) == 70
    assert len(bundle.val_indices) == 20
    assert len(bundle.test_indices) == 10


def test_robust_scaling_strategy():
    df = pd.DataFrame({"temp": [20.0, 21.0, 22.0, 20.5, 1000.0]})

    t_std = Tensorizer(
        numeric_columns=["temp"], categorical_columns=[], scaling_strategy="standard"
    )
    b_std = t_std.fit_transform(df)

    t_rob = Tensorizer(numeric_columns=["temp"], categorical_columns=[], scaling_strategy="robust")
    b_rob = t_rob.fit_transform(df)

    # Confronto sul valore centrale "normale" (22.0): robust e standard devono divergere
    assert b_rob.features[2, 0] != b_std.features[2, 0]


def test_full_bundle_export_with_splits(tmp_path):
    df = pd.DataFrame(
        {
            "temp": np.random.normal(20, 5, 100),
            "yield": np.random.normal(50, 10, 100),
            "date": pd.date_range("2024-01-01", periods=100),
        }
    )
    cleaner = AgriCleaner(CleanerConfig())
    df_clean = cleaner.clean(df)

    paths = export_ml_bundle(
        df_clean,
        cleaner,
        {},
        tmp_path,
        name="architect_test",
        split_ratios=(0.8, 0.1, 0.1),
    )

    assert (tmp_path / "architect_test_train.pt").exists()
    assert (tmp_path / "architect_test_val.pt").exists()
    assert (tmp_path / "architect_test_test.pt").exists()
    assert (tmp_path / "architect_test.json").exists()
    assert paths["zip"].exists()


def test_schema_lock_hash_consistency():
    # Stesse colonne in ordine diverso devono produrre lo stesso hash.
    df1 = pd.DataFrame({"temp": [20.0, 21.0], "ph": [7.0, 7.1]})
    df2 = pd.DataFrame({"ph": [7.5, 7.6], "temp": [21.0, 22.0]})
    df3 = pd.DataFrame({"temp": [20.0, 21.0], "rain": [5.0, 6.0]})

    t1 = Tensorizer(numeric_columns=["temp", "ph"], categorical_columns=[])
    b1 = t1.fit_transform(df1)

    t2 = Tensorizer(numeric_columns=["temp", "ph"], categorical_columns=[])
    b2 = t2.fit_transform(df2)

    t3 = Tensorizer(numeric_columns=["temp", "rain"], categorical_columns=[])
    b3 = t3.fit_transform(df3)

    assert b1.metadata["schema_lock_hash"] == b2.metadata["schema_lock_hash"]
    assert b1.metadata["schema_lock_hash"] != b3.metadata["schema_lock_hash"]
