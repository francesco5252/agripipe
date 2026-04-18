"""Test finali per il livello Data Architect (Split, Robust Scaling)."""

import pandas as pd
import numpy as np
import torch
import pytest
from agripipe.tensorizer import Tensorizer
from agripipe.export import export_ml_bundle
from agripipe.cleaner import AgriCleaner, CleanerConfig

def test_stratified_split_ratios():
    # Dataset di 100 righe
    df = pd.DataFrame({
        "temp": np.random.normal(20, 5, 100),
        "yield": np.random.normal(50, 10, 100),
        "date": pd.date_range("2024-01-01", periods=100)
    })
    
    t = Tensorizer(numeric_columns=["temp"], categorical_columns=[], target="yield")
    # Split 70/20/10
    bundle = t.fit_transform(df, split_ratios=(0.7, 0.2, 0.1))
    
    assert len(bundle.train_indices) == 70
    assert len(bundle.val_indices) == 20
    assert len(bundle.test_indices) == 10

def test_robust_scaling_strategy():
    # Dataset con un outlier enorme che "fregerebbe" lo StandardScaler
    df = pd.DataFrame({
        "temp": [20.0, 21.0, 22.0, 20.5, 1000.0] # 1000 è un outlier
    })
    
    # Standard: la media sarà altissima, i valori normali saranno schiacciati vicino a 0
    t_std = Tensorizer(numeric_columns=["temp"], categorical_columns=[], scaling_strategy="standard")
    b_std = t_std.fit_transform(df)
    
    # Robust: usa la mediana, i valori normali rimarranno ben spaziati
    t_rob = Tensorizer(numeric_columns=["temp"], categorical_columns=[], scaling_strategy="robust")
    b_rob = t_rob.fit_transform(df)
    
    # Nel robust scaling, la mediana (valore centrale) deve essere 0
    # I valori 20, 21, 22 sono vicini alla mediana
    assert b_rob.features[2, 0] != b_std.features[2, 0]

def test_full_bundle_export_with_splits(tmp_path):
    df = pd.DataFrame({
        "temp": np.random.normal(20, 5, 100),
        "yield": np.random.normal(50, 10, 100),
        "date": pd.date_range("2024-01-01", periods=100)
    })
    cleaner = AgriCleaner(CleanerConfig())
    
    paths = export_ml_bundle(
        df, cleaner, {}, tmp_path, 
        name="architect_test", 
        split_ratios=(0.8, 0.1, 0.1)
    )
    
    assert (tmp_path / "architect_test_train.pt").exists()
    assert (tmp_path / "architect_test_val.pt").exists()
    assert (tmp_path / "architect_test_test.pt").exists()
    assert (tmp_path / "architect_test.json").exists()

def test_float16_precision_optimization():
    df = pd.DataFrame({"temp": [20.0, 25.0], "yield": [5, 6]})
    # Richiediamo mezza precisione
    t = Tensorizer(numeric_columns=["temp"], categorical_columns=[], precision="float16")
    bundle = t.fit_transform(df)
    
    assert bundle.features.dtype == torch.float16
    assert bundle.metadata["precision"] == "float16"

def test_schema_lock_hash_consistency():
    df1 = pd.DataFrame({"temp": [20], "ph": [7]})
    df2 = pd.DataFrame({"ph": [7.5], "temp": [21]}) # Stesse colonne, ordine diverso
    df3 = pd.DataFrame({"temp": [20], "rain": [5]}) # Colonna diversa
    
    t1 = Tensorizer(numeric_columns=["temp", "ph"], categorical_columns=[])
    b1 = t1.fit_transform(df1)
    
    t2 = Tensorizer(numeric_columns=["temp", "ph"], categorical_columns=[])
    b2 = t2.fit_transform(df2)
    
    t3 = Tensorizer(numeric_columns=["temp", "rain"], categorical_columns=[])
    b3 = t3.fit_transform(df3)
    
    # b1 e b2 devono avere lo stesso schema_hash (stesse colonne)
    assert b1.metadata["schema_lock_hash"] == b2.metadata["schema_lock_hash"]
    # b3 deve essere diverso
    assert b1.metadata["schema_lock_hash"] != b3.metadata["schema_lock_hash"]
