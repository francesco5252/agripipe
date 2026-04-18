"""Test finali per il livello Production-Master (Batch, Log-Transform, Redundancy)."""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from agripipe.loader import load_from_dir
from agripipe.cleaner import AgriCleaner, CleanerConfig
from agripipe.tensorizer import Tensorizer

def test_batch_loading_from_dir(tmp_path):
    # Creiamo due file nella stessa cartella
    d = tmp_path / "batch_data"
    d.mkdir()
    df1 = pd.DataFrame({"date":["2024-01-01"], "field_id":["F1"], "temp":[20], "humidity":[50], "ph":[7], "yield":[5]})
    df2 = pd.DataFrame({"date":["2024-01-02"], "field_id":["F1"], "temp":[21], "humidity":[51], "ph":[7], "yield":[6]})
    df1.to_excel(d / "file1.xlsx", index=False)
    df2.to_csv(d / "file2.csv", index=False)
    
    df_batch = load_from_dir(d)
    assert len(df_batch) == 2
    assert "temp" in df_batch.columns

def test_auto_log_transform_skewed_data():
    # Creiamo dati molto sbilanciati (Log-Normal distribution)
    rng = np.random.default_rng(42)
    skewed_data = rng.exponential(scale=10, size=100) 
    df = pd.DataFrame({
        "rainfall": skewed_data,
        "temp": rng.normal(25, 2, 100),
        "date": pd.date_range("2024-01-01", periods=100),
        "field_id": ["F1"]*100
    })
    
    config = CleanerConfig(numeric_columns=["rainfall", "temp"], auto_log_transform=True)
    cleaner = AgriCleaner(config)
    df_clean = cleaner.clean(df)
    
    # Lo skewness della pioggia deve essere diminuito dopo il log
    assert "rainfall" in cleaner.diagnostics.log_transformed_columns
    assert df_clean["rainfall"].max() < 10.0 # log(1+x) di grandi numeri è piccolo

def test_redundancy_filter_zero_variance():
    df = pd.DataFrame({
        "useful": [1, 2, 3, 4, 5],
        "useless_constant": [10, 10, 10, 10, 10], # Varianza zero
        "useless_duplicate": [1, 2, 3, 4, 5] # Identica a 'useful'
    })
    
    t = Tensorizer(
        numeric_columns=["useful", "useless_constant", "useless_duplicate"],
        categorical_columns=[],
        drop_redundant=True
    )
    bundle = t.fit_transform(df)
    
    # Solo 'useful' (più le date cicliche se presenti) deve sopravvivere
    assert "useless_constant" not in t.feature_names
    assert "useless_duplicate" not in t.feature_names
    assert "useful" in t.feature_names
