"""Test per la Fase 3: Tensorizzazione avanzata (Sin/Cos, One-Hot)."""

import pandas as pd
import numpy as np
import torch
import pytest
from agripipe.tensorizer import Tensorizer

def test_cyclic_date_encoding():
    # Due date vicine (31 Dicembre e 1 Gennaio)
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-12-31", "2024-01-01"]),
        "field_id": ["F1", "F1"],
        "temp": [5.0, 6.0],
        "yield": [10, 11]
    })
    
    t = Tensorizer(numeric_columns=["temp"], categorical_columns=[], target="yield")
    bundle = t.fit_transform(df)
    
    assert "date_sin" in t.feature_names
    assert "date_cos" in t.feature_names
    
    # In un cerchio, 31/12 (fine) e 1/1 (inizio) hanno Coseno quasi identico (~1.0)
    idx_cos = t.feature_names.index("date_cos")
    cos_31_12 = bundle.features[0, idx_cos]
    cos_01_01 = bundle.features[1, idx_cos]
    assert cos_31_12 == pytest.approx(cos_01_01, abs=0.05)

def test_onehot_encoding_strategy():
    df = pd.DataFrame({
        "crop": ["Mais", "Vite", "Mais"],
        "temp": [20, 25, 22],
        "yield": [100, 200, 110]
    })
    
    t = Tensorizer(
        numeric_columns=["temp"], 
        categorical_columns=["crop"], 
        categorical_strategy="onehot"
    )
    bundle = t.fit_transform(df)
    
    assert "crop_Mais" in t.feature_names
    assert "crop_Vite" in t.feature_names
    assert bundle.features.shape[1] == 3

def test_bundle_metadata_injection():
    df = pd.DataFrame({"temp": [20, 25], "yield": [5, 6]})
    df.attrs["file_hash"] = "abc-123-xyz"
    
    t = Tensorizer(numeric_columns=["temp"], categorical_columns=[])
    bundle = t.fit_transform(df)
    
    assert bundle.metadata["file_hash"] == "abc-123-xyz"
    assert "scaler_params" in bundle.metadata
