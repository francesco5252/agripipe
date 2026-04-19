"""Test della Fase 3 — Tensorizer: one-hot, metadata bundle, schema hash."""

import pandas as pd

from agripipe.tensorizer import Tensorizer


def test_onehot_encoding_strategy():
    df = pd.DataFrame(
        {
            "crop": ["Mais", "Vite", "Mais"],
            "temp": [20.0, 25.0, 22.0],
            "yield": [100.0, 200.0, 110.0],
        }
    )

    t = Tensorizer(
        numeric_columns=["temp"],
        categorical_columns=["crop"],
        categorical_strategy="onehot",
    )
    bundle = t.fit_transform(df)

    assert "crop_Mais" in t.feature_names
    assert "crop_Vite" in t.feature_names
    # 1 numeric + 2 one-hot categories = 3 features totali
    assert bundle.features.shape[1] == 3


def test_bundle_metadata_injection():
    df = pd.DataFrame({"temp": [20.0, 25.0], "yield": [5.0, 6.0]})
    df.attrs["file_hash"] = "abc-123-xyz"

    t = Tensorizer(numeric_columns=["temp"], categorical_columns=[])
    bundle = t.fit_transform(df)

    assert bundle.metadata["file_hash"] == "abc-123-xyz"
    assert "scaler_params" in bundle.metadata
    # Nel caso standard scaler, i parametri contengono mean e scale
    assert bundle.metadata["scaler_params"]["type"] == "standard"
    assert "mean" in bundle.metadata["scaler_params"]
    assert "scale" in bundle.metadata["scaler_params"]
