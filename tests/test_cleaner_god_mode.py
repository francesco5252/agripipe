"""Test per le funzionalità God-Mode del Cleaner (KNN, Isolation Forest, Logic Checks)."""

import numpy as np
import pandas as pd
import pytest
from agripipe.cleaner import AgriCleaner, CleanerConfig

def test_knn_imputation_reconstructs_correlations():
    # Dataset più grande per stabilizzare KNN
    temp = np.linspace(20, 30, 20)
    yield_val = 2 * temp + np.random.normal(0, 0.1, 20)
    
    df = pd.DataFrame({"temp": temp, "yield": yield_val})
    df.loc[10, "yield"] = np.nan # Rimuoviamo un valore centrale
    
    config = CleanerConfig(
        numeric_columns=["temp", "yield"],
        missing_strategy="knn"
    )
    cleaner = AgriCleaner(config)
    df_clean = cleaner.clean(df)
    
    val_imputed = df_clean.loc[10, "yield"]
    expected = 2 * df_clean.loc[10, "temp"]
    assert val_imputed == pytest.approx(expected, rel=0.05)

def test_isolation_forest_detects_multivariate_outliers():
    # Dataset più grande (Isolation Forest richiede massa critica)
    rng = np.random.default_rng(42)
    temp = rng.normal(25, 2, 50)
    rain = rng.normal(5, 1, 50)
    
    df = pd.DataFrame({"temp": temp, "rainfall": rain})
    
    # Inseriamo un outlier "impossibile" multidimensionalmente
    # (idx 50): Temp bassissima e Pioggia altissima (0.0, 500.0)
    df.loc[50] = [0.0, 500.0] 
    
    config = CleanerConfig(
        numeric_columns=["temp", "rainfall"],
        outlier_method="ml",
        missing_strategy="median" # Per riempire il buco lasciato dall'outlier
    )
    cleaner = AgriCleaner(config)
    df_clean = cleaner.clean(df)
    
    # L'outlier deve essere stato rilevato
    assert cleaner.diagnostics.outliers_removed >= 1
    # Il valore finale a idx 50 non deve essere 0.0 (perché è stato rimpiazzato dalla mediana ~25.0)
    assert df_clean.loc[50, "temp"] > 10.0 

def test_logical_consistency_check():
    # Pioggia alta ma umidità suolo bassissima
    df = pd.DataFrame({
        "rainfall": [25.0, 0.0, 0.0],
        "soil_moisture": [10.0, 50.0, 50.0], 
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "field_id": ["F1", "F1", "F1"]
    })
    config = CleanerConfig(
        numeric_columns=["rainfall", "soil_moisture"],
        missing_strategy="median"
    )
    cleaner = AgriCleaner(config)
    df_clean = cleaner.clean(df)
    
    # L'incoerenza deve essere stata rilevata
    assert cleaner.diagnostics.inconsistent_rows > 0
    # La pioggia a riga 0 deve essere stata rimpiazzata (originale 25.0, mediana 0.0)
    assert df_clean.loc[0, "rainfall"] == 0.0

def test_seasonal_outlier_detection():
    # Creiamo un anno di dati con temperature normali (mediana ~20)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    temp = [20.0] * 100
    df = pd.DataFrame({"date": dates, "temp": temp, "field_id": ["F1"]*100})
    
    # Inseriamo un'anomalia stagionale: 35 gradi a GENNAIO
    df["temp"] = np.random.normal(20, 0.5, 100)
    df.loc[0, "temp"] = 35.0 
    
    config = CleanerConfig(numeric_columns=["temp"], enable_seasonal_checks=True)
    cleaner = AgriCleaner(config)
    df_clean = cleaner.clean(df)
    
    assert cleaner.diagnostics.seasonal_outliers >= 1
    assert df_clean.loc[0, "temp"] < 30.0

def test_peer_validation_across_fields():
    # 3 campi: due dicono 20 gradi, uno dice 40 gradi nella stessa data
    df = pd.DataFrame({
        "date": ["2024-01-01"] * 3,
        "field_id": ["F1", "F2", "F3"],
        "temp": [20.0, 21.0, 40.0], # F3 è un outlier rispetto ai suoi pari
        "humidity": [50, 51, 52],
        "ph": [7.0, 7.1, 7.2],
        "yield": [5, 5, 5]
    })
    config = CleanerConfig(enable_peer_validation=True, missing_strategy="median")
    cleaner = AgriCleaner(config)
    df_clean = cleaner.clean(df)
    
    # L'anomalia di F3 deve essere stata rilevata
    assert cleaner.diagnostics.peer_anomalies >= 1
    # Il valore di F3 non deve più essere 40.0 (rimpiazzato dalla mediana ~20.5)
    f3_val = df_clean[df_clean["field_id"]=="F3"]["temp"].iloc[0]
    assert f3_val == pytest.approx(20.5)

def test_target_bias_guard():
    # Dataset con resa (yield) identica per tutti: Bias estremo!
    df = pd.DataFrame({
        "yield": [10.0] * 20,
        "temp": np.random.normal(25, 2, 20),
        "date": pd.date_range("2024-01-01", periods=20),
        "field_id": ["F1"] * 20
    })
    cleaner = AgriCleaner(CleanerConfig())
    cleaner.clean(df)
    
    assert cleaner.diagnostics.target_bias_detected is True
