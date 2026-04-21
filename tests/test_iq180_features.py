import pandas as pd
from agripipe.cleaner import AgriCleaner


def test_gdd_and_soft_cleaning():
    """Test delle feature '180 IQ': GDD e Confidence Score."""

    # Dataset di una settimana per il Mais (t_base=10.0)
    data = {
        "date": pd.date_range("2025-05-01", periods=5),
        "temp": [15.0, 20.0, 10.0, 25.0, 5.0],  # GDD giornalieri: 5, 10, 0, 15, 0
        "yield": [0, 0, 0, 0, 50.0],  # Resa assurda al 5 Maggio (fuori stagione)
        "field_id": ["F1"] * 5,
    }
    df = pd.DataFrame(data)

    # Configurazione con Soft Cleaning e GDD
    # Usiamo il preset mais_intensivo_padano
    cleaner = AgriCleaner.from_preset("mais_intensivo_padano")
    cleaner.config.soft_cleaning = True
    cleaner.config.calculate_gdd = True

    out = cleaner.clean(df)

    # 1. Verifica GDD
    # Somma GDD attesa: 5 + 10 + 0 + 15 + 0 = 30
    assert "gdd_accumulated" in out.columns
    final_gdd = out.iloc[-1]["gdd_accumulated"]
    assert final_gdd == 30.0
    print(f"✅ GDD calcolati correttamente: {final_gdd}")

    # 2. Verifica Soft Cleaning
    # La resa a Maggio (indice 4) deve essere penalizzata nella confidence
    assert "confidence" in out.columns
    conf_bad = out.iloc[-1]["confidence"]
    assert conf_bad < 1.0
    # Il valore di yield deve essere ancora lì (non NaN)
    assert out.iloc[-1]["yield"] == 50.0
    print(f"✅ Soft Cleaning applicato: resa mantenuta con confidence {conf_bad}")


if __name__ == "__main__":
    test_gdd_and_soft_cleaning()
