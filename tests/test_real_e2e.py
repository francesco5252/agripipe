"""Test E2E su dataset reale con nomi colonna non-canonici [#9]."""

from pathlib import Path

import pandas as pd
import pytest

from agripipe.loader import load_raw


def test_loader_fails_on_non_canonical_columns(tmp_path: Path):
    """Verifica che il loader sollevi ValueError su nomi non-standard senza fuzzy matching."""
    p = tmp_path / "real_agri_data_raw.xlsx"
    # Dati con nomi italiani/sporchi comuni in Excel agronomici reali
    df = pd.DataFrame(
        {
            "Data": ["2024-05-10", "2024-05-11"],
            "ID Campo": ["Vigna_1", "Vigna_1"],
            "Temp_Aria_C": [22.5, 23.1],
            "Umidita_Relativa": [65.0, 68.0],
            "pH_Estratto": [7.2, 7.3],
            "Resa_Stima_kg": [12000, 12500],
        }
    )
    df.to_excel(p, index=False)

    with pytest.raises(ValueError, match="Colonne mancanti"):
        load_raw(p, fuzzy=False)


def test_loader_succeeds_on_non_canonical_with_fuzzy(tmp_path: Path):
    """Verifica che con fuzzy=True il loader riconosca le colonne dell'Excel reale."""
    p = tmp_path / "real_agri_data_fuzzy.xlsx"
    df = pd.DataFrame(
        {
            "Data": ["2024-05-10", "2024-05-11"],
            "ID Campo": ["Vigna_1", "Vigna_1"],
            "Temp_Aria_C": [22.5, 23.1],
            "Umidita_Relativa": [65.0, 68.0],
            "pH_Estratto": [7.2, 7.3],
            "Resa_Stima_kg": [12000, 12500],
        }
    )
    df.to_excel(p, index=False)

    # Questo test dovrebbe passare ora che il fuzzy matching è attivo
    df_loaded = load_raw(p, fuzzy=True)
    assert "temp" in df_loaded.columns
    assert "humidity" in df_loaded.columns
    assert "ph" in df_loaded.columns
    assert "yield" in df_loaded.columns
    assert len(df_loaded) == 2
