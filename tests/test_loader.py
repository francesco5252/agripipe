from pathlib import Path

import pandas as pd
import pytest

from agripipe.loader import load_raw


def test_load_raw_missing_file():
    with pytest.raises(FileNotFoundError):
        load_raw("does_not_exist.xlsx")


def test_load_raw_reads_xlsx(tmp_path: Path, dirty_df: pd.DataFrame):
    path = tmp_path / "sample.xlsx"
    dirty_df.to_excel(path, index=False)
    df = load_raw(path)
    assert len(df) == len(dirty_df)
    assert "ph" in df.columns


def test_load_raw_missing_columns(tmp_path: Path):
    pd.DataFrame({"foo": [1, 2]}).to_excel(tmp_path / "bad.xlsx", index=False)
    with pytest.raises(ValueError, match="Colonne mancanti"):
        load_raw(tmp_path / "bad.xlsx")


def test_load_raw_fuzzy_matching(tmp_path: Path):
    """Verifica che il fuzzy matching riconosca colonne non-canoniche."""
    p = tmp_path / "fuzzy.xlsx"
    df = pd.DataFrame(
        {
            "Data": pd.date_range("2025-01-01", periods=3),
            "Campo": ["A", "A", "A"],
            "Temperatura": [20.0, 21.0, 22.0],
            "Umidita": [50, 51, 52],
            "pH_suolo": [6.5, 6.6, 6.7],
            "Resa": [100, 110, 120],
        }
    )
    df.to_excel(p, index=False)

    # Senza fuzzy deve fallire
    with pytest.raises(ValueError, match="Colonne mancanti"):
        load_raw(p, fuzzy=False)

    # Con fuzzy deve passare
    df_loaded = load_raw(p, fuzzy=True)
    assert "temp" in df_loaded.columns
    assert "humidity" in df_loaded.columns
    assert "yield" in df_loaded.columns
    assert len(df_loaded) == 3
