"""Test per batch_load_raw: caricamento di tutti gli Excel di una cartella."""

from pathlib import Path

import pandas as pd
import pytest

from agripipe.loader import batch_load_raw


def _make_valid_xlsx(path: Path, n_rows: int = 5) -> None:
    """Crea un Excel valido minimale per i test di batch loading."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n_rows, freq="D"),
            "field_id": [f"F{i}" for i in range(n_rows)],
            "temp": [20.0 + i for i in range(n_rows)],
            "humidity": [55.0 + i for i in range(n_rows)],
            "ph": [6.5 + i * 0.1 for i in range(n_rows)],
            "yield": [100.0 + i * 10 for i in range(n_rows)],
        }
    )
    df.to_excel(path, index=False)


def test_batch_load_raw_empty_dir_raises(tmp_path: Path):
    """Una cartella vuota deve sollevare ValueError con messaggio esplicito."""
    with pytest.raises(ValueError, match="Nessun file Excel/CSV trovato"):
        batch_load_raw(tmp_path)


def test_batch_load_raw_missing_dir_raises(tmp_path: Path):
    """Una cartella inesistente deve sollevare FileNotFoundError."""
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        batch_load_raw(missing)


def test_batch_load_raw_single_file(tmp_path: Path):
    """Con un solo file, restituisce lo stesso contenuto di load_raw più la colonna source_file."""
    p = tmp_path / "a.xlsx"
    _make_valid_xlsx(p)
    df = batch_load_raw(tmp_path)
    assert len(df) == 5
    assert "source_file" in df.columns
    assert (df["source_file"] == "a.xlsx").all()


def test_batch_load_raw_multiple_files(tmp_path: Path):
    """Con più file, concatena tutti preservando la provenienza."""
    for name in ["a.xlsx", "b.xlsx", "c.xlsx"]:
        _make_valid_xlsx(tmp_path / name, n_rows=3)
    df = batch_load_raw(tmp_path)
    assert len(df) == 9
    assert set(df["source_file"].unique()) == {"a.xlsx", "b.xlsx", "c.xlsx"}


def test_batch_load_raw_skips_non_data_files(tmp_path: Path):
    """Ignora file che non sono Excel/CSV (es. .txt, .md)."""
    _make_valid_xlsx(tmp_path / "good.xlsx")
    (tmp_path / "readme.txt").write_text("not a data file")
    (tmp_path / "notes.md").write_text("# notes")
    df = batch_load_raw(tmp_path)
    assert len(df) == 5
    assert (df["source_file"] == "good.xlsx").all()


def test_batch_load_raw_continues_on_single_file_error(tmp_path: Path, caplog):
    """Se un file è malformato, salta quello e continua con gli altri loggando un warning."""
    _make_valid_xlsx(tmp_path / "good.xlsx")
    # File con schema rotto: mancano colonne obbligatorie
    pd.DataFrame({"foo": [1, 2]}).to_excel(tmp_path / "bad.xlsx", index=False)
    df = batch_load_raw(tmp_path, on_error="skip")
    assert len(df) == 5
    assert (df["source_file"] == "good.xlsx").all()
    assert any("bad.xlsx" in r.message for r in caplog.records)


def test_batch_load_raw_raises_on_error_default(tmp_path: Path):
    """Comportamento default: se un file è rotto, solleva l'errore."""
    _make_valid_xlsx(tmp_path / "good.xlsx")
    pd.DataFrame({"foo": [1, 2]}).to_excel(tmp_path / "bad.xlsx", index=False)
    with pytest.raises(ValueError):
        batch_load_raw(tmp_path)  # default: on_error="raise"
