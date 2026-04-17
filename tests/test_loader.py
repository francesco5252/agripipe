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
