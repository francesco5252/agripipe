from pathlib import Path

import pandas as pd

from agripipe.cleaner import AgriCleaner
from agripipe.report import generate_report
from agripipe.synth import SynthConfig, generate_dirty_excel


def test_generate_dirty_excel(tmp_path: Path):
    out = tmp_path / "dirty.xlsx"
    cfg = SynthConfig(n_rows=100, n_fields=3, seed=1)
    generate_dirty_excel(out, cfg)
    assert out.exists()
    df = pd.read_excel(out)
    # Ha le colonne attese
    for col in ["date", "field_id", "temp", "humidity", "ph", "yield"]:
        assert col in df.columns
    # Ha iniettato almeno qualche NaN
    assert df.isna().sum().sum() > 0


def test_generate_report(tmp_path: Path, dirty_df: pd.DataFrame, cleaner_config):
    df_clean = AgriCleaner(cleaner_config).clean(dirty_df)
    out = tmp_path / "report.html"
    generate_report(dirty_df, df_clean, out)
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "AgriPipe" in html
    assert "<table" in html
    assert str(len(dirty_df)) in html
