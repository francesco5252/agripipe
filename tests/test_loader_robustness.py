"""Test della robustezza del Loader (Fase 1)."""

import pandas as pd
import pytest
from pathlib import Path
from pydantic import BaseModel
from agripipe.loader import load_raw, RawSchema

def test_loader_skips_garbage_rows(tmp_path: Path):
    # Creiamo un Excel con "spazzatura" nelle prime righe
    xlsx_path = tmp_path / "garbage.xlsx"
    data = [
        ["RELAZIONE TECNICA CAMPO A", None, None],
        ["Data creazione: 2024-04-18", None, None],
        [None, None, None], # Riga vuota
        ["date", "field_id", "temp", "humidity", "ph", "yield"], # Header vero (riga 4)
        ["2024-01-01", "F1", 20.5, 60, 7.0, 5.0],
    ]
    df_build = pd.DataFrame(data)
    df_build.to_excel(xlsx_path, index=False, header=False)
    
    df = load_raw(xlsx_path)
    assert len(df) == 1
    assert "temp" in df.columns
    assert df["temp"].iloc[0] == 20.5

def test_loader_fuzzy_mapping(tmp_path: Path):
    xlsx_path = tmp_path / "fuzzy.xlsx"
    # L'utente usa nomi "umani" invece di quelli tecnici
    data = {
        "Data": ["2024-01-01"],
        "ID Campo": ["F1"],
        "T Media": [25.0],
        "Umidità (%)": [55],
        "Acidità (pH)": [6.5],
        "Resa (t/ha)": [12.0]
    }
    pd.DataFrame(data).to_excel(xlsx_path, index=False)
    
    df = load_raw(xlsx_path)
    # Deve mappare su nomi standard
    assert "date" in df.columns
    assert "temp" in df.columns
    assert "yield" in df.columns
    assert df["temp"].iloc[0] == 25.0

def test_loader_robust_csv_semicolon(tmp_path: Path):
    # CSV stile italiano con ;
    csv_path = tmp_path / "it.csv"
    content = "Data;Campo;Temperatura;Umidità;pH;Resa\n2024-04-18;F1;22.5;50;7.0;10.0\n"
    csv_path.write_text(content)
    
    df = load_raw(csv_path)
    assert df.shape[1] == 6
    assert "temp" in df.columns
    assert df["temp"].iloc[0] == 22.5

def test_loader_unit_conversion_fahrenheit(tmp_path: Path):
    # Dataset con temperatura in Fahrenheit
    xlsx_path = tmp_path / "usa_units.xlsx"
    data = {
        "date": ["2024-01-01"],
        "field_id": ["F1"],
        "temp (F)": [68.0], # 68F = 20C
        "humidity": [50],
        "ph": [7.0],
        "yield": [5.0]
    }
    pd.DataFrame(data).to_excel(xlsx_path, index=False)
    
    df = load_raw(xlsx_path)
    # Deve aver convertito 68 in 20
    assert df["temp"].iloc[0] == pytest.approx(20.0)

def test_loader_multi_sheet_merge(tmp_path: Path):
    # Creiamo un Excel con due fogli identici
    xlsx_path = tmp_path / "multi.xlsx"
    df1 = pd.DataFrame({
        "date": ["2024-01-01"], "field_id": ["F1"], "temp": [20], 
        "humidity": [50], "ph": [7], "yield": [5]
    })
    df2 = pd.DataFrame({
        "date": ["2024-01-02"], "field_id": ["F1"], "temp": [21], 
        "humidity": [55], "ph": [7], "yield": [6]
    })
    
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df1.to_excel(writer, sheet_name="Gennaio", index=False)
        df2.to_excel(writer, sheet_name="Febbraio", index=False)
    
    # Carichiamo con sheet_name=None per attivare il merge
    df = load_raw(xlsx_path, sheet_name=None)
    
    assert len(df) == 2
    assert "temp" in df.columns
    assert sorted(df["temp"].tolist()) == [20.0, 21.0]

def test_loader_filename_injection(tmp_path: Path):
    # File senza colonna field_id
    xlsx_path = tmp_path / "Campo-Nord_2024.xlsx"
    data = {
        "date": ["2024-01-01"],
        "temp": [20.0], "humidity": [50], "ph": [7.0], "yield": [5.0]
    }
    pd.DataFrame(data).to_excel(xlsx_path, index=False)
    
    df = load_raw(xlsx_path)
    # Deve aver iniettato 'Campo-Nord' o 'Campo' come field_id
    assert "field_id" in df.columns
    assert df["field_id"].iloc[0] == "Campo"
