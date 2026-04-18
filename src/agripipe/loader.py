"""Caricamento Production-Master: Batch Loading, Fingerprinting e Interoperabilità."""

from __future__ import annotations

import re
import hashlib
from pathlib import Path

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from agripipe.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Sinonimi agronomici standardizzati
COLUMN_SYNONYMS = {
    "date":      ["data", "periodo", "giorno", "timestamp", "datetime", "time"],
    "field_id":  ["campo", "lotto", "id campo", "id_campo", "appezzamento", "field", "stazione"],
    "temp":      ["temperatura", "t media", "t_media", "temp media", "deg c", "t air", "temp"],
    "humidity":  ["umidità", "hum", "rel humidity", "% umidità", "umidita"],
    "ph":        ["acidità", "ph suolo", "ph_suolo", "acidita"],
    "yield":     ["resa", "produzione", "raccolto", "yield t/ha", "ton ha"],
    "rainfall":  ["pioggia", "precipitazioni", "precip", "mm pioggia", "rain"],
    "n":         ["azoto", "nitrogen", "concimazione n", "kg n"],
    "lat":       ["latitudine", "latitude", "coord y", "gps y"],
    "lon":       ["longitudine", "longitude", "coord x", "gps x"],
}


class RawSchema(BaseModel):
    """Schema minimo per la pipeline."""
    required_columns: list[str] = Field(
        default_factory=lambda: ["date", "field_id", "temp", "humidity", "ph", "yield"]
    )


def load_raw(
    path: str | Path,
    sheet_name: str | int | None = 0,
    schema: RawSchema | None = None,
) -> pd.DataFrame:
    """Carica i dati con certificazione SHA-256 e Unit Intelligence totale."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    schema = schema or RawSchema()
    logger.info("Caricamento God-Mode: %s", path.name)

    file_hash = _generate_file_hash(path)

    if path.suffix.lower() == ".csv":
        df_raw = _load_csv_robustly(path, header=None)
        h_idx = _find_header_row(df_raw, schema.required_columns)
        df = _load_csv_robustly(path, skiprows=h_idx)
    else:
        excel_data = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl", header=None)
        if isinstance(excel_data, dict):
            df = _merge_excel_sheets(excel_data, schema.required_columns)
        else:
            h_idx = _find_header_row(excel_data, schema.required_columns)
            df = pd.read_excel(path, sheet_name=sheet_name or 0, engine="openpyxl", skiprows=h_idx)

    df = _convert_units(df)
    df = _apply_fuzzy_mapping(df)
    df = _inject_filename_info(df, path)
    integrity_report = _check_integrity(df, schema.required_columns)
    df = _normalize_dates(df)
    df = _early_deduplicate(df)

    df.attrs["file_hash"] = file_hash
    df.attrs["integrity_report"] = integrity_report

    return df


def load_from_dir(
    dir_path: str | Path,
    pattern: str = "*.*",
    schema: RawSchema | None = None,
) -> pd.DataFrame:
    """Carica e unisce tutti i file compatibili in una cartella."""
    dir_path = Path(dir_path)
    files = [f for f in dir_path.glob(pattern) if f.suffix.lower() in [".csv", ".xlsx", ".xls"]]
    
    if not files:
        raise ValueError(f"Nessun file CSV o Excel trovato in {dir_path}")
        
    logger.info("Batch Loading: trovati %d file in %s", len(files), dir_path.name)
    all_dfs = []
    
    for f in files:
        try:
            all_dfs.append(load_raw(f, schema=schema))
        except Exception as e:
            logger.warning("Salto file %s: %s", f.name, e)
            
    if not all_dfs:
        raise ValueError("Tutti i file nella cartella sono risultati incompatibili.")
        
    combined = pd.concat(all_dfs, ignore_index=True)
    # Deduplica globale finale
    combined = combined.drop_duplicates().reset_index(drop=True)
    logger.info("Batch Loading completato: %d righe totali da %d file.", len(combined), len(all_dfs))
    return combined


def _generate_file_hash(path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def _check_integrity(df: pd.DataFrame, numeric_targets: list[str]) -> dict:
    report = {}
    for col in df.columns:
        if any(target in col for target in ["temp", "ph", "yield", "humidity", "rainfall", "n"]):
            non_numeric = df[~df[col].apply(lambda x: isinstance(x, (int, float, np.number)))][col].unique()
            if len(non_numeric) > 0:
                report[col] = [str(x) for x in non_numeric if str(x).lower() != 'nan']
    return report


def _convert_units(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        col_name = str(col).lower()
        if "temp" in col_name and ("(f)" in col_name or "fahrenheit" in col_name):
            df[col] = (pd.to_numeric(df[col], errors='coerce') - 32) * 5/9
        if ("rain" in col_name or "pioggia" in col_name) and ("in" in col_name or "inch" in col_name):
            df[col] = pd.to_numeric(df[col], errors='coerce') * 25.4
        if "yield" in col_name and ("lbs" in col_name or "pound" in col_name):
            df[col] = pd.to_numeric(df[col], errors='coerce') * 0.00112085
    return df


def _inject_filename_info(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    fname = path.stem
    if "field_id" not in df.columns or df["field_id"].isna().all():
        field_candidate = fname.split('_')[0].split('-')[0]
        if field_candidate:
            df["field_id"] = field_candidate
    return df


def _early_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def _merge_excel_sheets(sheets_dict: dict[str, pd.DataFrame], required: list[str]) -> pd.DataFrame:
    valid_dfs = []
    for name, df_sheet in sheets_dict.items():
        h_idx = _find_header_row(df_sheet, required)
        df_clean = df_sheet.iloc[h_idx:].copy()
        df_clean.columns = df_sheet.iloc[h_idx]
        df_clean = df_clean.iloc[1:].reset_index(drop=True)
        valid_dfs.append(df_clean)
    if not valid_dfs:
        raise ValueError("Nessun foglio compatibile.")
    return pd.concat(valid_dfs, ignore_index=True)


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        return df
    def fix_excel_date(v):
        if isinstance(v, (int, float)) and 30000 < v < 60000:
            return pd.to_datetime(v, unit='D', origin='1899-12-30')
        return v
    df["date"] = df["date"].apply(fix_excel_date)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def _load_csv_robustly(path: Path, **kwargs) -> pd.DataFrame:
    for enc in ["utf-8", "latin1", "cp1252"]:
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, **kwargs)
                if df.shape[1] > 1: return df
            except: continue
    return pd.read_csv(path, **kwargs)


def _find_header_row(df: pd.DataFrame, required: list[str]) -> int:
    best_row = 0
    max_matches = -1
    all_keywords = set(required)
    for req in required:
        if req in COLUMN_SYNONYMS: all_keywords.update(COLUMN_SYNONYMS[req])
    for i in range(min(15, len(df))):
        row_values = [str(v).lower() for v in df.iloc[i].dropna().values]
        matches = sum(1 for v in row_values if any(k in v for k in all_keywords))
        if matches > max_matches:
            max_matches = matches
            best_row = i
    return best_row


def _apply_fuzzy_mapping(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for col in df.columns:
        clean_col = str(col).lower().strip()
        mapped = False
        if clean_col in COLUMN_SYNONYMS:
            new_cols[col] = clean_col
            continue
        for target, synonyms in COLUMN_SYNONYMS.items():
            if any(s in clean_col for s in synonyms):
                new_cols[col] = target
                mapped = True
                break
        if not mapped:
            new_cols[col] = re.sub(r'[^a-z0-9_]', '', clean_col.replace(' ', '_'))
    return df.rename(columns=new_cols)
