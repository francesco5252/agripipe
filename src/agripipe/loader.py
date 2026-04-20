"""Step 1 — Loader: carica Excel/CSV agronomici grezzi.

Responsabilità minimali, una sola funzione pubblica: ``load_raw``.

* Supporta Excel (``.xlsx``, ``.xls``) e CSV con separatori comuni (``,``, ``;``, TAB).
* Salta righe di "spazzatura" iniziali (intestazioni aziendali, note) e individua
  automaticamente la riga con l'header.
* Calcola un fingerprint SHA-256 del file (tracciabilità del dato).
* Valida lo schema: se mancano colonne obbligatorie ⇒ ``ValueError``.
* Normalizza la colonna ``date`` a ``datetime``.

NIENTE fuzzy mapping, conversione unità (Fahrenheit/pollici), batch-loading da
cartella o auto-iniezione del ``field_id`` dal nome file — funzionalità rimosse
per mantenere la pipeline prevedibile e facilmente debuggabile.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from agripipe.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Sinonimi riconosciuti SOLO per individuare la riga dell'header nei file
# con righe di intestazione aziendali prima del vero header.
_HEADER_HINTS = {
    "date",
    "data",
    "field_id",
    "campo",
    "temp",
    "temperatura",
    "humidity",
    "umidità",
    "umidita",
    "ph",
    "yield",
    "resa",
    "rainfall",
    "pioggia",
}


class RawSchema(BaseModel):
    """Schema minimo richiesto dalla pipeline."""

    required_columns: list[str] = Field(
        default_factory=lambda: ["date", "field_id", "temp", "humidity", "ph", "yield"]
    )


import yaml


def load_raw(
    path: str | Path,
    sheet_name: str | int | None = 0,
    schema: RawSchema | None = None,
    fuzzy: bool = False,
) -> pd.DataFrame:
    """Carica un file agronomico grezzo e restituisce un DataFrame validato.

    Args:
        path: Percorso al file ``.xlsx``, ``.xls`` o ``.csv``.
        sheet_name: Foglio Excel da leggere (default: primo foglio).
        schema: Schema atteso. Se ``None`` usa ``RawSchema()``.
        fuzzy: Se ``True``, prova a riconoscere i nomi delle colonne tramite
            fuzzy matching e sinonimi (es: "Temperatura" -> "temp").

    Returns:
        ``pandas.DataFrame`` con colonne validate. In ``df.attrs['file_hash']``
        viene salvato lo SHA-256 del file sorgente.

    Raises:
        FileNotFoundError: Se ``path`` non esiste.
        ValueError: Se mancano colonne obbligatorie dallo schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    schema = schema or RawSchema()
    logger.info("Caricamento: %s", path.name)
    file_hash = _generate_file_hash(path)

    if path.suffix.lower() == ".csv":
        df = _load_csv_with_header_detection(path, schema.required_columns)
    else:
        df = _load_excel_with_header_detection(path, sheet_name, schema.required_columns)

    if fuzzy:
        from agripipe.matching import fuzzy_rename_columns

        syn_path = Path("configs/column_synonyms.yaml")
        synonyms, threshold = {}, 85
        if syn_path.exists():
            with open(syn_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                synonyms = cfg.get("synonyms", {})
                threshold = cfg.get("threshold", 85)

        df, report = fuzzy_rename_columns(
            df, schema.required_columns, synonyms=synonyms, threshold=threshold
        )
        if report:
            logger.info("Fuzzy mapping applicato: %s", report)

    # Normalizzazione nomi colonne: lower + strip spazi
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Validazione schema — colonne obbligatorie presenti?
    missing = [c for c in schema.required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti nello schema: {missing}")

    df = _normalize_dates(df)
    df.attrs["file_hash"] = file_hash
    df.attrs["source_file"] = path.name
    return df


def _generate_file_hash(path: Path) -> str:
    """Calcola lo SHA-256 di un file leggendolo a blocchi da 8 KiB."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def _find_header_row(df_no_header: pd.DataFrame, required: list[str]) -> int:
    """Trova la riga che contiene l'header vero, saltando eventuale spazzatura."""
    hints = _HEADER_HINTS.union(c.lower() for c in required)
    best_row, max_matches = 0, -1
    for i in range(min(15, len(df_no_header))):
        row_values = [str(v).lower().strip() for v in df_no_header.iloc[i].dropna().values]
        matches = sum(1 for v in row_values if v in hints)
        if matches > max_matches:
            max_matches, best_row = matches, i
    return best_row


def _load_excel_with_header_detection(
    path: Path, sheet_name: str | int | None, required: list[str]
) -> pd.DataFrame:
    """Legge un Excel individuando la riga giusta da usare come header."""
    raw = pd.read_excel(path, sheet_name=sheet_name or 0, engine="openpyxl", header=None)
    header_idx = _find_header_row(raw, required)
    return pd.read_excel(path, sheet_name=sheet_name or 0, engine="openpyxl", skiprows=header_idx)


def _load_csv_with_header_detection(path: Path, required: list[str]) -> pd.DataFrame:
    """Legge un CSV provando (encoding × separatore) finché uno "tiene"."""
    last_err: Exception | None = None
    for enc in ["utf-8", "latin1", "cp1252"]:
        for sep in [",", ";", "\t"]:
            try:
                probe = pd.read_csv(path, sep=sep, encoding=enc, header=None)
                if probe.shape[1] <= 1:
                    continue
                header_idx = _find_header_row(probe, required)
                return pd.read_csv(path, sep=sep, encoding=enc, skiprows=header_idx)
            except Exception as e:  # noqa: BLE001 — probing, errori attesi
                last_err = e
                continue
    raise ValueError(f"Impossibile leggere il CSV {path.name}: {last_err}")


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Converte la colonna ``date`` in ``datetime``; righe senza data valida via."""
    if "date" not in df.columns:
        return df

    def _fix_excel_serial(v):
        # Excel serial date (es. 45000) → datetime
        if isinstance(v, (int, float)) and 30000 < v < 60000:
            return pd.to_datetime(v, unit="D", origin="1899-12-30")
        return v

    df = df.copy()
    df["date"] = df["date"].apply(_fix_excel_serial)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).reset_index(drop=True)


from typing import Literal


def batch_load_raw(
    input_dir: str | Path,
    sheet_name: str | int | None = 0,
    schema: RawSchema | None = None,
    on_error: Literal["raise", "skip"] = "raise",
    fuzzy: bool = False,
) -> pd.DataFrame:
    """Carica tutti gli Excel/CSV di una cartella e li concatena.

    Ogni riga del DataFrame risultante contiene una colonna ``source_file`` con
    il nome del file di provenienza, utile per tracciare la provenienza in
    analisi a valle.

    Args:
        input_dir: Percorso della cartella contenente i file da caricare.
        sheet_name: Foglio Excel da leggere per ogni file (default: 0).
        schema: Schema atteso. Se ``None`` usa ``RawSchema()``.
        on_error: Se ``"raise"``, solleva eccezione al primo file malformato.
            Se ``"skip"``, logga un warning e continua col file successivo.

    Returns:
        Un unico DataFrame consolidato.

    Raises:
        FileNotFoundError: Se ``input_dir`` non esiste.
        ValueError: Se non viene trovato alcun file Excel/CSV o se ``on_error="raise"``
            e un file è malformato.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Directory non trovata: {input_dir}")

    extensions = {".xlsx", ".xls", ".csv"}
    files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in extensions)

    if not files:
        raise ValueError(f"Nessun file Excel/CSV trovato in {input_dir}")

    logger.info("Batch loading da: %s (%d file trovati)", input_dir.name, len(files))
    all_dfs = []

    for f in files:
        try:
            df = load_raw(f, sheet_name=sheet_name, schema=schema, fuzzy=fuzzy)
            df["source_file"] = f.name
            all_dfs.append(df)
        except Exception as e:
            if on_error == "raise":
                raise ValueError(f"Errore fatale nel caricamento di {f.name}: {e}") from e
            logger.warning("Salto file malformato %s: %s", f.name, e)

    if not all_dfs:
        raise ValueError(f"Nessun file è stato caricato con successo da {input_dir}")

    return pd.concat(all_dfs, ignore_index=True)
