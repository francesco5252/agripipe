"""Modulo per il fuzzy matching dei nomi colonna."""

from __future__ import annotations

import pandas as pd
from rapidfuzz import process, utils

from agripipe.utils.logging_setup import get_logger

logger = get_logger(__name__)


def fuzzy_rename_columns(
    df: pd.DataFrame,
    required: list[str],
    synonyms: dict[str, list[str]] | None = None,
    threshold: int = 85,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rinomina le colonne del DataFrame usando fuzzy matching.

    Cerca di mappare ogni colonna esistente a uno dei nomi richiesti (o ai loro
    sinonimi). Se trova una corrispondenza con punteggio >= threshold, rinomina.

    Args:
        df: DataFrame originale.
        required: Lista di nomi colonna canonici richiesti.
        synonyms: Dizionario {canonico: [sinonimi]}.
        threshold: Punteggio minimo (0-100) per accettare un match.

    Returns:
        Tupla (df_rinominato, report_cambiamenti).
    """
    df = df.copy()
    report: dict[str, str] = {}
    synonyms = synonyms or {}

    # Costruiamo il set di "target" per il matching: ogni nome canonico
    # più i suoi sinonimi mappano al canonico stesso.
    target_map: dict[str, str] = {}
    for req in required:
        target_map[req.lower()] = req
        for syn in synonyms.get(req, []):
            target_map[syn.lower()] = req

    targets = list(target_map.keys())
    current_columns = [str(c) for c in df.columns]
    new_columns = list(df.columns)

    for i, col in enumerate(current_columns):
        col_clean = utils.default_process(col)
        if not col_clean:
            continue

        # Se la colonna normalizzata è già tra i target esatti, skip fuzzy
        if col_clean in target_map:
            canon_name = target_map[col_clean]
            # Rinominiamo comunque al canonico se differisce solo per case/spazi
            if col != canon_name:
                new_columns[i] = canon_name
                report[col] = canon_name
            continue

        # Altrimenti, proviamo il fuzzy match
        match = process.extractOne(col_clean, targets, score_cutoff=threshold)
        if match:
            best_target, score, _ = match
            canon_name = target_map[best_target]

            # Verifichiamo che non ci siano collisioni (es. due colonne mappano allo stesso canonico)
            if canon_name in new_columns:
                logger.warning(
                    "Collisione fuzzy: '%s' (score=%d) mappa a '%s', ma esiste già.",
                    col,
                    score,
                    canon_name,
                )
                continue

            new_columns[i] = canon_name
            report[col] = canon_name
            logger.info("Fuzzy match: '%s' -> '%s' (score=%d)", col, canon_name, score)

    df.columns = new_columns
    return df, report
