"""Conversione automatica di unità di misura non-SI → SI.

Rileva colonne con unità americane (Fahrenheit, inch, lb/acre) tramite due
meccanismi complementari:

1. **Suffisso nel nome colonna** — es. ``temp_f`` → Fahrenheit;
   ``rainfall_in`` → pollici; ``yield_lb_acre`` → libbre/acro.
2. **Euristica sul range numerico** (opt-in con ``use_range_heuristic=True``):
   una colonna ``temp`` dove tutti i valori stanno in range 50-120 è sospetta
   di essere Fahrenheit anziché Celsius.

Dopo la conversione la colonna viene rinominata al canonico AgriPipe
(``temp``, ``rainfall``, ``yield``), eliminando il suffisso.

Convertitori esposti:

* ``fahrenheit_to_celsius``
* ``inch_to_mm``
* ``lb_per_acre_to_kg_per_ha``

La mappa globale ``CONVERSIONS`` associa tuple ``(from_unit, to_unit)`` a
funzioni ``Callable[[float], float]``.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from agripipe.utils.logging_setup import get_logger

logger = get_logger(__name__)


def fahrenheit_to_celsius(f: float) -> float:
    """Converte Fahrenheit in Celsius."""
    return (f - 32.0) * 5.0 / 9.0


def inch_to_mm(i: float) -> float:
    """Converte pollici in millimetri."""
    return i * 25.4


def lb_per_acre_to_kg_per_ha(lba: float) -> float:
    """Converte libbre/acro in kg/ettaro."""
    return lba * 1.12085


CONVERSIONS: dict[tuple[str, str], Callable[[float], float]] = {
    ("fahrenheit", "celsius"): fahrenheit_to_celsius,
    ("inch", "mm"): inch_to_mm,
    ("lb_per_acre", "kg_per_ha"): lb_per_acre_to_kg_per_ha,
}


# Mappa suffisso/pattern nel nome colonna → conversione da applicare
# + nome canonico risultante (senza suffisso).
SUFFIX_RULES: dict[str, tuple[str, str, str]] = {
    # pattern_in_colonna → (from_unit, to_unit, canonical_name)
    "temp_f": ("fahrenheit", "celsius", "temp"),
    "temperature_f": ("fahrenheit", "celsius", "temp"),
    "rainfall_in": ("inch", "mm", "rainfall"),
    "rain_in": ("inch", "mm", "rainfall"),
    "yield_lb_acre": ("lb_per_acre", "kg_per_ha", "yield"),
    "yield_lb_per_acre": ("lb_per_acre", "kg_per_ha", "yield"),
}


# Colonne numeriche dove l'euristica range-based ha senso.
# Celsius: range tipico -20 / +50. Fahrenheit: range tipico +20 / +120.
# Se TUTTI i valori > 50 e < 120, è probabile che sia Fahrenheit.
RANGE_HEURISTICS: dict[str, tuple[tuple[float, float], tuple[str, str]]] = {
    "temp": ((50.0, 120.0), ("fahrenheit", "celsius")),
}


def detect_and_convert_units(
    df: pd.DataFrame,
    use_range_heuristic: bool = False,
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    """Rileva e converte colonne con unità non-SI in unità SI.

    Args:
        df: DataFrame da ispezionare.
        use_range_heuristic: Se ``True`` attiva la conversione basata sul
            range dei valori (es. ``temp > 50`` → probabile Fahrenheit).
            Default ``False`` per evitare falsi positivi silenziosi.

    Returns:
        Tupla ``(df_converted, report)`` dove ``report`` è un dict:
        ``{nome_colonna_orig: {"from": unit, "to": unit, "canonical": name}}``.
    """
    df = df.copy()
    report: dict[str, dict[str, str]] = {}

    # 1) Detection by column suffix
    for col in list(df.columns):
        col_lower = str(col).strip().lower()
        for pattern, (from_unit, to_unit, canonical) in SUFFIX_RULES.items():
            if (
                col_lower == pattern
                or col_lower.endswith(f"_{pattern}")
                or col_lower.endswith(pattern)
            ):
                converter = CONVERSIONS[(from_unit, to_unit)]
                df[canonical] = df[col].astype(float).apply(converter)
                df = df.drop(columns=[col])
                report[col] = {"from": from_unit, "to": to_unit, "canonical": canonical}
                logger.info(
                    "Unit conversion: %s (%s → %s) renamed to %s",
                    col,
                    from_unit,
                    to_unit,
                    canonical,
                )
                break

    # 2) Range heuristic (opt-in)
    if use_range_heuristic:
        for col, ((low, high), (from_unit, to_unit)) in RANGE_HEURISTICS.items():
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue
            if (series.min() >= low) and (series.max() <= high):
                converter = CONVERSIONS[(from_unit, to_unit)]
                df[col] = df[col].astype(float).apply(converter)
                report[col] = {"from": from_unit, "to": to_unit, "canonical": col}
                logger.warning(
                    "Range heuristic: %s values in [%.1f, %.1f] → converted %s → %s",
                    col,
                    series.min(),
                    series.max(),
                    from_unit,
                    to_unit,
                )

    return df, report
