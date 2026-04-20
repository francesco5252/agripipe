"""Test per la conversione automatica di unità SI."""

import pandas as pd
import pytest

from agripipe.units import (
    CONVERSIONS,
    detect_and_convert_units,
    fahrenheit_to_celsius,
    inch_to_mm,
    lb_per_acre_to_kg_per_ha,
)


def test_fahrenheit_to_celsius():
    assert fahrenheit_to_celsius(32.0) == pytest.approx(0.0)
    assert fahrenheit_to_celsius(212.0) == pytest.approx(100.0)
    assert fahrenheit_to_celsius(68.0) == pytest.approx(20.0, abs=0.01)


def test_inch_to_mm():
    assert inch_to_mm(1.0) == pytest.approx(25.4)
    assert inch_to_mm(10.0) == pytest.approx(254.0)


def test_lb_per_acre_to_kg_per_ha():
    # 1 lb/acre ≈ 1.12085 kg/ha
    assert lb_per_acre_to_kg_per_ha(1.0) == pytest.approx(1.12085, abs=0.001)


def test_detect_by_column_suffix_fahrenheit():
    """Colonna 'temp_f' viene auto-convertita in Celsius e rinominata 'temp'."""
    df = pd.DataFrame({"temp_f": [32.0, 212.0, 68.0], "humidity": [50, 60, 70]})
    converted, report = detect_and_convert_units(df)
    assert "temp" in converted.columns
    assert "temp_f" not in converted.columns
    assert converted["temp"].iloc[0] == pytest.approx(0.0)
    assert converted["temp"].iloc[1] == pytest.approx(100.0)
    assert "temp_f" in report
    assert report["temp_f"]["from"] == "fahrenheit"
    assert report["temp_f"]["to"] == "celsius"


def test_detect_by_column_suffix_inch():
    df = pd.DataFrame({"rainfall_in": [1.0, 2.0]})
    converted, report = detect_and_convert_units(df)
    assert "rainfall" in converted.columns
    assert converted["rainfall"].iloc[0] == pytest.approx(25.4)


def test_detect_by_range_suspicious_fahrenheit():
    """Colonna 'temp' con valori sospetti (>50) → probabile Fahrenheit, si converte."""
    df = pd.DataFrame({"temp": [70.0, 80.0, 90.0]})  # range sospetto per celsius
    converted, report = detect_and_convert_units(df, use_range_heuristic=True)
    # Heuristic conservativa: converte solo se TUTTI i valori sono > 50
    assert "temp" in converted.columns
    assert converted["temp"].iloc[0] == pytest.approx((70 - 32) * 5 / 9, abs=0.01)


def test_detect_skips_safe_celsius_range():
    """Valori normali per Celsius (0-40) NON vengono convertiti."""
    df = pd.DataFrame({"temp": [15.0, 20.0, 25.0]})
    converted, report = detect_and_convert_units(df, use_range_heuristic=True)
    assert converted["temp"].iloc[0] == pytest.approx(15.0)
    assert "temp" not in report


def test_detect_no_ambiguous_by_default():
    """Senza suffisso esplicito e senza heuristica, non converte niente."""
    df = pd.DataFrame({"temp": [70.0, 80.0]})
    converted, report = detect_and_convert_units(df, use_range_heuristic=False)
    assert converted["temp"].iloc[0] == pytest.approx(70.0)
    assert report == {}


def test_conversions_registry_complete():
    """Lo store CONVERSIONS contiene i 3 convertitori chiave documentati."""
    assert ("fahrenheit", "celsius") in CONVERSIONS
    assert ("inch", "mm") in CONVERSIONS
    assert ("lb_per_acre", "kg_per_ha") in CONVERSIONS
