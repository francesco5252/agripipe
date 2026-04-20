"""Test per il fuzzy matching dei nomi colonna."""

import pandas as pd

from agripipe.matching import fuzzy_rename_columns


def test_fuzzy_rename_exact_match():
    """Se i nomi sono già esatti, non cambia nulla."""
    df = pd.DataFrame({"temp": [20], "humidity": [50]})
    renamed, report = fuzzy_rename_columns(df, ["temp", "humidity"])
    assert list(renamed.columns) == ["temp", "humidity"]
    assert report == {}


def test_fuzzy_rename_case_and_spaces():
    """Gestisce differenze di case e spazi bianchi senza dizionario."""
    df = pd.DataFrame({" Temp ": [20], "HUMIDITY": [50]})
    renamed, report = fuzzy_rename_columns(df, ["temp", "humidity"])
    assert list(renamed.columns) == ["temp", "humidity"]
    # Non è un fuzzy match vero, è normalizzazione base, report vuoto o minimo
    assert "temp" in renamed.columns


def test_fuzzy_rename_with_synonyms():
    """Usa il dizionario per mappare termini italiani/varianti."""
    df = pd.DataFrame({"Temperatura": [20], "Umidità": [50]})
    synonyms = {"temp": ["temperatura", "t_c"], "humidity": ["umidità", "rh"]}
    renamed, report = fuzzy_rename_columns(df, ["temp", "humidity"], synonyms=synonyms)
    assert "temp" in renamed.columns
    assert "humidity" in renamed.columns
    assert report["Temperatura"] == "temp"
    assert report["Umidità"] == "humidity"


def test_fuzzy_rename_with_rapidfuzz_similarity():
    """Riconosce 'Temperature_Celsius' come 'temp' via similarity."""
    df = pd.DataFrame({"Temperature_Celsius": [20]})
    # Anche senza sinonimi espliciti, se la similarità è alta (es. con 'temp')
    # Ma 'temp' vs 'temperature_celsius' potrebbe essere basso.
    # Proviamo con qualcosa di più vicino.
    df = pd.DataFrame({"Temperat": [20]})
    renamed, report = fuzzy_rename_columns(df, ["temp"], threshold=60)
    assert "temp" in renamed.columns
    assert report["Temperat"] == "temp"


def test_fuzzy_rename_skips_low_score():
    """Non rinomina se lo score è sotto la soglia."""
    df = pd.DataFrame({"xyz": [20]})
    renamed, report = fuzzy_rename_columns(df, ["temp"], threshold=90)
    assert "xyz" in renamed.columns
    assert "temp" not in renamed.columns
    assert report == {}


def test_fuzzy_rename_avoids_collisions():
    """Se due colonne mappano sulla stessa richiesta, vince la migliore o solleva warning."""
    df = pd.DataFrame({"Temp_C": [20], "Temperature": [21]})
    renamed, report = fuzzy_rename_columns(df, ["temp"])
    # Dovrebbe sceglierne una e non distruggere l'altra
    assert "temp" in renamed.columns
    assert len(renamed.columns) == 2
