"""Motore di calcolo avanzato per Indici Agronomici e di Sostenibilità."""

from __future__ import annotations
import pandas as pd


def compute_agronomic_indices(df: pd.DataFrame, knowledge: dict) -> pd.DataFrame:
    """Arricchisce il DataFrame con indici agronomici per il Machine Learning.

    Calcola (dove le colonne sono presenti):
        - **GDD daily/accumulated**: Gradi Giorno rispetto a ``t_base`` per coltura.
        - **Huglin index**: Indice qualità vitivinicola (solo vite).
        - **drought_7d_score**: Bilancio idrico mobile a 7 giorni.
        - **n_efficiency**: Nitrogen Use Efficiency (yield / N).

    Args:
        df: DataFrame pulito. Colonne richieste: ``crop_type``/``crop``,
            ``temp``/``temperatura``. Opzionali: ``date``, ``field_id``,
            ``rainfall``, ``yield``, ``n``.
        knowledge: Dizionario caricato da ``agri_knowledge.yaml``. La chiave
            ``crops`` contiene ``t_base`` per coltura.

    Returns:
        DataFrame con le colonne originali + gli indici calcolati. Se mancano
        colonne essenziali (crop + temp) restituisce il df invariato.

    Example:
        >>> df_with_indices = compute_agronomic_indices(df, knowledge)
        >>> "gdd_accumulated" in df_with_indices.columns
        True
    """
    df = df.copy()

    # Identifica colonne necessarie
    crop_col = next((c for c in ["crop_type", "crop", "coltura"] if c in df.columns), None)
    temp_col = next((c for c in ["temp", "temperatura"] if c in df.columns), None)
    rain_col = next((c for c in ["rainfall", "pioggia"] if c in df.columns), None)
    date_col = next((c for c in ["date", "data"] if c in df.columns), None)
    field_col = next((c for c in ["field_id", "campo", "lotto"] if c in df.columns), None)
    yield_col = next((c for c in ["yield", "resa"] if c in df.columns), None)
    n_col = next((c for c in ["n", "azoto"] if c in df.columns), None)

    if not crop_col or not temp_col:
        return df

    # Ordiniamo i dati per campo e data (fondamentale per gli indici accumulati)
    if date_col and field_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=[field_col, date_col])

        # 1. GDD e INDICE DI HUGLIN (Vite)
        df["gdd_daily"] = 0.0
        df["huglin_daily"] = 0.0

        for crop_name, rules in knowledge.get("crops", {}).items():
            t_base = rules.get("t_base", 10.0)
            mask = df[crop_col].str.lower() == crop_name.lower()
            if not mask.any():
                continue

            # GDD classico
            df.loc[mask, "gdd_daily"] = (df.loc[mask, temp_col] - t_base).clip(lower=0)

            # Huglin (Semplificato per dati giornalieri: (T_media - 10 + T_max - 10)/2 * coefficiente_giorno)
            # Qui usiamo una versione base: (T_media - 10) con un piccolo bonus se fa molto caldo
            if "grape" in crop_name or "vite" in crop_name:
                df.loc[mask, "huglin_daily"] = (df.loc[mask, temp_col] - 10).clip(lower=0) * 1.02

        df["gdd_accumulated"] = df.groupby(field_col)["gdd_daily"].cumsum()
        df["huglin_index"] = df.groupby(field_col)["huglin_daily"].cumsum()

        # 2. STRESS IDRICO ACCUMULATO (Ultimi 7 giorni)
        if rain_col:
            # Calcoliamo il bilancio giornaliero
            df["daily_wb"] = df[rain_col] - (df[temp_col] * 0.2)
            # Somma mobile a 7 giorni: se il numero è molto negativo, c'è siccità
            df["drought_7d_score"] = df.groupby(field_col)["daily_wb"].transform(
                lambda x: x.rolling(7, min_periods=1).sum()
            )

    # 3. INDICI DI SOSTENIBILITÀ (Efficienza Nutrienti)
    if yield_col and n_col:
        # Nitrogen Use Efficiency (NUE): Tonnellate prodotte per ogni kg di Azoto
        # Più è alto, più l'azienda è sostenibile (produce di più con meno chimica)
        df["n_efficiency"] = df[yield_col] / (
            df[n_col] + 1.0
        )  # +1.0 per evitare divisioni per zero

    return df
