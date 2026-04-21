
import pandas as pd
import numpy as np
import pytest
from agripipe.cleaner import AgriCleaner

def test_giro_italia_validation():
    """Stress test finale: coerenza agronomica Nord, Centro, Sud."""
    
    # Dataset con anomalie contestuali
    data = {
        "date": [
            "2025-11-15",  # 1. Franciacorta a Novembre (ERRORE)
            "2025-09-20",  # 2. Chianti pH acido (ERRORE)
            "2025-07-10",  # 3. Pachino Salino (OK per Pachino!)
        ],
        "yield": [10.0, 8.0, 50.0],
        "ph": [7.0, 5.0, 7.2],
        "salinity": [0.5, 0.5, 4.5], # 4.5 è altissimo per quasi tutti, tranne Pachino
        "field_id": ["NORD", "CENTRO", "SUD"]
    }
    df = pd.DataFrame(data)

    # --- 1. TEST NORD: Franciacorta (Check Temporale) ---
    cleaner_nord = AgriCleaner.from_preset("vite_franciacorta_docg")
    out_nord = cleaner_nord.clean(df)
    # La resa a Novembre deve essere rimossa
    # Cerchiamo la riga "NORD" (indice 0)
    val_nord = out_nord.iloc[0]["yield"]
    assert pd.isna(val_nord) or val_nord != 10.0
    assert cleaner_nord.diagnostics.agronomic_outliers_removed >= 1

    # --- 2. TEST CENTRO: Chianti (Check pH Basico) ---
    cleaner_centro = AgriCleaner.from_preset("vite_sangiovese_chianti_brunello")
    out_centro = cleaner_centro.clean(df)
    # pH 5.0 è fuori dal range [7.5, 8.3] del Chianti
    val_ph = out_centro.iloc[1]["ph"]
    assert pd.isna(val_ph) or val_ph != 5.0
    assert cleaner_centro.diagnostics.out_of_bounds_removed >= 1

    # --- 3. TEST SUD: Pachino (Check Salinità Elevata) ---
    cleaner_sud = AgriCleaner.from_preset("pomodoro_pachino_igp")
    # Nota: Dobbiamo assicurarci che la colonna salinità venga processata
    # Se non è in physical_bounds, non viene filtrata a meno di outlier statistici.
    # Ma il preset Pachino ha salinity_tolerance: 5.0
    out_sud = cleaner_sud.clean(df)
    # Salinità 4.5 deve essere MANTENUTA (perché 4.5 < 5.0)
    val_sal = out_sud.iloc[2]["salinity"]
    assert val_sal == 4.5
    # Mentre se usassimo il preset Franciacorta, verrebbe filtrato se avessimo messo un limite basso
    # (Il preset Franciacorta non ha salinity_tolerance definito, quindi usa default)
    
    print("\n🇮🇹 Giro d'Italia completato con successo!")
