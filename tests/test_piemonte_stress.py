
import pandas as pd
from agripipe.cleaner import AgriCleaner

def test_stress_piemonte_anomalies():
    """Test 'cattivissimo' sulle colture piemontesi con dati impossibili."""
    
    # Creazione dataset sabotato
    data = {
        "date": [
            "2025-01-15", # 1. Nebbiolo a Gennaio (IMPOSSIBILE)
            "2025-08-20", # 2. Nocciola con resa 20 t/ha (IMPOSSIBILE)
            "2025-09-10", # 3. Riso Novarese con pH 9.0 (IMPOSSIBILE)
            "2025-07-05", # 4. Mais a 55 gradi (FUORI RANGE)
        ],
        "yield": [
            5.0,   # Nebbiolo fuori stagione
            20.0,  # Nocciola sovrumana
            7.0,   # Riso ok
            15.0   # Mais ok
        ],
        "ph": [
            7.5,   # Barolo ok
            7.0,   # Nocciola ok
            9.0,   # Riso Novarese (Dovrebbe essere acido!)
            6.5    # Mais ok
        ],
        "temp": [
            5.0,   # Gennaio ok
            25.0,  # Agosto ok
            20.0,  # Settembre ok
            55.0   # Luglio (Troppo caldo!)
        ],
        "field_id": ["F1", "F2", "F3", "F4"]
    }
    df_dirty = pd.DataFrame(data)

    # --- TEST 1: BAROLO (Filtro Temporale) ---
    cleaner_barolo = AgriCleaner.from_preset("vite_nebbiolo_barolo")
    # Righe originali: 0 (Gennaio)
    out_barolo = cleaner_barolo.clean(df_dirty)
    # Verifica che out_barolo esista e sia un DataFrame
    assert isinstance(out_barolo, pd.DataFrame)
    # La resa del Barolo a Gennaio (riga 0) deve essere stata azzerata (NaN) e poi imputata (mediana)
    # Ma dato che è l'unica riga di resa rilevante, guardiamo se il diagnostico l'ha presa
    assert cleaner_barolo.diagnostics.agronomic_outliers_removed >= 1
    
    # --- TEST 2: NOCCIOLA (Filtro Magnitudo) ---
    cleaner_nocciola = AgriCleaner.from_preset("nocciola_piemonte_alta_langa")
    out_nocciola = cleaner_nocciola.clean(df_dirty)
    # La resa 20.0 (riga 1) è > 2.8. Deve essere stata rimossa.
    # Usiamo pd.isna o verifichiamo che non sia più 20.0
    val = out_nocciola["yield"].iloc[1]
    assert pd.isna(val) or val != 20.0
    assert cleaner_nocciola.diagnostics.agronomic_outliers_removed >= 1

    # --- TEST 3: RISO NOVARESE (Filtro Chimico) ---
    cleaner_riso = AgriCleaner.from_preset("riso_novarese")
    out_riso = cleaner_riso.clean(df_dirty)
    # pH 9.0 (riga 2) è fuori dal range [5.0, 6.5]. Deve essere rimosso.
    val_ph = out_riso["ph"].iloc[2]
    assert pd.isna(val_ph) or val_ph != 9.0
    assert cleaner_riso.diagnostics.out_of_bounds_removed >= 1

    # --- TEST 4: MAIS (Filtro Fisico) ---
    cleaner_mais = AgriCleaner.from_preset("mais_cuneese_irriguo")
    out_mais = cleaner_mais.clean(df_dirty)
    # Temp 55.0 (riga 3) è fuori range [5, 40]. Deve essere rimossa.
    val_t = out_mais["temp"].iloc[3]
    assert pd.isna(val_t) or val_t != 55.0
    assert cleaner_mais.diagnostics.out_of_bounds_removed >= 1

    print("\n✅ AgriPipe ha superato lo Stress Test Piemonte!")
