from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import random_split

from agripipe.loader import load_raw
from agripipe.cleaner import AgriCleaner
from agripipe.dataset import AgriDataset
from agripipe.report import generate_report
from agripipe.synth import SynthConfig, generate_dirty_excel


def test_full_pipeline_e2e(tmp_path: Path, cleaner_config):
    """Test end-to-end della pipeline: Synth -> Load -> Clean -> Dataset -> Split -> Verify."""

    # 1. SETUP PATHS
    input_file = tmp_path / "raw_data.xlsx"
    output_pt = tmp_path / "dataset.pt"
    report_html = tmp_path / "report.html"

    # 2. SYNTH: Genera dati sporchi
    synth_cfg = SynthConfig(n_rows=200, n_fields=5, seed=42)
    generate_dirty_excel(input_file, synth_cfg)
    assert input_file.exists()

    # 3. LOAD: Caricamento e validazione schema
    df_raw = load_raw(input_file)
    assert len(df_raw) >= 200

    # 4. CLEAN: Pulizia outlier, missing e limiti fisici
    # Assicuriamoci che rainfall sia incluso per evitare NaN residui
    if "rainfall" not in cleaner_config.numeric_columns:
        cleaner_config.numeric_columns.append("rainfall")

    cleaner = AgriCleaner(cleaner_config)
    df_clean = cleaner.clean(df_raw)

    # Verifiche post-pulizia
    assert df_clean.isna().sum().sum() == 0

    # 5. REPORT: Generazione report di qualità
    generate_report(df_raw, df_clean, report_html)
    assert report_html.exists()

    # 6. DATASET: Trasformazione per PyTorch via AgriDataset
    dataset = AgriDataset(
        df=df_clean,
        numeric_columns=cleaner_config.numeric_columns,
        categorical_columns=cleaner_config.categorical_columns,
        target="yield",
    )

    assert len(dataset) == len(df_clean)
    features, target = dataset[0]
    assert isinstance(features, torch.Tensor)
    assert isinstance(target, torch.Tensor)

    # 7. SPLIT: Verifica compatibilità con utility PyTorch
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    assert len(train_ds) == train_size
    assert len(test_ds) == test_size

    # 8. SAVE/LOAD: Verifica persistenza del dataset intero o dei tensor
    # Salviamo i tensor estratti dal dataset per semplicità di caricamento futuro
    data_to_save = {
        "features": dataset.features,
        "target": dataset.target,
        "feature_names": dataset.feature_names,
    }
    torch.save(data_to_save, output_pt)
    assert output_pt.exists()

    loaded = torch.load(output_pt, weights_only=True)
    assert torch.equal(loaded["features"], dataset.features)
    assert loaded["feature_names"] == dataset.feature_names

    print("\n[E2E TEST] Pipeline completata con successo!")


def test_e2e_non_canonical_columns(tmp_path: Path):
    """Issue #9 — Test E2E su Excel reale con nomi colonna non-canonici.

    Simula un file Excel reale proveniente da un'azienda agricola con nomi
    italiani, spaziature e formati non-standard. Verifica che l'intera pipeline
    (load → fuzzy rename → clean → tensorize) funzioni senza errori manuali da
    parte dell'utente.

    Acceptance criteria:
    - Il loader con fuzzy=True riconosce tutti i nomi canonici.
    - Il cleaner rimuove outlier e imputa NaN.
    - Il tensor prodotto ha shape corretta e zero NaN.
    """
    # --- Crea un Excel "reale" con nomi colonna sporchi ---
    dirty_excel = tmp_path / "dirty_columns.xlsx"
    n_rows = 50
    df_dirty = pd.DataFrame(
        {
            "Data Rilevazione": pd.date_range("2024-05-01", periods=n_rows, freq="D"),
            "ID Appezzamento": [f"Campo_{i % 5}" for i in range(n_rows)],
            "Temp_Aria_C": [22.0 + (i % 8) for i in range(n_rows)],
            "Umidita_Relativa_%": [60.0 + (i % 20) for i in range(n_rows)],
            "pH_Estratto_Suolo": [6.5 + (i % 5) * 0.2 for i in range(n_rows)],
            "Resa_Stimata_kg_ha": [8000 + (i % 10) * 300 for i in range(n_rows)],
        }
    )
    # Inietta un outlier e un NaN realistici
    df_dirty.loc[5, "Temp_Aria_C"] = 200.0  # Valore palesemente errato
    df_dirty.loc[10, "Umidita_Relativa_%"] = float("nan")
    df_dirty.to_excel(dirty_excel, index=False)
    assert dirty_excel.exists()

    # --- Step 1: Load con fuzzy matching abilitato ---
    df_raw = load_raw(dirty_excel, fuzzy=True)

    # Verifiche base sul mapping
    assert "temp" in df_raw.columns, "fuzzy deve mappare Temp_Aria_C → temp"
    assert "humidity" in df_raw.columns, "fuzzy deve mappare Umidita_Relativa → humidity"
    assert "ph" in df_raw.columns, "fuzzy deve mappare pH_Estratto_Suolo → ph"
    assert "yield" in df_raw.columns, "fuzzy deve mappare Resa_Stimata → yield"
    assert len(df_raw) == n_rows

    # --- Step 2: Clean ---
    from agripipe.cleaner import CleanerConfig

    config = CleanerConfig(
        numeric_columns=["temp", "humidity", "ph", "yield"],
        categorical_columns=["field_id"] if "field_id" in df_raw.columns else [],
        missing_strategy="median",
        outlier_method="iqr",
        physical_bounds={"temp": (-5.0, 50.0), "humidity": (0.0, 100.0)},
    )
    cleaner = AgriCleaner(config)
    df_clean = cleaner.clean(df_raw)

    # Outlier rimosso e imputato
    assert df_clean["temp"].max() <= 50.0, "outlier temp non rimosso"
    # Nessun NaN nelle colonne numeriche
    for col in ["temp", "humidity", "ph", "yield"]:
        nan_count = df_clean[col].isna().sum()
        assert nan_count == 0, f"colonna {col} ha ancora {nan_count} NaN dopo clean"

    # --- Step 3: Tensorize ---
    dataset = AgriDataset(
        df=df_clean,
        numeric_columns=["temp", "humidity", "ph", "yield"],
        categorical_columns=[],
        target="yield",
    )
    features, target_val = dataset[0]
    assert isinstance(features, torch.Tensor)
    assert not torch.isnan(features).any(), "tensor features contiene NaN"
    assert dataset.features.shape[0] == len(df_clean)
    assert dataset.features.shape[1] >= 3  # almeno temp, humidity, ph

    print(
        f"\n[E2E #9] OK — {len(df_clean)} righe, "
        f"shape={tuple(dataset.features.shape)}, "
        f"outlier={cleaner.diagnostics.outliers_removed}, "
        f"imputed={cleaner.diagnostics.values_imputed}"
    )

