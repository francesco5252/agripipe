from pathlib import Path

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

    loaded = torch.load(output_pt)
    assert torch.equal(loaded["features"], dataset.features)
    assert loaded["feature_names"] == dataset.feature_names

    print("\n[E2E TEST] Pipeline completata con successo!")
