# Handoff AgriPipe — stato finale ML-Ops minimale

> Documento vivo per tenere traccia dello stato del progetto fra sessioni. Aggiornato dopo lo strip di Gemini e la riscrittura UI monolitica.

## Visione del progetto

**AgriPipe = pipeline ML-Ops in 3 step, niente agronomia.**

1. **Load** — Excel/CSV agronomici grezzi, con errori realistici (date serial Excel, righe di spazzatura, celle vuote, duplicati).
2. **Clean** — imputazione + outlier + limiti fisici, 100% statistica (niente regole agronomiche).
3. **Tensorize** — bundle PyTorch `.pt` + `metadata.json`, con split opzionale train/val/test.

Niente "green/yellow/red", niente scorecard di sostenibilità, niente interpretazione del dato.

## Architettura (file importanti)

```
src/agripipe/
├── __init__.py          # export pubblici (v0.2.0)
├── loader.py            # load_raw(): Excel/CSV → DataFrame validato (SHA-256)
├── cleaner.py           # AgriCleaner + CleanerConfig + diagnostics
├── tensorizer.py        # Tensorizer + TensorBundle (schema_lock_hash stabile)
├── dataset.py           # AgriDataset (wrapper torch.utils.data)
├── metadata.py          # build_metadata() + save_metadata_json()
├── export.py            # export_ml_bundle() — .pt + .json + .zip
├── synth.py             # generatore di dati sporchi (test + demo)
├── report.py            # report HTML (opzionale)
├── cli.py               # typer CLI (run/generate-sample/...)
├── config/              # preset YAML regionali
└── utils/logging_setup.py

app.py                   # Streamlit monolitico a 3 sezioni — UI ufficiale
tests/                   # 38 test, 82% coverage
```

### File rimossi

- `src/agripipe/pipeline.py` — pipeline "fluente" mai testata
- `src/agripipe/ui/` (theme.py, components.py, __init__.py) — UI multi-modulo
- `src/agripipe/app.py` — vecchia UI a 5 sezioni
- `tests/test_cleaner_god_mode.py`, `test_loader_robustness.py`, `test_production_master.py`, `test_ui_theme.py`

## Feature rimosse rispetto alla versione Gemini

Tutte le feature "pro" non essenziali sono state tagliate per mantenere la pipeline prevedibile:

- KNN / MICE imputation (resta solo mean/median/ffill/time/drop)
- Cyclic date encoding (sin/cos)
- Fahrenheit → Celsius, pollici → mm (niente conversione unità)
- Fuzzy mapping dei nomi di colonna
- Batch-loading da cartella
- Auto-iniezione `field_id` dal nome file
- Precision `float16` / `float32` / `float64` (resta `float32` di default)
- Target log transform
- `drop_redundant` correlation check
- Export Parquet (resta solo `.pt` + `.json` + `.zip`)
- Peer-validation, delta-checks, skewness guard, logical consistency
- Seasonal outliers, target bias detection
- Scorecard di sostenibilità (green/yellow/red)
- Indici agronomici (`indices.py`, `sustainability.py`)

## Stato verificato

- **Test**: 38 passed, 0 failed, 82% coverage (`pytest tests/`)
- **Lint**: ruff all green (`ruff check src tests app.py`)
- **Format**: black clean (`black --check src tests app.py`)
- **E2E**: pipeline verificata su `data/sample/pro_demo.xlsx`
  - 100 righe → 100 pulite, 4 fuori-range rimossi, 37 outlier, 41 imputati
  - Bundle ZIP con train/val/test `.pt` + `metadata.json` generato correttamente

## UI (app.py)

Pagina Streamlit monolitica con `st.session_state` per lo stato condiviso fra step. Avvio:

```bash
streamlit run app.py
```

- **Step 1 — Ingestion**: `st.file_uploader` → `load_raw` via tempfile → metriche righe/colonne/SHA + anteprima.
- **Step 2 — Refinery**: selectbox strategia imputazione + metodo outlier + limiti fisici numerici → bottone "Avvia pulizia" → metriche integrità + NaN prima/dopo.
- **Step 3 — Tensorizer**: dropdown target/scaler/encoding + slider split → bottone "Genera bundle ML" → `st.download_button` per `.zip` + `st.json` del metadata.

## Branch status

- `main` — stato precedente (prima del pivot ML-Ops finale)
- `gemini-cli` — branch di lavoro attuale (con tutte le modifiche)
- `claude-code` — storico

Il prossimo passo è **merge di `gemini-cli` → `main`**.

## Smoke test (per la prossima sessione)

```bash
pytest tests/                              # tutti passare
ruff check src tests app.py                 # green
black --check src tests app.py              # clean
streamlit run app.py                        # UI fruibile
python -c "from agripipe import load_raw, AgriCleaner, CleanerConfig; print('import OK')"
```
