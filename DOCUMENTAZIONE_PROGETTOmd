# 📝 Documentazione di Progetto — AgriPipe

> Doppio ruolo di questo documento:
> 1. **Sorgente canonica** da cui si riscrive il `README.md` e `README.en.md` quando il progetto evolve.
> 2. **Cartina geografica** per gli agenti AI (Gemini CLI, Claude Code) e per qualunque sviluppatore nuovo: orienta in `src/`, `configs/`, `tests/` e mostra come i pezzi si parlano.
>
> Il documento è organizzato in due parti: una **Mappa dichiarativa** in testa (cosa c'è e dove) seguita dalla **Narrazione in 3 step** (come è nato): **Loader** → **Cleaner** → **Tensorizer**.

---

## 🗺️ Mappa del progetto

### Moduli del core (`src/agripipe/`)

| File | Responsabilità | Input principale | Output principale | Dipende da |
|------|----------------|------------------|-------------------|------------|
| `loader.py` | Lettura Excel/CSV, auto-detect header, fingerprint SHA-256, validazione schema, normalizzazione date, batch loading | path file o cartella | `pandas.DataFrame` validato | `pandas`, `pydantic`, `matching` |
| `matching.py` | Fuzzy rename dei nomi colonna via dizionario sinonimi | `DataFrame` + `synonyms` | `DataFrame` con colonne rinominate | `rapidfuzz`, `pyyaml` |
| `units.py` | Conversione automatica a unità SI (F→°C, inch→mm, lb/acre→kg/ha) | `DataFrame` | `DataFrame` con unità normalizzate | — |
| `base.py` | Interfaccia astratta `AgriTransformer` (`fit` / `transform` / `fit_transform`) | — | — | — |
| `transformers.py` | 11 Transformer concreti disaccoppiati che ereditano da `AgriTransformer` (type coercion, physical bounds, outlier IQR/Z-score, imputazione numerica e categorica, dedup, ecc.) | `DataFrame` | `DataFrame` | `base`, `pandas`, `sklearn` |
| `pipeline.py` | Orchestratore stile scikit-learn (`Pipeline(steps=[...])`) che esegue i Transformer in sequenza | lista di `(name, transformer)` | `DataFrame` pulito | `base` |
| `cleaner.py` | API di alto livello `AgriCleaner` — carica `CleanerConfig` (Pydantic) da YAML/preset e assembla la Pipeline di Transformer | `CleanerConfig` | `AgriCleaner.clean(df) → DataFrame` | `pipeline`, `transformers` |
| `tensorizer.py` | Scaling (Standard/Robust), encoding categoriche, split train/val/test, schema hash | `DataFrame` pulito | tensor `torch.Tensor` + `feature_names` | `sklearn`, `torch` |
| `dataset.py` | Adattatore `AgriDataset` che avvolge le feature/target in una classe utilizzabile come `torch.utils.data.Dataset` | `DataFrame` + metadati | oggetto Dataset PyTorch | `torch` |
| `export.py` | Costruzione del bundle `.pt` + `.json` + `.zip` auto-contenuto | tensor + diagnostics | file `.zip` | `tensorizer`, `metadata` |
| `metadata.py` | Generazione del `metadata.json` (schema, stats, hash, snippet di caricamento) | stato pipeline | `dict` serializzabile | — |
| `tracking.py` | Integrazione opzionale MLflow: baseline Ridge come "safety benchmark" e logging automatico delle run | bundle + config | MLflow run | `mlflow`, `sklearn` |
| `report.py` | Report HTML di qualità (grafici before/after, statistiche) | `df_raw` + `df_clean` | file `.html` | `matplotlib`, `seaborn` |
| `synth.py` | Generatore di Excel sintetici "sporchi" per test e demo | `SynthConfig` | file `.xlsx` | `pandas`, `openpyxl` |
| `cli.py` | CLI Typer (`agripipe run / check / generate / report / list-presets / version`) | argomenti shell | side-effects | tutto sopra |
| `utils/logging_setup.py` | Logger strutturato condiviso | — | `logger` | `logging` |

### Interfaccia utente

| Percorso | Ruolo |
|----------|-------|
| `app.py` | Entry point Streamlit multi-page |
| `pages/1_📥_Ingestion.py` | Upload + preview + hash |
| `pages/2_🧹_Refinery.py` | Wizard di pulizia interattivo |
| `pages/3_📦_Tensorizer.py` | Split + scaler + download bundle |

### Configurazione e dati

| Percorso | Ruolo |
|----------|-------|
| `configs/default.yaml` | Config Cleaner di default |
| `configs/column_synonyms.yaml` | Dizionario IT/EN per fuzzy matching |
| `configs/presets/<regione>/<coltura>.yaml` | Preset regionali dell'Atlante Agronomico Italiano |
| `data/sample/` | Dataset di esempio per demo ed E2E |

### Test (`tests/`)

I test seguono lo schema `test_<modulo>.py`. I test E2E (`test_e2e.py`, `test_real_e2e.py`) coprono l'intera pipeline `Loader → Cleaner → Tensorizer → Export`. Attuale: **75 test passanti, copertura 88%**.

### Build, CI, docs

| File | Ruolo |
|------|-------|
| `pyproject.toml` | Dipendenze, metadata pacchetto, config `ruff`/`black`/`pytest`/`mypy` |
| `.pre-commit-config.yaml` | Hook locali allineati con CI (black 26, ruff 0.15) |
| `.github/workflows/ci.yml` | CI: lint + mypy + pytest su Python 3.10/3.11/3.12 |
| `mkdocs.yml` | Config sito documentazione |
| `docs/` | Sorgenti MkDocs (getting-started, configuration, screenshots, API) |

### Flusso dati end-to-end (vista compressa)

```
file .xlsx/.csv ──► loader.py ──► matching.py (opt) ──► units.py (opt) ──► DataFrame validato
                                                                                   │
                                                                                   ▼
                                                          cleaner.py ──► pipeline.py ──► transformers.py (×11)
                                                                                   │
                                                                                   ▼
                                                                        DataFrame pulito + diagnostics
                                                                                   │
                                                                                   ▼
                                                          tensorizer.py ──► dataset.py ──► export.py
                                                                                   │
                                                                                   ▼
                                                               bundle .pt + metadata.json + .zip
                                                                                   │
                                                                                   ▼
                                                                   tracking.py (opt) ──► MLflow run
```

---

## 🎯 Obiettivo del progetto

Costruire un tool che risolva un problema molto concreto: fare da ponte fra l'Excel agronomico reale (sporco, disomogeneo, pieno di errori umani e di sensore) e il formato rigido richiesto da un modello di Machine Learning in PyTorch.

Il risultato finale è un bundle `.zip` auto-documentato che contiene i tensor PyTorch, un manifest JSON con tutte le statistiche di trasformazione e un hash del file sorgente per la tracciabilità.

---

## 🏗 I 3 step della pipeline

| Step | Nome | Input | Output |
|------|------|-------|--------|
| 1 | **Loader** | `.xlsx` / `.xls` / `.csv` | `pandas.DataFrame` validato + fingerprint SHA-256 |
| 2 | **Cleaner** | `DataFrame` sporco | `DataFrame` pulito + diagnostica delle operazioni |
| 3 | **Tensorizer** | `DataFrame` pulito | Bundle PyTorch (`.pt` + `.json` + `.zip`) |

---

## 1️⃣ Step 1 — Loader: caricamento dei dati agricoli grezzi

**Obiettivo**: leggere file Excel/CSV reali (anche malformati) e consegnare un `DataFrame` valido, con uno schema minimo garantito e un fingerprint SHA-256 per la tracciabilità.

### 1.1 Definizione dello schema minimo

Prima ancora di scrivere il parser, ho fissato lo **schema minimo obbligatorio** per qualunque input AgriPipe:

```python
REQUIRED_COLUMNS = ["date", "field_id", "temp", "humidity", "ph", "yield"]
```

Questo schema rappresenta il contratto del progetto: se un file non espone queste colonne, non è un input valido. La scelta di una validazione rigida (invece che fuzzy matching nativo) è stata voluta — la tolleranza è opt-in via flag `--fuzzy` (vedi 1.8).

### 1.2 Gestione unificata di Excel e CSV

Il loader auto-rileva estensione e separatore:

```python
@dataclass
class LoaderConfig:
    header_row: int | None = None  # None = auto-detect
    sheet_name: str | int = 0
    csv_separator: str = ","
```

Se `header_row` è `None`, il loader cerca automaticamente la prima riga che contiene almeno 3 delle colonne obbligatorie. Questo permette di gestire gli Excel "aziendali" con 2-4 righe di intestazione sopra i dati veri (logo, autore, filiale, ecc.).

### 1.3 Riconoscimento delle intestazioni sporche

L'euristica di auto-detect legge le prime 15 righe del file "come dati" (senza header), poi per ogni riga calcola un punteggio di somiglianza con lo schema obbligatorio. La riga con score più alto diventa l'header; le righe precedenti vengono scartate come rumore.

```python
def _detect_header_row(df: pd.DataFrame, expected: list[str]) -> int:
    for i in range(min(15, len(df))):
        row_values = {str(v).strip().lower() for v in df.iloc[i].values}
        matches = sum(1 for col in expected if col in row_values)
        if matches >= 3:
            return i
    return 0
```

### 1.4 Fingerprint SHA-256 per la tracciabilità

Ogni file viene "firmato" alla lettura con un hash SHA-256 del contenuto byte-per-byte. Il fingerprint viene incluso nel `metadata.json` del bundle finale: se il file cambia di una sola cella, il bundle prodotto avrà un hash diverso ed è quindi identificabile come non-riproducibile.

```python
def _compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
```

### 1.5 Normalizzazione delle date

La colonna `date` può arrivare in 3 formati diversi nel mondo reale:
- stringa ISO (`"2024-01-15"`)
- stringa localizzata (`"15/01/2024"`)
- **seriale Excel** (un intero come `45123`, che Excel usa internamente)

Il loader li normalizza tutti a `pandas.Timestamp`, riconoscendo il seriale Excel quando il valore è numerico e dentro un range plausibile:

```python
if pd.api.types.is_numeric_dtype(df["date"]):
    df["date"] = pd.to_datetime(df["date"], origin="1899-12-30", unit="D")
else:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
```

### 1.6 Validazione dello schema

Dopo il caricamento, si controlla che tutte le colonne obbligatorie siano presenti; in caso contrario viene sollevata un'eccezione parlante:

```python
missing = set(REQUIRED_COLUMNS) - set(df.columns)
if missing:
    raise ValueError(f"Schema non valido: mancano le colonne {sorted(missing)}")
```

### 1.7 Batch loading da cartella

In produzione, un operatore agritech riceve spesso molti piccoli file (es. uno al giorno) invece di un unico dataset consolidato. Chiamare la CLI per ogni file è inefficiente.

- Funzione `batch_load_raw(input_dir)` in `src/agripipe/loader.py`.
- Riconosce automaticamente file `.xlsx`, `.xls` e `.csv` nella cartella.
- Concatena tutti i file validi in un unico DataFrame.
- Aggiunge una colonna `source_file` per preservare la tracciabilità della riga.
- Opzione `on_error="skip"` per ignorare file corrotti e continuare il caricamento.
- Integrazione CLI via flag `--input-dir` (o `-d`) nel comando `agripipe run`.

```bash
agripipe run --input-dir ./data/daily_exports/ --preset ulivo_ligure --output bundle.pt
```

### 1.8 Fuzzy matching dei nomi colonna

Lo schema rigido è la base del contratto AgriPipe. Ma in produzione un operatore agritech riceve Excel con nomi come `Temperatura_C`, `Temp °C`, `Umidità_%`, `pH_suolo` — varianti perfettamente comprensibili a un umano ma che il loader rigetterebbe.

Layer opzionale di **fuzzy matching** basato su `rapidfuzz`:

- Dizionario IT/EN di sinonimi agronomici in `configs/column_synonyms.yaml`.
- `fuzzy_rename_columns(df, required, synonyms)` in `src/agripipe/matching.py` usa `WRatio` per confrontare ogni colonna coi sinonimi e rinominare quelle che superano lo score threshold (default 85/100).
- Integrazione in `load_raw(path, fuzzy=True)` come **opt-in** (default `False` per retrocompatibilità).
- Flag CLI: `agripipe run --input file.xlsx --preset ulivo_ligure --fuzzy`.

```python
load_raw("excel_italiano.xlsx")              # ValueError: colonne mancanti
load_raw("excel_italiano.xlsx", fuzzy=True)  # Temperatura→temp, Umidità→humidity, Resa→yield
```

**Scelta di design**: il rename è irreversibile e loggato — la tracciabilità è preservata via `df.attrs['file_hash']`, e il matching è explicit (si attiva solo con `fuzzy=True`). Niente magia nascosta.

### 1.9 Conversione automatica a unità SI

Dati da sensori americani, macchinari misti o export multi-fornitore arrivano frequentemente in unità non-SI: Fahrenheit, inch, lb/acre. Un errore di unità è un bug silenzioso che scopri solo a modello addestrato — il tipo di bug più costoso.

Modulo `src/agripipe/units.py` con registry estendibile:

```python
CONVERSIONS = {
    "F_to_C":         lambda x: (x - 32) * 5 / 9,
    "inch_to_mm":     lambda x: x * 25.4,
    "lb_per_acre_to_kg_per_ha": lambda x: x * 1.12085,
}
```

Rilevamento doppio:
1. **per suffisso del nome colonna** (`temp_F`, `rainfall_inch`, `fertilizer_lb_acre`);
2. **per range numerico fuori norma** (es. una `temp` con valori attorno a 80 è quasi certamente Fahrenheit).

Attivazione opt-in via flag CLI `--auto-units` o via `CleanerConfig.auto_unit_conversion = True`.

### ✅ Verifica dello step 1

Test in `tests/test_loader.py`, `tests/test_loader_batch.py`, `tests/test_matching.py`, `tests/test_units.py`:
- Excel con 4 righe di intestazione sporca → header rilevato correttamente
- CSV con separatore `;` → parsing ok
- date in formato seriale Excel → conversione corretta a `Timestamp`
- schema incompleto → `ValueError` con messaggio descrittivo
- hash SHA-256 riproducibile su letture successive dello stesso file
- fuzzy matching: `Temperatura_C` → `temp` con score ≥ 85
- unit conversion: colonna `temp_F=95` → `temp=35.0` in Celsius

---

## 2️⃣ Step 2 — Cleaner: pulizia automatica delle anomalie

**Obiettivo**: prendere un `DataFrame` caricato dal Loader e consegnarlo pronto per la tensorizzazione, con tutte le anomalie rilevate, contate e gestite in modo trasparente.

### 2.1 Configurazione dichiarativa

Il `Cleaner` è pilotato da un `CleanerConfig` che separa la policy dal codice. La config dichiara quali colonne sono numeriche/categoriche/date, la strategia di imputazione, il metodo di outlier detection e gli eventuali bound fisici:

```python
class CleanerConfig(BaseModel):
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    date_columns: list[str] = []
    dedup_keys: list[str] = []
    missing_strategy: ImputationStrategy = "median"
    outlier_method: OutlierMethod = "iqr"
    outlier_iqr_multiplier: float = 1.5
    physical_bounds: dict[str, tuple[float, float]] = {}
    auto_unit_conversion: bool = False
```

La configurazione si carica da YAML (`AgriCleaner.from_yaml(path)`) o da preset regionali predefiniti (`AgriCleaner.from_preset("ulivo_pugliese")`).

### 2.2 Coercizione dei tipi

Prima di qualunque statistica, si forza il tipo corretto per ogni colonna. I valori non convertibili diventano `NaN` e vengono gestiti in fase di imputazione.

```python
for col in config.numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
for col in config.date_columns:
    df[col] = pd.to_datetime(df[col], errors="coerce")
```

### 2.3 Limiti fisici configurabili

Per ogni colonna numerica si possono specificare limiti fisici (`physical_bounds`). I valori fuori range vengono rimossi prima di qualunque altra analisi:

```python
for col, (low, high) in config.physical_bounds.items():
    mask = (df[col] >= low) & (df[col] <= high)
    diagnostics.out_of_bounds_removed += (~mask).sum()
    df = df[mask | df[col].isna()]
```

Esempio: umidità al 150% viene scartata; pH a -3 viene scartato; temperatura a 87°C viene scartata. Il conteggio finisce nei diagnostics.

### 2.4 Rilevamento outlier (IQR o Z-score)

```python
# IQR (default, robusto)
q1, q3 = series.quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
mask = (series >= lower) & (series <= upper)

# Z-score (adatto a distribuzioni gaussiane)
z = (series - series.mean()) / series.std()
mask = z.abs() <= 3
```

Gli outlier finiscono in `diagnostics.outliers_removed`.

### 2.5 Imputazione dei valori mancanti

Quattro strategie supportate: `mean`, `median` (default), `ffill`, `time` (interpolazione temporale).

Se è richiesta `time` ma nessuna colonna data è disponibile, fallback automatico a `median` con nota nei diagnostics.

### 2.6 Imputazione delle categoriche

Le categoriche vengono riempite con la **moda** colonna per colonna.

### 2.7 Deduplicazione

Righe duplicate (per chiavi configurabili, tipicamente `date + field_id`) rimosse mantenendo la prima occorrenza.

### 2.8 Diagnostica completa

Ogni operazione contribuisce a un oggetto `CleanerDiagnostics` restituito insieme al DataFrame pulito. Questo oggetto viene serializzato nel `metadata.json` del bundle finale.

### 2.9 Smontaggio in Pipeline + Transformer

La versione iniziale di `AgriCleaner` era una God-class con tutti i passaggi in metodi privati (`_coerce_types`, `_apply_physical_bounds`, `_impute`, ecc.). All'aumentare dei passaggi è diventata difficile da estendere e testare isolatamente.

Refactor in architettura modulare:

- `src/agripipe/base.py` definisce l'interfaccia `AgriTransformer` (`fit`, `transform`, `fit_transform`) stile scikit-learn.
- `src/agripipe/transformers.py` contiene **11 Transformer concreti disaccoppiati**:
  1. `DateCoercionTransformer`
  2. `AutoUnitConversionTransformer`
  3. `TypeCoercionTransformer`
  4. `PhysicalBoundsTransformer`
  5. `OutlierIQRTransformer`
  6. `OutlierZScoreTransformer`
  7. `NumericImputationTransformer`
  8. `TimeSeriesImputationTransformer`
  9. `CategoricalImputationTransformer`
  10. `DeduplicationTransformer`
  11. `GrowingDegreeDaysTransformer` (calcolo dinamico GDD se la base biologica è impostata)
- `src/agripipe/pipeline.py` espone `Pipeline(steps=[(name, transformer), ...])` che richiama `fit_transform` a cascata.
- `AgriCleaner` diventa una facciata sottile che legge `CleanerConfig` e assembla la `Pipeline` giusta.

**Vantaggi**: ogni Transformer ha il proprio test isolato, si possono disattivare singoli step via config, l'ordine di esecuzione è esplicito e ispezionabile (`pipeline.steps`).

### 2.10 Migrazione `CleanerConfig` da dataclass a Pydantic

`CleanerConfig` era in origine un `@dataclass`. Con l'aumentare dei campi e l'introduzione di preset YAML caricati a runtime è diventato necessario validare i valori (strategie ammesse, range dei moltiplicatori, tipi dei bounds).

La config è stata migrata a `pydantic.BaseModel`:
- validazione automatica al caricamento YAML (`CleanerConfig(**yaml_dict)` solleva `ValidationError` se un campo è invalido)
- tipi `Literal["mean", "median", "ffill", "time"]` al posto di stringhe libere
- `model_config = {"protected_namespaces": ()}` dove necessario per evitare conflitti con prefissi `model_`

### ✅ Verifica dello step 2

Test in `tests/test_cleaner.py`, `tests/test_cleaner_time_imputation.py`:
- umidità a 150% rimossa dai physical_bounds
- outlier di temperatura a 45°C flaggato dall'IQR
- NaN imputati con mediana corrispondono al valore atteso
- fallback da `time` a `median` quando manca la colonna data
- duplicati esatti rimossi, diagnostica coerente
- ogni Transformer ha test isolato che verifica l'idempotenza su input già pulito

---

## 3️⃣ Step 3 — Tensorizer: trasformazione in tensor PyTorch

**Obiettivo**: consegnare un bundle `.pt` pronto per essere caricato in PyTorch, con scaling e encoding riproducibili e un manifest JSON che documenta ogni trasformazione.

### 3.1 Validazione pre-tensorizzazione

Prima di fare qualunque cosa, si verifica che il DataFrame non contenga NaN o Inf residui (il Cleaner dovrebbe averli eliminati, ma la difesa in profondità non costa nulla):

```python
arr = df[cols_to_check].to_numpy(dtype=float, na_value=np.nan)
if not np.isfinite(arr).all():
    raise ValueError(
        "Il DataFrame contiene NaN o Inf nelle colonne usate dal Tensorizer: "
        f"{cols_to_check}. Eseguire prima AgriCleaner.clean()."
    )
```

### 3.2 Scaling delle feature numeriche

Due scaler supportati (via scikit-learn):

```python
scaler = StandardScaler()   # media 0, std 1 (default)
scaler = RobustScaler()     # mediana 0, IQR 1 (resistente a outlier residui)
features_scaled = scaler.fit_transform(df[numeric_cols])
```

I parametri dello scaler (`mean_` e `scale_` per Standard, `center_` e `scale_` per Robust) vengono salvati nel bundle `.pt`: questo permette di applicare la stessa trasformazione al dataset di inferenza senza ri-fittare — **zero Data Leakage**.

### 3.3 Encoding delle categoriche

Due encoder supportati: `LabelEncoder` (una colonna intera, compatto, adatto a tree models) e `OneHotEncoder` (N colonne binarie, adatto a reti neurali).

### 3.4 Split train/val/test

Suddivisione in due passaggi con `train_test_split` per ottenere proporzioni configurabili:

```python
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
val_ratio_adj = val_ratio / (1 - test_ratio)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio_adj, random_state=42)
```

`random_state=42` è fisso per garantire riproducibilità.

### 3.5 Schema hash per il lock

Ogni bundle include uno `schema_lock_hash`: stesse colonne nello stesso ordine → stesso hash. Se la forma del dataset cambia, l'hash cambia e puoi rilevarlo automaticamente.

### 3.6 Costruzione del bundle `.pt`

Il bundle è un dizionario Python serializzato con `torch.save`:

```python
bundle = {
    "features": torch.tensor(X, dtype=torch.float32),
    "target": torch.tensor(y, dtype=torch.float32),
    "feature_names": feature_names,
    "scaler_mean": torch.tensor(mean, dtype=torch.float32),
    "scaler_scale": torch.tensor(scale, dtype=torch.float32),
    "metadata": metadata_dict,
}
torch.save(bundle, output_path)
```

Caricamento lato utente:

```python
bundle = torch.load("agripipe_export.pt", weights_only=False)
```

### 3.7 Export completo (`.pt` + `.json` + `.zip`)

Oltre ai tensor, `src/agripipe/export.py` genera un `metadata.json` con:
- schema delle colonne e unità di misura inferite
- statistiche per colonna (media, std, min, max, quantili)
- matrice di correlazione delle feature numeriche
- diagnostica del Cleaner (righe rimosse, imputazioni applicate)
- fingerprint SHA-256 del file sorgente
- `schema_lock_hash`
- snippet Python di esempio per caricare il bundle in PyTorch

Tutto impacchettato in un singolo `.zip`.

### 3.8 Safety benchmark + logging MLflow

Un bundle tecnicamente "corretto" (nessun NaN, shape giusta) può ancora essere inutile per il ML downstream: se le feature non hanno alcun segnale verso il target, il modello finale non imparerà nulla.

`src/agripipe/tracking.py` addestra automaticamente un **baseline Ridge Regression** sul bundle e ne misura R² e RMSE. Questi valori fungono da *safety benchmark*: se un baseline stupido non predice nulla, il problema non è nel modello — è nei dati.

Se il pacchetto `mlflow` è installato e una tracking URI è configurata, la run viene loggata automaticamente (metriche, parametri, artefatti). L'integrazione è **opt-in e non blocca la pipeline**: se MLflow non è disponibile, il safety benchmark viene comunque calcolato e stampato in console.

### ✅ Verifica dello step 3

Test in `tests/test_tensorizer.py`, `tests/test_tensorizer_robustness.py`, `tests/test_export.py`, `tests/test_data_architect.py`:
- DataFrame con NaN residui → `ValueError` chiaro
- StandardScaler applicato → media ≈ 0 e std ≈ 1 su ogni colonna scalata
- Split 70/15/15 → proporzioni corrette entro tolleranza
- Riproducibilità: stesso input + stesso seed → stesso output
- Caricamento del `.pt` da zero e ricostruzione del `TensorDataset` PyTorch
- Bundle `.zip` contiene `.pt` + `.json` nominati correttamente
- Baseline Ridge calcolato senza crash se `mlflow` non è installato

---

## 4️⃣ Atlante Agronomico Italiano — preset regionali

Il vero valore agronomico di AgriPipe vive nei **preset regionali** sotto `configs/presets/<regione>/<coltura>.yaml`. Ogni preset codifica domain knowledge che un non-agronomo non saprebbe scrivere: range fisici realistici, strategie di imputazione adatte al ciclo biologico, chiavi di deduplicazione pertinenti alla coltura.

Struttura di un preset tipo:

```yaml
crop_display: "Ulivo — zona Colline Liguri"
region: "Liguria"
zona: "Collinare interna"
numeric_columns: [temp, humidity, ph, rainfall, yield]
categorical_columns: [variety, soil_type]
date_columns: [date]
dedup_keys: [date, field_id]
missing_strategy: time
outlier_method: iqr
physical_bounds:
  temp:     [-5, 42]
  humidity: [0, 100]
  ph:       [5.5, 8.5]
  rainfall: [0, 250]
  yield:    [0, 8000]
```

La CLI `agripipe list-presets --region Liguria` scorre tutti i preset esposti, raggruppati per regione. L'obiettivo a lungo termine è coprire le colture principali di tutte le regioni italiane (vite, olivo, grano, riso, agrumi, ortaggi), rendendo AgriPipe il "Wikipedia dei preset agronomici italiani".

---

## 🧭 Verifica E2E sulla pipeline completa

Verifica end-to-end eseguita sul dataset `data/sample/pro_demo.xlsx`:

```
LOADER     → righe caricate, schema valido, SHA-256 calcolato
CLEANER    → physical bounds + outlier IQR + imputazione median applicati, diagnostics coerenti
TENSORIZER → bundle train/val/test generato, .pt + .json + .zip compressi
TRACKING   → safety benchmark Ridge calcolato (R² e RMSE in log)
```

Il bundle viene ricaricato in PyTorch con `torch.load` e trasformato in `TensorDataset` senza ulteriori conversioni. Il ciclo è chiuso. Test E2E in `tests/test_e2e.py` e `tests/test_real_e2e.py` eseguono l'intero flusso a ogni push in CI.

---

## 🧪 Disciplina di sviluppo

Pratiche applicate in modo sistematico:

- **TDD** — ogni funzione pubblica ha un test scritto prima dell'implementazione. Copertura attuale: **88% su 75 test**.
- **Commit atomici e convenzionali** — ogni commit copre una sola responsabilità, messaggi secondo [Conventional Commits](https://www.conventionalcommits.org) (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`).
- **Linting e formattazione automatici** — `ruff` per il linting, `black` per la formattazione, `mypy` per i tipi. Tutti bloccanti in CI. `pre-commit` locale pinnato sulle stesse versioni di CI per evitare drift di formattazione.
- **Type hints e docstrings Google-style** — tutte le API pubbliche sono tipizzate e documentate.
- **Branch separati per agenti AI** — `gemini-cli` per Gemini, `claude-code` per Claude; merge su `main` solo via Pull Request.

La CI GitHub Actions esegue lint + mypy + suite test su Python 3.10, 3.11 e 3.12 a ogni push e Pull Request verso `main`.

---

## 📂 Struttura finale del progetto

```
agripipe/
├── src/agripipe/
│   ├── __init__.py
│   ├── loader.py          # Step 1 — Excel/CSV → DataFrame validato
│   ├── matching.py        # Step 1 (opt) — fuzzy rename colonne
│   ├── units.py           # Step 1 (opt) — conversione unità SI
│   ├── base.py            # Interfaccia AgriTransformer
│   ├── transformers.py    # Step 2 — 11 Transformer concreti
│   ├── pipeline.py        # Step 2 — orchestratore Pipeline
│   ├── cleaner.py         # Step 2 — facciata AgriCleaner + CleanerConfig (Pydantic)
│   ├── tensorizer.py      # Step 3 — scaling, encoding, split
│   ├── dataset.py         # Step 3 — AgriDataset PyTorch
│   ├── export.py          # Step 3 — bundle .pt + .json + .zip
│   ├── metadata.py        # Step 3 — generazione metadata.json
│   ├── tracking.py        # Step 3 (opt) — baseline Ridge + MLflow
│   ├── report.py          # Report HTML di qualità
│   ├── synth.py           # Generatore Excel sintetici
│   ├── cli.py             # CLI Typer
│   └── utils/logging_setup.py
├── tests/                 # 75 test, copertura 88%
├── configs/
│   ├── default.yaml
│   ├── column_synonyms.yaml
│   └── presets/<regione>/<coltura>.yaml
├── data/sample/           # Dataset di esempio per demo e test E2E
├── pages/                 # Streamlit multi-page
│   ├── 1_📥_Ingestion.py
│   ├── 2_🧹_Refinery.py
│   └── 3_📦_Tensorizer.py
├── docs/                  # MkDocs source
├── app.py                 # Entry point Streamlit
├── pyproject.toml
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── README.md              # Documentazione utente (IT)
├── README.en.md           # Documentazione utente (EN)
└── DOCUMENTAZIONE_PROGETTO.md  # Questo file
```

---

<sub>Questo documento è la mappa e la cronaca del progetto AgriPipe. Va aggiornato a ogni feature significativa e usato come sorgente canonica per la scrittura di `README.md` e `README.en.md`.</sub>
