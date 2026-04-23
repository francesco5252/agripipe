# 📝 Documentazione di Progetto — AgriPipe

> Percorso passo-passo seguito per costruire **AgriPipe**: una pipeline riproducibile che porta dati agricoli grezzi da Excel a tensor PyTorch validati.
>
> Il documento è organizzato nei **3 step cardine** della pipeline: **Loader** → **Cleaner** → **Tensorizer**. Ogni step è suddiviso nei sotto-passaggi implementativi nell'ordine in cui sono stati realizzati.

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

Questo schema rappresenta il contratto del progetto: se un file non espone queste colonne, non è un input valido. La scelta di una validazione rigida (invece che fuzzy matching) è stata voluta — vedi sezione "Limiti noti" nel README.

### 1.2 Gestione unificata di Excel e CSV

Ho implementato il `LoaderConfig` (dataclass) con auto-detect del formato:

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

Ho aggiunto la funzionalità di **batch loading**:

- Nuova funzione `batch_load_raw(input_dir)` in `src/agripipe/loader.py`.
- Riconosce automaticamente file `.xlsx`, `.xls` e `.csv` nella cartella.
- Concatena tutti i file validi in un unico DataFrame.
- Aggiunge una colonna `source_file` per preservare la tracciabilità della riga.
- Opzione `on_error="skip"` per ignorare file corrotti e continuare il caricamento.
- Integrazione CLI via flag `--input-dir` (o `-d`) nel comando `agripipe run`.

```bash
# Esempio d'uso
agripipe run --input-dir ./data/daily_exports/ --preset ulivo_ligure --output bundle.pt
```

Questa aggiunta sposta AgriPipe da tool per singoli file a piccola data pipeline per batch di dati reali.

### 1.8 Fuzzy matching dei nomi colonna

Lo schema rigido è la base del contratto AgriPipe: se mancano colonne, errore. Ma in produzione un operatore agritech riceve Excel con nomi come `Temperatura_C`, `Temp °C`, `Umidità_%`, `pH_suolo` — varianti perfettamente comprensibili a un umano ma che il loader rigetterebbe.

Ho aggiunto un layer opzionale di **fuzzy matching** basato su `rapidfuzz`:

- Dizionario IT/EN di sinonimi agronomici in `configs/column_synonyms.yaml`.
- Funzione `fuzzy_rename_columns(df, required, synonyms)` in `src/agripipe/matching.py` che usa `WRatio` per confrontare ogni colonna coi sinonimi e rinominare quelle che superano lo score threshold (default 85/100).
- Integrazione in `load_raw(path, fuzzy=True)` come **opt-in** (default `False` per retrocompatibilità).
- Flag CLI: `agripipe run --input file.xlsx --preset ulivo_ligure --fuzzy`.

```python
# Prima (rigetta):
load_raw("excel_italiano.xlsx")  # ValueError: Colonne mancanti ['temp', 'humidity', 'yield']

# Dopo (riconosce e rinomina):
load_raw("excel_italiano.xlsx", fuzzy=True)
# Temperatura → temp, Umidità → humidity, Resa → yield
```

**Scelta di design**: il rename è irreversibile e loggato — la tracciabilità è preservata via `df.attrs['file_hash']`, e il matching è explicit (si attiva solo con `fuzzy=True`). Niente magia nascosta.

### ✅ Verifica dello step 1

Test unitari in `tests/test_loader.py`:
- lettura di Excel con 4 righe di intestazione sporca → header rilevato correttamente
- lettura di CSV con separatore `;` → parsing ok
- lettura di Excel con date in formato seriale → conversione corretta a `Timestamp`
- schema incompleto → `ValueError` con messaggio descrittivo
- hash SHA-256 riproducibile su letture successive dello stesso file

---

## 2️⃣ Step 2 — Cleaner: pulizia automatica delle anomalie

**Obiettivo**: prendere un `DataFrame` caricato dal Loader e consegnarlo pronto per la tensorizzazione, con tutte le anomalie rilevate, contate e gestite in modo trasparente.

### 2.1 Configurazione dichiarativa

Il `Cleaner` è pilotato da un `CleanerConfig` (dataclass) che separa la policy dal codice:

```python
@dataclass
class CleanerConfig:
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    date_columns: list[str] = field(default_factory=list)
    dedup_keys: list[str] = field(default_factory=list)
    missing_strategy: ImputationStrategy = "median"
    outlier_method: OutlierMethod = "iqr"
    outlier_iqr_multiplier: float = 1.5
    physical_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
```

La configurazione può essere caricata da YAML (`Cleaner.from_yaml(path)`) oppure da preset regionali predefiniti (`Cleaner.from_preset("ulivo_pugliese")`), pensati per coprire casi tipici dell'agricoltura italiana.

### 2.2 Coercizione dei tipi

Prima di qualunque statistica, si forza il tipo corretto per ogni colonna:

```python
def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
    for col in self.config.numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in self.config.date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df
```

I valori non convertibili diventano `NaN` e vengono gestiti in fase di imputazione. Questo è fondamentale: garantisce che dopo il Cleaner tutte le colonne numeriche siano effettivamente `float64`.

### 2.3 Limiti fisici configurabili

Per ogni colonna numerica si possono specificare limiti fisici (`physical_bounds`). I valori fuori range vengono rimossi prima di qualunque altra analisi:

```python
def _apply_physical_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
    for col, (low, high) in self.config.physical_bounds.items():
        if col in df.columns:
            mask = (df[col] >= low) & (df[col] <= high)
            self.diagnostics.out_of_bounds_removed += (~mask).sum()
            df = df[mask | df[col].isna()]
    return df
```

Esempio: umidità al 150% viene scartata; pH a -3 viene scartato; temperatura a 87°C viene scartata. Il conteggio finisce nei diagnostics.

### 2.4 Rilevamento outlier (IQR o Z-score)

Due metodi statistici sono supportati:

```python
# IQR (default, robusto agli outlier estremi)
q1, q3 = series.quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
mask = (series >= lower) & (series <= upper)

# Z-score (più permissivo, adatto a distribuzioni gaussiane)
z = (series - series.mean()) / series.std()
mask = z.abs() <= 3
```

Gli outlier non vengono eliminati in silenzio: finiscono in `diagnostics.outliers_removed`.

### 2.5 Imputazione dei valori mancanti

Quattro strategie supportate:
- `mean` — media della colonna
- `median` — mediana (default, più robusto)
- `ffill` — forward-fill (propaga l'ultimo valore valido)
- `time` — interpolazione temporale (richiede una colonna data)

Se è richiesta `time` ma nessuna colonna data è disponibile, c'è un fallback automatico a `median` con avviso nei diagnostics:

```python
if self.config.missing_strategy == "time" and not date_col:
    self.diagnostics.imputation_strategy_used = "median (fallback da time)"
    strategy = "median"
```

### 2.6 Imputazione delle categoriche

Le categoriche vengono trattate separatamente con la **moda**:

```python
def _impute_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
    for col in self.config.categorical_columns:
        if col in df.columns and df[col].isna().any():
            mode = df[col].mode()
            if len(mode) > 0:
                df[col] = df[col].fillna(mode.iloc[0])
    return df
```

### 2.7 Deduplicazione

Le righe duplicate (per chiavi configurabili, tipicamente `date + field_id`) vengono rimosse mantenendo la prima occorrenza:

```python
before = len(df)
df = df.drop_duplicates(subset=self.config.dedup_keys, keep="first")
self.diagnostics.duplicates_removed += before - len(df)
```

### 2.8 Diagnostica completa

Ogni operazione contribuisce a un oggetto `CleanerDiagnostics` restituito insieme al DataFrame pulito:

```python
@dataclass
class CleanerDiagnostics:
    total_rows: int = 0
    current_preset_name: str | None = None
    imputation_strategy_used: str = ""
    values_imputed: int = 0
    outliers_removed: int = 0
    out_of_bounds_removed: int = 0
    duplicates_removed: int = 0
```

Questo oggetto viene serializzato nel `metadata.json` del bundle finale: l'utente può sempre risalire a quante righe sono state rimosse e perché.

### ✅ Verifica dello step 2

Test unitari in `tests/test_cleaner.py`:
- umidità a 150% viene rimossa dai physical_bounds
- outlier di temperatura a 45°C in un dataset estivo normale viene flaggato dall'IQR
- NaN imputati con mediana corrispondono al valore atteso
- fallback da `time` a `median` quando manca la colonna data
- duplicati esatti vengono rimossi, diagnostica coerente

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
# StandardScaler: media 0, std 1 (default)
scaler = StandardScaler()

# RobustScaler: mediana 0, IQR 1 (resistente a outlier residui)
scaler = RobustScaler()

features_scaled = scaler.fit_transform(df[numeric_cols])
```

I parametri dello scaler (`mean_` e `scale_` per Standard, `center_` e `scale_` per Robust) vengono salvati nel bundle `.pt`. Questo permette di applicare la stessa trasformazione al dataset di inferenza senza ri-fittare:

```python
def _scaler_params(scaler) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(scaler, StandardScaler):
        return scaler.mean_, scaler.scale_
    if isinstance(scaler, RobustScaler):
        return scaler.center_, scaler.scale_
```

### 3.3 Encoding delle categoriche

Due encoder supportati:
- `LabelEncoder` — una colonna intera per feature (compatto, adatto a tree models)
- `OneHotEncoder` — N colonne binarie per feature (adatto a reti neurali)

L'encoder scelto viene applicato in-place e le nuove colonne concatenate alle feature numeriche già scalate.

### 3.4 Split train/val/test

Suddivisione in due passaggi con `train_test_split` per ottenere proporzioni configurabili:

```python
# Primo split: separa il test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=test_ratio, random_state=42
)
# Secondo split: separa train da val nel resto
val_ratio_adj = val_ratio / (1 - test_ratio)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_ratio_adj, random_state=42
)
```

`random_state=42` è fisso per garantire riproducibilità: stesso input → stesso split.

### 3.5 Schema hash per il lock

Ogni bundle include un `schema_lock_hash`: se domani riprocessi un file con le stesse colonne nello stesso ordine, otterrai lo stesso hash. Se la forma del dataset cambia, l'hash cambia e puoi rilevarlo in automatico:

```python
def _compute_schema_hash(columns: list[str]) -> str:
    schema_str = ",".join(sorted(str(c) for c in columns))
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]
```

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

Il caricamento lato utente è una sola riga:

```python
bundle = torch.load("agripipe_export.pt", weights_only=False)
```

### 3.7 Export completo (`.pt` + `.json` + `.zip`)

Oltre ai tensor, viene generato un `metadata.json` con:
- schema delle colonne e unità di misura inferite
- statistiche per colonna (media, std, min, max, quantili)
- matrice di correlazione delle feature numeriche
- diagnostica del Cleaner (righe rimosse, imputazioni applicate)
- fingerprint SHA-256 del file sorgente
- `schema_lock_hash`
- snippet Python di esempio per caricare il bundle in PyTorch

Tutto viene infine impacchettato in un singolo `.zip` per la distribuzione:

```python
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(pt_path, arcname=pt_path.name)
    zf.write(json_path, arcname=json_path.name)
```

### ✅ Verifica dello step 3

Test unitari in `tests/test_tensorizer.py`:
- DataFrame con NaN residui → `ValueError` chiaro
- StandardScaler applicato → media ≈ 0 e std ≈ 1 su ogni colonna scalata
- Split 70/15/15 → proporzioni corrette entro tolleranza
- Riproducibilità: stesso input + stesso seed → stesso output
- Caricamento del `.pt` da zero e ricostruzione del `TensorDataset` PyTorch

---

## 🧭 Verifica E2E sulla pipeline completa

A valle dei tre step ho eseguito una verifica end-to-end sul dataset `data/sample/pro_demo.xlsx` (100 righe, 12 colonne):

```
LOADER     → 100 righe caricate, schema valido, SHA-256 calcolato
CLEANER    → 4 valori out-of-bounds, 37 outlier (IQR), 41 imputazioni (median)
TENSORIZER → bundle train/val/test generato, .pt + .json + .zip (~9 KB)
```

Il bundle è stato ricaricato in PyTorch con `torch.load` e trasformato in `TensorDataset` senza ulteriori conversioni. Il ciclo è chiuso.

---

## 🧪 Disciplina di sviluppo

Quattro pratiche applicate in modo sistematico durante tutto il progetto:

- **TDD (Test-Driven Development)** — ogni funzione pubblica ha un test scritto prima dell'implementazione. Copertura attuale: ~82% sui 38 test.
- **Commit atomici e convenzionali** — ogni commit copre una sola responsabilità, con messaggi secondo [Conventional Commits](https://www.conventionalcommits.org) (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`).
- **Linting e formattazione automatici** — `ruff` per il linting, `black` per la formattazione. Entrambi bloccanti in CI.
- **Type hints e docstrings Google-style** — tutte le API pubbliche sono tipizzate e documentate, così la pipeline è leggibile senza leggere il codice.

La CI GitHub Actions esegue l'intera suite di test + lint + mkdocs build su Python 3.10, 3.11 e 3.12 a ogni push sul branch `main`.

---

## 📂 Struttura finale del progetto

```
agripipe/
├── src/agripipe/
│   ├── __init__.py
│   ├── loader.py          # Step 1 — lettura Excel/CSV + validazione schema
│   ├── cleaner.py         # Step 2 — bounds, outlier, imputazione, dedup
│   ├── tensorizer.py      # Step 3 — scaling, encoding, split, bundle
│   ├── cli.py             # Interfaccia a riga di comando (Typer)
│   └── presets/           # Preset YAML regionali (ulivo, vite, grano, ...)
├── tests/
│   ├── test_loader.py
│   ├── test_cleaner.py
│   └── test_tensorizer.py
├── data/sample/           # Dataset di esempio per demo e test E2E
├── docs/screenshots/      # Screenshot della UI Streamlit
├── app.py                 # UI Streamlit a 3 step
├── pyproject.toml         # Config progetto + dipendenze
├── README.md              # Documentazione utente (IT)
├── README.en.md           # Documentazione utente (EN)
└── DOCUMENTAZIONE_PROGETTO.md  # Questo file
```

---

<sub>Questo log documenta il percorso di sviluppo di AgriPipe come esperienza di ML-Ops rigorosa: dalla definizione dello schema alla produzione del bundle riproducibile. Ogni step è verificabile, ogni operazione è tracciata.</sub>
