# Handoff per Gemini CLI — AgriPipe Pro Upgrade

> **Lettore**: Gemini CLI (Gemini 2.5 Pro), dispiegato per completare 3 task meccanici del piano AgriPipe Pro Upgrade mentre Claude Code lavora sui moduli core in parallelo.

## Contesto progetto

**AgriPipe** è una pipeline Python che trasforma file Excel agronomici grezzi in tensor PyTorch pronti per Machine Learning.

- **Stack**: pandas, PyTorch, Streamlit, pytest
- **Target**: X Farm — agronomi professionisti italiani
- **Philosophy**: "Come posso far risparmiare tempo all'agronomo di X Farm oggi?"
- **Repo**: https://github.com/francesco5252/agripipe

Stato attuale: Task 1-3 del piano (CleanerDiagnostics dataclass + counting + time-series imputation) sono già completati e mergiati su `main`.

## Branch strategy — IMPORTANTE

- **Lavora SOLO sul branch `gemini-cli`**. È già configurato e tracka `origin/gemini-cli`.
- Verifica con: `git branch --show-current` → deve dire `gemini-cli`
- Se per errore sei su un altro branch: `git checkout gemini-cli`
- Al termine: `git push origin gemini-cli` + apri Pull Request verso `main` su GitHub

## Divisione del lavoro Claude vs Gemini

**Claude Code** sta lavorando su `claude-code` branch su questi moduli (NON toccarli):
- `src/agripipe/sustainability.py` (NEW)
- `src/agripipe/metadata.py` (NEW)
- `src/agripipe/export.py` (NEW)
- `src/agripipe/ui/theme.py` (NEW)
- `src/agripipe/ui/components.py` (NEW)
- `src/agripipe/app.py` (REWRITE)
- `src/agripipe/cleaner.py` (già modificato Task 1-3, Claude lo integrerà)

**Gemini CLI** (tu) si occupa di:
- `configs/agri_knowledge.yaml` (Task 7 + 8 — expand presets)
- `src/agripipe/indices.py` (Task 16 — docstring)
- `src/agripipe/loader.py` (Task 16 — docstring)
- `src/agripipe/tensorizer.py` (Task 16 — docstring)
- `src/agripipe/dataset.py` (Task 16 — docstring)
- `tests/test_presets_load.py` (Task 8 — nuovo test)

## Regole operative

1. **TDD dove applicabile**: Task 8 include test nuovi. Scrivi il test prima, verifica che fallisca, poi implementa, verifica che passi, commit. Task 7 e 16 sono additivi (non rompono test esistenti) — basta verificare che `pytest` resti verde dopo ogni modifica.

2. **Comandi Python su Windows**: usa sempre `python -m pytest` (NON `pytest` diretto). Lo stesso per altri tool: `python -m ruff`, `python -m black`.

3. **Commit frequenti**: un commit per step logico. Messaggio in formato Conventional Commits:
   ```
   <type>(<scope>): <subject>
   ```
   Esempi concreti che userai:
   - `feat(config): add region and crop_display to existing presets`
   - `feat(config): add 7 curated presets (10 regions, 12 total) + crop rules`
   - `test(presets): verify all regional presets load with required fields`
   - `docs: adopt Google-style docstrings on core modules`

   **Nota firma**: non aggiungere `Co-Authored-By: Claude` — sei Gemini, firma normalmente o usa il tuo pattern standard.

4. **Verifica test prima di ogni commit**: `python -m pytest --tb=short` deve restare verde. Baseline attuale: 20/20 passed.

5. **Nessuna refactorizzazione non richiesta**. Fai SOLO ciò che la spec chiede. Niente "migliorie" opportunistiche.

6. **Encoding file**: tutti i file vanno letti/scritti in UTF-8. Il codice è misto italiano/inglese — mantieni coerenza con quanto trovi nei file.

---

## 🔨 Task 7 — Espandere YAML con `region` e `crop_display`

### File da modificare
- `configs/agri_knowledge.yaml`

### Cosa fare

Il YAML attuale ha 5 preset sotto `regional_presets:` (ulivo_ligure, ulivo_pugliese, grano_siciliano, grano_emiliano, vite_piemontese). Per ognuno aggiungi 2 nuovi campi: `region` e `crop_display`. Lascia invariati tutti gli altri campi esistenti.

#### Contenuto finale dei 5 preset

```yaml
  ulivo_ligure:
    region: "Liguria"
    crop: "olive"
    crop_display: "Olivo DOP Taggiasca"
    zona: "Liguria (Riviera)"
    suolo_tessitura: "Medio impasto / Scheletro"
    max_yield: 5.0
    temp_range: [-5, 38]
    ideal_ph: [6.5, 7.5]
    note: "Focus su terreni terrazzati e bassa meccanizzazione."

  ulivo_pugliese:
    region: "Puglia"
    crop: "olive"
    crop_display: "Olivo Intensivo (Salento)"
    zona: "Puglia (Salento/Bari)"
    suolo_tessitura: "Terra rossa / Calcareo"
    max_yield: 12.0
    temp_range: [-2, 45]
    ideal_ph: [7.0, 8.5]
    salinity_tolerance: 4.0
    note: "Pianure assolate, alta tolleranza al calore."

  grano_siciliano:
    region: "Sicilia"
    crop: "durum_wheat"
    crop_display: "Grano Duro Antico Siciliano"
    zona: "Sicilia (Entroterra)"
    suolo_tessitura: "Argilloso (Vertisuoli)"
    max_yield: 4.5
    temp_range: [-5, 45]
    harvest_months: [5, 6]
    ideal_ph: [7.5, 8.5]
    note: "Grano duro antico, suoli pesanti e fessurati in estate."

  grano_emiliano:
    region: "Emilia-Romagna"
    crop: "soft_wheat"
    crop_display: "Grano Tenero Emiliano"
    zona: "Emilia-Romagna (Pianura Padana)"
    suolo_tessitura: "Limoso / Alluvionale"
    max_yield: 10.0
    temp_range: [-15, 35]
    harvest_months: [6, 7]
    ideal_ph: [6.0, 7.5]
    note: "Grano tenero per panificazione, suoli fertili e freschi."

  vite_piemontese:
    region: "Piemonte"
    crop: "wine_grape_docg"
    crop_display: "Vite Nebbiolo (Barolo DOCG)"
    zona: "Langhe / Roero"
    suolo_tessitura: "Marne / Argille calcaree"
    max_yield: 8.0
    temp_range: [-15, 36]
    harvest_months: [9, 10]
    ideal_ph: [7.0, 8.0]
    note: "Grandi vini rossi (DOCG), focus su microclimi di collina."
```

### Verifica

```bash
python -c "import yaml; yaml.safe_load(open('configs/agri_knowledge.yaml', encoding='utf-8'))"
python -m pytest --tb=short
```
Entrambi devono eseguire senza errori. Test esistenti: 20 passed.

### Commit
```bash
git add configs/agri_knowledge.yaml
git commit -m "feat(config): add region and crop_display to existing presets"
```

---

## 🔨 Task 8 — Aggiungere 7 nuovi preset regionali + test verifiche

### File da modificare
- `configs/agri_knowledge.yaml` (append 7 nuovi preset sotto `regional_presets:`, dopo `vite_piemontese`)
- `tests/test_presets_load.py` (NUOVO file)

### Step 1: Append dei 7 nuovi preset

Aggiungi **dopo `vite_piemontese`** (e prima della sezione top-level `crops:` se presente):

```yaml
  # --- LOMBARDIA ---
  riso_lombardo:
    region: "Lombardia"
    crop: "rice"
    crop_display: "Riso Carnaroli"
    zona: "Lomellina (Pavia)"
    suolo_tessitura: "Sommerso / Franco-limoso"
    max_yield: 7.0
    temp_range: [-8, 35]
    harvest_months: [9, 10]
    ideal_ph: [5.5, 7.0]
    note: "Risaie allagate, richiede gestione idrica precisa."

  # --- VENETO ---
  prosecco_veneto:
    region: "Veneto"
    crop: "wine_grape_docg"
    crop_display: "Vite Prosecco DOCG"
    zona: "Valdobbiadene / Conegliano"
    suolo_tessitura: "Vulcanico / Marnoso-arenaceo"
    max_yield: 13.5
    temp_range: [-10, 35]
    harvest_months: [8, 9]
    ideal_ph: [6.5, 7.5]
    note: "Colline UNESCO, microclimi ventilati."

  # --- TRENTINO-ALTO ADIGE ---
  mela_trentina:
    region: "Trentino-Alto Adige"
    crop: "apple"
    crop_display: "Mela Melinda DOP"
    zona: "Val di Non"
    suolo_tessitura: "Morenico / Ghiaioso"
    max_yield: 60.0
    temp_range: [-20, 32]
    harvest_months: [9, 10]
    ideal_ph: [6.0, 7.0]
    note: "Richiede fabbisogno freddo invernale (>1200 ore)."

  # --- EMILIA-ROMAGNA (secondo preset) ---
  pomodoro_emiliano:
    region: "Emilia-Romagna"
    crop: "tomato"
    crop_display: "Pomodoro da Industria"
    zona: "Parma / Piacenza"
    suolo_tessitura: "Alluvionale / Limoso"
    max_yield: 90.0
    temp_range: [-5, 38]
    harvest_months: [7, 8, 9]
    ideal_ph: [6.0, 7.0]
    note: "Pianura Padana irrigua, filiera industriale."

  # --- TOSCANA ---
  vite_toscana:
    region: "Toscana"
    crop: "wine_grape_docg"
    crop_display: "Vite Chianti DOCG (Sangiovese)"
    zona: "Chianti Classico"
    suolo_tessitura: "Galestro / Alberese"
    max_yield: 9.0
    temp_range: [-10, 38]
    harvest_months: [9, 10]
    ideal_ph: [6.5, 7.5]
    note: "Colline toscane, varietà Sangiovese."

  ulivo_toscano:
    region: "Toscana"
    crop: "olive"
    crop_display: "Olivo DOP Toscano"
    zona: "Colline Pisane/Senesi"
    suolo_tessitura: "Argilloso calcareo"
    max_yield: 4.5
    temp_range: [-8, 38]
    ideal_ph: [6.5, 7.5]
    note: "Olio IGP, cultivar Frantoio/Leccino/Moraiolo."

  # --- CAMPANIA ---
  pomodoro_sanmarzano:
    region: "Campania"
    crop: "tomato"
    crop_display: "Pomodoro San Marzano DOP"
    zona: "Agro Nocerino-Sarnese"
    suolo_tessitura: "Vulcanico (Vesuviano)"
    max_yield: 50.0
    temp_range: [0, 40]
    harvest_months: [7, 8, 9]
    ideal_ph: [6.0, 7.0]
    note: "Pomodoro lungo DOP, coltivazione tradizionale."
```

### Step 2: Estendere la sezione `crops:`

Il YAML attuale (mergiato da Task 1-3) ha già una sezione `crops:` al top-level con 4 colture (olive, durum_wheat, soft_wheat, wine_grape_docg) + `general:`. I 7 nuovi preset introducono 3 nuove colture: `rice`, `apple`, `tomato`. Aggiungile alla sezione `crops:` esistente (inserisci prima della sezione `general:`):

```yaml
  rice:
    t_base: 12.0
    max_yield: 9.0
    flowering_months: [7, 8]
    frost_danger_months: [4, 5]
    critical_temp_flowering: 35

  apple:
    t_base: 4.0
    max_yield: 70.0
    flowering_months: [4, 5]
    frost_danger_months: [4]
    critical_temp_flowering: 35
    chill_hours_required: 1200

  tomato:
    t_base: 10.0
    max_yield: 100.0
    flowering_months: [6, 7]
    frost_danger_months: [4, 5]
    critical_temp_flowering: 38
```

Verifica sintassi YAML:
```bash
python -c "import yaml; d = yaml.safe_load(open('configs/agri_knowledge.yaml', encoding='utf-8')); print('Presets:', len(d['regional_presets'])); print('Crops:', list(d['crops'].keys()))"
```
Deve stampare: `Presets: 12` e `Crops: ['olive', 'durum_wheat', 'soft_wheat', 'wine_grape_docg', 'rice', 'apple', 'tomato']`.

### Step 3: Scrivere il test nuovo (TDD)

Crea **file nuovo** `tests/test_presets_load.py`:

```python
"""Smoke test: tutti i preset territoriali sono validi e caricabili."""

from pathlib import Path

import yaml


def test_all_presets_have_required_fields():
    data = yaml.safe_load(Path("configs/agri_knowledge.yaml").read_text(encoding="utf-8"))
    presets = data["regional_presets"]
    assert len(presets) >= 12
    for key, p in presets.items():
        assert "region" in p, f"{key}: missing region"
        assert "crop" in p, f"{key}: missing crop"
        assert "crop_display" in p, f"{key}: missing crop_display"
        assert "max_yield" in p, f"{key}: missing max_yield"
        assert "ideal_ph" in p, f"{key}: missing ideal_ph"


def test_presets_cover_at_least_ten_regions():
    data = yaml.safe_load(Path("configs/agri_knowledge.yaml").read_text(encoding="utf-8"))
    regions = {p["region"] for p in data["regional_presets"].values()}
    assert len(regions) >= 10, f"Solo {len(regions)} regioni: {regions}"


def test_crops_biological_rules_exist():
    data = yaml.safe_load(Path("configs/agri_knowledge.yaml").read_text(encoding="utf-8"))
    crops = data.get("crops", {})
    for crop_key in ["olive", "durum_wheat", "soft_wheat", "wine_grape_docg",
                     "rice", "apple", "tomato"]:
        assert crop_key in crops, f"Missing biological rules for {crop_key}"
        assert "t_base" in crops[crop_key]
```

### Step 4: Verifica

```bash
python -m pytest tests/test_presets_load.py -v
```
Deve dare: **3 passed**.

```bash
python -m pytest --tb=short
```
Deve dare: **23 passed** (20 esistenti + 3 nuovi).

> ⚠️ Nota: il conteggio "10 regioni" del secondo test si basa sui 12 preset finali. Verifica se è vero contando le region uniche: Liguria, Puglia, Sicilia, Emilia-Romagna, Piemonte, Lombardia, Veneto, Trentino-Alto Adige, Toscana, Campania = **10 regioni**. OK.

### Commit
```bash
git add configs/agri_knowledge.yaml tests/test_presets_load.py
git commit -m "feat(config): add 7 curated presets (10 regions, 12 total) + crop rules"
```

---

## 🔨 Task 16 — Docstring Google-style sui 4 moduli core

### File da modificare
- `src/agripipe/indices.py`
- `src/agripipe/loader.py`
- `src/agripipe/tensorizer.py`
- `src/agripipe/dataset.py`

### Principio generale

Usa il formato Google-style con blocchi `Args:`, `Returns:`, `Raises:`, `Example:` (dove applicabile). Mantieni la lingua italiana nelle spiegazioni semantiche ma le keyword di sezione (Args/Returns/Raises/Example) in inglese.

### Step 1: `src/agripipe/indices.py`

Sostituisci la docstring della funzione `compute_agronomic_indices` con:

```python
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
```

### Step 2: `src/agripipe/loader.py`

Leggi la docstring esistente di `load_raw`. Se **già segue Google style** (ha blocchi Args/Returns/Raises), non modificare. Se no, refattorizza per conformità. Esempio di conformità accettabile:

```python
def load_raw(path: str | Path) -> pd.DataFrame:
    """Carica e valida un file Excel/CSV agronomico.

    Args:
        path: Percorso al file ``.xlsx``, ``.xls``, o ``.csv``.

    Returns:
        DataFrame grezzo con tipi inferiti da pandas.

    Raises:
        FileNotFoundError: Se il path non esiste.
        ValueError: Se il formato non è supportato o il file è vuoto.
    """
```

### Step 3: `src/agripipe/tensorizer.py`

**Sostituisci la docstring della classe `Tensorizer`**:

```python
class Tensorizer:
    """Converte un DataFrame pulito in tensor PyTorch normalizzati.

    Normalizza le colonne numeriche con ``StandardScaler`` (media 0, dev std 1)
    e codifica le colonne categoriche con ``LabelEncoder``. Serializzabile per
    inferenza futura.

    Attributes:
        numeric_columns: Colonne continue da normalizzare.
        categorical_columns: Colonne categoriche da codificare.
        target: Colonna target (opzionale, esclusa dalle features).
        target_dtype: Tipo del tensor target (``"float32"`` | ``"long"``).
        scaler: ``StandardScaler`` fit sui numeric_columns.
        encoders: Dict ``{col_name: LabelEncoder}`` per le categoriche.
    """
```

**E la docstring del metodo `transform`**:

```python
    def transform(self, df: pd.DataFrame) -> TensorBundle:
        """Applica la trasformazione già fit a un nuovo DataFrame.

        Args:
            df: DataFrame con le stesse colonne usate in ``fit_transform``.

        Returns:
            ``TensorBundle`` con ``features``, ``target`` (o None) e
            ``feature_names``.

        Raises:
            RuntimeError: Se ``_fit`` non è ancora stato chiamato.
            ValueError: Se il tensor risultante contiene NaN/Inf.
        """
```

### Step 4: `src/agripipe/dataset.py`

**Sostituisci la docstring della classe `AgriDataset`**:

```python
class AgriDataset(Dataset):
    """PyTorch Dataset costruito sopra un DataFrame già pulito.

    Wrapper leggero attorno a ``Tensorizer`` che espone ``__len__`` e
    ``__getitem__`` per l'uso con ``torch.utils.data.DataLoader``.

    Attributes:
        tensorizer: Istanza di ``Tensorizer`` fit sui dati.
        features: Tensor 2D ``[N, D]`` float32, già normalizzato.
        target: Tensor 1D ``[N]`` o None per dataset unsupervised.
        feature_names: Ordine delle colonne nelle features.

    Example:
        >>> ds = AgriDataset(df_clean, numeric_columns=["temp", "ph"], target="yield")
        >>> loader = DataLoader(ds, batch_size=32, shuffle=True)
    """
```

### Verifica finale
```bash
python -m pytest --tb=short
```
Deve ancora dare **23 passed** (nessuna regressione — i docstring non cambiano comportamento).

### Commit
```bash
git add src/agripipe/indices.py src/agripipe/loader.py src/agripipe/tensorizer.py src/agripipe/dataset.py
git commit -m "docs: adopt Google-style docstrings on core modules"
```

---

## 🏁 Chiusura: Pull Request su GitHub

Quando hai completato tutti e 3 i task (Task 7, 8, 16):

1. Verifica lo stato finale:
   ```bash
   python -m pytest --tb=short
   ```
   Deve dare: **23 passed**.

2. Verifica che tutto sia committato:
   ```bash
   git status
   ```
   Deve dire `nothing to commit, working tree clean`.

3. Verifica i commit effettuati:
   ```bash
   git log origin/main..gemini-cli --oneline
   ```
   Devono esserci 3-4 commit (uno per task, eventualmente più piccoli per step TDD).

4. Push sul remote:
   ```bash
   git push origin gemini-cli
   ```

5. Apri la Pull Request:
   - URL: https://github.com/francesco5252/agripipe/compare/main...gemini-cli
   - **Titolo**: `feat: Task 7, 8, 16 — YAML presets + Google docstrings`
   - **Body** (copia e incolla questo template):
     ```markdown
     ## Summary

     Completa tre task meccanici del piano AgriPipe Pro Upgrade, assegnati a Gemini CLI per risparmiare crediti Claude sui task complessi.

     ## Task completati

     - **Task 7**: Aggiunto `region` e `crop_display` ai 5 preset regionali esistenti
     - **Task 8**: Aggiunti 7 nuovi preset (12 totali, 10 regioni italiane coperte) + 3 nuove colture biologiche (rice/apple/tomato) + `tests/test_presets_load.py` con 3 smoke test
     - **Task 16**: Docstring Google-style su `indices.py`, `loader.py`, `tensorizer.py`, `dataset.py`

     ## Test

     - Baseline prima: 20 passed
     - Baseline dopo: 23 passed (3 nuovi smoke test in `tests/test_presets_load.py`)
     - Comando: `python -m pytest --tb=short`

     ## Riferimenti

     - Plan originale: `docs/superpowers/plans/2026-04-18-agripipe-pro-upgrade.md`
     - Handoff doc: `docs/superpowers/handoff-gemini.md`
     ```

6. Una volta aperta la PR, **non mergere da solo** — aspetta review di Claude Code o dell'utente.

---

## Cosa fare se qualcosa va storto

| Situazione | Azione |
|---|---|
| Un test esistente fallisce dopo le tue modifiche | `git diff` per capire, poi ripristina la parte che rompe. Il YAML e i docstring sono additivi — non dovrebbero rompere nulla. |
| La PR ha conflitti con `main` | Probabilmente Claude ha mergiato qualcosa prima. `git fetch origin && git rebase origin/main` sul branch gemini-cli. |
| Dubbio semantico su un preset (es. `max_yield` per il pomodoro) | Usa i valori nel plan originale come source of truth. Non cambiarli. |
| Errore `yaml.safe_load` | Probabilmente indentazione sbagliata. Il YAML è sensibile a spazi — usa 2 spazi per livello, mai tab. |
| pytest non trova moduli | Usa `python -m pytest`, mai `pytest` da solo (differenza cruciale su Windows). |

## Risorse

- **Plan completo**: `docs/superpowers/plans/2026-04-18-agripipe-pro-upgrade.md` (sezioni Task 7, 8, 16 contengono tutti i dettagli, ma questo handoff è già self-contained)
- **Config base**: `configs/agri_knowledge.yaml` (stato post Task 1-3)
- **Struttura codice**: `src/agripipe/`
- **Test esistenti**: `tests/`

Buon lavoro! 🌿
