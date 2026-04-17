# AgriPipe Pro Upgrade — Design Spec

**Data**: 2026-04-17
**Autore**: Francesco (studente Agricoltura Sostenibile) + Claude
**Target**: Agritech X Farm — pipeline da Excel agronomico a tensor PyTorch ML-ready
**Stato**: Approvato, pronto per writing-plans

---

## 1. Contesto e obiettivo

AgriPipe è un tool che trasforma Excel agricoli "sporchi" in due output pronti all'uso:
- **Excel Pulito** per agronomi e gestionali
- **Bundle PyTorch** (`.pt` + `metadata.json`) pronto per training Machine Learning

Il prototipo attuale è funzionante: pipeline modulare (loader → cleaner → indices → dataset → tensorizer → report), UI Streamlit con selezione territoriale, 5 preset regionali italiani.

**Obiettivo di questo upgrade**: trasformare AgriPipe in un'applicazione professionale, visivamente curata e facilissima da usare per i dipendenti di X Farm. Filosofia guida, applicata a ogni decisione:

> *"Come posso far risparmiare tempo all'agronomo di X Farm oggi?"*

---

## 2. Scope (4 workstream, consegna unificata — Approccio A)

1. **UI/UX Restyle** — palette Clean & Nature, step numerati, messaggi motivazionali
2. **Imputazione time-series** nel cleaner — rispetto del ciclo colturale
3. **Sustainability Score Card** — 4 badge semaforo (Azoto, Peronospora, Irrigazione, Suolo)
4. **Robustezza tecnica** — bundle ML con `metadata.json`, docstring Google su tutte le funzioni pubbliche

L'ordine di esecuzione rispetta le dipendenze tecniche:
cleaner (diagnostics) → export (metadata) → theme/components → app.py (composizione) → UI/Scorecard.

---

## 3. Decisioni prese (e perché)

| # | Decisione | Scelta | Motivazione |
|---|---|---|---|
| D1 | Fallback interpolazione se manca `date` | Median + warning (A) | Tool deve restare "facilissimo da usare", non rompere su dati sporchi |
| D2 | Soglie Score Card | Semaforo pragmatico (A) | 0% è irrealistico; il giallo rappresenta "accettabile" |
| D3 | Numero badge | 4 (aggiunto Suolo) | Griglia 2×2 equilibrata visivamente |
| D4 | Selettore territorio | Cascata Regione → Coltura | UX più chiara, scalabile a 20 regioni |
| D5 | Copertura preset | 10 regioni, 12 preset (B) | Eccellenze DOP/DOCG, showcase credibile senza inventare dati |
| D6 | Strategia scope | Spec unificato, ordine dipendenze (A) | Upgrade coeso, una review, filosofia "professionale" |
| D7 | UI tooling | Skill `frontend-design` al build time | Qualità visiva professionale con CSS custom in Streamlit |

---

## 4. Architettura

### File toccati

| File | Azione |
|---|---|
| `configs/agri_knowledge.yaml` | Modifica: aggiungi `region`, `crop_display`, 7 preset nuovi + regole colture |
| `src/agripipe/cleaner.py` | Modifica: `"time"` strategy + `CleanerDiagnostics` + docstring |
| `src/agripipe/indices.py` | Modifica: docstring Google |
| `src/agripipe/loader.py` | Modifica: docstring Google |
| `src/agripipe/tensorizer.py` | Modifica: docstring Google |
| `src/agripipe/dataset.py` | Modifica: docstring Google |
| `src/agripipe/sustainability.py` | **Nuovo**: compute_scorecard + overall_message |
| `src/agripipe/export.py` | **Nuovo**: export_ml_bundle (pt + json + zip) |
| `src/agripipe/metadata.py` | **Nuovo**: build_metadata, save_metadata_json |
| `src/agripipe/ui/__init__.py` | **Nuovo**: package marker |
| `src/agripipe/ui/theme.py` | **Nuovo**: palette + inject_css |
| `src/agripipe/ui/components.py` | **Nuovo**: render_hero, render_step, render_scorecard, render_download_row |
| `src/agripipe/app.py` | Riscrittura composizione (~100 righe) |
| `src/agripipe/cli.py` | Integrazione bundle export |
| `tests/test_cleaner_time_interpolation.py` | **Nuovo** |
| `tests/test_sustainability.py` | **Nuovo** |
| `tests/test_export.py` | **Nuovo** |
| `tests/test_e2e.py` | Modifica: aggiungi training-readiness test |

### Principio di isolamento

Ogni modulo ha **una responsabilità**:
- `sustainability.py` → solo calcolo badge (zero I/O, zero Streamlit)
- `export.py` → solo bundling output
- `metadata.py` → solo costruzione/salvataggio metadata
- `ui/theme.py` → solo costanti visive + CSS injector
- `ui/components.py` → solo rendering componenti

La logica agronomica resta in `cleaner.py` / `indices.py` (single source of truth).

### Flusso dati post-modifica

```
Excel sporco
    ↓
load_raw() → df_raw
    ↓
AgriCleaner.clean()
    ├── df_clean (con indici GDD/Huglin/Drought/NUE)
    └── cleaner.diagnostics (conteggi violazioni)
         ↓
    ┌────┴────┬────────────┐
    ↓         ↓            ↓
Score      Excel       export_ml_bundle()
Card        pulito       ├── .pt (tensor bundle)
(UI)        (download)   ├── metadata.json
                         └── .zip (bundle ML)
```

---

## 5. Palette e sistema visivo (Clean & Nature)

### Palette

| Ruolo | Nome | Hex |
|---|---|---|
| Primario | Verde Salvia | `#7FA77F` |
| Primario scuro | Verde Bosco | `#3D5A3D` |
| Secondario | Marrone Terra | `#8B6F47` |
| Accento | Blu Acqua | `#4A90A4` |
| Sfondo | Bianco Latte | `#FAFAF7` |
| Card | Bianco Puro | `#FFFFFF` |
| Badge verde | Verde Foglia | `#6BAF6B` |
| Badge giallo | Giallo Grano | `#D4A64A` |
| Badge rosso | Rosso Melograno | `#B84A3E` |
| Testo | Antracite | `#2B2B2B` |
| Testo tenue | Grigio Pietra | `#6B6B6B` |

### Principi

- Padding generoso (1.5rem nelle card, 1rem gap tra elementi)
- Bordi sottili (`1px solid earth @ 20% opacity`) invece di ombre pesanti
- Border-radius 8px ovunque
- Max-width 1200px sul container
- Icone sempre accompagnate da testo (accessibilità)
- Font system-default (zero latenza, zero web font)

### Tipografia

- Titoli: weight 600, Verde Bosco
- Corpo: weight 400, Antracite, line-height 1.6
- Caption: 0.85rem, Grigio Pietra

### Tutti i valori vivono in

`src/agripipe/ui/theme.py` come costanti Python + funzione `inject_css()`. Cambiare un colore = una sola modifica.

### Implementazione UI

In fase di build, invocare la skill **`frontend-design`** per applicare principi di design system, micro-interazioni e accessibilità al CSS iniettato in Streamlit.

---

## 6. Layout UI (5 step lineari)

```
┌─ Hero: 🌱 AgriPipe — "Da Excel sporco a dati ML-ready in 30 secondi"
├─ Step 1 📍 Inquadramento Territoriale
│    Dropdown Regione → Dropdown Coltura → 3 card (suolo/resa/pH)
├─ Step 2 🌾 Dati del Campo
│    Banner motivazionale → File uploader → Bottone "Avvia Ottimizzazione"
├─ Step 3 🧪 Risultati
│    3 metriche → Download row [📥 Excel] [💾 Bundle ML (zip)]
├─ Step 4 🌱 Sustainability Score Card
│    Griglia 2×2 badge + messaggio di sintesi dinamico
└─ Step 5 📊 Analisi Visiva
     Expander per colonna: boxplot + distribuzione prima/dopo
```

### Selettore territorio (cascata)

Step 1 ha due dropdown in sequenza:
1. **🗺️ Regione** → elenca solo regioni con ≥1 preset
2. **🌾 Coltura** → filtra per regione selezionata, usa `crop_display`

Colture che si ripetono (Olivo Ligure vs Pugliese vs Toscano) applicano regole agronomiche **diverse** via `physical_bounds` del preset e la sezione biologica `knowledge.crops` del YAML (t_base, flowering_months, ecc.).

### Step 4: Sustainability Score Card

Griglia 2×2, ciascuna card contiene:
- Icona + titolo badge
- Pallino colorato (Verde/Giallo/Rosso) + etichetta
- Numero evidente della violazione (es. "2 eventi", "0%")
- Micro-consiglio agronomico in 1 riga

Sotto la griglia, messaggio di sintesi dinamico:
- 4 verdi → "🌱 Gestione esemplare: pratiche pienamente sostenibili"
- 2–3 verdi → "👍 Buona gestione con margine di miglioramento"
- ≤1 verde → "⚠️ Rivedi le aree critiche per allinearti agli standard"

---

## 7. Logica cleaner (time interpolation + diagnostics)

### A. Nuova strategia `"time"`

```python
ImputationStrategy = Literal["mean", "median", "ffill", "drop", "time"]
```

### B. Comportamento

Quando `missing_strategy == "time"` e `date_col` è presente:
1. Ordina per `(field_col, date_col)` se possibile
2. Per ogni campo (groupby), interpola via `method="time"` con `limit=3`
3. `ffill().bfill()` finale per bordi

Fallback cascade:
- `time` + date + field → interpolazione per-field ✅
- `time` + date, no field → interpolazione globale ⚠️ warning
- `time` senza date → fallback median, warning
- `time` con <3 righe → fallback median

### C. `CleanerDiagnostics`

Dataclass popolata durante `clean()`. Campi:

```
total_rows, imputation_strategy_used, values_imputed,
outliers_removed, out_of_bounds_removed,
nitrogen_violations, peronospora_events, irrigation_inefficient,
soil_organic_low, heat_stress_flowering, late_frost_events
```

Reset a ogni chiamata `clean()`. **Non cambia il comportamento**, aggiunge solo esposizione dati per Score Card e metadata.

### D. Retrocompatibilità

Default resta `"median"`. `"time"` è opt-in. Test esistenti passano senza modifiche.

---

## 8. Sustainability Scorecard

### Modulo `sustainability.py`

Funzione pura, zero I/O:

```python
def compute_scorecard(diagnostics, total_rows) -> dict[str, Badge]:
    """Calcola i 4 badge dalle diagnostics del cleaner."""
```

### Soglie

| Badge | Metrica | Verde | Giallo | Rosso |
|---|---|---|---|---|
| 💧 Azoto | `nitrogen_violations / total_rows` | 0% | ≤5% | >5% |
| 🍇 Peronospora | `peronospora_events` | 0 | 1–3 | >3 |
| 🚿 Irrigazione | `irrigation_inefficient / total_rows` | 0% | ≤10% | >10% |
| 🌰 Suolo | `soil_organic_low / total_rows` | 0% | ≤15% | >15% |

`overall_message(badges)` sintetizza lo stato complessivo.

---

## 9. Export ML Bundle

### Modulo `export.py`

```python
def export_ml_bundle(df_clean, cleaner, preset, output_dir, name) -> dict[str, Path]:
    """Crea .pt + metadata.json + zip nella output_dir."""
```

### Contenuto `.pt`

```python
torch.save({
    "features": Tensor[N, D] float32,
    "target":   Tensor[N]    float32,
    "feature_names": list[str],
    "scaler_mean":   Tensor[D],
    "scaler_scale":  Tensor[D],
}, pt_path)
```

Bundle completo per `DataLoader` senza ulteriore preparazione.

### Contenuto `metadata.json`

```
schema_version: 1
generated_at: ISO-8601 timestamp
dataset_info: {name, rows, features, target, target_unit, task}
columns: [{name, index, unit, description, normalized}, ...]
agronomic_context: {crop, region, zona, cleaning_rules[]}
pytorch_usage: {example_code}
```

### Bundle Zip

Un singolo `.zip` con `.pt` + `.json` per download dall'UI. Un click, zero confusione.

---

## 10. Preset colture (10 regioni, 12 preset)

Curati da fonti agronomiche note (CREA, ARPA, disciplinari DOP/DOCG/IGP):

| Regione | Preset | Note |
|---|---|---|
| Liguria | Olivo DOP Taggiasca | esistente |
| Piemonte | Vite Nebbiolo (Barolo DOCG) | esistente |
| Lombardia | Riso Carnaroli | 🆕 Lomellina |
| Veneto | Vite Prosecco DOCG | 🆕 Valdobbiadene |
| Trentino-Alto Adige | Mela Melinda DOP | 🆕 Val di Non |
| Emilia-Romagna | Grano Tenero | esistente |
| Emilia-Romagna | Pomodoro da industria | 🆕 |
| Toscana | Vite Chianti DOCG | 🆕 |
| Toscana | Olivo DOP | 🆕 |
| Campania | Pomodoro San Marzano DOP | 🆕 |
| Puglia | Olivo intensivo | esistente |
| Sicilia | Grano Duro Antico | esistente |

Per ogni coltura nuova aggiungo due blocchi al YAML:
1. Entry in `regional_presets` (territorio): `region`, `crop`, `crop_display`, `zona`, `suolo_tessitura`, `max_yield`, `ideal_ph`, `temp_range`, `note`.
2. Entry in `crops` (biologia): `t_base`, `max_yield`, `flowering_months`, `frost_danger_months`, `critical_temp_flowering` + regole specifiche (es. `rule_10_temp` / `rule_10_rain` per Peronospora vite).

---

## 11. Docstring style (Google)

Tutte le funzioni pubbliche e le classi di questi moduli ricevono docstring Google-style:
`cleaner.py`, `indices.py`, `loader.py`, `tensorizer.py`, `dataset.py`, `sustainability.py`, `export.py`, `metadata.py`, `ui/theme.py`, `ui/components.py`.

Pattern: descrizione breve, sezione Args, Returns, Raises (dove applicabile), Example (per le funzioni più rilevanti).

---

## 12. Test plan

### Nuovi file

- **`tests/test_cleaner_time_interpolation.py`**: 3 test
  - Interpolation rispetta i boundary dei campi
  - Fallback a median quando manca `date`
  - Non interpola buchi oltre `limit=3`

- **`tests/test_sustainability.py`**: 6 test
  - Una soglia per badge (green/yellow/red)
  - `overall_message` per ogni caso (4/3/1 verdi)

- **`tests/test_export.py`**: 3 test
  - Bundle `.pt` contiene tutti i campi attesi
  - `metadata.json` è JSON valido e copre tutte le colonne
  - Zip contiene entrambi i file

### Modifiche

- **`tests/test_e2e.py`**: aggiungo `test_exported_bundle_is_training_ready`
  - Simula un training reale: `DataLoader → nn.Linear → mse_loss` → asserisci `.isfinite()`

### Copertura attesa

≥85% per i moduli nuovi, 100% per `sustainability.py` (funzione pura critica).

---

## 13. Rischi e mitigazioni

| Rischio | Probabilità | Mitigazione |
|---|---|---|
| Test E2E esistenti si rompono con nuova strategy | Bassa | Default resta `"median"`, `"time"` è opt-in |
| CSS Streamlit si comporta in modo inconsistente tra browser | Media | Uso solo proprietà CSS standard, test manuale su Chrome/Firefox |
| Preset nuovi hanno regole agronomiche sbagliate | Media | Cito fonti nel YAML (es. "CREA 2023"), revisione da studente Agricoltura |
| metadata.json troppo verboso/rumoroso | Bassa | Schema JSON minimale ma completo, versionato (`schema_version: 1`) |
| Export `.pt` non caricabile su altre macchine | Bassa | Test E2E carica e simula training |

---

## 14. Success criteria

L'upgrade è considerato riuscito quando:

1. ✅ Un dipendente X Farm può caricare un Excel, scaricare un bundle ML funzionante in <1 minuto senza istruzioni
2. ✅ Un Data Scientist X Farm apre il `.pt` + `metadata.json` e avvia il training in <3 righe di codice
3. ✅ La Score Card comunica lo stato di sostenibilità in <10 secondi di lettura
4. ✅ Tutte le funzioni pubbliche hanno docstring Google leggibili
5. ✅ Test suite passa al 100%, copertura ≥85% sui moduli nuovi
6. ✅ Il progetto rende evidente il valore aggiunto rispetto a una pipeline generica (regole territoriali italiane)

---

## 15. Prossimo passo

Dopo l'approvazione utente di questo spec: invocare la skill `superpowers:writing-plans` per generare il piano di implementazione dettagliato con task ordinati, criteri di acceptance e file checklist.
