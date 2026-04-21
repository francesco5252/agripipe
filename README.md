# 🌱 AgriPipe

[![CI](https://github.com/francesco5252/agripipe/actions/workflows/ci.yml/badge.svg)](https://github.com/francesco5252/agripipe/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Da Excel agronomico sporco a tensor PyTorch validati. Tre step, riproducibili, tracciabili.**

> 🇬🇧 English version: [README.en.md](README.en.md)

---

## 🎯 Per chi è questo progetto

AgriPipe nasce per colmare un vuoto concreto nel mondo dell'agricoltura digitale: la distanza fra i **dati raccolti in campo** (sensori, registri cartacei digitalizzati, fogli Excel compilati a mano) e il **formato rigido richiesto dai modelli di Machine Learning**.

È pensato per tre profili di utente:

- **👨‍🔬 Data scientist e ricercatori agritech** che ricevono Excel agronomici di qualità variabile e devono trasformarli in dataset ML-ready in modo riproducibile.
- **🎓 Studenti di agronomia, scienze ambientali e agricoltura sostenibile** che vogliono portare un dataset reale a un modello PyTorch senza scrivere codice di pulizia da zero.
- **🌾 Operatori agritech e sviluppatori di aziende del settore** (come X Farm) che hanno bisogno di una pipeline prevedibile e auditabile per alimentare i propri modelli di previsione della resa.

Non serve essere esperti di PyTorch per usarlo: la UI Streamlit copre tutto il flusso con pochi click.

---

## 💡 Il problema che risolve

Un Excel agronomico tipico è un campo minato:

- Date in formato *seriale Excel* (`45123` invece di `2024-01-15`).
- Umidità registrata come `150%` (impossibile fisicamente).
- Tre o quattro righe di intestazione aziendale prima del vero header.
- Righe duplicate per errori di sincronizzazione dei sensori.
- Valori separatore decimale `,` invece di `.` (retaggio italiano).
- NaN sparsi ovunque, a volte indicati con `-`, `n.d.`, o celle vuote.

Fare Machine Learning su dati così richiede **ore di pulizia manuale** e introduce bug silenziosi difficili da rintracciare. AgriPipe automatizza tutto il processo in una pipeline trasparente a 3 step, generando un bundle `.zip` auto-documentato con tensor PyTorch, metadata JSON e parametri dello scaler pronti per la fase di training o inferenza.

---

## 🚀 Come si usa

AgriPipe offre due modalità d'uso, entrambe supportate:

### 🖥 Via UI Streamlit (consigliato per esplorare)

```bash
streamlit run app.py
```

Si apre una web app a 3 step: carichi il file, configuri la pulizia, scarichi il bundle `.zip`. Zero righe di codice.

![AgriPipe UI — 3 step](docs/screenshots/agripipe_ui.png)

### ⚙️ Via CLI (consigliato per pipeline automatiche)

```bash
# Pulizia + tensorizzazione con preset regionale
agripipe run --input dati.xlsx --preset ulivo_pugliese --output model_input.pt

# Export bundle ML completo (.pt + .json + .zip)
agripipe run -i dati.xlsx -p vite_piemontese -e ./export/

# Generazione di dati sintetici per test
agripipe generate --rows 1000 --output data/synthetic.xlsx
```

Esegui `agripipe --help` per la lista completa dei comandi.

---

## 📦 Cosa produce

Alla fine della pipeline ottieni un archivio **`<nome>.zip`** che contiene:

| File | Contenuto |
|------|-----------|
| `<nome>.pt` *(o `<nome>_train.pt`, `_val.pt`, `_test.pt` se attivi lo split)* | Bundle PyTorch con `features`, `target`, `feature_names`, `scaler_mean`, `scaler_scale`, `metadata` |
| `<nome>.json` | Manifest completo: schema, unità, statistiche per colonna, correlazioni, diagnostica pulizia, esempio PyTorch |

Il tutto è tracciabile: il `metadata.json` include l'hash SHA-256 del file sorgente e uno `schema_lock_hash` che ti permette di verificare quando un dataset cambia forma.

### Caricamento in PyTorch (5 righe)

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

bundle = torch.load("agripipe_export.pt", weights_only=False)
dataset = TensorDataset(bundle["features"], bundle["target"])
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

Il modello PyTorch è pronto per l'addestramento senza ulteriori trasformazioni.

---

## 🇮🇹 Atlante Agronomico Italiano Integrato

AgriPipe non è più solo un tool statistico: ora include una base di conoscenza agronomica che copre l'intero territorio nazionale. Grazie all'**Atlante Agronomico Integrato**, il sistema è in grado di validare i dati non solo numericamente, ma biologicamente.

L'Atlante comprende oltre **50 preset regionali** iper-localizzati, tra cui:
- **Nord:** Riso Vercellese/Novarese (suoli acidi vs argillosi), Nebbiolo delle Langhe vs Valtellina, Mele del Trentino, Radicchio di Treviso (coltura invernale).
- **Centro:** Sangiovese del Chianti e Brunello (suoli Galestro/Alberese), Zafferano dell'Aquila (alta quota), Kiwi di Latina, Tabacco Kentucky.
- **Sud e Isole:** Pomodoro San Marzano DOP, Olivo Coratina pugliese, Bergamotto reggino, Vite dell'Etna (suoli vulcanici acidi), Vermentino di Gallura (granito).

Ogni preset applica automaticamente:
- **Validazione Temporale:** Azzeramento rese fuori dalle finestre di raccolta reali.
- **Identità del Suolo:** Check di coerenza su pH e tessitura (es. sassi, argille, tufi).
- **Soglie di Magnitudo:** Limiti di resa calibrati sui disciplinari DOCG/IGP reali.

---

## 🏗 Come funziona: i 4 motori

```
┌─────────────┐   ┌─────────────┐   ┌──────────────┐   ┌────────────────┐
│ Excel / CSV │──▶│  1. LOADER  │──▶│  2. CLEANER  │──▶│ 3. TENSORIZER  │──▶ .pt + .json + .zip
└─────────────┘   └─────────────┘   └──────────────┘   └────────────────┘
                    Fuzzy Match       Validazione        Scaling
                    Batch Load        Agronomica         Encoding cat.
                    Unit Conv.        Imputazione        Train/Val/Test
```

1. **Loader** — Legge Excel o CSV, gestisce il **batch loading** da intere cartelle, applica il **fuzzy matching** per riconoscere colonne scritte male o in italiano, e converte automaticamente le unità (es. Fahrenheit → Celsius).

2. **Cleaner** — Il "cuore agronomico". Oltre alla pulizia statistica (IQR/Z-score), applica le regole dell'Atlante Italiano per eliminare dati biologicamente impossibili.

3. **Tensorizer** — Scala le feature e codifica le variabili categoriche, generando tensor pronti per PyTorch.

---

## 🛠 Installazione

```bash
# Clona il repository
git clone https://github.com/francesco5252/agripipe.git
cd agripipe

# Installa in modalità sviluppo (include dipendenze di test)
pip install -e ".[dev]"
```

Requisiti: **Python 3.10+**, sistema operativo qualsiasi (testato su Windows, Linux, macOS).

---

## 🧪 Sviluppo e test

Il progetto segue una disciplina TDD con test rigorosi:

```bash
pytest                        # 38 test, ~82% coverage
ruff check src tests app.py    # linting
black --check src tests app.py # formattazione
```

La CI GitHub Actions esegue automaticamente test + lint su Python 3.10, 3.11 e 3.12 a ogni push.

---

## ⚠️ Limiti noti (onestà intellettuale)

Conoscere i limiti di uno strumento è parte della sua qualità. AgriPipe **non fa**:

- **Fuzzy matching dei nomi colonna** — lo schema minimo (`date`, `field_id`, `temp`, `humidity`, `ph`, `yield`) è obbligatorio. Se nel tuo Excel la colonna si chiama `Temperatura_C`, devi rinominarla prima.
- **Conversione di unità di misura** — niente Fahrenheit → Celsius, niente pollici → mm. I dati si assumono già nelle unità canoniche (SI dove possibile).
- **Batch loading da cartelle** — un file alla volta. La combinazione di più file è una scelta di workflow esterno.
- **Modelli agronomici interpretativi** — nessun indice di sostenibilità, nessuna scorecard "green/yellow/red". AgriPipe produce dati puliti, non giudizi agronomici. Questa era una scelta di design: separare la preparazione del dato dall'interpretazione.
- **Imputazione ML-based (KNN, MICE)** — resta su metodi statistici classici per trasparenza e riproducibilità.

Queste esclusioni sono **intenzionali**: mantengono la pipeline prevedibile, debuggabile e facile da validare scientificamente.

---

## 🗺️ Roadmap & contributi

Dove sta andando AgriPipe: [`ROADMAP.md`](ROADMAP.md) — visione a 3 orizzonti (0-3 mesi, 3-12 mesi, 12+ mesi).

Vuoi contribuire? Le task pronte da prendere in mano sono le [good first issues](https://github.com/francesco5252/agripipe/labels/good-first-issue). Per il setup di sviluppo in locale vedi [`docs/contributing.md`](docs/contributing.md).

---

## 📄 Licenza

Distribuito sotto licenza **MIT**. Vedere il file [`LICENSE`](LICENSE) per i dettagli.

---

<sub>Progetto sviluppato con un approccio ML-Ops rigoroso. Per il percorso di sviluppo completo passo-passo, consulta [`DOCUMENTAZIONE_LOG.md`](DOCUMENTAZIONE_LOG.md).</sub>
