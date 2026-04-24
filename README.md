# AgriPipe: Data-to-Tensor

> 🇬🇧 **[Read this in English →](README.en.md)**

AgriPipe è IL prototipo di una  piattaforma **MLOps** ideata per traghettare i dati agronomici grezzi (provenienti da excel, droni, stazioni meteo o operatori in campo) allo stato di ecosistemi vettoriali blindati (`.pt` PyTorch tensors) per addestramenti previsionali in Deep Learning.

---

## 1. Per chi è questo progetto

* **Data Scientist e AI Engineers nel panorama AgriTech**: che necessitano di iterare rapidamente e scalare su modelli pesanti senza perdersi nella giungla dei preprocessing "ad-hoc" ogni volta che arriva un nuovo Excel o cambiano i sensori.
* **Agronomi Digitali & Centri di Ricerca**: a cui serve un'interfaccia usabile e garantita a garanzia della storicizzazione del test che validi automaticamente dati sporchi applicando regole di pulizia agronomica.
* **Team Operations**: che mirano a standardizzare un bridge perfetto tra chi raccoglie in campo e chi sviluppa i layer inferenziali in Python.

---

## 2. Il problema che risolve

I dati agricoli raw sono caotici: l'umidità viene intesa come ratio (`0.2`) ed espressa l'anno dopo in percetuale (`20%`), il pH spesso presenta errori stratosferici di battitura (`pH 45.0` anziché `4.5`), e sono gremiti di _Not-a-Number (NaN)_.

Gestire a mano e separatamente questi bug tra i laboratori ed il team Machine Learning rallenta lo scaling algoritmico o porta i modelli a overfittare rumore statistico.
**AgriPipe risolve il collo di bottiglia del preprocessing manuale**: garantisce una pipeline type-safe e modulare che esegue data-cleaning massivo preservando l'integrità del senso agronomico e validando i file per creare split (train, val, test) sicuri e inalterabili tramite standardizzazione `scikit-learn` in tempo reale.

---

## 3. Come si usa

Il repository è concepito attorno a una User Interface solida tramite **Streamlit** (multipage architecture).
Dalla cartella del progetto basta avviare:

```bash
streamlit run app.py
```

Il browser ti accompagnerà in tre pagine, disposte in step rigidi per impedire misconfigurazioni:
1. `1_📥_Ingestion`: fai upload del tuo CSV / Excel `(xls/xlsx)`. Esplora l'anteprima visiva raw e verifica l'hash di ingresso del file.
2. `2_🧹_Refinery`: esegui i wizard dinamici scegliendo criteri di outlier detection (IQR, Z-Score o None), imputazione per missing values e limiti fisici di tolleranza customizzati per impedire al Deep Learning di interpretare assurdi numerici biologici.
3. `3_📦_Tensorizer`: qui affetti il dataset nei canonici set (Training, Validation e Testing), decidi lo Scaler desiderato (`Standard` / `Robust`) e gli applichi i Tensorizer categorical/numeric. Infine l'interfaccia estrarrà e ti farà scaricare un file `.zip` auto-contenuto.

---

## 4. Cosa produce

Alla fine del ciclo di processamento, generi il **Machine Learning Bundle**.
È uno `.zip` esatto che tu potrai spedire allo sviluppo modello con i seguenti output:
* **Tensor Multipli** (`_train.pt`, `_val.pt`, `_test.pt`): Oggetti serializzati Pytorch composti dai vettori di `features`, la label del `target`, e persino `scaler_mean / scaler_scale`. Tutto calcolato live assecondando zero Data Leakage (i test vengono calcolati sugli estimatori del train!).
* **Manifest `metadata.json`**: Uno spaccato indelebile dell'algoritmo che hai lanciato. Mostra in chiaro i dropout, le feature selezionate, i preset algoritmici applicati originariamente.
* **Logging Tracciato su MLflow (Dietro le quinte)**: Agripipe valuta tacitamente l'integrità matematica del pacchetto finale calcolando un safety benchmark (Ridge Regression baseline) e tracciando i record tramite runtime MLOps a latere se `mlflow` è attivo.

---

## 5. Come funzionano i 3 Motori (Loader, Cleaner, Tensorizer)

Dietro l'interfaccia amichevole, Agripipe implementa architettura enterprise rigorosamente controllata da Pydantic:

* **Engine I: Loader (`src/agripipe/loader.py`)**
  Il demone dell'ingestione. Sfrutta il fuzzy matching e una profonda intelligenza lessicale per riconoscere sinonimi (Es: se il file legge `data_raccolta`, il Loader coercisce internamente la colonna in `date`). Supporta il **batch loading** da intere cartelle (`--input-dir`) e la **conversione automatica a unità SI** (Fahrenheit → Celsius, inch → mm, lb/acre → kg/ha) via flag `--auto-units`. Gestisce cache hashing.

* **Engine II: Cleaner (`src/agripipe/cleaner.py` / `transformers.py`)**
  È il colosso statistico disassemblato in una pipeline scikit-learn. Sotto al cofano ci sono **11 moduli seriali disaccoppiati** ("Transformers"), in ordine formale: coercizione date, Auto-Unit conversion, bound checker `pydantic` su scale fisiche del pH/Temp/Hum, rilevamento outlier (IQR o Z-score), imputazione numerica (`median` / `mean` / `ffill` / interpolazione temporale), imputazione categorica via moda, deduplicazione. Ritorna il DataFrame sanificato e pronto. Inietta il calcolo dinamico dei Gradi Giorno Accumulati (GDD) se impostata la base biologica.

* **Engine III: Tensorizer (`src/agripipe/tensorizer.py` & `export.py`)**
  La cerniera tra Pandas Analytics e PyTorch AI. Abbatte e rimpiazza la pipeline string/integer categorica in matrici normalizzate. Scalifica tutto incapsulando il modello di `StandardScaler` o `RobustScaler`. L'istanza generata è un modulo incapsulato e sicuro esportabile. In coda alla tensorizzazione, `tracking.py` calcola un **safety benchmark** (baseline Ridge Regression) con logging opzionale su **MLflow**: se il baseline non predice nulla, il problema è nei dati, non nel modello.

---

## 6. Istruzioni per l'installazione

Agripipe abbraccia `pyproject.toml` per l'installazione nativa delle dipendenze. Usa Python 3.11+.

```bash
# 1. Clona il codice
git clone https://github.com/francesco5252/agripipe.git
cd agripipe

# 2. Imposta un ambiente pulito
python -m venv venv

# Su macOS/Linux:    source venv/bin/activate
# Su Windows:        venv\Scripts\activate

# 3. Installa il repository in editable mode (include tutte le app deps)
pip install -e "."

# 4. Compila (opzionale se sei uno sviluppatore backend della piattaforma, in quel caso lancia `pip install -e ".[dev]"`)

# 5. Avvia il terminale Data-to-Tensor
streamlit run app.py
```
