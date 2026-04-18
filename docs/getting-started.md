# Guida Rapida

Questa guida ti aiuterà a configurare ed eseguire AgriPipe per la prima volta.

## 1. Generazione di Dati Sintetici

Se non hai ancora un dataset, AgriPipe può generarne uno "sporco" per testare la pipeline:

```bash
agripipe generate --output data/sample/test_data.xlsx --rows 100
```

## 2. Esecuzione della Pipeline

Il comando `run` esegue il caricamento, la pulizia e la generazione dei tensor in un unico step:

```bash
agripipe run -i data/sample/test_data.xlsx -o output/tensors.pt -r output/report.html
```

### Parametri principali:
- `-i, --input`: Percorso del file Excel/CSV di input.
- `-o, --output`: Dove salvare i tensor PyTorch (`.pt`).
- `-r, --report`: (Opzionale) Percorso per il report HTML di qualità dei dati.

## 3. Utilizzo in Python

Puoi anche integrare AgriPipe direttamente nei tuoi script:

```python
from agripipe.loader import load_raw
from agripipe.cleaner import AgriCleaner

# Caricamento
df = load_raw("mio_file.xlsx")

# Pulizia basata su config YAML
cleaner = AgriCleaner.from_yaml("configs/default.yaml")
df_clean = cleaner.clean(df)
```
