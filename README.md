# 🌱 AgriPipe

> Data pipeline salva-tempo per dati agronomici: da Excel sporco a tensor PyTorch pronti per il ML.

[![CI](https://github.com/YOUR_USER/agripipe/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USER/agripipe/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Perché

Chi lavora su ML in agritech passa troppo tempo a pulire Excel malformati, gestire NaN e convertire dataframe in tensor. **AgriPipe** automatizza tutto in un solo comando e riduce al minimo i crash a runtime.

## Pipeline

```
Excel grezzo ──► Loader ──► Cleaner ──► Tensorizer ──► torch.Tensor
                   │           │            │
                 validazione  outlier    normalize
                   schema     missing    encode
                              dedup      split
```

## Quickstart

```bash
pip install -e .

# 1. Genera un Excel sporco di esempio (se non ne hai uno tuo)
agripipe generate --output data/sample/synthetic_dirty.xlsx --rows 500

# 2. Pipeline completa con report HTML di qualità
agripipe run -i data/sample/synthetic_dirty.xlsx -o out/tensors.pt -r out/report.html

# 3. Solo report di qualità, senza tensor
agripipe report -i data/sample/synthetic_dirty.xlsx -o out/report.html
```

Vedi anche `notebooks/demo.ipynb` per un tour end-to-end interattivo.

In Python:

```python
from agripipe import load_raw, AgriCleaner, AgriDataset

df = load_raw("data/raw.xlsx")
df = AgriCleaner.from_yaml("configs/default.yaml").clean(df)
ds = AgriDataset(df, target="yield")
```

## Struttura

```
src/agripipe/     # codice core
tests/            # unit test pytest
configs/          # YAML di configurazione pipeline
data/sample/      # Excel di esempio per test end-to-end
```

## Sviluppo

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

## Licenza

MIT
