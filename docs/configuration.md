# Configurazione Pipeline

AgriPipe utilizza file YAML per definire le regole di pulizia e trasformazione. Questo permette di mantenere la logica di business separata dal codice.

## Struttura dello YAML

Un file di configurazione tipico si trova in `configs/default.yaml`. Ecco i parametri principali:

### 1. `cleaner`
Definisce come pulire le colonne.

```yaml
cleaner:
  columns:
    yield:
      outlier_strategy: iqr  # iqr o zscore
      impute_strategy: mean  # mean, median, zero
      min_val: 0             # Limite fisico minimo
      max_val: 100           # Limite fisico massimo
    moisture:
      outlier_strategy: null # Nessuna rimozione outlier
      impute_strategy: median
```

### 2. `tensorizer`
Definisce come preparare i dati per PyTorch.

```yaml
tensorizer:
  target_column: "yield"
  test_size: 0.2
  scaler: "standard" # standard o minmax
```

## Esempio Completo

Puoi trovare un esempio completo in `configs/default.yaml` nella root del progetto.
